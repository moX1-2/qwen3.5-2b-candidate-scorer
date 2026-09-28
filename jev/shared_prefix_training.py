"""保留梯度的单题公共前缀缓存。每次参数更新后重新计算，不跨题缓存。"""
import copy
import gc
import os
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, LinearAttentionLayer
from common import Example, branch_prompts
from PIL import Image
from decision_template import render_image_decision


class FunctionalLinearLayer(LinearAttentionLayer):
    # 原生缓存 copy_ 会修改反向传播所需状态；训练缓存采用函数式更新。
    def update_conv_state(self, states, state_idx=0, conv_kernel_size=None, **kwargs):
        size = conv_kernel_size or self.conv_kernel_size.get(state_idx) or states.shape[-1]
        if self.has_previous_state.get(state_idx, False):
            full = torch.cat([self.conv_states[state_idx], states], dim=-1)
        else:
            full = states
            if full.shape[-1] < size:
                full = F.pad(full, (size - full.shape[-1], 0))
        self.conv_states[state_idx] = full[..., -size:].clone()
        self.conv_kernel_size[state_idx] = size
        self.is_conv_states_initialized[state_idx] = True
        self.has_previous_state[state_idx] = True
        self.device, self.dtype = states.device, states.dtype
        return full

    def update_recurrent_state(self, states, state_idx=0, **kwargs):
        self.recurrent_states[state_idx] = states.clone()
        self.is_recurrent_states_initialized[state_idx] = True
        return self.recurrent_states[state_idx]


def training_cache(config):
    cache = DynamicCache(config=config)
    for i, layer in enumerate(cache.layers):
        if isinstance(layer, LinearAttentionLayer):
            new = FunctionalLinearLayer()
            new.__dict__.update(copy.copy(layer.__dict__))
            cache.layers[i] = new
    return cache


def clone_cache(cache):
    result = copy.copy(cache)
    result.layers = []
    for layer in cache.layers:
        new = copy.copy(layer)
        for key, value in layer.__dict__.items():
            if torch.is_tensor(value):
                setattr(new, key, value.clone())
            elif isinstance(value, dict):
                setattr(new, key, {k: v.clone() if torch.is_tensor(v) else v for k, v in value.items()})
        result.layers.append(new)
    return result


def encode_branches(processor, record, data_root, device, max_length, mm_template='qwen'):
    if mm_template not in ('qwen', 'decision'):
        raise ValueError(f'未知图文模板: {mm_template}')
    example = Example(record['id'], record['description'], tuple(record['candidates']),
                      record['gold_index'], record.get('language', 'en'),
                      record.get('source_dataset', 'unknown'), record.get('domain', 'unknown'))
    texts = branch_prompts(example)
    image_path = record.get('image') or record.get('image_path')
    image = None
    if image_path:
        with Image.open(data_root / image_path) as original:
            image = original.convert('RGB')
    encoded = []
    for text in texts:
        if image is not None:
            if mm_template == 'decision':
                text = render_image_decision(processor, text)
            else:
                text = processor.apply_chat_template([{'role': 'user', 'content': [
                    {'type': 'image'}, {'type': 'text', 'text': text}]}], tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[text], images=[image], return_tensors='pt', add_special_tokens=False)
        else:
            inputs = processor.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        if inputs.input_ids.shape[-1] > max_length:
            raise ValueError(f"{example.id}: 长度 {inputs.input_ids.shape[-1]} 超过 {max_length}，禁止截断候选")
        inputs._prefix_cpu_ids = inputs.input_ids[0].tolist()
        encoded.append(inputs.to(device, non_blocking=True))
    return encoded


def positions(model, inputs):
    result = model.compute_3d_position_ids(
        input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'),
        inputs_embeds=None,
        image_grid_thw=inputs.get('image_grid_thw'),
        mm_token_type_ids=inputs.get('mm_token_type_ids'))
    if result is None:
        result = torch.arange(inputs['input_ids'].shape[-1], device=inputs['input_ids'].device)[None, None].expand(3, 1, -1)
    return result


def score_question(model, head, encoded, shared=False, cache_alignment=1, cache_min_suffix=0):
    pos = [positions(model, item) for item in encoded]
    if not shared:
        reuse_vision = (os.environ.get('JEV_REUSE_VISION', '0') == '1'
            and all(item.get('pixel_values') is not None for item in encoded)
            and not model.visual.training and not any(p.requires_grad for p in model.visual.parameters()))
        if reuse_vision:
            first = encoded[0]
            for item in encoded[1:]:
                if not torch.equal(first['pixel_values'], item['pixel_values']) or not torch.equal(first['image_grid_thw'], item['image_grid_thw']):
                    raise ValueError('同题图像特征复用要求图像与网格完全一致')
            features = model.get_image_features(first['pixel_values'], first['image_grid_thw'], return_dict=True)
            image_features = torch.cat(features.pooler_output, dim=0)
            vectors = []
            for item, p in zip(encoded, pos):
                embed = model.get_input_embeddings()(item['input_ids'])
                mask, _ = model.get_placeholder_mask(item['input_ids'], inputs_embeds=embed, image_features=image_features)
                embed = embed.masked_scatter(mask, image_features.to(embed))
                out = model.language_model(inputs_embeds=embed, attention_mask=item.get('attention_mask'), position_ids=p, use_cache=False)
                vectors.append(out.last_hidden_state[0, -1])
        else:
            vectors = [model(**item, position_ids=p, use_cache=False).last_hidden_state[0, -1]
                       for item, p in zip(encoded, pos)]
    else:
        if model.language_model.is_gradient_checkpointing:
            raise ValueError('共享 KV 训练不能启用梯度检查点')
        ids = [getattr(item, '_prefix_cpu_ids', None) or item['input_ids'][0].detach().cpu().tolist() for item in encoded]
        prefix_len = min(len(x) for x in ids) - 1
        for i in range(prefix_len):
            if any(x[i] != ids[0][i] for x in ids[1:]):
                prefix_len = i
                break
        if prefix_len < 1:
            raise ValueError('无公共前缀')
        if cache_alignment > 1:
            prefix_len = (prefix_len // cache_alignment) * cache_alignment
            while prefix_len > cache_alignment and min(len(x) for x in ids) - prefix_len < cache_min_suffix:
                prefix_len -= cache_alignment
            if prefix_len == 0:
                return score_question(model, head, encoded, shared=False)
        if any(item.get('image_grid_thw') is not None or item.get('video_grid_thw') is not None for item in encoded):
            for p in pos[1:]:
                if not torch.equal(pos[0][..., :prefix_len], p[..., :prefix_len]):
                    raise ValueError('公共前缀位置编码不一致')
        cache = training_cache(model.config.text_config)
        embeddings = None
        if cache_alignment > 1:
            # 完整图像特征只算一次，再允许在视觉 token 内按分块边界切分。
            image_features = None
            first = encoded[0]
            if first.get('pixel_values') is not None:
                features = model.get_image_features(first['pixel_values'], first['image_grid_thw'], return_dict=True)
                image_features = torch.cat(features.pooler_output, dim=0)
            embeddings = []
            for item in encoded:
                embed = model.get_input_embeddings()(item['input_ids'])
                if image_features is not None:
                    if not torch.equal(first['image_grid_thw'], item['image_grid_thw']):
                        raise ValueError('候选分支图像网格不同')
                    mask, _ = model.get_placeholder_mask(item['input_ids'], inputs_embeds=embed, image_features=image_features)
                    embed = embed.masked_scatter(mask, image_features.to(embed))
                embeddings.append(embed)
            out = model.language_model(inputs_embeds=embeddings[0][..., :prefix_len, :],
                attention_mask=first['attention_mask'][..., :prefix_len], position_ids=pos[0][..., :prefix_len],
                past_key_values=cache, use_cache=True)
        else:
            prefix = dict(encoded[0])
            for key in ('input_ids', 'attention_mask', 'mm_token_type_ids', 'token_type_ids'):
                if key in prefix:
                    prefix[key] = prefix[key][..., :prefix_len]
            out = model(**prefix, position_ids=pos[0][..., :prefix_len], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        del out  # 保留缓存计算图，释放不参与评分的前缀输出。
        if torch.is_grad_enabled() and model.training:
            states = []
            for layer in cache.layers:
                for name in ('keys', 'values'):
                    tensor = getattr(layer, name, None)
                    if torch.is_tensor(tensor) and tensor.numel():
                        states.append(tensor)
                for name in ('conv_states', 'recurrent_states'):
                    states.extend(t for t in getattr(layer, name, {}).values() if torch.is_tensor(t))
            if not states or not all(t.requires_grad for t in states):
                raise RuntimeError('共享训练的公共前缀缓存未完整保留计算图')
        vectors = score_suffixes(model, encoded, pos, cache, prefix_len, embeddings)
    return head(torch.stack(vectors).float()).squeeze(-1)



def expand_branch_cache(cache, count):
    """复制到候选 batch，repeat 保留前缀梯度累加。"""
    result = copy.copy(cache)
    result.layers = []
    def expand(value):
        if torch.is_tensor(value):
            if value.ndim and value.shape[0] == 1:
                return value.repeat(count, *([1] * (value.ndim - 1)))
            return value.clone()
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value
    for layer in cache.layers:
        new = copy.copy(layer)
        new.__dict__.update({k: expand(v) for k, v in layer.__dict__.items()})
        result.layers.append(new)
    return result


def score_suffix_batch(model, encoded, pos, cache, prefix_len, indices, embeddings):
    lengths = [encoded[i]['input_ids'].shape[-1] - prefix_len for i in indices]
    maximum = max(lengths)
    masks, positions_list, inputs = [], [], []
    for i, length in zip(indices, lengths):
        mask = encoded[i].get('attention_mask')
        if mask is None:
            mask = torch.ones_like(encoded[i]['input_ids'])
        masks.append(F.pad(mask, (0, maximum - length), value=0))
        positions_list.append(F.pad(pos[i][..., prefix_len:], (0, maximum - length), value=0))
        if embeddings is None:
            inputs.append(F.pad(encoded[i]['input_ids'][..., prefix_len:], (0, maximum - length), value=0))
        else:
            inputs.append(F.pad(embeddings[i][..., prefix_len:, :], (0, 0, 0, maximum - length), value=0))
    kwargs = {'input_ids' if embeddings is None else 'inputs_embeds': torch.cat(inputs, dim=0)}
    out = model.language_model(**kwargs, attention_mask=torch.cat(masks, dim=0),
        position_ids=torch.cat(positions_list, dim=1),
        past_key_values=expand_branch_cache(cache, len(indices)), use_cache=True)
    return [out.last_hidden_state[j, length - 1] for j, length in enumerate(lengths)]


def score_suffixes(model, encoded, pos, cache, prefix_len, embeddings):
    # 默认为已验证的串行路径；平台数值验收通过后由脚本启用 batch=4。
    batch_size = int(os.environ.get('JEV_SUFFIX_BATCH', '1'))
    if not 1 <= batch_size <= 8:
        raise ValueError('JEV_SUFFIX_BATCH 必须在 1 至 8 之间')
    lengths = [item['input_ids'].shape[-1] - prefix_len for item in encoded]
    order = sorted(range(len(encoded)), key=lambda i: lengths[i]) if batch_size > 1 else list(range(len(encoded)))
    vectors = [None] * len(encoded)
    cursor = 0
    while cursor < len(order):
        indices = order[cursor:cursor + batch_size]
        # 控制补齐放大，长后缀保留单候选路径。
        while len(indices) > 1 and (max(lengths[i] for i in indices) > 512 or
                max(lengths[i] for i in indices) * len(indices) > 1.5 * sum(lengths[i] for i in indices)):
            indices = indices[:-1]
        if len(indices) > 1:
            failed = False
            try:
                values = score_suffix_batch(model, encoded, pos, cache, prefix_len, indices, embeddings)
            except torch.cuda.OutOfMemoryError:
                failed = True
            if failed:
                # 离开 except 后释放失败前向的 traceback，再执行串行路径。
                gc.collect()
                torch.cuda.empty_cache()
                values = [score_suffix_batch(model, encoded, pos, cache, prefix_len, [i], embeddings)[0] for i in indices]
        else:
            values = [score_suffix_batch(model, encoded, pos, cache, prefix_len, indices, embeddings)[0]]
        for i, value in zip(indices, values):
            vectors[i] = value
        cursor += len(indices)
    return vectors
