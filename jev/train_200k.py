"""低学习率图文候选训练，支持阶段热启、完整恢复、验证和共享前缀实验。"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from train_cluster import ScoreHead, make_scheduler, collate_micro_batch, score_batched
from common import Example, stratified_metrics
from shared_prefix_training import encode_branches, score_question
from cpu_preprocess import CPUWindowPrefetch, CPUPreprocessingError, move_cpu_branches

STOP = False


def stop(*_):
    global STOP
    STOP = True


def read(path):
    with path.open(encoding='utf-8') as f:
        records = [json.loads(s) for s in f if s.strip()]
    ids = set()
    for r in records:
        if r['id'] in ids or not isinstance(r['gold_index'], int) or not 0 <= r['gold_index'] < len(r['candidates']):
            raise ValueError(f"样本格式错误: {r['id']}")
        if not 2 <= len(r['candidates']) <= 26 or any(not c.strip() for c in r['candidates']):
            raise ValueError(f"候选格式错误: {r['id']}")
        ids.add(r['id'])
    if not records:
        raise ValueError(f'空数据: {path}')
    return records


def load_model(a):
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    processor.tokenizer.padding_side = 'right'
    processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': 262144}
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    full = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16,
            device_map={'': device}, local_files_only=True, attn_implementation=os.environ.get('JEV_ATTENTION_IMPL', 'eager'))
    from fast_runtime import configure_delta_backend
    backend_info = configure_delta_backend(os.environ.get('JEV_DELTA_BACKEND', 'native'))
    print(json.dumps({'event': 'fast_backend', **backend_info, 'attention': os.environ.get('JEV_ATTENTION_IMPL', 'eager'), 'suffix_batch': int(os.environ.get('JEV_SUFFIX_BATCH', '1'))}, ensure_ascii=False), flush=True)
    model = full.model
    del full
    gc.collect()
    for p in model.parameters():
        p.requires_grad = False
    latest = a.output_dir / 'latest.json'
    inference_checkpoint = getattr(a, 'inference_checkpoint', None)
    ckpt = inference_checkpoint or (a.output_dir / json.loads(latest.read_text())['checkpoint'] if latest.exists() else a.init_checkpoint)
    model.language_model = PeftModel.from_pretrained(model.language_model, ckpt / 'adapter', is_trainable=True)
    model.language_model.peft_config['default'].base_model_name_or_path = str(a.model)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    head = ScoreHead(model.config.text_config.hidden_size).to(device)
    head.load_state_dict(torch.load(ckpt / 'score_head.pt', map_location=device, weights_only=True))
    if a.shared_kv or getattr(a, 'text_prefix_sharing', False):
        model.language_model.enable_input_require_grads()
    if a.gradient_checkpointing:
        if a.shared_kv:
            raise ValueError('共享 KV 与梯度检查点不能同时开启')
        model.language_model.enable_input_require_grads()
        model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return processor, model, head, ckpt if latest.exists() and not inference_checkpoint else None


def evaluate(a, processor, model, head, records):
    model.eval()
    head.eval()
    predictions = []
    with torch.no_grad():
        for r in records:
            if STOP:
                return None
            inputs = encode_branches(processor, r, a.validation_data.parent, next(head.parameters()).device, a.max_length, a.mm_template)
            scores = score_question(model, head, inputs, shared=False)
            probs = scores.softmax(0).cpu().tolist()
            pred = scores.argmax().item()
            predictions.append({'id': r['id'], 'language': r.get('language', 'en'),
                'source_dataset': r.get('source_dataset', 'unknown'), 'domain': r.get('domain', 'unknown'),
                'candidate_count': len(probs), 'gold_position': r['gold_index'],
                'predicted_position': pred, 'correct': pred == r['gold_index'], 'probabilities': probs})
    return {'metrics': stratified_metrics(predictions), 'predictions': predictions}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('models/Qwen3.5-2B'))
    p.add_argument('--train-data', type=Path, default=Path('data/mix200k/train.jsonl'))
    p.add_argument('--validation-data', type=Path, default=Path('data/mix200k/validation.jsonl'))
    p.add_argument('--init-checkpoint', type=Path, default=Path('checkpoints/init-r16'))
    p.add_argument('--output-dir', type=Path, default=Path('runs/mix200k-stable-r16-lr1e5'))
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--gradient-accumulation', type=int, default=16)
    p.add_argument('--micro-batch-size', type=int, default=4, help='连续纯文本题批处理，图文按题完整前向')
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--patience', type=int, default=3, help='仅兼容旧配置，训练已取消早停')
    p.add_argument('--max-steps', type=int)
    p.add_argument('--max-train-examples', type=int)
    p.add_argument('--max-validation-examples', type=int)
    p.add_argument('--shared-kv', action='store_true')
    p.add_argument('--text-prefix-sharing', action='store_true', help='可选的纯文本前缀共享训练，图文始终完整前向')
    p.add_argument('--mm-template', choices=['qwen', 'decision'], default='qwen', help='图文聊天模板或自定义评分模板')
    p.add_argument('--multimodal-shared-kv', action='store_true', help='自定义图文模板按 64-token 分块共享前缀与图像特征')
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=20260926)
    a = p.parse_args()
    if a.shared_kv or a.multimodal_shared_kv:
        raise ValueError('请移除旧共享参数；纯文本共享训练使用 --text-prefix-sharing，图文训练不共享')
    if a.micro_batch_size < 1:
        raise ValueError('micro-batch-size 必须为正数')
    if min(a.epochs, a.gradient_accumulation, a.eval_every, a.save_every) < 1 or a.learning_rate <= 0:
        raise ValueError('训练参数必须为正数')
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / 'train.pid').write_text(str(os.getpid()))
    def log(data):
        data = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), **data}
        print(json.dumps(data, ensure_ascii=False), flush=True)
        with (a.output_dir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    training, validation = read(a.train_data), read(a.validation_data)
    if a.max_train_examples:
        training = training[:a.max_train_examples]
    if a.max_validation_examples:
        validation = validation[:a.max_validation_examples]
    if {r['id'] for r in training} & {r['id'] for r in validation}:
        raise ValueError('训练验证编号重叠')
    signature = {k: str(v) for k, v in vars(a).items() if k not in ['max_steps', 'output_dir']}
    for key in ('train_data', 'validation_data'):
        signature[key + '_sha256'] = hashlib.sha256(getattr(a, key).read_bytes()).hexdigest()
    config_file = a.output_dir / 'config.json'
    if config_file.exists():
        existing = json.loads(config_file.read_text())
        # 旧更新包没有模板字段，实际采用 qwen；保持旧任务恢复兼容。
        existing.setdefault('mm_template', 'qwen')
        existing.setdefault('multimodal_shared_kv', False)
        existing.setdefault('micro_batch_size', 1)
        existing.setdefault('text_prefix_sharing', False)
        if existing != signature:
            raise ValueError('恢复配置或数据变化，请使用新的输出目录')
    config_file.write_text(json.dumps(signature, indent=2))
    processor, model, head, resume = load_model(a)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.learning_rate, weight_decay=0.01)
    total = math.ceil(len(training) / a.gradient_accumulation) * a.epochs
    scheduler = make_scheduler(opt, total, 0.05)
    epoch, position, step, best, bad = 0, 0, 0, float('inf'), 0
    if resume:
        state = torch.load(resume / 'training_state.pt', map_location='cpu', weights_only=False)
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch, position, step, best, bad = [state[k] for k in ('epoch', 'position', 'step', 'best', 'bad')]
        torch.set_rng_state(state['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state['cuda_rng'])
    log({'event': 'start', 'train': len(training), 'validation': len(validation), 'step': step,
         'lr': a.learning_rate, 'trainable': sum(p.numel() for p in params), 'shared_kv': a.shared_kv, 'early_stopping': False})
    def save(e, pos):
        relative = f'checkpoints/step-{step:06d}'
        destination = a.output_dir / relative
        temporary = destination.with_name(destination.name + '.tmp')
        temporary.mkdir(parents=True, exist_ok=True)
        model.language_model.save_pretrained(temporary / 'adapter')
        torch.save(head.state_dict(), temporary / 'score_head.pt')
        torch.save({'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                    'epoch': e, 'position': pos, 'step': step, 'best': best, 'bad': bad,
                    'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None},
                   temporary / 'training_state.pt')
        if destination.exists():
            # 同一步验证后 best/bad 或整轮结束的 epoch 可能更新，原子替换恢复状态。
            import shutil
            (temporary / 'training_state.pt').replace(destination / 'training_state.pt')
            shutil.rmtree(temporary)
        else:
            temporary.rename(destination)
        pointer = a.output_dir / 'latest.tmp'
        pointer.write_text(json.dumps({'checkpoint': relative, 'step': step}))
        pointer.replace(a.output_dir / 'latest.json')
        return relative
    if not (a.output_dir / 'baseline.json').exists():
        baseline = evaluate(a, processor, model, head, validation)
        if baseline is None:
            save(epoch, position)
            log({'event': 'stop_during_baseline', 'step': step})
            return
        best = baseline['metrics']['overall']['nll']
        (a.output_dir / 'baseline.json').write_text(json.dumps(baseline, ensure_ascii=False))
        log({'event': 'baseline', 'metrics': baseline['metrics']})
    # 保存周期是运行时策略，不改变原训练签名或恢复状态。
    checkpoint_seconds = float(os.environ.get('JEV_CHECKPOINT_SECONDS', '600'))
    if checkpoint_seconds <= 0:
        raise ValueError('JEV_CHECKPOINT_SECONDS 必须为正数')
    cpu_workers = int(os.environ.get('JEV_CPU_WORKERS', '0'))
    cpu_depth = int(os.environ.get('JEV_CPU_PREFETCH_WINDOWS', '2'))
    if cpu_workers:
        torch.set_num_threads(int(os.environ.get('JEV_CPU_TORCH_THREADS', '1')))
    full_batch_mode = os.environ.get('JEV_FULL_BATCH', '0') == '1'
    last_checkpoint = time.monotonic()
    while epoch < a.epochs:
        order = list(range(len(training)))
        random.Random(a.seed + epoch).shuffle(order)
        with CPUWindowPrefetch(processor, training, order, position, a.gradient_accumulation,
                encode_branches, a.train_data.parent, a.max_length, a.mm_template,
                workers=cpu_workers, depth=cpu_depth) as cpu_pool:
            while position < len(order):
                model.train()
                model.visual.eval()
                head.train()
                opt.zero_grad(set_to_none=True)
                window = order[position:position + a.gradient_accumulation]
                losses = torch.zeros((), device=next(head.parameters()).device, dtype=torch.float64)
                text_shared_count, image_full_count, text_full_count = 0, 0, 0
                start = time.monotonic()
                cursor = 0
                cpu_wait_seconds, cpu_prepare_seconds = 0.0, 0.0
                try:
                    if full_batch_mode:
                        from full_batch_runtime import full_window
                        losses, text_full_count, image_full_count = full_window(
                            model, head, [training[i] for i in window], processor,
                            a.train_data.parent, a.max_length, a.mm_template, log)
                    else:
                        prepared_window = cpu_pool.take(position, len(window))
                        prepared_cpu = prepared_window.examples if prepared_window is not None else None
                        if prepared_window is not None:
                            cpu_wait_seconds = prepared_window.wait_seconds
                            cpu_prepare_seconds = prepared_window.cpu_seconds
                        prepared_text = {}
                        if a.text_prefix_sharing and os.environ.get('JEV_PREFETCH_TEXT', '0') == '1':
                            for offset, index in enumerate(window):
                                record = training[index]
                                if not (record.get('image') or record.get('image_path')):
                                    prepared_text[offset] = (move_cpu_branches(prepared_cpu[offset], next(head.parameters()).device)
                                        if prepared_cpu is not None else encode_branches(processor, record, a.train_data.parent, next(head.parameters()).device, a.max_length, a.mm_template))
                        window_golds = torch.tensor([training[index]['gold_index'] for index in window], device=next(head.parameters()).device)
                        while cursor < len(window):
                            r = training[window[cursor]]
                            if r.get('image') or r.get('image_path'):
                                inputs = prepared_text.get(cursor)
                                if inputs is None:
                                    inputs = (move_cpu_branches(prepared_cpu[cursor], next(head.parameters()).device) if prepared_cpu is not None else encode_branches(processor, r, a.train_data.parent, next(head.parameters()).device, a.max_length, a.mm_template))
                                scores = score_question(model, head, inputs, shared=False)
                                loss = F.cross_entropy(scores[None], window_golds[cursor:cursor + 1])
                                (loss / len(window)).backward()
                                losses += loss.detach()
                                image_full_count += 1
                                cursor += 1
                            elif a.text_prefix_sharing:
                                model.language_model.gradient_checkpointing_disable()
                                inputs = prepared_text.get(cursor)
                                if inputs is None:
                                    inputs = (move_cpu_branches(prepared_cpu[cursor], next(head.parameters()).device) if prepared_cpu is not None else encode_branches(processor, r, a.train_data.parent, next(head.parameters()).device, a.max_length, a.mm_template))
                                scores = score_question(model, head, inputs, shared=True)
                                loss = F.cross_entropy(scores[None], window_golds[cursor:cursor + 1])
                                (loss / len(window)).backward()
                                losses += loss.detach()
                                text_shared_count += 1
                                if a.gradient_checkpointing:
                                    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                                cursor += 1
                            else:
                                items = []
                                while cursor < len(window) and len(items) < a.micro_batch_size:
                                    r = training[window[cursor]]
                                    if r.get('image') or r.get('image_path'):
                                        break
                                    inputs = (move_cpu_branches(prepared_cpu[cursor], next(head.parameters()).device) if prepared_cpu is not None else encode_branches(processor, r, a.train_data.parent, next(head.parameters()).device, a.max_length, a.mm_template))
                                    ex = Example(r['id'], r['description'], tuple(r['candidates']), r['gold_index'],
                                        r.get('language', 'en'), r.get('source_dataset', 'unknown'), r.get('domain', 'unknown'))
                                    items.append((ex, [x['input_ids'][0] for x in inputs]))
                                    cursor += 1
                                ids, mask, counts, golds = collate_micro_batch(items, processor.tokenizer.pad_token_id)
                                scores_list = score_batched(model.language_model, head, ids, mask, counts, train_backbone=True)
                                loss = sum(F.cross_entropy(scores[None], torch.tensor([gold], device=scores.device))
                                           for scores, gold in zip(scores_list, golds))
                                (loss / len(window)).backward()
                                losses += loss.detach()
                                text_full_count += len(items)
                except CPUPreprocessingError as error:
                    opt.zero_grad(set_to_none=True)
                    save(epoch, position)
                    log({'event': 'stop_preprocessing', 'step': step, 'position': position, 'reason': str(error)})
                    raise SystemExit(76)
                except torch.cuda.OutOfMemoryError:
                    # 当前窗口尚未执行 opt.step，保存上一个完整更新的位置。
                    opt.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    save(epoch, position)
                    log({'event': 'stop_oom', 'step': step, 'position': position,
                         'reason': '当前更新未提交，已保存恢复状态'})
                    raise SystemExit(75)
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                opt.step()
                scheduler.step()
                step += 1
                position += len(window)
                log({'event': 'train', 'step': step, 'loss': losses.item() / len(window), 'lr': scheduler.get_last_lr()[0],
                     'forward_mode': 'full_batch_dynamic' if full_batch_mode else ('text_shared_image_full' if a.text_prefix_sharing else 'full_branches'), 'text_micro_batch_size': a.micro_batch_size,
                     'text_shared_examples': text_shared_count, 'image_full_examples': image_full_count, 'text_full_examples': text_full_count,
                     'cpu_wait_seconds': cpu_wait_seconds, 'cpu_prepare_seconds': cpu_prepare_seconds, 'cpu_workers': cpu_workers,
                     'seconds': time.monotonic() - start, 'vram_gib': torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0})
                should_eval = step % a.eval_every == 0 or position == len(order)
                if should_eval:
                    save(epoch, position)
                    result = evaluate(a, processor, model, head, validation)
                    if result is None:
                        save(epoch, position)
                        log({'event': 'stop_during_eval', 'step': step})
                        return
                    (a.output_dir / f'eval-{step:06d}.json').write_text(json.dumps(result, ensure_ascii=False))
                    nll = result['metrics']['overall']['nll']
                    improved = nll < best
                    bad = 0 if improved else bad + 1
                    best = min(best, nll)
                    log({'event': 'eval', 'step': step, 'metrics': result['metrics'], 'bad_evals': bad})
                    relative = save(epoch, position)
                    last_checkpoint = time.monotonic()
                    if improved:
                        (a.output_dir / 'best.json').write_text(json.dumps({'checkpoint': relative, 'nll': best}))
                timed_save = time.monotonic() - last_checkpoint >= checkpoint_seconds
                if step % a.save_every == 0 or timed_save or STOP or (a.max_steps and step >= a.max_steps):
                    relative = save(epoch, position)
                    last_checkpoint = time.monotonic()
                    log({'event': 'checkpoint', 'step': step, 'position': position,
                         'checkpoint': relative, 'timed_save': timed_save})
                if STOP or (a.max_steps and step >= a.max_steps):
                    log({'event': 'stop', 'step': step, 'bad_evals': bad})
                    return
        epoch += 1
        position = 0
    save(epoch, position)
    log({'event': 'complete', 'step': step})


if __name__ == '__main__':
    main()
