"""在当前 GPU 比较全分支/共享前缀的评分、损失和 LoRA 梯度。"""
import argparse
import json
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jev'))
from train_200k import load_model
from shared_prefix_training import encode_branches, score_question


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('models/Qwen3.5-2B'))
    p.add_argument('--init-checkpoint', type=Path, default=Path('checkpoints/init-r16'))
    p.add_argument('--output-dir', type=Path, default=Path('runs/shared-verification'))
    p.add_argument('--mm-template', choices=['qwen', 'decision'], default='qwen')
    p.add_argument('--multimodal-shared-kv', action='store_true')
    p.add_argument('--text-only', action='store_true')
    a = p.parse_args()
    a.gradient_checkpointing, a.shared_kv = False, True
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / 'PASS.json').unlink(missing_ok=True)
    processor, model, head, _ = load_model(a)
    model.train()
    model.visual.eval()
    head.train()
    Image.new('RGB', (64, 64), 'red').save(a.output_dir / 'red.png')
    records = [
        {'id': 'text', 'description': '18 加 27 等于多少？', 'candidates': ['45', '46', '44'], 'gold_index': 0, 'language': 'zh'},
        {'id': 'image', 'description': '图像主要是什么颜色？', 'candidates': ['红色', '蓝色', '绿色'], 'gold_index': 0, 'language': 'zh', 'image': 'red.png'},
        {'id': 'unequal', 'description': 'Choose the animal.', 'candidates': ['cat', 'an inanimate stone on a mountain'], 'gold_index': 0, 'language': 'en'},
    ]
    real_root = Path('data/mix200k')
    if (real_root / 'train.jsonl').exists():
        selected = {}
        with (real_root / 'train.jsonl').open() as f:
            for line in f:
                r = json.loads(line)
                source = r['source_dataset']
                if source == 'race' and source not in selected and 1500 < len(r['description']) < 2500:
                    selected[source] = r
                if source == 'scienceqa' and source not in selected and len(r['description']) < 350:
                    selected[source] = r
                if len(selected) == 2:
                    break
        records.extend(selected.values())
    params = {n: v for n, v in model.named_parameters() if v.requires_grad}
    if a.text_only:
        records = [r for r in records if not r.get('image')]
    report = []
    for r in records:
        encoded = encode_branches(processor, r, real_root if ':' in r['id'] else a.output_dir, next(head.parameters()).device, 4096, a.mm_template)
        snapshots = []
        durations = []
        for shared in [False, True]:
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            if shared:
                model.language_model.gradient_checkpointing_disable()
            else:
                model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.monotonic()
            scores = score_question(model, head, encoded, shared,
                cache_alignment=64 if r.get('image') and a.multimodal_shared_kv else 1)
            loss = F.cross_entropy(scores[None], torch.tensor([r['gold_index']], device=scores.device))
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            durations.append(time.monotonic() - started)
            grads = {n: v.grad.detach().float().cpu().clone() for n, v in params.items() if v.grad is not None}
            if len(grads) != len(params) or not all(torch.isfinite(g).all() for g in grads.values()):
                raise ValueError('缺失或非有限 LoRA 梯度')
            snapshots.append((scores.detach().float().cpu(), loss.item(), grads))
        full, shared = snapshots
        probability_error = (full[0].softmax(0) - shared[0].softmax(0)).abs().max().item()
        print(json.dumps({'id': r['id'], 'full_scores': full[0].tolist(), 'shared_scores': shared[0].tolist(),
                          'probability_max_error': probability_error}), flush=True)
        is_image = bool(r.get('image'))
        check_route = not is_image or a.multimodal_shared_kv
        if check_route:
            torch.testing.assert_close(full[0], shared[0], rtol=0.02, atol=0.1)
            if probability_error > 0.02:
                raise ValueError('文本概率偏差超过两个百分点')
            if full[0].argmax() != shared[0].argmax():
                raise ValueError('文本候选排序不同')
        dot = sum((full[2][n] * shared[2][n]).sum().item() for n in params)
        norm1 = sum(g.square().sum().item() for g in full[2].values()) ** 0.5
        norm2 = sum(g.square().sum().item() for g in shared[2].values()) ** 0.5
        cosine = dot / max(norm1 * norm2, 1e-20)
        relative = sum((full[2][n] - shared[2][n]).square().sum().item() for n in params) ** 0.5 / max(norm1, 1e-20)
        print(json.dumps({'id': r['id'], 'gradient_cosine': cosine, 'gradient_relative_error': relative}), flush=True)
        strict_gradient_pass = cosine >= 0.995 and relative <= 0.1
        bounded_bf16_scale = (is_image and a.multimodal_shared_kv and full[1] < 0.05
                             and cosine >= 0.995 and relative <= 0.15 and probability_error <= 0.005)
        if check_route and not (strict_gradient_pass or bounded_bf16_scale):
            raise ValueError(f'梯度偏差过大: cosine={cosine}, relative={relative}')
        if check_route and bounded_bf16_scale and not strict_gradient_pass:
            print(json.dumps({'warning': 'BF16 高置信度样本梯度幅度告警，预测与方向检查通过',
                              'id': r['id'], 'gradient_relative_error': relative}), flush=True)
        report.append({'id': r['id'], 'score_max_error': (full[0] - shared[0]).abs().max().item(),
                       'full_loss': full[1], 'shared_loss': shared[1], 'probability_max_error': probability_error,
                       'gradient_cosine': cosine, 'gradient_relative_error': relative,
                       'production_route': 'aligned_shared_prefix' if is_image and a.multimodal_shared_kv else ('full_forward' if is_image else 'shared_prefix'),
                       'raw_shared_gradient_pass': strict_gradient_pass,
                       'bounded_bf16_scale_warning': check_route and bounded_bf16_scale and not strict_gradient_pass})
        report[-1]['full_forward_backward_seconds'], report[-1]['shared_forward_backward_seconds'] = durations
        print(json.dumps(report[-1], ensure_ascii=False), flush=True)
    (a.output_dir / 'PASS.json').write_text(json.dumps({'status': 'PASS_SHARED_TRAINING' if a.multimodal_shared_kv else 'PASS_TEXT_ROUTE',
        'multimodal_shared_kv_enabled': a.multimodal_shared_kv, 'cases': report}, indent=2))


if __name__ == '__main__':
    main()
