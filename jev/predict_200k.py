"""候选推理入口，可在无梯度模式下共享图像特征及公共前缀缓存。"""
import argparse
import json
from pathlib import Path
import time
import torch
from train_200k import load_model
from shared_prefix_training import encode_branches, score_question


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('models/Qwen3.5-2B'))
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/init-r16'))
    p.add_argument('--question', required=True)
    p.add_argument('--candidate', action='append', required=True)
    p.add_argument('--image', type=Path)
    p.add_argument('--language', choices=['zh', 'en'], default='zh')
    p.add_argument('--mm-template', choices=['qwen', 'decision'], default='decision')
    p.add_argument('--full-forward', action='store_true')
    a = p.parse_args()
    if not 2 <= len(a.candidate) <= 26:
        raise ValueError('候选数量必须为 2 至 26')
    a.inference_checkpoint, a.init_checkpoint = a.checkpoint, a.checkpoint
    a.shared_kv, a.gradient_checkpointing = False, False
    a.output_dir = Path('runs/inference')
    processor, model, head, _ = load_model(a)
    model.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    r = {'id': 'manual', 'description': a.question, 'candidates': a.candidate,
         'gold_index': 0, 'language': a.language,
         'image': str(a.image.resolve()) if a.image else None}
    start = time.monotonic()
    with torch.inference_mode():
        inputs = encode_branches(processor, r, Path.cwd(), next(head.parameters()).device, 4096, a.mm_template)
        scores = score_question(model, head, inputs, shared=not a.full_forward,
                                cache_alignment=64 if a.image else 1)
        probabilities = scores.softmax(0).cpu().tolist()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    result = {'mode': 'full_forward' if a.full_forward else 'shared_inference', 'gradients_enabled': False,
              'best_index': scores.argmax().item(), 'best_candidate': a.candidate[scores.argmax().item()],
              'scores': scores.cpu().tolist(), 'probabilities': probabilities,
              'seconds': time.monotonic()-start}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
