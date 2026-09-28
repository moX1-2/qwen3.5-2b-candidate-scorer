"""Evaluate newly received 200k checkpoints on the fixed text holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jev"))
from train_200k import evaluate, load_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", choices=("step-007500", "step-009000", "step-010000", "step-012500"))
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.5-2B")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "result/unpacked/mix200k-20260928")
    parser.add_argument("--heldout", type=Path, default=ROOT / "result/evaluation-2026-09-25/heldout_200.jsonl")
    parser.add_argument("--custom", type=Path, default=ROOT / "result/evaluation-2026-09-25/custom_20.jsonl")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    checkpoints = {
        "step-007500": args.checkpoint_dir / "step-007500",
        "step-009000": args.checkpoint_dir / "step-009000",
        "step-010000": args.checkpoint_dir / "step-010000",
        "step-012500": args.checkpoint_dir / "step-012500",
    }
    selected = list(dict.fromkeys(args.checkpoint or checkpoints))
    sets = {
        "heldout_200": args.heldout,
        "custom_20": args.custom,
    }
    for name in selected:
        checkpoint = checkpoints[name]
        for relative in ("adapter/adapter_model.safetensors", "adapter/adapter_config.json", "score_head.pt"):
            if not (checkpoint / relative).is_file():
                raise FileNotFoundError(checkpoint / relative)

    first = checkpoints[selected[0]]
    setup = SimpleNamespace(
        model=args.model,
        inference_checkpoint=first,
        init_checkpoint=first,
        output_dir=args.output_dir,
        shared_kv=False,
        text_prefix_sharing=False,
        gradient_checkpointing=False,
    )
    processor, model, head, _ = load_model(setup)
    model.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    for name in selected[1:]:
        model.language_model.load_adapter(checkpoints[name] / "adapter", adapter_name=name, is_trainable=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with torch.inference_mode():
        for name in selected:
            checkpoint = checkpoints[name]
            model.language_model.set_adapter("default" if name == selected[0] else name)
            state = torch.load(checkpoint / "score_head.pt", map_location="cpu", weights_only=True)
            head.load_state_dict(state)
            results[name] = {}
            for set_name, path in sets.items():
                records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if args.limit is not None:
                    records = records[: args.limit]
                config = SimpleNamespace(validation_data=path, max_length=4096, mm_template="decision")
                result = evaluate(config, processor, model, head, records)
                if result is None:
                    raise RuntimeError(f"Evaluation interrupted: {name}/{set_name}")
                results[name][set_name] = result["metrics"]
                predictions_path = args.output_dir / f"{name}_{set_name}_predictions.json"
                predictions_path.write_text(json.dumps(result["predictions"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(name, set_name, result["metrics"]["overall"], flush=True)

    manifest = {
        "device": torch.cuda.get_device_name(0),
        "max_length": 4096,
        "mm_template": "decision",
        "shared_kv": False,
        "limit_per_set": args.limit,
        "checkpoint_paths": {k: str(checkpoints[k]) for k in selected},
        "test_set_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in sets.items()},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
