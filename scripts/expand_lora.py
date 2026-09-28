"""保持初始增量不变，将普通 LoRA 的秩扩为两倍。"""
import argparse
import json
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def expand(source, destination, seed=20260926):
    if destination.exists():
        raise FileExistsError(destination)
    cfg = json.loads((source / "adapter/adapter_config.json").read_text())
    if any(cfg.get(k) for k in ["use_rslora", "use_dora", "rank_pattern", "alpha_pattern", "modules_to_save", "lora_bias"]):
        raise ValueError("仅支持本项目的普通、统一秩 LoRA")
    torch.manual_seed(seed)
    weights = load_file(str(source / "adapter/adapter_model.safetensors"))
    old_r = cfg["r"]
    expanded = {}
    max_error = 0.0
    for key, tensor in weights.items():
        if ".lora_A." in key:
            extra = torch.empty_like(tensor)
            torch.nn.init.kaiming_uniform_(extra, a=5 ** 0.5)
            expanded[key] = torch.cat([tensor, extra], dim=0)
        elif ".lora_B." in key:
            expanded[key] = torch.cat([tensor, torch.zeros_like(tensor)], dim=1)
        else:
            raise ValueError(f"不支持的 adapter 权重: {key}")
    # 用随机投影验证每个模块增量输出，避免构建巨大的 B@A 矩阵。
    for key, A in weights.items():
        if ".lora_A." not in key:
            continue
        bkey = key.replace(".lora_A.", ".lora_B.")
        x = torch.randn(A.shape[1], 3)
        before = weights[bkey].float() @ (A.float() @ x)
        after = expanded[bkey].float() @ (expanded[key].float() @ x)
        error = (before - after).abs().max().item()
        max_error = max(max_error, error)
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)
    cfg["r"] *= 2
    cfg["lora_alpha"] *= 2
    # 共享前缀训练要求 dropout=0，正式热启时统一使用此设置。
    cfg["lora_dropout"] = 0.0
    cfg['base_model_name_or_path'] = 'models/Qwen3.5-2B'
    (destination / "adapter").mkdir(parents=True)
    save_file(expanded, str(destination / "adapter/adapter_model.safetensors"))
    (destination / "adapter/adapter_config.json").write_text(json.dumps(cfg, indent=2))
    shutil.copy2(source / "score_head.pt", destination / "score_head.pt")
    report = {"source": str(source), "old_rank": old_r, "new_rank": cfg['r'],
              "new_alpha": cfg['lora_alpha'], "dropout": 0, "seed": seed,
              "max_projection_error": max_error,
              "adapter_parameters_before": sum(t.numel() for t in weights.values()),
              "adapter_parameters_after": sum(t.numel() for t in expanded.values())}
    (destination / "expansion.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    expand(a.source, a.destination)
