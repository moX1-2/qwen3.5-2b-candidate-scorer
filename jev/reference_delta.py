"""诊断用的原生 PyTorch DeltaNet 内核，保留上游方程与自动微分。"""
import inspect
import torch


def enable_reference_delta():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    qwen.torch_chunk_gated_delta_rule = inspect.unwrap(qwen.torch_chunk_gated_delta_rule)
    qwen.torch_recurrent_gated_delta_rule = inspect.unwrap(qwen.torch_recurrent_gated_delta_rule)
    qwen.l2norm = inspect.unwrap(qwen.l2norm)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
