#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
task_python="${JEV_PYTHON:-jev/.venv/bin/python}"
task_model="${JEV_MODEL:-models/Qwen3.5-2B}"
task_text_shared="${TEXT_PREFIX_SHARING:-1}"
task_default_output="runs/mix200k-stable-r16-lr1e5"
if [[ "$task_text_shared" == 1 ]]; then task_default_output="runs/mix200k-textshared-r16-lr1e5"; fi
task_output="${OUTPUT_DIR:-$task_default_output}"
task_template="${MM_TEMPLATE:-decision}"
# 稳定训练始终完整前向，不读取旧 SHARED_KV / MM_SHARED_KV 环境变量。
args=(--model "$task_model" --output-dir "$task_output" --mm-template "$task_template" --gradient-checkpointing)
if [[ "$task_text_shared" == 1 ]]; then
    "$task_python" scripts/verify_shared_training.py --model "$task_model" --text-only --output-dir "$task_output/preflight"
    args+=(--text-prefix-sharing)
fi
exec "$task_python" -u jev/train_200k.py "${args[@]}" "$@"
