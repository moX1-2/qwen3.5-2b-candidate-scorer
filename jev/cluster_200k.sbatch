#!/usr/bin/env bash
#SBATCH --job-name=jev-200k-r16
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=runs/slurm-200k-%j.out
#SBATCH --error=runs/slurm-200k-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?请在更新包根目录提交}"
# 文本共享前缀训练可选，图文训练始终完整前向。
export TEXT_PREFIX_SHARING="${TEXT_PREFIX_SHARING:-1}"
bash jev/run_200k.sh &
task_pid=$!
trap 'kill -TERM "$task_pid" 2>/dev/null || true; wait "$task_pid" || true' TERM INT
wait "$task_pid"
