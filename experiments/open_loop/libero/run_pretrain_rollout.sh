#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash experiments/open_loop/libero/run_pretrain_rollout.sh <relative_or_absolute_ckpt_path> [task_suite] [cuda]"
  echo "Example: bash experiments/open_loop/libero/run_pretrain_rollout.sh exp_pretrain/my_ckpt libero_spatial 0"
  exit 1
fi

CKPT_PATH="$1"
TASK_SUITE_NAME="${2:-libero_spatial}"
CUDA_ID="${3:-0}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/LIBERO:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/transformers:$PYTHONPATH"

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CKPT_PATH" \
  --task_suite_name "$TASK_SUITE_NAME" \
  --cuda "$CUDA_ID" \
  --use_latent True \
  --latent_size 8 \
  --seed 0
