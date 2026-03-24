#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="$(date '+%m%d_%H%M%S')"
OUTPUT_DIR="$ROOT_DIR/experiments/open_loop/libero/outputs/$RUN_TAG"

TASK_SUITE_NAME="${1:-libero_spatial}"
CUDA_ID="${2:-0}"
EPISODE_PATH="${3:-/home/robot/zhy/last0/datasets/libero_spatial_no_noops}"
MODEL_ROOT="${4:-/home/robot/zhy/last0/exp/libero_spatial_test1/libero_spatial/checkpoint-41-276906}"
HORIZON="${5:-16}"
STRIDE="${6:-16}"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/transformers:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "RUN_TAG=$RUN_TAG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TASK_SUITE_NAME=$TASK_SUITE_NAME"
echo "EPISODE_PATH=$EPISODE_PATH"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "HORIZON=$HORIZON"
echo "STRIDE=$STRIDE"

python experiments/open_loop/libero/compare_single_ckpt_gt.py \
  --model-root "$MODEL_ROOT" \
  --episode-path "$EPISODE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --cuda "$CUDA_ID" \
  --task-suite-name "$TASK_SUITE_NAME" \
  --horizon "$HORIZON" \
  --stride "$STRIDE" \
  2>&1 | tee "$OUTPUT_DIR/console.log"
