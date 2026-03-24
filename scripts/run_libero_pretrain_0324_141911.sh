#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="$(date '+%m%d_%H%M%S')"
OUTPUT_DIR="$ROOT_DIR/experiments/open_loop/libero/outputs/$RUN_TAG"

TASK_SUITE_NAME="${1:-libero_spatial}"
CUDA_ID="${2:-0}"
SOURCE_CKPT_DIR="/home/robot/zhy/last0/weights/Pretrain/tfmr"
LOCAL_CKPT_LINK="$ROOT_DIR/weights/Pretrain/tfmr"

mkdir -p "$ROOT_DIR/weights/Pretrain"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/rollouts"

if [[ -L "$LOCAL_CKPT_LINK" || -e "$LOCAL_CKPT_LINK" ]]; then
  rm -rf "$LOCAL_CKPT_LINK"
fi
ln -s "$SOURCE_CKPT_DIR" "$LOCAL_CKPT_LINK"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/transformers:${PYTHONPATH:-}"
export LAST0_OUTPUT_DIR="$OUTPUT_DIR"
export LAST0_RUN_TAG="$RUN_TAG"

echo "RUN_TAG=$RUN_TAG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CKPT_LINK=$LOCAL_CKPT_LINK -> $SOURCE_CKPT_DIR"

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$LOCAL_CKPT_LINK" \
  --task_suite_name "$TASK_SUITE_NAME" \
  --cuda "$CUDA_ID" \
  --use_latent True \
  --latent_size 8 \
  --seed 0 \
  --run_id_note "$RUN_TAG" \
  --local_log_dir "$OUTPUT_DIR/logs" \
  2>&1 | tee "$OUTPUT_DIR/console.log"
