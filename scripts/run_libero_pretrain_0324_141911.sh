#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="$(date '+%m%d_%H%M%S')"
OUTPUT_DIR="$ROOT_DIR/experiments/open_loop/libero/outputs/$RUN_TAG"

TASK_SUITE_NAME="${1:-libero_spatial}"
CUDA_ID="${2:-0}"
UNNORM_KEY="oxe_pretrain"
SOURCE_CKPT_DIR="/home/robot/zhy/last0/weights/Pretrain/tfmr"
SOURCE_STATS_PATH="/home/robot/zhy/last0/weights/Pretrain/stats_data.json"
LOCAL_CKPT_LINK="$ROOT_DIR/weights/Pretrain/tfmr"
LOCAL_STATS_LINK="$ROOT_DIR/weights/Pretrain/stats_data.json"
LIBERO_CANDIDATES=(
  "/Disk1/zhy/last0/LIBERO"
  "/home/robot/zhy/last0/LIBERO"
  "$ROOT_DIR/LIBERO"
)

mkdir -p "$ROOT_DIR/weights/Pretrain"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/rollouts"

if [[ -L "$LOCAL_CKPT_LINK" || -e "$LOCAL_CKPT_LINK" ]]; then
  rm -rf "$LOCAL_CKPT_LINK"
fi
ln -s "$SOURCE_CKPT_DIR" "$LOCAL_CKPT_LINK"

if [[ -L "$LOCAL_STATS_LINK" || -e "$LOCAL_STATS_LINK" ]]; then
  rm -rf "$LOCAL_STATS_LINK"
fi
ln -s "$SOURCE_STATS_PATH" "$LOCAL_STATS_LINK"

cd "$ROOT_DIR"
for libero_dir in "${LIBERO_CANDIDATES[@]}"; do
  if [[ -d "$libero_dir" ]]; then
    export PYTHONPATH="$libero_dir:${PYTHONPATH:-}"
    echo "LIBERO_ROOT=$libero_dir"
    break
  fi
done
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/transformers:${PYTHONPATH:-}"
export LAST0_OUTPUT_DIR="$OUTPUT_DIR"
export LAST0_RUN_TAG="$RUN_TAG"

echo "RUN_TAG=$RUN_TAG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CKPT_LINK=$LOCAL_CKPT_LINK -> $SOURCE_CKPT_DIR"
echo "STATS_LINK=$LOCAL_STATS_LINK -> $SOURCE_STATS_PATH"
echo "UNNORM_KEY=$UNNORM_KEY"

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$LOCAL_CKPT_LINK" \
  --task_suite_name "$TASK_SUITE_NAME" \
  --unnorm_key "$UNNORM_KEY" \
  --cuda "$CUDA_ID" \
  --use_latent True \
  --latent_size 8 \
  --seed 0 \
  --run_id_note "$RUN_TAG" \
  --local_log_dir "$OUTPUT_DIR/logs" \
  2>&1 | tee "$OUTPUT_DIR/console.log"
