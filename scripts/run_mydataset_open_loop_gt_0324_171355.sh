#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="$(date '+%m%d_%H%M%S')"

CUDA_ID="${1:-0}"
EPISODE_ID="${2:-0}"
JSONL_PATH="${3:-/home/robot/play/MyDataset/20260323_173241_last0_20260127/galaxea_json/20260127_last0_train.jsonl}"
MODEL_ROOT="${4:-/home/robot/zhy/last0/exp_0127/libero_spatial_test1/libero_spatial/checkpoint-19-121280}"
HORIZON="${5:-16}"
STRIDE="${6:-16}"

DATA_TAG="$(basename "$JSONL_PATH" .jsonl)"
MODEL_TAG="$(basename "$MODEL_ROOT")"
OUTPUT_DIR="$ROOT_DIR/experiments/open_loop/libero/outputs/${RUN_TAG}__${DATA_TAG}__${MODEL_TAG}__ep${EPISODE_ID}__h${HORIZON}__s${STRIDE}"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/transformers:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "RUN_TAG=$RUN_TAG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "JSONL_PATH=$JSONL_PATH"
echo "EPISODE_ID=$EPISODE_ID"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "HORIZON=$HORIZON"
echo "STRIDE=$STRIDE"

python experiments/open_loop/libero/compare_galaxea_jsonl_gt.py \
  --model-root "$MODEL_ROOT" \
  --jsonl-path "$JSONL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --episode-id "$EPISODE_ID" \
  --cuda "$CUDA_ID" \
  --horizon "$HORIZON" \
  --stride "$STRIDE" \
  2>&1 | tee "$OUTPUT_DIR/console.log"
