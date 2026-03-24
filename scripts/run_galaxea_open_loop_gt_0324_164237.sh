#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="$(date '+%m%d_%H%M%S')"
OUTPUT_DIR="$ROOT_DIR/experiments/open_loop/libero/outputs/$RUN_TAG"

CUDA_ID="${1:-0}"
EPISODE_ID="${2:-0}"
JSONL_PATH="${3:-/home/robot/play/galaxea_tools_outputs/20260320_145013_convert_arrange_fruits_to_last0_libero16_14/galaxea_json/Arrange_Fruits_20250819_011_last0_train.jsonl}"
MODEL_ROOT="${4:-/home/robot/zhy/last0/exp/libero_spatial_test1/libero_spatial/checkpoint-41-276906}"
HORIZON="${5:-16}"
STRIDE="${6:-16}"

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
