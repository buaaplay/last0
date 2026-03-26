#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="$(date '+%m%d_%H%M%S')"

CUDA_ID="${1:-0}"
EPISODE_ID="${2:-0}"
JSONL_PATH="${3:-/home/robot/play/galaxea_tools_outputs/20260320_145013_convert_arrange_fruits_to_last0_libero16_14/galaxea_json/Arrange_Fruits_20250819_011_last0_train.jsonl}"
MODEL_ROOT="${4:-/home/robot/zhy/last0/exp/libero_spatial_test1/libero_spatial/checkpoint-41-276906}"
MODEL_ALIAS="${5:-galaxea_finetune}"
HORIZON="${6:-16}"
STRIDE="${7:-16}"

DATA_TAG="$(basename "$JSONL_PATH" .jsonl)"
MODEL_TAG="$(basename "$MODEL_ROOT")"
OUTPUT_DIR="$ROOT_DIR/experiments/open_loop/libero/outputs/${RUN_TAG}__${MODEL_ALIAS}__${DATA_TAG}__${MODEL_TAG}__ep${EPISODE_ID}__h${HORIZON}__s${STRIDE}"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/transformers:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

python experiments/open_loop/libero/compare_jsonl_gt_raw.py \
  --model-root "$MODEL_ROOT" \
  --model-alias "$MODEL_ALIAS" \
  --jsonl-path "$JSONL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --episode-id "$EPISODE_ID" \
  --cuda "$CUDA_ID" \
  --horizon "$HORIZON" \
  --stride "$STRIDE" \
  2>&1 | tee "$OUTPUT_DIR/console.log"
