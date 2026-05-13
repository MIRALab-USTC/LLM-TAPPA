#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="${MODEL_NAME:-qwen-2.5-7b}"
MODEL_PATH="${MODEL_PATH:-PATH_TO_QWEN25_7B}"
DATASET_PATH="${DATASET_PATH:-}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

ARGS=(
  --model-name "$MODEL_NAME"
  --model-path "$MODEL_PATH"
  --max-samples "$MAX_SAMPLES"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --output "prune_orders/raw_scores/${MODEL_NAME}_bi.json"
)

if [[ -n "$DATASET_PATH" ]]; then
  ARGS+=(--dataset-path "$DATASET_PATH")
fi

python compute_bi.py "${ARGS[@]}"
