#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="${MODEL_NAME:-qwen-2.5-7b}"

ARGS=(
  --model-name "$MODEL_NAME" \
  --bi-file "prune_orders/raw_scores/${MODEL_NAME}_bi.json" \
  --output "prune_orders/generated/${MODEL_NAME}_shortgpt_bi.json"
)

python build_shortgpt_order.py "${ARGS[@]}"
