#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="${MODEL_NAME:-qwen-2.5-7b}"
BETA="${BETA:-0.1}"

ARGS=(
  --model-name "$MODEL_NAME" \
  --bi-file "prune_orders/raw_scores/${MODEL_NAME}_bi.json" \
  --q-file "prune_orders/raw_scores/${MODEL_NAME}_q_similarity.json" \
  --beta "$BETA" \
  --output "prune_orders/generated/${MODEL_NAME}_tappa_beta${BETA}.json"
)

python build_prune_order.py "${ARGS[@]}"
