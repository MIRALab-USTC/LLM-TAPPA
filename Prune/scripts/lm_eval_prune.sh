#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen-2.5-7b}"
MODEL_PATH="${MODEL_PATH:-PATH_TO_QWEN25_7B}"
METHOD="${METHOD:-TAPPA}"
ORDER_NAME="${ORDER_NAME:-BI+0.1q}"
ORDER_FILE="${ORDER_FILE:-prune_orders/generated/qwen-2.5-7b_tappa_beta0.1.json}"
N_PRUNE_LAYERS="${N_PRUNE_LAYERS:-11}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
SAVE_DIR="${SAVE_DIR:-results}"

python eval_lm.py \
  --model "$MODEL" \
  --model-path "$MODEL_PATH" \
  --method "$METHOD" \
  --order-name "$ORDER_NAME" \
  --order-file "$ORDER_FILE" \
  --n-prune-layers "$N_PRUNE_LAYERS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --save-dir "$SAVE_DIR" \
  --tasks piqa hellaswag arc_easy winogrande
