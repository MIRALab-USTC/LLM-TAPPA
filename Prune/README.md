# Prune: TAPPA-Guided Structural Pruning

This folder contains the structural pruning code used for the TAPPA-guided pruning experiments. It keeps ShortGPT as the BI baseline and adds TAPPA's q-similarity guided update for layer ranking.

## Scope

* Compute ShortGPT block influence (BI) for each transformer layer.
* Compute TAPPA q-similarity for each layer.
* Combine normalized BI and q-similarity with beta:

```text
BI_prime = BI_norm + beta * (1 - q_similarity_norm)
```

* Evaluate a model after skipping selected transformer blocks during loading. The checkpoint weights are not physically rewritten.

Lower scores are pruned first.

## Directory Structure

* `compute_bi.py`: Offline BI computation.
* `compute_q_similarity.py`: Offline q-similarity computation.
* `build_shortgpt_order.py`: Builds a ShortGPT BI pruning order.
* `build_prune_order.py`: Builds TAPPA `BI_prime` scores and a pruning order.
* `eval_lm.py`: lm-eval entrypoint for ShortGPT or TAPPA pruning orders.
* `prune/`: Shared utilities for score computation, order lookup, and layer removal.
* `prune_orders/prune_orders.json`: Preserved table layer sets kept for reference, backward-compatible evaluation, and optional manual layer selection.
* `prune_orders/raw_scores/`: Raw BI and q-similarity JSON files in the offline-script output schema.
* `prune_orders/generated/`: ShortGPT and TAPPA order files generated from raw scores.
* `scripts/`: Example shell scripts.
* `config/`: Public model path and task placeholders.

## Setup

```bash
cd Prune
pip install -r requirements.txt
```

The original experiments used Hugging Face models for:

* `llama-2-7b`
* `llama-3.1-8b`
* `qwen-2.5-7b`

Edit `config/model2path.json` or pass `MODEL_PATH` to the scripts.

## Offline BI Computation

```bash
MODEL_NAME=qwen-2.5-7b \
MODEL_PATH=/path/to/Qwen2.5-7B \
DATASET_PATH=/path/to/pg19_validation \
MAX_SAMPLES=1000 \
bash scripts/compute_bi.sh
```

This writes:

```text
prune_orders/raw_scores/qwen-2.5-7b_bi.json
```

## Build a ShortGPT Pruning Order

```bash
MODEL_NAME=qwen-2.5-7b \
bash scripts/build_shortgpt_order.sh
```

This writes:

```text
prune_orders/generated/qwen-2.5-7b_shortgpt_bi.json
```

## Offline q-Similarity Computation

```bash
MODEL_NAME=qwen-2.5-7b \
MODEL_PATH=/path/to/Qwen2.5-7B \
DATASET_PATH=/path/to/pg19_validation \
MAX_SAMPLES=128 \
bash scripts/compute_q_similarity.sh
```

The q-similarity collector works with standard Hugging Face Llama/Qwen-style attention modules by hooking each layer's `q_proj` output.

## Build a TAPPA Pruning Order

```bash
MODEL_NAME=qwen-2.5-7b \
BETA=0.1 \
bash scripts/build_tappa_order.sh
```

The output stores raw scores, normalized scores, beta, `BI_prime`, and the full pruning order.

## Evaluate With Existing Orders

Available pruning depths match the table entries in `prune_orders/prune_orders.json`.

```bash
MODEL=qwen-2.5-7b \
MODEL_PATH=/path/to/Qwen2.5-7B \
METHOD=TAPPA \
ORDER_FILE=prune_orders/generated/qwen-2.5-7b_tappa_beta0.1.json \
N_PRUNE_LAYERS=11 \
bash scripts/lm_eval_prune.sh
```

ShortGPT baseline:

```bash
MODEL=qwen-2.5-7b \
MODEL_PATH=/path/to/Qwen2.5-7B \
METHOD=ShortGPT \
ORDER_FILE=prune_orders/generated/qwen-2.5-7b_shortgpt_bi.json \
N_PRUNE_LAYERS=11 \
bash scripts/lm_eval_prune.sh
```

`eval_lm.py` reads generated order files directly and uses the first `N_PRUNE_LAYERS` entries from the top-level `prune_order`. For ShortGPT, `prune_order` is the simple ascending order of BI scores.

`eval_lm.py` removes the selected transformer blocks from the loaded model object and refreshes attention layer indices. The checkpoint on disk remains unchanged.


## Data Notes

Precomputed BI and q-similarity values are stored under `prune_orders/raw_scores/` using the same schema as the offline scripts. Generated order files are stored under `prune_orders/generated/`.

Generated ShortGPT and TAPPA score files are the direct reproduction entrypoints for evaluation. `prune_orders/prune_orders.json` is kept as a reference file and optional manual layer-order database.

## End-to-End Example

```bash
MODEL_NAME=qwen-2.5-7b \
MODEL_PATH=/model/qwen-2.5-7b \
DATASET_PATH=/path/to/pg19_validation \
bash scripts/compute_bi.sh

MODEL_NAME=qwen-2.5-7b \
bash scripts/build_shortgpt_order.sh

MODEL=qwen-2.5-7b \
MODEL_PATH=/model/qwen-2.5-7b \
METHOD=ShortGPT \
ORDER_FILE=prune_orders/generated/qwen-2.5-7b_shortgpt_bi.json \
N_PRUNE_LAYERS=11 \
bash scripts/lm_eval_prune.sh

MODEL_NAME=qwen-2.5-7b \
MODEL_PATH=/model/qwen-2.5-7b \
DATASET_PATH=/path/to/pg19_validation \
bash scripts/compute_q_similarity.sh

MODEL_NAME=qwen-2.5-7b \
BETA=0.1 \
bash scripts/build_tappa_order.sh

MODEL=qwen-2.5-7b \
MODEL_PATH=/model/qwen-2.5-7b \
METHOD=TAPPA \
ORDER_FILE=prune_orders/generated/qwen-2.5-7b_tappa_beta0.1.json \
N_PRUNE_LAYERS=11 \
bash scripts/lm_eval_prune.sh
```
