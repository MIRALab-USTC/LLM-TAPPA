# KVCache: KV Cache Compression Budget Allocation

This folder contains the code used in the TAPPA paper for the **KV cache compression budget allocation** experiment. It runs LongBench-style evaluation with a unified entry script and supports multiple KV-cache methods (TAPPA, CAKE and several baselines).

## Scope

* Focus: KV cache compression under a fixed budget (`max_capacity_prompts`) and LongBench inference.
* This README documents only what is inside `KVCache/` and how to run the default script.

## Directory structure

* `pred_kvcache.py`: Main entry script.

  * Loads model and LongBench tasks.
  * Applies method-specific monkeypatching.
  * Runs generation and writes per-task predictions.
  * Invokes `eval.py` at the end.
* `script/longbench_tappa.sh`: Example shell script with a default TAPPA run.
* `tappa/`: TAPPA-specific implementation.

  * `patch.py`: Monkeypatch for the Hugging Face attention forward.
  * `model/llama_forward.py`, `model/qwen2_forward.py`: TAPPA-enabled forward implementations.
  * `tappa_qsim.py`: q-similarity utilities.
* `cake/`: Integrated CAKE implementation (upstream code with local adaptation).
* `pyramidkv/`: Integrated baselines from KVCache-Factory (upstream code with local adaptation).
* `config/`: JSON mapping files (models, datasets, prompts, max lengths, and related settings).
* `eval.py` and `metrics.py`: Evaluation and metrics scripts adapted from the official LongBench evaluation code.

Upstream repositories for integrated folders:

```text
CAKE-KV: https://github.com/antgroup/cakekv
KVCache-Factory: https://github.com/Zefan-Cai/KVCache-Factory
```

## Setup

### Environments

```text
CUDA 12.4 
transformers==4.44.2
torch==2.5.0
flash-attn==2.8.1
```

### Installation

```bash
pip install -r requirements.txt
```

## Data layout

LongBench data files are expected in JSONL format and loaded via `datasets.load_dataset("json", ...)`.

Place task files under the following relative path:

```text
KVCache/data/LongBench/<task>.jsonl
```

Each `<task>.jsonl` should follow the LongBench JSONL schema used by the official evaluation scripts.


## Quickstart

Run the provided example script from the repository root:

```bash
bash KVCache/script/longbench_tappa.sh
```

The script configures a TAPPA run with:

* `method="TAPPA"`
* `max_capacity_prompts=1024` (KV cache compression budget)
* `window_size=32` (used for computing compression metrics)
* `alpha="inf"` (TAPPA hyperparameter; `inf` means using q-similarity only)

## Key arguments

Arguments are defined in `pred_kvcache.py`.

### General

* `--model`: Model name (must match a key in `config/model2path.json`).
* `--method`: KV-cache method. Supported values include `FullKV`, `Cake`, `TAPPA`, `PyramidKV`, `SnapKV`, `H2O`, `StreamingLLM`.
* `--task`: LongBench task list as a space-separated string.
* `--pred_name`: Experiment name used in the output directory.
* `--device`: CUDA device index.

### Budget-related

* `--max_capacity_prompts`: KV cache budget parameter used by compression methods.
* `--window_size`: Recent-token window size used for computing compression metrics.



### TAPPA hyperparameters

* `--alpha`: TAPPA parameter controlling the weight of q-similarity in the preference score. `inf` is supported and means using q-similarity only.

### CAKE hyperparameters

These flags follow the CAKE official defaults (as stated in `pred_kvcache.py`).

* `--cascading`: Enable cascading cache management.
* `--tau1`, `--tau2`: CAKE parameters.
* `--gamma`: CAKE parameter.

## Outputs

`pred_kvcache.py` creates an output directory under:

```text
KVCache/pred_result/<cache_name>/<pred_name>/seed<seed>/<timestamp>/<model>/
```

For each task, a `<task>.jsonl` file is written containing predictions and metadata. `eval.py` then produces a `result.json` in the same directory.
