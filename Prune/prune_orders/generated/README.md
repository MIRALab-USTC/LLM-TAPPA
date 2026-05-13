# Generated Pruning Orders

This directory stores generated pruning-order files used directly by `eval_lm.py`.

ShortGPT files are generated from BI scores by `build_shortgpt_order.py`.

TAPPA files are generated from BI and q-similarity scores by `build_prune_order.py`; each TAPPA file stores raw BI, raw q-similarity, normalized values, beta, TAPPA scores, and the full layer pruning order.

`eval_lm.py` can consume these files directly through `--order-file`. It removes the first `--n-prune-layers` entries from the top-level `prune_order`.

`prune_orders/prune_orders.json` is kept as a reference file and optional manual layer-order database.
