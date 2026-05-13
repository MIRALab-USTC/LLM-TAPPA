# Raw Score Files

This directory stores precomputed BI and q-similarity values in the same JSON schema produced by the offline score scripts.

Current files:

* `llama-2-7b_bi.json`
* `llama-2-7b_q_similarity.json`
* `llama-3.1-8b_bi.json`
* `llama-3.1-8b_q_similarity.json`
* `qwen-2.5-7b_bi.json`
* `qwen-2.5-7b_q_similarity.json`

The schema is produced by:

```bash
python compute_bi.py ...
python compute_q_similarity.py ...
```

Each file follows this schema:

```json
{
  "score_type": "BI",
  "model": "qwen-2.5-7b",
  "scores": [0.1, 0.2, 0.3]
}
```

Use `score_type: "q_similarity"` for q-similarity values.
