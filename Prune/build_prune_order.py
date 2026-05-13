from __future__ import annotations

import argparse
import json
from pathlib import Path

from prune.orders import build_tappa_scores, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TAPPA pruning order from BI and q-similarity files.")
    parser.add_argument("--bi-file", required=True, help="JSON file produced by compute_bi.py.")
    parser.add_argument("--q-file", required=True, help="JSON file produced by compute_q_similarity.py.")
    parser.add_argument("--beta", type=float, required=True, help="TAPPA beta in BI' = BI + beta * (1 - q).")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bi_payload = load_json(args.bi_file)
    q_payload = load_json(args.q_file)
    scores = build_tappa_scores(bi_payload["scores"], q_payload["scores"], args.beta)
    model_name = args.model_name or bi_payload.get("model") or q_payload.get("model")
    payload = {
        "method": "TAPPA",
        "formula": "BI_prime = BI_norm + beta * (1 - q_similarity_norm)",
        "model": model_name,
        "bi_file": str(Path(args.bi_file)),
        "q_file": str(Path(args.q_file)),
        **scores,
    }
    save_json(args.output, payload)
    print(json.dumps({"output": args.output, "num_layers": len(payload["prune_order"])}, indent=2))


if __name__ == "__main__":
    main()
