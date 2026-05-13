from __future__ import annotations

import argparse
import json
from pathlib import Path

from prune.orders import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a ShortGPT pruning order from a BI score file.")
    parser.add_argument("--bi-file", required=True, help="JSON file produced by compute_bi.py.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bi_payload = load_json(args.bi_file)
    scores = [float(v) for v in bi_payload["scores"]]
    model_name = args.model_name or bi_payload.get("model")
    payload = {
        "method": "ShortGPT",
        "score_type": "BI",
        "model": model_name,
        "bi_file": str(Path(args.bi_file)),
        "bi": scores,
        "prune_order": sorted(range(len(scores)), key=lambda idx: (scores[idx], idx)),
    }
    save_json(args.output, payload)
    print(json.dumps({"output": args.output, "num_layers": len(payload["prune_order"])}, indent=2))


if __name__ == "__main__":
    main()
