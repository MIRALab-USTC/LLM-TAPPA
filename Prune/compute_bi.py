from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from prune.model_utils import ShortGPTModel
from prune.orders import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ShortGPT block influence scores.")
    parser.add_argument("--model-path", required=True, help="Hugging Face model name or local model path.")
    parser.add_argument("--model-name", required=True, help="Name stored in the output metadata.")
    parser.add_argument("--dataset-path", default=None, help="Local datasets.load_from_disk path.")
    parser.add_argument("--dataset-name", default="emozilla/pg19", help="HF dataset name used when dataset-path is absent.")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layers-path", default="model.layers")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--angular", action="store_true")
    parser.add_argument("--output", default="prune_orders/raw_scores/bi_scores.json")
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def load_text_dataset(args: argparse.Namespace):
    if args.dataset_path:
        return load_from_disk(args.dataset_path)
    return load_dataset(args.dataset_name, split=args.dataset_split)


def batched_texts(dataset, text_field: str, batch_size: int, max_samples: int):
    batch = []
    used = 0
    for row in dataset:
        text = row[text_field]
        if not text:
            continue
        batch.append(text)
        used += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
        if used >= max_samples:
            break
    if batch:
        yield batch


def main() -> None:
    args = parse_args()
    dataset = load_text_dataset(args)
    model = ShortGPTModel(
        model_name_or_path=args.model_path,
        layers_path=args.layers_path,
        device=args.device,
        dtype=args.dtype,
        token=args.hf_token,
    )

    for prompts in tqdm(batched_texts(dataset, args.text_field, args.batch_size, args.max_samples)):
        model.eval_bi(
            prompts=prompts,
            max_seq_len=args.max_seq_len,
            stride=args.stride,
            max_gen_len=0,
            angular=args.angular,
        )

    scores = model.mean_bi()
    payload = {
        "score_type": "BI",
        "model": args.model_name,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "text_field": args.text_field,
        "max_samples": args.max_samples,
        "max_seq_len": args.max_seq_len,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "layers_path": args.layers_path,
        "angular": args.angular,
        "scores": scores,
        "prune_order": sorted(range(len(scores)), key=lambda idx: (scores[idx], idx)),
    }
    save_json(args.output, payload)
    print(json.dumps({"output": str(Path(args.output)), "num_layers": len(scores)}, indent=2))


if __name__ == "__main__":
    main()
