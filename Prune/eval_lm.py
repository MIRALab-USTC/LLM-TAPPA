from __future__ import annotations

import argparse
import json
from pathlib import Path

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer

from prune.model_utils import remove_layers
from prune.orders import resolve_prune_order_from_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a layer-pruned model with lm-eval.")
    parser.add_argument("--model", required=True, help="Model key used in the pruning order file.")
    parser.add_argument("--model-path", required=True, help="Hugging Face model name or local path.")
    parser.add_argument("--method", choices=["ShortGPT", "TAPPA"], default="TAPPA")
    parser.add_argument("--order-name", default=None, help="Order name such as BI, BI+0.2q, or BI+0.3q.")
    parser.add_argument("--beta", type=float, default=None, help="Resolve TAPPA order as BI+{beta}q when set.")
    parser.add_argument("--n-prune-layers", type=int, default=10)
    parser.add_argument(
        "--order-file",
        default="prune_orders/prune_orders.json",
        help=(
            "Either a generated TAPPA score file with a top-level prune_order, "
            "or the legacy prune_orders/prune_orders.json database."
        ),
    )
    parser.add_argument("--layers-path", default="model.layers")
    parser.add_argument("--tasks", nargs="+", default=["piqa", "hellaswag", "arc_easy", "winogrande"])
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--save-dir", default="results")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def metric_value(result: dict) -> float:
    if "acc_norm,none" in result:
        return result["acc_norm,none"]
    if "acc,none" in result:
        return result["acc,none"]
    return next(iter(result.values()))


def main() -> None:
    args = parse_args()
    layers_to_remove = resolve_prune_order_from_file(
        order_file=args.order_file,
        model=args.model,
        method=args.method,
        n_prune_layers=args.n_prune_layers,
        order_name=args.order_name,
        beta=args.beta,
    )

    torch_dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    removed = remove_layers(model, layers_to_remove, layers_path=args.layers_path)
    model.to(args.device)
    model.eval()

    task_manager = TaskManager()
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    results = lm_eval.simple_evaluate(
        model=hflm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        task_manager=task_manager,
    )["results"]

    metrics = {task: round(metric_value(result), 4) for task, result in results.items()}
    metrics["average"] = round(sum(metrics.values()) / len(metrics), 4) if metrics else 0.0
    payload = {
        "model": args.model,
        "method": args.method,
        "order_name": args.order_name,
        "beta": args.beta,
        "order_file": args.order_file,
        "n_prune_layers": args.n_prune_layers,
        "removed_layers": removed,
        "tasks": args.tasks,
        "metrics": metrics,
    }

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{args.model}_{args.method}_{args.n_prune_layers}_layers.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
