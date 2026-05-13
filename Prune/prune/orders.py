from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def normalize_scores(values: Iterable[float]) -> List[float]:
    values = [float(v) for v in values]
    if not values:
        raise ValueError("Cannot normalize an empty score list")
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def build_tappa_scores(bi: Iterable[float], q: Iterable[float], beta: float) -> Dict[str, Any]:
    bi = [float(v) for v in bi]
    q = [float(v) for v in q]
    if len(bi) != len(q):
        raise ValueError(f"BI and q lengths differ: {len(bi)} vs {len(q)}")

    bi_norm = normalize_scores(bi)
    q_norm = normalize_scores(q)
    tappa = [bi_i + beta * (1.0 - q_i) for bi_i, q_i in zip(bi_norm, q_norm)]
    order = sorted(range(len(tappa)), key=lambda idx: (tappa[idx], idx))
    return {
        "beta": beta,
        "bi": bi,
        "q_similarity": q,
        "bi_norm": bi_norm,
        "q_similarity_norm": q_norm,
        "tappa_score": tappa,
        "prune_order": order,
    }


def resolve_order_name(method_payload: Dict[str, Any], order_name: Optional[str], beta: Optional[float]) -> str:
    if order_name:
        return order_name
    if beta is not None:
        beta_key = f"BI+{beta:g}q"
        if beta_key in method_payload:
            return beta_key

    preferred = ["BI+0.2q", "BI+0.3q", "BI+0.1q", "BI+0.5q", "BI"]
    for candidate in preferred:
        if candidate in method_payload:
            return candidate
    return next(iter(method_payload))


def lookup_prune_order(
    order_db: Dict[str, Any],
    model: str,
    method: str,
    n_prune_layers: int,
    order_name: Optional[str] = None,
    beta: Optional[float] = None,
) -> List[int]:
    orders = order_db.get("orders", {})
    if model not in orders:
        available = ", ".join(sorted(orders))
        raise KeyError(f"Unknown model '{model}'. Available models: {available}")

    model_payload = orders[model]
    if method not in model_payload:
        available = ", ".join(sorted(model_payload))
        raise KeyError(f"Unknown method '{method}' for {model}. Available methods: {available}")

    method_payload = model_payload[method]
    resolved_name = resolve_order_name(method_payload, order_name, beta)
    if resolved_name not in method_payload:
        available = ", ".join(sorted(method_payload))
        raise KeyError(f"Unknown order '{resolved_name}' for {model}/{method}. Available orders: {available}")

    order_payload = method_payload[resolved_name]
    n_key = str(n_prune_layers)
    if n_key not in order_payload["orders"]:
        available = ", ".join(sorted(order_payload["orders"], key=int))
        raise KeyError(f"No {n_prune_layers}-layer order for {model}/{method}/{resolved_name}. Available: {available}")
    return [int(i) for i in order_payload["orders"][n_key]]


def resolve_prune_order_from_file(
    order_file: str | Path,
    model: str,
    method: str,
    n_prune_layers: int,
    order_name: Optional[str] = None,
    beta: Optional[float] = None,
) -> List[int]:
    payload = load_json(order_file)

    for explicit_key in ("layers_to_remove", "removed_layers"):
        if explicit_key in payload:
            layers = [int(i) for i in payload[explicit_key]]
            if n_prune_layers > len(layers):
                raise ValueError(
                    f"Requested {n_prune_layers} pruned layers, but {order_file} only contains {len(layers)}"
                )
            return layers[:n_prune_layers]

    if "orders" in payload:
        n_key = str(n_prune_layers)
        if n_key not in payload["orders"]:
            available = ", ".join(sorted(payload["orders"], key=int))
            raise KeyError(f"No {n_prune_layers}-layer order in {order_file}. Available: {available}")
        return [int(i) for i in payload["orders"][n_key]]

    if "prune_order" in payload:
        payload_model = payload.get("model")
        if payload_model is not None and payload_model != model:
            raise ValueError(f"Order file is for model '{payload_model}', but --model is '{model}'")

        payload_method = payload.get("method")
        if payload_method is not None and payload_method != method:
            raise ValueError(f"Order file is for method '{payload_method}', but --method is '{method}'")

        full_order = [int(i) for i in payload["prune_order"]]
        if n_prune_layers > len(full_order):
            raise ValueError(
                f"Requested {n_prune_layers} pruned layers, but {order_file} only contains {len(full_order)}"
            )
        return full_order[:n_prune_layers]

    return lookup_prune_order(
        order_db=payload,
        model=model,
        method=method,
        n_prune_layers=n_prune_layers,
        order_name=order_name,
        beta=beta,
    )
