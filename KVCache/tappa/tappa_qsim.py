import math

import torch


def compute_q_sim_torch(
    q: torch.Tensor,
    mode: str = "cosine",
    rbf_sigma: float = 1.0,
    kl_eps: float = 1e-8,
) -> float:
    """
    Compute TAPPA q_sim from a query time series.

    Parameters
    ----------
    q:
        Query states with shape [T, D] or [H, T, D].
    mode:
        Similarity / distance metric:
        - "cosine"   : average cosine similarity (default)
        - "dot"      : dot product
        - "euclidean": L2 distance
        - "rbf"      : RBF kernel similarity
        - "l1"       : L1 distance
        - "pearson"  : Pearson correlation
        - "angular"  : angular similarity
        - "kl"       : symmetric KL divergence (treat each vector as a distribution)
    rbf_sigma:
        Sigma for the RBF kernel.
    kl_eps:
        Numerical constant for KL computation.

    Returns
    -------
    float:
        A scalar q_sim score.
    """
    if q.dim() == 2:
        q = q.unsqueeze(0)  # [1, T, D]
    elif q.dim() != 3:
        raise ValueError("q must have shape [T, D] or [H, T, D].")

    H, T, D = q.shape
    if T <= 1:
        return float(q.new_tensor(0.0).item())

    eps = 1e-8

    def avg_offdiag(mat: torch.Tensor) -> torch.Tensor:
        # mat: [H, T, T] -> [H]
        total = mat.sum(dim=(-1, -2))
        diag = mat.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return (total - diag) / (T * (T - 1))

    if mode == "cosine":
        norm = q.norm(dim=-1, keepdim=True).clamp_min(eps)  # [H, T, 1]
        qn = q / norm
        sim = torch.matmul(qn, qn.transpose(-1, -2))  # [H, T, T]
        q_sim = avg_offdiag(sim).mean()
    elif mode == "dot":
        sim = torch.matmul(q, q.transpose(-1, -2))
        q_sim = avg_offdiag(sim).mean()
    elif mode == "euclidean":
        q_i = q.unsqueeze(-2)  # [H, T, 1, D]
        q_j = q.unsqueeze(-3)  # [H, 1, T, D]
        dist = torch.norm(q_i - q_j, dim=-1)  # [H, T, T]
        q_sim = avg_offdiag(dist).mean()
    elif mode == "l1":
        q_i = q.unsqueeze(-2)
        q_j = q.unsqueeze(-3)
        dist = torch.abs(q_i - q_j).sum(dim=-1)
        q_sim = avg_offdiag(dist).mean()
    elif mode == "rbf":
        q_i = q.unsqueeze(-2)
        q_j = q.unsqueeze(-3)
        dist2 = ((q_i - q_j) ** 2).sum(dim=-1)
        sim = torch.exp(-dist2 / (2 * (rbf_sigma ** 2)))
        q_sim = avg_offdiag(sim).mean()
    elif mode == "pearson":
        q_centered = q - q.mean(dim=-1, keepdim=True)
        denom = (q_centered.norm(dim=-1, keepdim=True).clamp_min(eps))
        qn = q_centered / denom
        corr = torch.matmul(qn, qn.transpose(-1, -2)) / D
        q_sim = avg_offdiag(corr).mean()
    elif mode == "angular":
        norm = q.norm(dim=-1, keepdim=True).clamp_min(eps)
        qn = q / norm
        cos_sim = torch.matmul(qn, qn.transpose(-1, -2)).clamp(-1.0, 1.0)
        ang = torch.acos(cos_sim) / math.pi
        q_sim = avg_offdiag(ang).mean()
    elif mode == "kl":
        logits = q.to(torch.float32)
        probs = torch.softmax(logits, dim=-1)
        probs = probs.clamp_min(kl_eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        log_probs = torch.log(probs)

        P_i = probs.unsqueeze(-2)     # [H, T, 1, D]
        P_j = probs.unsqueeze(-3)     # [H, 1, T, D]
        logP_i = log_probs.unsqueeze(-2)
        logP_j = log_probs.unsqueeze(-3)

        diff_log = logP_i - logP_j
        sym_kl = 0.5 * ((P_i - P_j) * diff_log).sum(dim=-1)  # [H, T, T]
        q_sim = avg_offdiag(sym_kl).mean()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return float(q_sim.item())
