from __future__ import annotations

import gc
from collections import defaultdict

import numpy as np
import torch

from .tensors import parameter_block


def _mean(values: dict[str, list[float]]) -> float:
    means = [float(np.mean(v)) for v in values.values() if v]
    return float(np.mean(means)) if means else 0.0


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return resolved


def signed_cosine(a: torch.Tensor, b: torch.Tensor, *, device: str | torch.device | None = "auto") -> float:
    dev = resolve_device(device)
    av = a.reshape(-1).to(device=dev, dtype=torch.float32, non_blocking=dev.type == "cuda")
    bv = b.reshape(-1).to(device=dev, dtype=torch.float32, non_blocking=dev.type == "cuda")
    aa = torch.dot(av, av)
    bb = torch.dot(bv, bv)
    if float(aa) < 1e-30 or float(bb) < 1e-30:
        return 0.0
    return float((torch.dot(av, bv) / torch.sqrt(aa * bb)).item())


def topk_right_basis(
    matrix: torch.Tensor,
    topk: int,
    *,
    device: str | torch.device | None = "auto",
    svd_niter: int = 2,
) -> torch.Tensor:
    k = min(topk, min(matrix.shape) - 1)
    if k < 1:
        return torch.empty(0, matrix.shape[1])
    dev = resolve_device(device)
    mat = matrix.to(device=dev, dtype=torch.float32, non_blocking=dev.type == "cuda")
    _, _, v = torch.svd_lowrank(
        mat,
        q=min(k + 5, min(mat.shape) - 1),
        niter=svd_niter,
    )
    return v[:, :k].T.cpu().contiguous()


def subspace_overlap(v1: torch.Tensor, v2: torch.Tensor) -> float:
    if v1.numel() == 0 or v2.numel() == 0:
        return 0.0
    k = v1.shape[0]
    mat = v1 @ v2.T
    return float((mat * mat).sum().item()) / max(1, k)


def triplet_overlap(
    anchor_weights: dict[str, torch.Tensor],
    target_weights: dict[str, torch.Tensor],
    witness_weights: dict[str, torch.Tensor],
    *,
    topk: int = 16,
    metrics: tuple[str, ...] = ("svd", "cosine"),
    device: str | torch.device | None = "auto",
    svd_niter: int = 2,
) -> dict:
    if isinstance(metrics, str):
        metrics = (metrics,)
    requested = set(metrics)
    unknown = requested - {"svd", "cosine"}
    if unknown:
        raise ValueError(f"Unknown metric(s): {sorted(unknown)}")
    dev = resolve_device(device)
    by_block_svd: dict[str, list[float]] = defaultdict(list)
    by_block_cos: dict[str, list[float]] = defaultdict(list)
    n_tensors = 0

    with torch.no_grad():
        for name in sorted(set(anchor_weights) & set(target_weights) & set(witness_weights)):
            ta = anchor_weights[name]
            tb = target_weights[name]
            tw = witness_weights[name]
            if not (ta.shape == tb.shape == tw.shape):
                continue
            d_ab = tb.float() - ta.float()
            d_aw = tw.float() - ta.float()
            if d_ab.ndim != 2 or min(d_ab.shape) < 2:
                continue
            block = parameter_block(name)
            if "cosine" in requested:
                by_block_cos[block].append(signed_cosine(d_ab, d_aw, device=dev))
            if "svd" in requested:
                try:
                    v_ab = topk_right_basis(d_ab, topk, device=dev, svd_niter=svd_niter)
                    v_aw = topk_right_basis(d_aw, topk, device=dev, svd_niter=svd_niter)
                    by_block_svd[block].append(subspace_overlap(v_ab, v_aw))
                except RuntimeError:
                    pass
            n_tensors += 1
            del d_ab, d_aw

    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    result = {"n_tensors": n_tensors}
    if "svd" in requested:
        result["mean_overlap"] = _mean(by_block_svd)
        result["by_block_overlap"] = {k: float(np.mean(v)) for k, v in by_block_svd.items() if v}
    if "cosine" in requested:
        result["overlap_cos_signed"] = _mean(by_block_cos)
        result["by_block_cos_signed"] = {k: float(np.mean(v)) for k, v in by_block_cos.items() if v}
    return result
