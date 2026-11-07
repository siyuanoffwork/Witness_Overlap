from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ..feature_store import METRIC_KEYS, lookup_feature, write_json
from ..metrics import triplet_overlap
from ..tensors import load_projection_tensors


def base_fixed_anchor_score(
    features: dict,
    *,
    anchor: str,
    base_label: str,
    stages: list[str],
    key: str,
) -> tuple[float | None, list[dict]]:
    values = []
    rows = []
    for witness in stages:
        if witness == anchor:
            continue
        value = lookup_feature(features, anchor, base_label, witness, key)
        rows.append({"target": base_label, "witness": witness, "overlap": value})
        if value is not None:
            values.append(value)
    return (float(np.mean(values)) if values else None, rows)


def base_fixed_anchor_order(features: dict, chain: list[str], key: str) -> dict:
    """Order post-base checkpoints using base-fixed anchor overlaps.

    The first checkpoint in ``chain`` is treated as the base. For a post-base
    checkpoint A, the score averages O(A; base, W) over all other post-base
    checkpoints W. Lower scores are interpreted as earlier post-base stages.
    """
    base_label = chain[0]
    stages = chain[1:]
    rows = []
    for candidate in stages:
        score, overlaps = base_fixed_anchor_score(
            features,
            anchor=candidate,
            base_label=base_label,
            stages=stages,
            key=key,
        )
        rows.append(
            {
                "model_id": candidate,
                "mean_base_fixed_anchor_overlap": score,
                "base_fixed_overlaps": overlaps,
            }
        )
    available = [row for row in rows if row["mean_base_fixed_anchor_overlap"] is not None]
    predicted_stages = sorted(available, key=lambda row: row["mean_base_fixed_anchor_overlap"])
    return {
        "base_label": base_label,
        "score_direction": (
            "base is fixed as one endpoint; lower mean_base_fixed_anchor_overlap = "
            "earlier post-base stage; higher = later post-base stage"
        ),
        "predicted_order": [base_label, *[row["model_id"] for row in predicted_stages]],
        "scores": rows,
    }


def parse_chain_paths(items: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    seen = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"chain path must have the form label=/path/to/checkpoint: {item}")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty chain label in: {item}")
        if label in seen:
            raise ValueError(f"duplicate chain label: {label}")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        parsed.append((label, path))
        seen.add(label)
    if len(parsed) < 3:
        raise ValueError("chain must contain a base plus at least two post-base checkpoints")
    return parsed


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_features_from_paths(
    chain_paths: list[tuple[str, Path]],
    *,
    metrics: list[str],
    topk: int,
    device: str,
    svd_niter: int,
    seed: int,
) -> dict:
    set_seed(seed)
    weights = {}
    for label, path in chain_paths:
        weights[label] = load_projection_tensors(path)
        n_params = sum(t.numel() for t in weights[label].values())
        print(f"loaded {label}: {len(weights[label])} tensors, {n_params * 4 / 1e9:.2f} GB fp32")

    labels = [label for label, _ in chain_paths]
    base_label = labels[0]
    stages = labels[1:]
    features = {}
    for anchor in stages:
        block = {}
        for witness in stages:
            if witness == anchor:
                continue
            block[f"{base_label}__VS__{witness}"] = triplet_overlap(
                weights[anchor],
                weights[base_label],
                weights[witness],
                topk=topk,
                metrics=tuple(metrics),
                device=device,
                svd_niter=svd_niter,
            )
            print(f"computed {anchor} | {base_label}__VS__{witness}")
        features[anchor] = block
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover a single chain order by average anchor scoring.")
    parser.add_argument(
        "--chain-paths",
        nargs="+",
        required=True,
        metavar="LABEL=PATH",
        help="Ordered local checkpoints, e.g. base=/path/base sft0=/path/stage0/model.",
    )
    parser.add_argument("--out", default="results/single_chain_order.json")
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRIC_KEYS), default=None)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--svd-niter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chain_paths = parse_chain_paths(args.chain_paths)
    chain = [label for label, _ in chain_paths]
    requested_metrics = args.metrics or list(METRIC_KEYS)
    features = compute_features_from_paths(
        chain_paths,
        metrics=requested_metrics,
        topk=args.topk,
        device=args.device,
        svd_niter=args.svd_niter,
        seed=args.seed,
    )

    result = {
        "true_order": chain,
        "base_label": chain[0],
        "chain_paths": {label: str(path) for label, path in chain_paths},
        "method_note": (
            "The first checkpoint is treated as the base. Each post-base checkpoint is used "
            "as an anchor against pairs of the form base|other_post_base_checkpoint. Lower "
            "mean base-fixed anchor overlap is earlier, and higher mean overlap is later."
        ),
        "metrics": {},
    }
    for metric_name in requested_metrics:
        key = METRIC_KEYS[metric_name]
        result["metrics"][metric_name] = {
            "base_fixed_anchor_order": base_fixed_anchor_order(features, chain, key),
        }
    write_json(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
