from __future__ import annotations

import argparse
import itertools
import math
import random
from collections import defaultdict

import numpy as np

from ..feature_store import METRIC_KEYS, feature_map, lookup_feature, metric_items, read_json, write_json
from ..registry import families, load_registry, parent_child_edges


def direction_records(features: dict, grouped: dict, edges: list[tuple[str, str, str]], key: str) -> dict:
    records = {}
    for family, parent, child in edges:
        witnesses = [
            model["model_id"]
            for model in grouped[family]
            if model["model_id"] not in {parent, child}
        ]
        rows = []
        for witness in witnesses:
            parent_score = lookup_feature(features, parent, child, witness, key)
            child_score = lookup_feature(features, child, parent, witness, key)
            if parent_score is None or child_score is None:
                continue
            rows.append(
                {
                    "witness": witness,
                    "parent_score": parent_score,
                    "child_score": child_score,
                    "margin": child_score - parent_score,
                    "correct": child_score > parent_score,
                }
            )
        records[f"{parent}__TO__{child}"] = {"family": family, "parent": parent, "child": child, "witnesses": rows}
    return records


def choose_combinations(items: list, k: int, samples: int | None, rng: random.Random):
    n_total = math.comb(len(items), k)
    if samples is None or samples <= 0 or n_total <= samples:
        yield from itertools.combinations(items, k)
        return
    seen = set()
    while len(seen) < samples:
        idx = tuple(sorted(rng.sample(range(len(items)), k)))
        if idx in seen:
            continue
        seen.add(idx)
        yield [items[i] for i in idx]


def k_sweep(records: dict, max_k: int, samples: int | None, seed: int) -> list[dict]:
    """Average witness-combination accuracy within each edge, then across edges."""
    rng = random.Random(seed)
    rows = []
    for k in range(1, max_k + 1):
        edge_accuracies = []
        edge_margins = []
        n_decisions = 0
        for record in records.values():
            witnesses = record["witnesses"]
            if len(witnesses) < k:
                continue
            decisions = []
            margins = []
            for combo in choose_combinations(witnesses, k, samples, rng):
                margin = float(np.mean([row["margin"] for row in combo]))
                margins.append(margin)
                decisions.append(margin > 0.0)
            edge_accuracies.append(float(np.mean(decisions)))
            edge_margins.append(float(np.mean(margins)))
            n_decisions += len(decisions)
        if edge_accuracies:
            rows.append(
                {
                    "N_w": k,
                    "accuracy": float(np.mean(edge_accuracies)),
                    "mean_margin": float(np.mean(edge_margins)),
                    "n_decisions": n_decisions,
                    "n_edges": len(edge_accuracies),
                    "accuracy_weighting": "equal_per_edge",
                }
            )

    decisions = []
    margins = []
    for record in records.values():
        witnesses = record["witnesses"]
        if not witnesses:
            continue
        margin = float(np.mean([row["margin"] for row in witnesses]))
        margins.append(margin)
        decisions.append(margin > 0.0)
    rows.append(
        {
            "N_w": "all",
            "accuracy": float(np.mean(decisions)) if decisions else None,
            "mean_margin": float(np.mean(margins)) if margins else None,
            "n_decisions": len(decisions),
            "n_edges": len(decisions),
            "accuracy_weighting": "equal_per_edge",
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Parent-child direction and N_w sweep.")
    parser.add_argument("--registry", default="data/model_registry.json")
    parser.add_argument("--features", default="results/triplet_features.json")
    parser.add_argument("--out", default="results/parent_child_direction.json")
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "Optional Monte-Carlo cap per edge and N_w. Omit it, or pass 0, "
            "to enumerate all witness combinations."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRIC_KEYS), default=None)
    args = parser.parse_args()

    models = load_registry(args.registry)
    grouped = families(models)
    edges = parent_child_edges(models)
    features = feature_map(read_json(args.features))
    result = {"n_edges": len(edges), "metrics": {}}

    for metric_name, key in metric_items(features, args.metrics):
        records = direction_records(features, grouped, edges, key)
        single = [row["correct"] for rec in records.values() for row in rec["witnesses"]]
        result["metrics"][metric_name] = {
            "single_witness_triplet_accuracy": float(np.mean(single)) if single else None,
            "n_single_witness_triplets": len(single),
            "k_sweep": k_sweep(records, args.max_k, args.samples, args.seed),
            "per_edge": records,
        }

    write_json(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
