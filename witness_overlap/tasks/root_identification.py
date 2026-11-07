from __future__ import annotations

import argparse
import itertools

import numpy as np

from ..feature_store import METRIC_KEYS, feature_map, lookup_feature, metric_items, read_json, write_json
from ..registry import families, load_registry


def root_score(features: dict, candidate: str, others: list[str], key: str) -> float | None:
    values = []
    for a, b in itertools.combinations(others, 2):
        value = lookup_feature(features, candidate, a, b, key)
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Root identification with Witness Overlap.")
    parser.add_argument("--registry", default="data/model_registry.json")
    parser.add_argument("--features", default="results/triplet_features.json")
    parser.add_argument("--out", default="results/root_identification.json")
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRIC_KEYS), default=None)
    args = parser.parse_args()

    grouped = families(load_registry(args.registry))
    features = feature_map(read_json(args.features))
    result = {}

    for metric_name, key in metric_items(features, args.metrics):
        per_family = {}
        for family, models in grouped.items():
            ids = [model["model_id"] for model in models]
            true_root = next(model["model_id"] for model in models if model["role"] == "base")
            scores = {
                candidate: root_score(features, candidate, [x for x in ids if x != candidate], key)
                for candidate in ids
            }
            scores = {candidate: score for candidate, score in scores.items() if score is not None}
            if not scores:
                continue
            predicted = min(scores, key=scores.get)
            per_family[family] = {
                "true_root": true_root,
                "predicted_root": predicted,
                "correct": predicted == true_root,
                "scores": scores,
            }
        correct = sum(int(row["correct"]) for row in per_family.values())
        result[metric_name] = {
            "root_correct": correct,
            "n_families": len(per_family),
            "root_accuracy": correct / max(1, len(per_family)),
            "per_family": per_family,
        }

    write_json(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
