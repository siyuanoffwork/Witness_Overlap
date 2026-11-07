from __future__ import annotations

import argparse
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

from ..feature_store import METRIC_KEYS, feature_map, lookup_feature, metric_items, read_json, write_json
from ..registry import families, load_registry, model_index


def is_parent_child(a: str, b: str, by_id: dict) -> bool:
    return by_id[a].get("claimed_parent_model_id") == b or by_id[b].get("claimed_parent_model_id") == a


def build_rows(features: dict, grouped: dict, by_id: dict, key: str) -> list[dict]:
    rows = []
    for family, models in grouped.items():
        ids = [model["model_id"] for model in models]
        for a, b in itertools.combinations(ids, 2):
            label = int(is_parent_child(a, b, by_id))
            for witness in ids:
                if witness in {a, b}:
                    continue
                score_a = lookup_feature(features, a, b, witness, key)
                score_b = lookup_feature(features, b, a, witness, key)
                if score_a is None or score_b is None:
                    continue
                rows.append(
                    {
                        "family": family,
                        "a": a,
                        "b": b,
                        "witness": witness,
                        "label_pc": label,
                        "feature": min(score_a, score_b),
                    }
                )
    return rows


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def evaluate_lofo(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_samples": 0,
            "n_pc": 0,
            "n_sibling": 0,
            "lofo_auroc": None,
            "lofo_balanced_accuracy": None,
            "per_family": {},
        }
    labels = np.array([row["label_pc"] for row in rows], dtype=int)
    x = np.array([[row["feature"]] for row in rows], dtype=float)
    raw_feature = x[:, 0]
    fams = np.array([row["family"] for row in rows])
    in_auroc = float(roc_auc_score(labels, -raw_feature))
    in_threshold = -youden_threshold(labels, -raw_feature)
    in_pred = (raw_feature <= in_threshold).astype(int)
    in_balanced_accuracy = float(balanced_accuracy_score(labels, in_pred))
    per_family = {}
    for family in sorted(set(fams.tolist())):
        train = fams != family
        test = fams == family
        if len(set(labels[train].tolist())) < 2 or len(set(labels[test].tolist())) < 2:
            continue
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(x[train], labels[train])
        train_probability = classifier.predict_proba(x[train])[:, 1]
        test_probability = classifier.predict_proba(x[test])[:, 1]
        threshold = youden_threshold(labels[train], train_probability)
        pred = (test_probability >= threshold).astype(int)
        auroc = float(roc_auc_score(labels[test], test_probability))
        balanced = float(balanced_accuracy_score(labels[test], pred))
        per_family[family] = {
            "threshold": threshold,
            "coef": float(classifier.coef_[0, 0]),
            "intercept": float(classifier.intercept_[0]),
            "auroc": auroc,
            "balanced_accuracy": balanced,
            "n_test": int(test.sum()),
            "n_pc_test": int(labels[test].sum()),
        }
    aurocs = [v["auroc"] for v in per_family.values()]
    balanced_scores = [v["balanced_accuracy"] for v in per_family.values()]
    return {
        "n_samples": len(rows),
        "n_pc": int(labels.sum()),
        "n_sibling": int((labels == 0).sum()),
        "in_auroc_raw_feature": in_auroc,
        "in_balanced_accuracy_raw_feature": in_balanced_accuracy,
        "in_raw_feature_threshold": in_threshold,
        "classifier": "LogisticRegression(max_iter=1000)",
        "decision_threshold": "Youden threshold selected on training-family predicted probabilities",
        "lofo_auroc": float(np.mean(aurocs)) if aurocs else None,
        "lofo_balanced_accuracy": float(np.mean(balanced_scores)) if balanced_scores else None,
        "per_family": per_family,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sibling versus parent-child pair discrimination.")
    parser.add_argument("--registry", default="data/model_registry.json")
    parser.add_argument("--features", default="results/triplet_features.json")
    parser.add_argument("--out", default="results/pair_discrimination.json")
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRIC_KEYS), default=None)
    args = parser.parse_args()

    models = load_registry(args.registry)
    grouped = families(models)
    by_id = model_index(models)
    features = feature_map(read_json(args.features))
    result = {}
    for metric_name, key in metric_items(features, args.metrics):
        rows = build_rows(features, grouped, by_id, key)
        result[metric_name] = evaluate_lofo(rows)
    write_json(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
