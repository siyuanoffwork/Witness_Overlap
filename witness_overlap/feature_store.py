from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRIC_KEYS = {
    "svd": "mean_overlap",
    "cosine": "overlap_cos_signed",
}


def read_json(path: str | Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def feature_map(payload: dict) -> dict:
    return payload.get("features", payload)


def lookup_feature(features: dict, anchor: str, a: str, b: str, key: str) -> float | None:
    block = features.get(anchor, {})
    rec = block.get(f"{a}__VS__{b}") or block.get(f"{b}__VS__{a}")
    if rec is None:
        return None
    return rec.get(key)


def available_metrics(features: dict) -> list[str]:
    found = set()
    for block in features.values():
        if not isinstance(block, dict):
            continue
        for record in block.values():
            if not isinstance(record, dict):
                continue
            for name, key in METRIC_KEYS.items():
                if record.get(key) is not None:
                    found.add(name)
            if len(found) == len(METRIC_KEYS):
                break
    return [name for name in METRIC_KEYS if name in found]


def metric_items(features: dict, requested: list[str] | None = None) -> list[tuple[str, str]]:
    available = set(available_metrics(features))
    if requested:
        missing = [name for name in requested if name not in available]
        if missing:
            raise ValueError(f"Requested metric(s) not available in feature file: {missing}")
        names = requested
    else:
        names = [name for name in METRIC_KEYS if name in available]
    selected = [(name, METRIC_KEYS[name]) for name in names if name in available]
    if not selected:
        raise ValueError("No requested metrics are available in the feature file.")
    return selected
