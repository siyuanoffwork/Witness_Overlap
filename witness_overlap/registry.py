from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dataset_root() -> Path:
    env = os.environ.get("DATASET_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def default_registry_path() -> Path:
    return project_root() / "data" / "model_registry.json"


def load_registry(
    path: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    registry_path = Path(path) if path is not None else default_registry_path()
    root = Path(dataset_root) if dataset_root is not None else default_dataset_root()
    rows = json.load(open(registry_path))
    out = []
    for rec in rows:
        item = dict(rec)
        item["role"] = item.get("role") or item.get("stage")
        raw_path = item.get("local_path") or item.get("path")
        if raw_path is None:
            raise KeyError(f"missing path/local_path for {item.get('model_id')}")
        model_path = Path(raw_path)
        item["local_path"] = str(model_path if model_path.is_absolute() else root / model_path)
        out.append(item)
    return out


def families(models: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in models:
        grouped[model["family"]].append(model)
    return dict(grouped)


def model_index(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {model["model_id"]: model for model in models}


def descendant_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [model for model in models if model.get("role") in {"sft", "rlhf"}]


def parent_child_edges(models: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Return (family, parent_id, child_id) direct edges from claimed_parent_model_id."""
    known = model_index(models)
    edges = []
    for child in descendant_models(models):
        parent = child.get("claimed_parent_model_id")
        if parent and parent in known:
            edges.append((child["family"], parent, child["model_id"]))
    return edges
