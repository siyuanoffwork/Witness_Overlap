from __future__ import annotations

import argparse
import gc
import itertools
import random
import time
from pathlib import Path

import numpy as np
import torch

from .feature_store import feature_map, read_json, write_json
from .metrics import resolve_device, triplet_overlap
from .registry import families, load_registry
from .tensors import load_projection_tensors

METRIC_KEYS = {
    "svd": "mean_overlap",
    "cosine": "overlap_cos_signed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Witness Overlap triplet features.")
    parser.add_argument("--registry", default="data/model_registry.json")
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--out", default="results/triplet_features.json")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRIC_KEYS), default=["svd", "cosine"])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--svd-niter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample-layers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="Recompute records even if they already exist.")
    return parser.parse_args()


def has_requested_metrics(record: dict, metrics: list[str]) -> bool:
    return record.get("n_tensors", 0) > 0 and all(METRIC_KEYS[name] in record for name in metrics)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    device_note = str(device)
    if device.type == "cuda":
        device_note += f" ({torch.cuda.get_device_name(device)})"
    print(
        f"device={device_note}; metrics={','.join(args.metrics)}; "
        f"topk={args.topk}; svd_niter={args.svd_niter}",
        flush=True,
    )
    models = load_registry(args.registry, args.models_root)
    grouped = families(models)
    if args.families:
        grouped = {name: grouped[name] for name in args.families}

    out_path = Path(args.out)
    payload = read_json(out_path) if args.resume and out_path.exists() else {}
    features = feature_map(payload) if payload else {}

    for family, family_models in grouped.items():
        if len(family_models) < 3:
            print(f"[skip] {family}: need at least 3 checkpoints")
            continue
        print(f"\n=== {family}: {len(family_models)} checkpoints ===", flush=True)
        loaded: dict[str, dict] = {}

        def get(model: dict) -> dict:
            model_id = model["model_id"]
            if model_id not in loaded:
                t0 = time.time()
                loaded[model_id] = load_projection_tensors(
                    model["local_path"],
                    subsample_layers=args.subsample_layers,
                )
                n_params = sum(t.numel() for t in loaded[model_id].values())
                print(
                    f"  load {model_id}: {len(loaded[model_id])} tensors, "
                    f"{n_params * 4 / 1e9:.2f} GB, {time.time() - t0:.1f}s",
                    flush=True,
                )
            return loaded[model_id]

        def evict(keep: set[str]) -> None:
            for model_id in list(loaded):
                if model_id not in keep:
                    del loaded[model_id]
            gc.collect()

        for anchor in family_models:
            anchor_id = anchor["model_id"]
            anchor_block = features.get(anchor_id, {})
            others = [m for m in family_models if m["model_id"] != anchor_id]
            for target, witness in itertools.combinations(others, 2):
                key = f"{target['model_id']}__VS__{witness['model_id']}"
                if (
                    not args.force
                    and key in anchor_block
                    and has_requested_metrics(anchor_block[key], args.metrics)
                ):
                    continue
                keep = {anchor_id, target["model_id"], witness["model_id"]}
                evict(keep)
                t0 = time.time()
                record = dict(anchor_block.get(key, {}))
                record.update(
                    triplet_overlap(
                        get(anchor),
                        get(target),
                        get(witness),
                        topk=args.topk,
                        metrics=tuple(args.metrics),
                        device=device,
                        svd_niter=args.svd_niter,
                    )
                )
                anchor_block[key] = record
                print(f"  {anchor_id} | {key}: {time.time() - t0:.1f}s", flush=True)
            features[anchor_id] = anchor_block
            write_json(
                out_path,
                {
                    "metadata": {
                        "registry": str(args.registry),
                        "topk": args.topk,
                        "metrics": args.metrics,
                        "device": str(device),
                        "svd_niter": args.svd_niter,
                        "seed": args.seed,
                        "subsample_layers": args.subsample_layers,
                    },
                    "features": features,
                },
            )
            evict(set())

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
