from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from .registry import default_dataset_root, load_registry


def is_merged_adapter_entry(row: dict) -> bool:
    notes = " ".join(str(note) for note in row.get("notes", []))
    return "sft_merged" in Path(row["path"]).parts or "Merged from LoRA adapter" in notes


def adapter_path_for(row: dict, models_root: Path) -> Path:
    repo_parts = row["hf_repo_id"].split("/")
    stage = row.get("stage") or row.get("role") or "adapter"
    return models_root / "models" / row["family"] / f"{stage}_adapter" / Path(*repo_parts)


def dtype_from_name(name: str):
    import torch

    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def merge_peft_adapter(
    *,
    base_dir: Path,
    adapter_dir: Path,
    out_dir: Path,
    dtype_name: str,
    merge_device: str,
    device_map: str | None,
    max_shard_size: str,
) -> None:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Merging LoRA/PEFT adapters requires `transformers` and `peft`. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc

    import torch

    torch_dtype = dtype_from_name(dtype_name)
    if merge_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA merge was requested, but torch.cuda.is_available() is False.")
    kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if merge_device == "cuda":
        kwargs["device_map"] = {"": "cuda:0"}
    elif merge_device == "cpu":
        kwargs["device_map"] = {"": "cpu"}
    elif device_map:
        kwargs["device_map"] = device_map

    print(f"  merge base    = {base_dir}")
    print(f"  merge adapter = {adapter_dir}")
    print(f"  merge output  = {out_dir}")
    print(f"  merge device  = {merge_device}; cuda_available = {torch.cuda.is_available()}")
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), **kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size=max_shard_size)

    tokenizer = None
    for source in (adapter_dir, base_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)
            break
        except Exception:
            tokenizer = None
    if tokenizer is not None:
        tokenizer.save_pretrained(str(out_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all HuggingFace repositories listed in model_registry.json "
            "to their registry-relative paths under --models-root."
        )
    )
    parser.add_argument("--registry", default="data/model_registry.json")
    parser.add_argument(
        "--models-root",
        default=None,
        help=(
            "Dataset root used to resolve registry paths. If a registry path is "
            "'models/family/...', files are downloaded to '<models-root>/models/family/...'. "
            "Defaults to DATASET_ROOT or the parent dataset directory."
        ),
    )
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--model-ids", nargs="*", default=None)
    parser.add_argument("--token", default=None, help="Optional HuggingFace token. Defaults to HF_TOKEN.")
    parser.add_argument("--force", action="store_true", help="Download even if the target directory exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without downloading.")
    parser.add_argument("--revision", default=None, help="Optional HuggingFace revision for all downloads.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--skip-merge", action="store_true", help="Download PEFT adapters but do not merge them.")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--merge-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map used while merging PEFT adapters when --merge-device auto.",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_root = Path(args.models_root).expanduser().resolve() if args.models_root else default_dataset_root()
    rows = load_registry(args.registry, models_root)
    token = args.token or os.environ.get("HF_TOKEN")

    if args.families:
        keep = set(args.families)
        rows = [row for row in rows if row["family"] in keep]
    if args.model_ids:
        keep = set(args.model_ids)
        rows = [row for row in rows if row["model_id"] in keep]

    print(f"models_root = {models_root}")
    print(f"registry    = {args.registry}")
    print(f"n_models    = {len(rows)}")

    all_rows = load_registry(args.registry, models_root)
    all_by_id = {row["model_id"]: row for row in all_rows}
    completed: set[str] = set()

    def download_repo(repo_id: str, local_dir: Path) -> None:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=args.revision,
            local_dir=str(local_dir),
            token=token,
            max_workers=args.max_workers,
        )

    def handle_row(row: dict) -> None:
        if row["model_id"] in completed:
            return
        repo_id = row["hf_repo_id"]
        local_dir = Path(row["local_path"])
        merged_adapter = is_merged_adapter_entry(row)

        if local_dir.exists() and any(local_dir.iterdir()) and not args.force:
            print(f"[skip] {row['model_id']}: {local_dir}")
            completed.add(row["model_id"])
            return

        if merged_adapter:
            parent_id = row.get("claimed_parent_model_id")
            if not parent_id or parent_id not in all_by_id:
                raise RuntimeError(f"{row['model_id']} is marked merged but has no known parent in the registry")
            parent = all_by_id[parent_id]
            handle_row(parent)
            adapter_dir = adapter_path_for(row, models_root)
            print(f"[adapter] {row['model_id']}  {repo_id} -> {adapter_dir}")
            print(f"[merge]   {row['model_id']}  {parent_id} + adapter -> {local_dir}")
            if args.dry_run:
                completed.add(row["model_id"])
                return
            adapter_dir.mkdir(parents=True, exist_ok=True)
            if args.force or not (adapter_dir.exists() and any(adapter_dir.iterdir())):
                download_repo(repo_id, adapter_dir)
            if not args.skip_merge:
                merge_peft_adapter(
                    base_dir=Path(parent["local_path"]),
                    adapter_dir=adapter_dir,
                    out_dir=local_dir,
                    dtype_name=args.torch_dtype,
                    merge_device=args.merge_device,
                    device_map=args.device_map,
                    max_shard_size=args.max_shard_size,
                )
            completed.add(row["model_id"])
            return

        print(f"[download] {row['model_id']}  {repo_id} -> {local_dir}")
        if args.dry_run:
            completed.add(row["model_id"])
            return
        local_dir.mkdir(parents=True, exist_ok=True)
        download_repo(repo_id, local_dir)
        completed.add(row["model_id"])

    for row in rows:
        handle_row(row)


if __name__ == "__main__":
    main()
