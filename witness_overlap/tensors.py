from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import safe_open

SKIP_BLOCKS = {"emb", "lm_head", "layernorm", "other"}
LAYER_RE = re.compile(r"\.(\d+)\.")


def _is_aux_file(name: str) -> bool:
    low = name.lower()
    return any(token in low for token in ("adapter", "lora", "optimizer", "scheduler"))


def _looks_like_lora(keys: list[str]) -> bool:
    sample = keys[:32]
    return bool(sample) and all(
        ("lora_A" in key)
        or ("lora_B" in key)
        or ("lora_up" in key)
        or ("lora_down" in key)
        or key.endswith(".alpha")
        for key in sample
    )


def iter_named_tensors(model_dir: Path) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield tensors from common HuggingFace checkpoint layouts."""
    shards = sorted(model_dir.glob("*.safetensors"))
    if shards:
        for shard in shards:
            if _is_aux_file(shard.name):
                continue
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                if _looks_like_lora(keys):
                    continue
                for key in keys:
                    yield key, handle.get_tensor(key)
        return

    bins = sorted(model_dir.glob("pytorch_model*.bin"))
    if not bins and (model_dir / "model.safetensors").exists():
        bins = []
    for bin_path in bins:
        if _is_aux_file(bin_path.name):
            continue
        state = torch.load(bin_path, map_location="cpu")
        for key, tensor in state.items():
            if torch.is_tensor(tensor):
                yield key, tensor


def canonical_name(name: str) -> str:
    return name[len("transformer.") :] if name.startswith("transformer.") else name


def parameter_block(name: str) -> str:
    n = name.lower()
    if (
        "layernorm" in n
        or "layer_norm" in n
        or "rmsnorm" in n
        or ".ln_" in n
        or n.endswith(".norm.weight")
        or n.endswith(".norm.bias")
        or ".norm." in n
    ):
        return "layernorm"
    if n.endswith("lm_head.weight") or ".lm_head." in n:
        return "lm_head"
    if (
        "embed_tokens" in n
        or n.endswith(".wte.weight")
        or n.endswith(".wpe.weight")
        or "token_embedding" in n
        or "position_embedding" in n
    ):
        return "emb"
    if "c_attn" in n or "qkv" in n:
        return "qkv_fused"
    if "q_proj" in n:
        return "q_proj"
    if "k_proj" in n:
        return "k_proj"
    if "v_proj" in n:
        return "v_proj"
    if "o_proj" in n or "out_proj" in n:
        return "o_proj"
    if "gate_proj" in n:
        return "gate_proj"
    if "up_proj" in n:
        return "up_proj"
    if "down_proj" in n:
        return "down_proj"
    if "c_proj" in n:
        return "c_proj"
    if "c_fc" in n or "fc1" in n:
        return "fc1"
    if "fc2" in n:
        return "fc2"
    if "mlp" in n:
        return "other_mlp"
    return "other"


def load_projection_tensors(
    model_dir: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    subsample_layers: int = 1,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in iter_named_tensors(Path(model_dir)):
        cname = canonical_name(name)
        block = parameter_block(cname)
        if block in SKIP_BLOCKS:
            continue
        if tensor.ndim != 2 or min(tensor.shape) < 2:
            continue
        if subsample_layers > 1:
            match = LAYER_RE.search(cname)
            if match is not None and int(match.group(1)) % subsample_layers != 0:
                continue
        if cname not in out:
            out[cname] = tensor.to(dtype)
    return out
