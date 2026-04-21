# scripts/export_verl_checkpoint.py
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def has_hf_weights(hf_dir: Path) -> bool:
    weight_files = [
        hf_dir / "model.safetensors",
        hf_dir / "pytorch_model.bin",
        hf_dir / "model.safetensors.index.json",
        hf_dir / "pytorch_model.bin.index.json",
    ]
    return any(p.exists() for p in weight_files)


def copy_hf_metadata(src_hf_dir: Path, dst_hf_dir: Path) -> None:
    dst_hf_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]:
        src = src_hf_dir / name
        if src.exists():
            shutil.copy2(src, dst_hf_dir / name)


def inspect_checkpoint(path: Path) -> None:
    obj = torch.load(path, map_location="cpu")
    print(f"\n=== Inspect: {path} ===")
    print(f"type={type(obj)}")
    if isinstance(obj, dict):
        print(f"top-level keys ({len(obj)}):")
        for k in list(obj.keys())[:50]:
            print("  ", k)
        for candidate in ["model", "module", "state_dict", "actor", "weights"]:
            if candidate in obj and isinstance(obj[candidate], dict):
                print(
                    f"\nNested dict under key '{candidate}' ({len(obj[candidate])} keys):"
                )
                for kk in list(obj[candidate].keys())[:50]:
                    print("  ", kk)
    print("=== End Inspect ===\n")


def to_plain_tensor(x: Any) -> torch.Tensor:
    """
    Convert checkpoint tensor-like objects to a plain CPU torch.Tensor.

    Handles:
    - torch.distributed.tensor.DTensor
    - objects exposing .to_local()
    - regular torch.Tensor
    """

    # IMPORTANT:
    # DTensor is a subclass of torch.Tensor, so we must check DTensor first.
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(x, DTensor):
            return x.to_local().detach().cpu()
    except Exception:
        pass

    # Some sharded tensor-like objects also expose to_local()
    if hasattr(x, "to_local"):
        local = x.to_local()
        if isinstance(local, torch.Tensor):
            return local.detach().cpu()

    # Plain tensor
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()

    raise TypeError(f"Unsupported tensor type: {type(x)}")



def extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    """
    Try a few common checkpoint layouts and convert all tensors to plain CPU tensors.
    """
    if isinstance(obj, dict):
        # Case 1: already a plain-ish state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            tensor_items = {}
            for k, v in obj.items():
                try:
                    tensor_items[k] = to_plain_tensor(v)
                except TypeError:
                    pass
            if tensor_items:
                return tensor_items

        # Case 2: nested common names
        for key in ["model", "module", "state_dict", "actor", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                inner = {}
                for k, v in obj[key].items():
                    try:
                        inner[k] = to_plain_tensor(v)
                    except TypeError:
                        pass
                if inner:
                    return inner

    raise ValueError("Could not extract a tensor state_dict from checkpoint object")


def strip_prefixes(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = [
        "module.",
        "_forward_module.",
        "actor.",
    ]

    out = {}
    for k, v in sd.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        out[nk] = v
    return out


def collect_state_dicts(rank_ckpts: list[Path]) -> dict[str, list[torch.Tensor]]:
    per_key: dict[str, list[torch.Tensor]] = {}

    for ckpt_path in rank_ckpts:
        obj = torch.load(ckpt_path, map_location="cpu")
        sd = strip_prefixes(extract_state_dict(obj))

        for k, v in sd.items():
            per_key.setdefault(k, []).append(v)

    return per_key


def tensors_all_equal(ts: list[torch.Tensor]) -> bool:
    first = ts[0]
    return all(t.shape == first.shape and torch.equal(t, first) for t in ts[1:])


def resolve_tensor_for_key(
    key: str,
    tensors: list[torch.Tensor],
    target_shape: torch.Size,
) -> torch.Tensor:
    """
    Resolve one parameter from possibly multiple rank-local tensors.

    Strategy:
    - one tensor -> use it
    - identical tensors across ranks -> keep one
    - otherwise try cat on dim=0
    - then try cat on dim=1 for matrix weights
    """
    if len(tensors) == 1:
        return tensors[0]

    if tensors_all_equal(tensors):
        return tensors[0]

    try:
        cat0 = torch.cat(tensors, dim=0)
        if tuple(cat0.shape) == tuple(target_shape):
            return cat0
    except Exception:
        pass

    if len(target_shape) >= 2:
        try:
            cat1 = torch.cat(tensors, dim=1)
            if tuple(cat1.shape) == tuple(target_shape):
                return cat1
        except Exception:
            pass

    raise ValueError(
        f"Could not resolve key={key}, "
        f"candidate_shapes={[tuple(t.shape) for t in tensors]}, "
        f"target_shape={tuple(target_shape)}"
    )


def merge_rank_checkpoints(
    rank_ckpts: list[Path],
    model: AutoModelForCausalLM,
) -> dict[str, torch.Tensor]:
    """
    Merge rank checkpoints into a single HF-compatible state_dict.

    This function:
    - loads every rank file
    - extracts plain CPU tensors
    - groups tensors by parameter name
    - uses target HF model parameter shapes to decide whether to keep,
      deduplicate, or concatenate shards
    """
    per_key = collect_state_dicts(rank_ckpts)
    target_sd = model.state_dict()
    merged: dict[str, torch.Tensor] = {}

    for k, target_tensor in target_sd.items():
        if k not in per_key:
            continue
        merged[k] = resolve_tensor_for_key(k, per_key[k], target_tensor.shape)

    return merged


def load_model_skeleton(
    hf_dir: Path, torch_dtype: torch.dtype
) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        dtype=torch_dtype,
    )
    return model


def export_verl_actor(
    actor_dir: Path,
    output_dir: Path,
    dtype: str = "bfloat16",
    inspect_only: bool = False,
) -> None:
    actor_dir = actor_dir.resolve()
    hf_dir = actor_dir / "huggingface"
    if not hf_dir.exists():
        raise FileNotFoundError(f"Missing huggingface metadata dir: {hf_dir}")

    rank_ckpts = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
    if not rank_ckpts:
        raise FileNotFoundError(f"No rank checkpoints found under {actor_dir}")

    if inspect_only:
        inspect_checkpoint(rank_ckpts[0])
        return

    if has_hf_weights(hf_dir):
        print(f"HF weights already exist in {hf_dir}, copying directly to {output_dir}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(hf_dir, output_dir)
        return

    copy_hf_metadata(hf_dir, output_dir)

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    print("Building HF model skeleton...")
    model = load_model_skeleton(hf_dir, torch_dtype=torch_dtype)

    print("Merging rank checkpoints...")
    merged_sd = merge_rank_checkpoints(rank_ckpts, model)

    missing, unexpected = model.load_state_dict(merged_sd, strict=False)

    if missing:
        print("\nMissing keys:")
        for k in missing[:100]:
            print("  ", k)
    if unexpected:
        print("\nUnexpected keys:")
        for k in unexpected[:100]:
            print("  ", k)

    if missing:
        raise ValueError(
            "Model load still has missing keys. "
            "The verl checkpoint layout likely needs additional key remapping "
            "or a more specific shard merge rule."
        )

    print("Saving HF model...")
    model.save_pretrained(output_dir, safe_serialization=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Tokenizer save skipped: {e}")

    print(f"Done. HF export written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    export_verl_actor(
        actor_dir=args.actor_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        inspect_only=args.inspect_only,
    )


if __name__ == "__main__":
    main()
