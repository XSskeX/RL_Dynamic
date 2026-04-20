# scripts/export_verl_checkpoint.py
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
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
                print(f"\nNested dict under key '{candidate}' ({len(obj[candidate])} keys):")
                for kk in list(obj[candidate].keys())[:50]:
                    print("  ", kk)
    print("=== End Inspect ===\n")


def extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    """
    Try a few common checkpoint layouts.
    You may need to extend this after inspecting a real verl checkpoint.
    """
    if isinstance(obj, dict):
        # Case 1: already a plain state_dict
        if all(isinstance(k, str) for k in obj.keys()) and any(
            isinstance(v, torch.Tensor) for v in obj.values()
        ):
            return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}

        # Case 2: nested common names
        for key in ["model", "module", "state_dict", "actor", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                inner = obj[key]
                if any(isinstance(v, torch.Tensor) for v in inner.values()):
                    return {k: v for k, v in inner.items() if isinstance(v, torch.Tensor)}

    raise ValueError("Could not extract a tensor state_dict from checkpoint object")


def strip_prefixes(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = [
        "module.",
        "_forward_module.",
        "model.",
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


def merge_rank_checkpoints(rank_ckpts: list[Path]) -> dict[str, torch.Tensor]:
    """
    Conservative merge strategy:
    - load every rank file
    - extract state_dict
    - if a key only appears once, keep it
    - if a key appears in multiple ranks and tensors are identical, keep one
    - if a key appears in multiple ranks with different shapes/values, fail fast

    This works if verl saved duplicated full params or rank-partitioned dicts
    without requiring tensor concatenation.
    It will NOT solve arbitrary FSDP flatten-shard layouts automatically.
    """
    merged: dict[str, torch.Tensor] = {}

    for ckpt_path in rank_ckpts:
        obj = torch.load(ckpt_path, map_location="cpu")
        sd = strip_prefixes(extract_state_dict(obj))

        for k, v in sd.items():
            if k not in merged:
                merged[k] = v
                continue

            old = merged[k]
            if old.shape == v.shape and torch.equal(old, v):
                continue

            raise ValueError(
                f"Conflicting tensor for key '{k}'. "
                f"Existing shape={tuple(old.shape)}, new shape={tuple(v.shape)}. "
                "This likely means the checkpoint is true FSDP-sharded and needs "
                "a format-specific consolidation path."
            )

    return merged


def load_model_skeleton(hf_dir: Path, torch_dtype: torch.dtype) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
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

    print("Merging rank checkpoints...")
    merged_sd = merge_rank_checkpoints(rank_ckpts)

    print("Building HF model skeleton...")
    model = load_model_skeleton(hf_dir, torch_dtype=torch_dtype)

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
            "The verl checkpoint layout likely needs an additional key remapping."
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
