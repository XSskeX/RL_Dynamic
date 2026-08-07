from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diffing.utils.dictionary import load_dictionary_model


def extract_crosscoder_decoder(
    model_path: str | Path,
    output_path: str | Path,
    model_index: int,
    ckpt_name: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Path:
    """Extract one model slice from an n-way crosscoder decoder.

    The n-way crosscoder decoder is expected to have shape:
        [num_models, num_features, activation_dim]

    The saved decoder_weight has shape:
        [num_features, activation_dim]
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    crosscoder = load_dictionary_model(model_path).eval()
    decoder_weight = crosscoder.decoder.weight.detach().cpu()
    if decoder_weight.ndim != 3:
        raise ValueError(
            "Expected crosscoder.decoder.weight shape "
            f"[num_models, num_features, activation_dim], got {tuple(decoder_weight.shape)}"
        )
    if not 0 <= model_index < decoder_weight.shape[0]:
        raise IndexError(
            f"model_index={model_index} is out of range for decoder shape "
            f"{tuple(decoder_weight.shape)}"
        )

    decoder_slice = decoder_weight[model_index].to(dtype=dtype).contiguous()
    payload: dict[str, Any] = {
        "decoder_weight": decoder_slice,
        "metadata": {
            "source_model_path": str(model_path),
            "model_index": model_index,
            "ckpt_name": ckpt_name or f"model_{model_index}",
            "source_decoder_shape": tuple(decoder_weight.shape),
            "saved_decoder_shape": tuple(decoder_slice.shape),
            "dtype": str(decoder_slice.dtype),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(payload["metadata"], indent=2),
        encoding="utf-8",
    )
    return output_path


def load_extracted_decoder(path: str | Path) -> torch.Tensor:
    """Load a decoder matrix saved by extract_crosscoder_decoder."""
    payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload
    if not isinstance(payload, dict) or "decoder_weight" not in payload:
        raise ValueError(f"{path} is not an extracted decoder payload")
    return torch.as_tensor(payload["decoder_weight"])


def compute_decoder_feature_norms(decoder_matrix: torch.Tensor) -> torch.Tensor:
    """Compute the L2 norm of every feature direction in a decoder matrix.

    Args:
        decoder_matrix: Tensor with shape [num_features, activation_dim].

    Returns:
        Tensor with shape [num_features], where norms[i] is ||decoder_matrix[i]||_2.
    """
    decoder_matrix = torch.as_tensor(decoder_matrix).detach().cpu().to(torch.float32)
    if decoder_matrix.ndim != 2:
        raise ValueError(
            "Expected decoder matrix shape [num_features, activation_dim], got "
            f"{tuple(decoder_matrix.shape)}"
        )
    return decoder_matrix.norm(dim=1)


def validate_decoder_feature_norms(
    decoder_path: str | Path,
    norms_output_path: str | Path | None = None,
) -> torch.Tensor:
    """Load an extracted decoder and verify per-feature decoder norms are finite.

    If norms_output_path is provided, norms are saved as .pt, .csv, .tsv, or .txt.
    """
    decoder = load_extracted_decoder(decoder_path)
    norms = compute_decoder_feature_norms(decoder)
    if not torch.isfinite(norms).all():
        bad_count = int((~torch.isfinite(norms)).sum().item())
        raise ValueError(f"Found {bad_count} non-finite decoder feature norm(s)")

    if norms_output_path is not None:
        save_feature_norms(norms, norms_output_path)
    return norms


def save_feature_norms(norms: torch.Tensor, output_path: str | Path) -> Path:
    """Save per-feature norms to .pt, .csv, .tsv, or .txt."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    norms = torch.as_tensor(norms).detach().cpu().to(torch.float32).flatten()

    suffix = output_path.suffix.lower()
    if suffix == ".pt":
        torch.save(norms, output_path)
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        lines = [f"feature_id{sep}decoder_norm"]
        lines.extend(
            f"{feature_id}{sep}{float(norm):.9g}"
            for feature_id, norm in enumerate(norms)
        )
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif suffix == ".txt":
        output_path.write_text(
            "\n".join(f"{float(norm):.9g}" for norm in norms) + "\n",
            encoding="utf-8",
        )
    else:
        raise ValueError(
            f"Unsupported norms output suffix '{suffix}'. Use .pt, .csv, .tsv, or .txt."
        )
    return output_path


def summarize_feature_norms(norms: torch.Tensor) -> dict[str, float | int]:
    """Return compact statistics for per-feature decoder norms."""
    norms = torch.as_tensor(norms).detach().cpu().to(torch.float32).flatten()
    if norms.numel() == 0:
        raise ValueError("Cannot summarize empty norms tensor")
    return {
        "num_features": int(norms.numel()),
        "min": float(norms.min().item()),
        "mean": float(norms.mean().item()),
        "max": float(norms.max().item()),
        "zero_count": int((norms == 0).sum().item()),
    }


def get_decoder_vectors(
    decoder_path: str | Path,
    feature_ids: list[int] | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return decoder directions for selected feature ids.

    Output shape is [len(feature_ids), activation_dim].
    """
    decoder = load_extracted_decoder(decoder_path).to(device=device, dtype=dtype)
    feature_ids = torch.as_tensor(feature_ids, dtype=torch.long, device=device).flatten()
    if feature_ids.numel() == 0:
        raise ValueError("feature_ids must not be empty")
    if feature_ids.min() < 0 or feature_ids.max() >= decoder.shape[0]:
        raise IndexError(
            f"Feature ids must be in [0, {decoder.shape[0] - 1}], got "
            f"min={int(feature_ids.min())}, max={int(feature_ids.max())}"
        )
    return decoder[feature_ids]


def build_steering_vector_from_decoder(
    decoder_path: str | Path,
    feature_ids: list[int] | torch.Tensor,
    feature_weights: list[float] | torch.Tensor | None = None,
    *,
    decoder_normalization: str = "none",
    output_norm: float | None = 1.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Combine extracted decoder directions into one steering vector."""
    vectors = get_decoder_vectors(decoder_path, feature_ids, device=device)
    if decoder_normalization == "unit":
        vectors = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1e-8)
    elif decoder_normalization != "none":
        raise ValueError(f"Unknown decoder_normalization: {decoder_normalization}")

    if feature_weights is None:
        weights = torch.ones(vectors.shape[0], device=vectors.device)
    else:
        weights = torch.as_tensor(
            feature_weights,
            dtype=vectors.dtype,
            device=vectors.device,
        ).flatten()
    if weights.numel() != vectors.shape[0]:
        raise ValueError(
            f"feature_weights has {weights.numel()} values but selected "
            f"{vectors.shape[0]} feature ids"
        )

    steering_vector = (weights[:, None] * vectors).sum(dim=0)
    if output_norm is not None:
        steering_vector = (
            steering_vector / steering_vector.norm().clamp_min(1e-8) * output_norm
        )
    return steering_vector.detach().cpu()


def _dtype_from_string(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one checkpoint/model decoder matrix from an n-way crosscoder."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Path to the crosscoder model_final.pt.",
    )
    parser.add_argument(
        "--model-index",
        required=True,
        type=int,
        help="Index along decoder.weight[model_index]. For n-way RL ckpts this is the ckpt order.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Where to save the extracted decoder .pt file.",
    )
    parser.add_argument(
        "--ckpt-name",
        default=None,
        help="Optional human-readable label, e.g. global_step_120.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype used for the saved decoder matrix.",
    )
    parser.add_argument(
        "--norms-output-path",
        type=Path,
        default=None,
        help="Optional path to save per-feature decoder norms (.pt, .csv, .tsv, or .txt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = extract_crosscoder_decoder(
        model_path=args.model_path,
        output_path=args.output_path,
        model_index=args.model_index,
        ckpt_name=args.ckpt_name,
        dtype=_dtype_from_string(args.dtype),
    )
    decoder = load_extracted_decoder(output_path)
    norms = validate_decoder_feature_norms(output_path, args.norms_output_path)
    norm_summary = summarize_feature_norms(norms)
    print(f"Saved decoder to: {output_path}")
    print(f"Decoder shape: {tuple(decoder.shape)}")
    print(
        "Decoder feature norms: "
        f"num_features={norm_summary['num_features']}, "
        f"min={norm_summary['min']:.6g}, "
        f"mean={norm_summary['mean']:.6g}, "
        f"max={norm_summary['max']:.6g}, "
        f"zero_count={norm_summary['zero_count']}"
    )
    if args.norms_output_path is not None:
        print(f"Saved decoder feature norms to: {args.norms_output_path}")


if __name__ == "__main__":
    main()
