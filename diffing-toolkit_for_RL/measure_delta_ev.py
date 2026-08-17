"""Measure how well a CrossCoder reconstructs checkpoint-to-reference changes.

The input activations must be aligned across checkpoints and have the order used
when training the CrossCoder.  For checkpoint m and reference r, this script
computes

    delta_h       = h_m - h_r
    delta_h_hat   = h_hat_m - h_hat_r

and reports the energy explained by delta_h_hat, in addition to ordinary
per-checkpoint reconstruction FVE.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
TOOLKIT_ROOT_DEFAULT = REPO_ROOT / "cache" / "diffing-toolkit_for_RL"


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Both LABEL and PATH must be non-empty")
    return label.strip(), Path(raw_path).expanduser()


def _load_measure_helpers(toolkit_root: Path):
    """Import the cache/tensor helpers used by measure_rl_changes.py."""
    src_dir = HERE.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import measure_rl_changes as drift

    return drift


def _load_crosscoder(model_path: Path, toolkit_root: Path, device: str):
    src_dir = toolkit_root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Toolkit source directory does not exist: {src_dir}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from diffing.utils.dictionary import load_dictionary_model

    model = load_dictionary_model(model_path, is_sae=False)
    return model.to(device).eval()


def _load_activation_batches(
    args: argparse.Namespace, drift: Any
) -> tuple[Iterable[torch.Tensor], list[str], bool]:
    """Return batches shaped [batch, checkpoints, hidden_dim]."""
    if args.activation_cache:
        specs: list[tuple[str, Path]] = args.activation_cache
        cache_cls = drift._import_activation_cache_tuple(args.toolkit_root)
        cache = cache_cls(
            *(str(path) for _, path in specs),
            submodule_name=args.submodule_name,
        )
        labels = [label for label, _ in specs]
        tokens_verified = drift._verify_cache_tokens(cache, labels)
        from torch.utils.data import DataLoader

        batches: Iterable[torch.Tensor] = DataLoader(
            cache,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        return batches, labels, tokens_verified

    specs = args.activation_tensor
    tensors = [drift._load_activation_tensor(path) for _, path in specs]
    first_shape = tensors[0].shape
    for (label, _), tensor in zip(specs[1:], tensors[1:]):
        if tensor.shape != first_shape:
            raise ValueError(
                f"Activation tensor {label!r} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(first_shape)}"
            )
    labels = [label for label, _ in specs]
    return drift._tensor_batches(tensors, args.batch_size), labels, False


class Metrics:
    def __init__(self, label: str, device: torch.device | str):
        self.label = label
        self.device = torch.device(device)
        self.metric_dtype = (
            torch.float64 if self.device.type == "cpu" else torch.float32
        )
        self.tokens = 0
        self.hidden_dim: int | None = None
        self.delta_sq = torch.zeros((), device=self.device, dtype=self.metric_dtype)
        self.delta_hat_sq = torch.zeros(
            (), device=self.device, dtype=self.metric_dtype
        )
        self.delta_error_sq = torch.zeros(
            (), device=self.device, dtype=self.metric_dtype
        )
        self.delta_dot = torch.zeros((), device=self.device, dtype=self.metric_dtype)
        self.target_sum: torch.Tensor | None = None
        self.target_sq_sum: torch.Tensor | None = None
        self.error_sum: torch.Tensor | None = None
        self.error_sq_sum: torch.Tensor | None = None

    def update(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        delta: torch.Tensor,
        delta_reconstruction: torch.Tensor,
    ) -> None:
        if target.shape != reconstruction.shape:
            raise ValueError(
                f"Target/reconstruction shape mismatch: {tuple(target.shape)} != "
                f"{tuple(reconstruction.shape)}"
            )
        if delta.shape != delta_reconstruction.shape:
            raise ValueError("Delta/reconstructed-delta shape mismatch")
        if target.ndim != 2:
            raise ValueError("Expected tensors shaped [tokens, hidden_dim]")

        target = target.detach().to(device=self.device, dtype=torch.float32)
        reconstruction = reconstruction.detach().to(
            device=self.device, dtype=torch.float32
        )
        delta = delta.detach().to(device=self.device, dtype=torch.float32)
        delta_reconstruction = delta_reconstruction.detach().to(
            device=self.device, dtype=torch.float32
        )
        error = target - reconstruction
        delta_error = delta - delta_reconstruction

        batch_tokens, hidden_dim = target.shape
        if self.hidden_dim is None:
            self.hidden_dim = hidden_dim
            stats_kwargs = {"device": self.device, "dtype": self.metric_dtype}
            self.target_sum = torch.zeros(hidden_dim, **stats_kwargs)
            self.target_sq_sum = torch.zeros(hidden_dim, **stats_kwargs)
            self.error_sum = torch.zeros(hidden_dim, **stats_kwargs)
            self.error_sq_sum = torch.zeros(hidden_dim, **stats_kwargs)
        elif self.hidden_dim != hidden_dim:
            raise ValueError("Hidden dimension changed between batches")

        target_metric = target.to(self.metric_dtype)
        error_metric = error.to(self.metric_dtype)
        delta_metric = delta.to(self.metric_dtype)
        delta_reconstruction_metric = delta_reconstruction.to(self.metric_dtype)
        delta_error_metric = delta_error.to(self.metric_dtype)
        self.tokens += batch_tokens
        self.delta_sq += delta_metric.square().sum()
        self.delta_hat_sq += delta_reconstruction_metric.square().sum()
        self.delta_error_sq += delta_error_metric.square().sum()
        self.delta_dot += (delta_metric * delta_reconstruction_metric).sum()
        self.target_sum += target_metric.sum(dim=0)
        self.target_sq_sum += target_metric.square().sum(dim=0)
        self.error_sum += error_metric.sum(dim=0)
        self.error_sq_sum += error_metric.square().sum(dim=0)

    def finalize(self) -> dict[str, Any]:
        if self.tokens < 1 or self.hidden_dim is None:
            raise ValueError(f"No tokens accumulated for {self.label}")
        delta_sq = float(self.delta_sq.item())
        delta_hat_sq = float(self.delta_hat_sq.item())
        delta_error_sq = float(self.delta_error_sq.item())
        delta_dot = float(self.delta_dot.item())
        delta_norm = delta_sq**0.5
        delta_error_norm = delta_error_sq**0.5
        delta_hat_norm = delta_hat_sq**0.5
        cosine_denominator = delta_norm * delta_hat_norm

        # This matches the training FVE definition up to the common unbiased
        # variance divisor, which cancels in the ratio.
        total_variance = self.target_sq_sum - self.target_sum.square() / self.tokens
        residual_variance = self.error_sq_sum - self.error_sum.square() / self.tokens
        total_variance_sum = float(total_variance.sum().item())
        residual_variance_sum = float(residual_variance.sum().item())

        return {
            "target": self.label,
            "token_count": self.tokens,
            "hidden_dim": self.hidden_dim,
            "delta_norm": delta_norm,
            "delta_hat_norm": delta_hat_norm,
            "delta_error_norm": delta_error_norm,
            "delta_relative_error": (
                delta_error_norm / delta_norm if delta_norm > 0 else None
            ),
            "delta_energy_explained": (
                1.0 - delta_error_sq / delta_sq
                if delta_sq > 0
                else None
            ),
            "delta_global_cosine": (
                delta_dot / cosine_denominator
                if cosine_denominator > 0
                else None
            ),
            "reconstruction_fve": (
                1.0 - residual_variance_sum / total_variance_sum
                if total_variance_sum > 0
                else None
            ),
        }


@torch.no_grad()
def compute_delta_metrics(
    crosscoder: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    *,
    labels: Sequence[str],
    reference_index: int,
    max_tokens: int | None,
) -> list[dict[str, Any]]:
    num_models = int(crosscoder.decoder.weight.shape[0])
    if len(labels) != num_models:
        raise ValueError(
            f"Received {len(labels)} activation streams but CrossCoder has "
            f"{num_models} decoder slices"
        )
    device = crosscoder.device
    accumulators = {
        index: Metrics(labels[index], device)
        for index in range(num_models)
        if index != reference_index
    }
    processed = 0
    for batch in batches:
        batch = torch.as_tensor(batch)
        if batch.ndim != 3 or batch.shape[1] != num_models:
            raise ValueError(
                "Expected activation batch [tokens, crosscoder_models, hidden_dim], "
                f"got {tuple(batch.shape)}"
            )
        if max_tokens is not None:
            remaining = max_tokens - processed
            if remaining <= 0:
                break
            batch = batch[:remaining]
        if batch.shape[0] == 0:
            continue

        model_input = batch.to(device=device, dtype=crosscoder.dtype)
        reconstruction = crosscoder(model_input)
        reference = model_input[:, reference_index, :]
        reference_reconstruction = reconstruction[:, reference_index, :]
        for index, accumulator in accumulators.items():
            target = model_input[:, index, :]
            target_reconstruction = reconstruction[:, index, :]
            delta = target - reference
            delta_reconstruction = target_reconstruction - reference_reconstruction
            accumulator.update(
                target,
                target_reconstruction,
                delta,
                delta_reconstruction,
            )
        processed += batch.shape[0]

    if processed == 0:
        raise ValueError("No activation tokens were processed")
    return [
        {"reference": labels[reference_index], **accumulators[index].finalize()}
        for index in sorted(accumulators)
    ]


def write_records(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "delta_metrics.json"
    csv_path = output_dir / "delta_metrics.csv"
    json_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(records[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure CrossCoder reconstruction quality on checkpoint deltas."
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--activation-cache", type=parse_label_path, action="append")
    sources.add_argument("--activation-tensor", type=parse_label_path, action="append")
    parser.add_argument("--crosscoder", type=Path, required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--toolkit-root", type=Path, default=TOOLKIT_ROOT_DEFAULT)
    parser.add_argument("--submodule-name", default="layer_13_out")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.activation_cache is None and args.activation_tensor is None:
        raise ValueError("Provide --activation-cache or --activation-tensor")
    specs = args.activation_cache or args.activation_tensor
    if len(specs) < 2:
        raise ValueError("At least two activation streams are required")
    if args.reference_label not in [label for label, _ in specs]:
        raise ValueError(f"Reference label {args.reference_label!r} is not present")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)

    drift = _load_measure_helpers(args.toolkit_root)
    batches, labels, tokens_verified = _load_activation_batches(args, drift)
    crosscoder = _load_crosscoder(args.crosscoder, args.toolkit_root, args.device)
    records = compute_delta_metrics(
        crosscoder,
        batches,
        labels=labels,
        reference_index=labels.index(args.reference_label),
        max_tokens=args.max_tokens,
    )
    for record in records:
        record["submodule_name"] = args.submodule_name
        record["tokens_verified"] = tokens_verified
    write_records(records, args.output_dir)


if __name__ == "__main__":
    main()
