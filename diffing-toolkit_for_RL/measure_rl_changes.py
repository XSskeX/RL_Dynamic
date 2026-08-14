from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


DEFAULT_LAYER_REGEX = r"(?:^|\.)layers\.(\d+)\."


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Both LABEL and PATH must be non-empty")
    return label.strip(), Path(raw_path).expanduser()


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


@dataclass
class TensorComparisonAccumulator:
    tolerances: tuple[float, ...]
    numel: int = 0
    exact_changed: int = 0
    changed_by_tolerance: dict[float, int] = field(default_factory=dict)
    reference_sq_sum: float = 0.0
    target_sq_sum: float = 0.0
    delta_sq_sum: float = 0.0
    dot_sum: float = 0.0

    def __post_init__(self) -> None:
        self.changed_by_tolerance = {tol: 0 for tol in self.tolerances}

    def update(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
        *,
        chunk_numel: int,
    ) -> None:
        if reference.shape != target.shape:
            raise ValueError(
                f"Tensor shape mismatch: {tuple(reference.shape)} != {tuple(target.shape)}"
            )

        reference_flat = reference.detach().reshape(-1)
        target_flat = target.detach().reshape(-1)
        self.numel += reference_flat.numel()

        for start in range(0, reference_flat.numel(), chunk_numel):
            stop = min(start + chunk_numel, reference_flat.numel())
            reference_raw = reference_flat[start:stop]
            target_raw = target_flat[start:stop]
            self.exact_changed += int(
                torch.count_nonzero(reference_raw != target_raw).item()
            )

            reference_float = reference_raw.to(device="cpu", dtype=torch.float32)
            target_float = target_raw.to(device="cpu", dtype=torch.float32)
            delta = target_float - reference_float
            abs_delta = delta.abs()

            for tolerance in self.tolerances:
                self.changed_by_tolerance[tolerance] += int(
                    torch.count_nonzero(abs_delta > tolerance).item()
                )

            reference_double = reference_float.to(torch.float64)
            target_double = target_float.to(torch.float64)
            delta_double = delta.to(torch.float64)
            self.reference_sq_sum += float(reference_double.square().sum().item())
            self.target_sq_sum += float(target_double.square().sum().item())
            self.delta_sq_sum += float(delta_double.square().sum().item())
            self.dot_sum += float((reference_double * target_double).sum().item())

    def merge(self, other: "TensorComparisonAccumulator") -> None:
        if self.tolerances != other.tolerances:
            raise ValueError("Cannot merge accumulators with different tolerances")
        self.numel += other.numel
        self.exact_changed += other.exact_changed
        for tolerance in self.tolerances:
            self.changed_by_tolerance[tolerance] += other.changed_by_tolerance[
                tolerance
            ]
        self.reference_sq_sum += other.reference_sq_sum
        self.target_sq_sum += other.target_sq_sum
        self.delta_sq_sum += other.delta_sq_sum
        self.dot_sum += other.dot_sum

    def finalize(self) -> dict[str, Any]:
        reference_norm = math.sqrt(self.reference_sq_sum)
        target_norm = math.sqrt(self.target_sq_sum)
        delta_norm = math.sqrt(self.delta_sq_sum)
        result: dict[str, Any] = {
            "numel": self.numel,
            "exact_changed_count": self.exact_changed,
            "exact_nonzero_density": _finite_ratio(self.exact_changed, self.numel),
            "reference_frobenius_norm": reference_norm,
            "target_frobenius_norm": target_norm,
            "delta_frobenius_norm": delta_norm,
            "parameter_relative_l2": _finite_ratio(delta_norm, reference_norm),
            "parameter_cosine": _finite_ratio(
                self.dot_sum, reference_norm * target_norm
            ),
        }
        for tolerance in self.tolerances:
            suffix = f"{tolerance:.0e}"
            changed = self.changed_by_tolerance[tolerance]
            result[f"changed_count_atol_{suffix}"] = changed
            result[f"nonzero_density_atol_{suffix}"] = _finite_ratio(
                changed, self.numel
            )
        return result


def _validate_parameter_sets(
    reference: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> None:
    reference_names = set(reference)
    target_names = set(target)
    missing_in_target = sorted(reference_names - target_names)
    missing_in_reference = sorted(target_names - reference_names)
    if missing_in_target or missing_in_reference:
        raise ValueError(
            "Parameter names differ. "
            f"Missing in target: {missing_in_target[:10]}; "
            f"missing in reference: {missing_in_reference[:10]}"
        )


def compare_named_parameters(
    reference: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    tolerances: Sequence[float] = (1e-5, 1e-4),
    layer_regex: str = DEFAULT_LAYER_REGEX,
    chunk_numel: int = 4_000_000,
) -> list[dict[str, Any]]:
    """Compute element-weighted parameter metrics for the model and every layer."""
    if chunk_numel <= 0:
        raise ValueError("chunk_numel must be positive")
    tolerances_tuple = tuple(sorted(set(float(tol) for tol in tolerances)))
    if any(tol < 0 for tol in tolerances_tuple):
        raise ValueError("tolerances must be non-negative")

    _validate_parameter_sets(reference, target)
    pattern = re.compile(layer_regex)
    accumulators: dict[str, TensorComparisonAccumulator] = {
        "all": TensorComparisonAccumulator(tolerances_tuple)
    }

    for name, reference_parameter in reference.items():
        target_parameter = target[name]
        parameter_accumulator = TensorComparisonAccumulator(tolerances_tuple)
        parameter_accumulator.update(
            reference_parameter, target_parameter, chunk_numel=chunk_numel
        )
        accumulators["all"].merge(parameter_accumulator)

        match = pattern.search(name)
        if match is None:
            continue
        group = f"layer_{int(match.group(1))}"
        layer_accumulator = accumulators.setdefault(
            group, TensorComparisonAccumulator(tolerances_tuple)
        )
        layer_accumulator.merge(parameter_accumulator)

    def sort_key(item: tuple[str, TensorComparisonAccumulator]) -> tuple[int, int]:
        group = item[0]
        if group == "all":
            return (0, -1)
        return (1, int(group.removeprefix("layer_")))

    return [
        {"group": group, **accumulator.finalize()}
        for group, accumulator in sorted(accumulators.items(), key=sort_key)
    ]


@dataclass
class ActivationComparisonAccumulator:
    label: str
    token_count: int = 0
    token_relative_count: int = 0
    token_cosine_count: int = 0
    hidden_dim: int | None = None
    reference_sq_sum: float = 0.0
    target_sq_sum: float = 0.0
    delta_sq_sum: float = 0.0
    dot_sum: float = 0.0
    token_relative_l2_sum: float = 0.0
    token_cosine_sum: float = 0.0

    def update(self, reference: torch.Tensor, target: torch.Tensor) -> None:
        if reference.shape != target.shape:
            raise ValueError(
                f"Activation shape mismatch: {tuple(reference.shape)} != {tuple(target.shape)}"
            )
        if reference.ndim != 2:
            raise ValueError(
                f"Expected activation batch [tokens, hidden_dim], got {tuple(reference.shape)}"
            )

        reference = reference.detach().to(device="cpu", dtype=torch.float32)
        target = target.detach().to(device="cpu", dtype=torch.float32)
        delta = target - reference
        reference_double = reference.to(torch.float64)
        target_double = target.to(torch.float64)
        delta_double = delta.to(torch.float64)

        batch_tokens, hidden_dim = reference.shape
        if self.hidden_dim is None:
            self.hidden_dim = hidden_dim
        elif self.hidden_dim != hidden_dim:
            raise ValueError(
                f"Hidden dimension changed: {self.hidden_dim} != {hidden_dim}"
            )

        self.token_count += batch_tokens
        self.reference_sq_sum += float(reference_double.square().sum().item())
        self.target_sq_sum += float(target_double.square().sum().item())
        self.delta_sq_sum += float(delta_double.square().sum().item())
        self.dot_sum += float((reference_double * target_double).sum().item())

        reference_token_norm = reference.norm(dim=-1)
        target_token_norm = target.norm(dim=-1)
        delta_token_norm = delta.norm(dim=-1)
        valid_reference = reference_token_norm > 0
        if valid_reference.any():
            self.token_relative_l2_sum += float(
                (delta_token_norm[valid_reference] / reference_token_norm[valid_reference])
                .sum()
                .item()
            )
            self.token_relative_count += int(valid_reference.sum().item())

        valid_cosine = (reference_token_norm > 0) & (target_token_norm > 0)
        if valid_cosine.any():
            token_dot = (reference[valid_cosine] * target[valid_cosine]).sum(dim=-1)
            token_cosine = token_dot / (
                reference_token_norm[valid_cosine]
                * target_token_norm[valid_cosine]
            )
            self.token_cosine_sum += float(token_cosine.sum().item())
            self.token_cosine_count += int(valid_cosine.sum().item())

    def finalize(self) -> dict[str, Any]:
        reference_norm = math.sqrt(self.reference_sq_sum)
        target_norm = math.sqrt(self.target_sq_sum)
        delta_norm = math.sqrt(self.delta_sq_sum)
        return {
            "target": self.label,
            "token_count": self.token_count,
            "hidden_dim": self.hidden_dim,
            "reference_frobenius_norm": reference_norm,
            "target_frobenius_norm": target_norm,
            "delta_frobenius_norm": delta_norm,
            "activation_relative_l2": _finite_ratio(delta_norm, reference_norm),
            "activation_global_cosine": _finite_ratio(
                self.dot_sum, reference_norm * target_norm
            ),
            "mean_token_relative_l2": _finite_ratio(
                self.token_relative_l2_sum, self.token_relative_count
            ),
            "mean_token_cosine": _finite_ratio(
                self.token_cosine_sum, self.token_cosine_count
            ),
        }


def compute_activation_metrics(
    batches: Iterable[torch.Tensor],
    *,
    labels: Sequence[str],
    reference_index: int = 0,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Compute activation drift from batches shaped [tokens, models, hidden_dim]."""
    if not labels:
        raise ValueError("labels must not be empty")
    if not 0 <= reference_index < len(labels):
        raise IndexError("reference_index is out of range")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    accumulators = {
        model_index: ActivationComparisonAccumulator(label)
        for model_index, label in enumerate(labels)
        if model_index != reference_index
    }
    processed_tokens = 0

    for batch in batches:
        batch = torch.as_tensor(batch)
        if batch.ndim != 3:
            raise ValueError(
                "Expected activation batch [tokens, models, hidden_dim], got "
                f"{tuple(batch.shape)}"
            )
        if batch.shape[1] != len(labels):
            raise ValueError(
                f"Batch contains {batch.shape[1]} models but got {len(labels)} labels"
            )

        if max_tokens is not None:
            remaining = max_tokens - processed_tokens
            if remaining <= 0:
                break
            batch = batch[:remaining]
        if batch.shape[0] == 0:
            continue

        reference = batch[:, reference_index, :]
        for model_index, accumulator in accumulators.items():
            accumulator.update(reference, batch[:, model_index, :])
        processed_tokens += batch.shape[0]

    if processed_tokens == 0:
        raise ValueError("No activation tokens were processed")

    reference_label = labels[reference_index]
    return [
        {"reference": reference_label, **accumulators[index].finalize()}
        for index in sorted(accumulators)
    ]


def _load_model(
    model_path: Path,
    *,
    dtype_name: str,
    device_map: str,
    cache_dir: Path | None,
    trust_remote_code: bool,
):
    from transformers import AutoModelForCausalLM

    dtype: str | torch.dtype
    if dtype_name == "auto":
        dtype = "auto"
    else:
        dtype = getattr(torch, dtype_name)
    from importlib.util import find_spec

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "trust_remote_code": trust_remote_code,
    }
    if find_spec("accelerate") is not None:
        load_kwargs["device_map"] = device_map
        load_kwargs["low_cpu_mem_usage"] = True
    elif device_map != "cpu":
        raise ImportError(
            "Non-CPU --device-map requires the accelerate package"
        )

    return AutoModelForCausalLM.from_pretrained(
        str(model_path), **load_kwargs
    ).eval()


def _named_parameter_mapping(model: Any) -> dict[str, torch.Tensor]:
    return {name: parameter.detach() for name, parameter in model.named_parameters()}


def _annotate_parameter_records(
    records: list[dict[str, Any]], reference: str, target: str
) -> list[dict[str, Any]]:
    return [
        {"reference": reference, "target": target, **record}
        for record in records
    ]


def run_parameter_comparisons(args: argparse.Namespace) -> list[dict[str, Any]]:
    checkpoint_specs: list[tuple[str, Path]] = args.checkpoint
    load_kwargs = {
        "dtype_name": args.torch_dtype,
        "device_map": args.device_map,
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    all_records: list[dict[str, Any]] = []

    if args.comparison in {"base", "both"}:
        reference_model = _load_model(args.reference_model, **load_kwargs)
        reference_parameters = _named_parameter_mapping(reference_model)
        for target_label, target_path in checkpoint_specs:
            target_model = _load_model(target_path, **load_kwargs)
            records = compare_named_parameters(
                reference_parameters,
                _named_parameter_mapping(target_model),
                tolerances=args.tolerances,
                layer_regex=args.layer_regex,
                chunk_numel=args.chunk_numel,
            )
            all_records.extend(
                _annotate_parameter_records(
                    records, args.reference_label, target_label
                )
            )
            del target_model
            gc.collect()
        del reference_parameters, reference_model
        gc.collect()

    if args.comparison in {"adjacent", "both"}:
        previous_label = args.reference_label
        previous_model = _load_model(args.reference_model, **load_kwargs)
        for target_label, target_path in checkpoint_specs:
            target_model = _load_model(target_path, **load_kwargs)
            records = compare_named_parameters(
                _named_parameter_mapping(previous_model),
                _named_parameter_mapping(target_model),
                tolerances=args.tolerances,
                layer_regex=args.layer_regex,
                chunk_numel=args.chunk_numel,
            )
            all_records.extend(
                _annotate_parameter_records(records, previous_label, target_label)
            )
            del previous_model
            gc.collect()
            previous_model = target_model
            previous_label = target_label
        del previous_model
        gc.collect()

    return all_records


def _import_activation_cache_tuple(toolkit_root: Path):
    for path in (toolkit_root / "vendor", toolkit_root / "src"):
        if not path.exists():
            raise FileNotFoundError(f"Toolkit path does not exist: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from dictionary_learning.cache import ActivationCacheTuple

    return ActivationCacheTuple


def _get_cache_tokens(model_cache: Any) -> torch.Tensor | None:
    try:
        return model_cache.tokens
    except AttributeError:
        return None


def _verify_cache_tokens(cache: Any, labels: Sequence[str]) -> bool:
    activation_caches = cache.activation_caches
    reference_tokens = _get_cache_tokens(activation_caches[0])
    if reference_tokens is None:
        return False
    for index, model_cache in enumerate(activation_caches[1:], start=1):
        tokens = _get_cache_tokens(model_cache)
        if tokens is None:
            return False
        if reference_tokens.shape != tokens.shape or not torch.equal(
            reference_tokens, tokens
        ):
            raise ValueError(
                f"Token cache mismatch between {labels[0]!r} and {labels[index]!r}"
            )
    return True


def _load_activation_tensor(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        if "activations" not in payload:
            raise ValueError(
                f"Activation tensor dict {path} must contain an 'activations' key"
            )
        payload = payload["activations"]
    tensor = torch.as_tensor(payload)
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected {path} to contain [tokens, hidden_dim], got {tuple(tensor.shape)}"
        )
    return tensor


def _tensor_batches(
    tensors: Sequence[torch.Tensor], batch_size: int
) -> Iterable[torch.Tensor]:
    token_count = tensors[0].shape[0]
    for start in range(0, token_count, batch_size):
        stop = min(start + batch_size, token_count)
        yield torch.stack(
            [tensor[start:stop] for tensor in tensors], dim=1
        )


def run_activation_comparison(args: argparse.Namespace) -> list[dict[str, Any]]:
    from torch.utils.data import DataLoader

    tokens_verified = False
    if args.activation_cache:
        specs: list[tuple[str, Path]] = args.activation_cache
        ActivationCacheTuple = _import_activation_cache_tuple(args.toolkit_root)
        cache = ActivationCacheTuple(
            *(str(path) for _, path in specs),
            submodule_name=args.submodule_name,
        )
        labels = [label for label, _ in specs]
        tokens_verified = _verify_cache_tokens(cache, labels)
        batches: Iterable[torch.Tensor] = DataLoader(
            cache,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        specs = args.activation_tensor
        tensors = [_load_activation_tensor(path) for _, path in specs]
        first_shape = tensors[0].shape
        for (label, _), tensor in zip(specs[1:], tensors[1:]):
            if tensor.shape != first_shape:
                raise ValueError(
                    f"Activation tensor {label!r} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(first_shape)}"
                )
        labels = [label for label, _ in specs]
        batches = _tensor_batches(tensors, args.batch_size)

    if len(labels) < 2:
        raise ValueError("At least two activation caches/tensors are required")
    if args.reference_label not in labels:
        raise ValueError(
            f"Reference label {args.reference_label!r} is not in {labels}"
        )
    records = compute_activation_metrics(
        batches,
        labels=labels,
        reference_index=labels.index(args.reference_label),
        max_tokens=args.max_tokens,
    )
    return [
        {
            **record,
            "submodule_name": args.submodule_name,
            "tokens_verified": tokens_verified,
        }
        for record in records
    ]


def write_records(
    records: list[dict[str, Any]], output_dir: Path, stem: str
) -> None:
    if not records:
        raise ValueError("No records to write")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure element-weighted RL parameter update density, relative parameter "
            "L2, and activation relative L2."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parameter_parser = subparsers.add_parser(
        "parameters", help="Compare Hugging Face model parameters."
    )
    parameter_parser.add_argument("--reference-model", type=Path, required=True)
    parameter_parser.add_argument("--reference-label", default="base")
    parameter_parser.add_argument(
        "--checkpoint",
        type=parse_label_path,
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parameter_parser.add_argument(
        "--comparison", choices=["base", "adjacent", "both"], default="base"
    )
    parameter_parser.add_argument(
        "--tolerances", type=float, nargs="+", default=[1e-5, 1e-4]
    )
    parameter_parser.add_argument("--layer-regex", default=DEFAULT_LAYER_REGEX)
    parameter_parser.add_argument("--chunk-numel", type=int, default=4_000_000)
    parameter_parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parameter_parser.add_argument("--device-map", default="cpu")
    parameter_parser.add_argument("--cache-dir", type=Path, default=None)
    parameter_parser.add_argument("--trust-remote-code", action="store_true")
    parameter_parser.add_argument("--output-dir", type=Path, required=True)

    activation_parser = subparsers.add_parser(
        "activations", help="Compare aligned n-way activation caches or .pt tensors."
    )
    activation_sources = activation_parser.add_mutually_exclusive_group(required=True)
    activation_sources.add_argument(
        "--activation-cache",
        type=parse_label_path,
        action="append",
        metavar="LABEL=PATH",
        help="Repeat for each cache directory above layer_13_out.",
    )
    activation_sources.add_argument(
        "--activation-tensor",
        type=parse_label_path,
        action="append",
        metavar="LABEL=PATH",
        help="Repeat for .pt tensors shaped [tokens, hidden_dim].",
    )
    activation_parser.add_argument("--reference-label", required=True)
    activation_parser.add_argument("--submodule-name", default="layer_13_out")
    activation_parser.add_argument(
        "--toolkit-root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "cache"
            / "diffing-toolkit_for_RL"
        ),
    )
    activation_parser.add_argument("--batch-size", type=int, default=4096)
    activation_parser.add_argument("--num-workers", type=int, default=0)
    activation_parser.add_argument("--max-tokens", type=int, default=None)
    activation_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "parameters":
        records = run_parameter_comparisons(args)
        write_records(records, args.output_dir, "parameter_metrics")
    elif args.command == "activations":
        records = run_activation_comparison(args)
        write_records(records, args.output_dir, "activation_metrics")
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
