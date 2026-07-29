from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch as th
from loguru import logger

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from torch.nn.functional import cosine_similarity
from torch.utils.data import ConcatDataset, DataLoader, Subset

from diffing.utils.activations import (
    calculate_samples_per_dataset,
    get_layer_indices,
    load_n_activation_datasets_from_config,
)
from diffing.utils.configs import (
    CONFIGS_DIR,
    get_dataset_configurations,
    get_nway_model_configurations,
)
from diffing.utils.dictionary import load_dictionary_model


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return None
    return [float(item) for item in items]


def _load_feature_ids(path: str | Path) -> th.Tensor:
    path = Path(path)
    if path.suffix.lower() == ".pt":
        values = th.load(path, map_location="cpu")
        return th.as_tensor(values, dtype=th.long).flatten()

    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if "feature_id" in df.columns:
            values = df["feature_id"]
        elif "latent_id" in df.columns:
            values = df["latent_id"]
        else:
            values = df.iloc[:, 0]
        return th.as_tensor(values.to_numpy(), dtype=th.long).flatten()

    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(int(line.split(",")[0].strip()))
    return th.tensor(values, dtype=th.long)


def _load_feature_weights(path: str | Path) -> th.Tensor:
    path = Path(path)
    if path.suffix.lower() == ".pt":
        values = th.load(path, map_location="cpu")
        return th.as_tensor(values, dtype=th.float32).flatten()

    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        for column in ("score", "weight", "feature_weight"):
            if column in df.columns:
                return th.as_tensor(df[column].to_numpy(), dtype=th.float32).flatten()
        return th.as_tensor(df.iloc[:, -1].to_numpy(), dtype=th.float32).flatten()

    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(float(line.split(",")[-1].strip()))
    return th.tensor(values, dtype=th.float32)


def _select_value_columns(
    df: pd.DataFrame,
    column_regex: str | None,
) -> list[str]:
    excluded = {"feature_id", "latent_id", "range", "score", "weight"}
    numeric_columns = [
        column for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
    if column_regex is None:
        return numeric_columns

    pattern = re.compile(column_regex)
    return [column for column in numeric_columns if pattern.search(column)]


def _standardize(x: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(eps)


def compute_feature_scores(
    stats_csv: str | Path,
    method: str,
    top_k: int,
    column_regex: str | None = None,
    aime_accuracies: list[float] | None = None,
    min_abs_score: float = 0.0,
) -> pd.DataFrame:
    """Rank features from per-model crosscoder activation statistics."""
    df = pd.read_csv(stats_csv)
    if "feature_id" not in df.columns:
        if df.columns[0].startswith("Unnamed"):
            df = df.rename(columns={df.columns[0]: "feature_id"})
        else:
            raise ValueError("stats_csv must contain a feature_id column")

    value_columns = _select_value_columns(df, column_regex)
    if not value_columns:
        raise ValueError("No numeric per-model columns were selected from stats_csv")

    values = th.tensor(df[value_columns].to_numpy(), dtype=th.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected values shape [features, models], got {tuple(values.shape)}")

    if method == "final_minus_base":
        scores = values[:, -1] - values[:, 0]
    elif method == "range_signed":
        direction = th.sign(values[:, -1] - values[:, 0])
        scores = direction * (values.max(dim=1).values - values.min(dim=1).values)
    elif method in {"accuracy_corr", "accuracy_slope"}:
        if aime_accuracies is None:
            raise ValueError(f"{method} requires --aime_accuracies")
        acc = th.tensor(aime_accuracies, dtype=th.float32)
        if acc.numel() != values.shape[1]:
            raise ValueError(
                f"Got {acc.numel()} AIME accuracies but selected {values.shape[1]} columns: {value_columns}"
            )
        centered_acc = acc - acc.mean()
        centered_values = values - values.mean(dim=1, keepdim=True)
        if method == "accuracy_corr":
            numerator = (centered_values * centered_acc).mean(dim=1)
            denominator = (
                values.std(dim=1, unbiased=False) * acc.std(unbiased=False)
            ).clamp_min(1e-8)
            scores = numerator / denominator
            scores = scores * values.std(dim=1, unbiased=False)
        else:
            scores = (centered_values * centered_acc).sum(dim=1) / (
                centered_acc.square().sum().clamp_min(1e-8)
            )
    else:
        raise ValueError(f"Unknown scoring method: {method}")

    ranked = pd.DataFrame(
        {
            "feature_id": df["feature_id"].astype(int).to_numpy(),
            "score": scores.numpy(),
            "abs_score": scores.abs().numpy(),
        }
    )
    for column in value_columns:
        ranked[column] = df[column].to_numpy()

    ranked = ranked[ranked["abs_score"] >= min_abs_score]
    ranked = ranked.sort_values("abs_score", ascending=False)
    if top_k > 0:
        ranked = ranked.head(top_k)
    return ranked.reset_index(drop=True)


def build_steering_vector(
    model_path: str | Path,
    feature_ids: th.Tensor,
    feature_weights: th.Tensor,
    base_model_index: int = 0,
    decoder_normalization: str = "none",
    normalize_weights: bool = False,
    output_norm: float | None = 1.0,
    device: str | None = None,
) -> th.Tensor:
    """Combine selected base decoder directions into a single steering vector."""
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    crosscoder = load_dictionary_model(Path(model_path)).to(device).eval()
    decoder_weight = crosscoder.decoder.weight.detach().to(device=device, dtype=th.float32)
    if decoder_weight.ndim != 3:
        raise ValueError(
            "Expected crosscoder.decoder.weight shape "
            f"[num_models, num_features, activation_dim], got {tuple(decoder_weight.shape)}"
        )
    if not 0 <= base_model_index < decoder_weight.shape[0]:
        raise IndexError(
            f"base_model_index={base_model_index} is out of range for decoder shape {tuple(decoder_weight.shape)}"
        )

    feature_ids = feature_ids.to(device=device, dtype=th.long).flatten()
    feature_weights = feature_weights.to(device=device, dtype=th.float32).flatten()
    if feature_ids.numel() != feature_weights.numel():
        raise ValueError(
            f"feature_ids has {feature_ids.numel()} elements but feature_weights has {feature_weights.numel()}"
        )
    if feature_ids.numel() == 0:
        raise ValueError("No features selected for steering")

    if feature_ids.min() < 0 or feature_ids.max() >= decoder_weight.shape[1]:
        raise IndexError(
            f"Feature ids must be in [0, {decoder_weight.shape[1] - 1}], got "
            f"min={int(feature_ids.min())}, max={int(feature_ids.max())}"
        )

    base_decoder = decoder_weight[base_model_index, feature_ids, :]
    decoder_norms = base_decoder.norm(dim=1)
    if decoder_normalization == "unit":
        base_decoder = base_decoder / decoder_norms.clamp_min(1e-8).unsqueeze(1)
    elif decoder_normalization != "none":
        raise ValueError(f"Unknown decoder_normalization: {decoder_normalization}")

    weights = feature_weights
    if normalize_weights:
        weights = _standardize(weights)

    steering_vector = (weights.unsqueeze(1) * base_decoder).sum(dim=0)
    raw_norm = steering_vector.norm()
    if output_norm is not None:
        steering_vector = steering_vector / raw_norm.clamp_min(1e-8) * output_norm

    logger.info(f"Selected {feature_ids.numel()} features")
    logger.info(f"Base decoder direction norms: mean={decoder_norms.mean().item():.6g}, max={decoder_norms.max().item():.6g}")
    logger.info(f"Raw steering norm before output scaling: {raw_norm.item():.6g}")
    logger.info(f"Final steering norm: {steering_vector.norm().item():.6g}")
    return steering_vector.detach().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a base-model steering vector from an n-way crosscoder."
    )
    parser.add_argument("--model_path", required=True, help="Path to crosscoder model_final.pt")
    parser.add_argument("--output_path", required=True, help="Where to save the steering vector .pt")
    parser.add_argument(
        "--stats_csv",
        help="CSV from activation_analysis.py, e.g. per_model_encoder_contrib_stats.csv",
    )
    parser.add_argument(
        "--feature_ids",
        help="Optional txt/csv/pt file of feature ids. If set, stats_csv ranking is skipped.",
    )
    parser.add_argument(
        "--feature_weights",
        help="Optional txt/csv/pt file of feature weights matching --feature_ids. Defaults to 1.",
    )
    parser.add_argument(
        "--score_method",
        default="accuracy_slope",
        choices=["accuracy_slope", "accuracy_corr", "final_minus_base", "range_signed"],
    )
    parser.add_argument(
        "--aime_accuracies",
        help="Comma-separated accuracies ordered like the selected per-model CSV columns.",
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--column_regex",
        help="Regex selecting per-model columns from stats_csv. Defaults to all numeric non-id columns.",
    )
    parser.add_argument("--min_abs_score", type=float, default=0.0)
    parser.add_argument("--base_model_index", type=int, default=0)
    parser.add_argument(
        "--decoder_normalization",
        default="none",
        choices=["none", "unit"],
        help="Use 'unit' if your feature scores already include decoder norm.",
    )
    parser.add_argument("--normalize_weights", action="store_true")
    parser.add_argument(
        "--output_norm",
        type=float,
        default=1.0,
        help="Final vector norm. Use a negative value to skip output normalization.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--metadata_csv",
        help="Where to save selected feature ids and weights. Defaults to output_path with .csv suffix.",
    )
    return parser.parse_args()


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    args = parse_args()
    model_cfgs = get_nway_model_configurations(cfg)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.feature_ids is not None:
        feature_ids = _load_feature_ids(args.feature_ids)
        if args.feature_weights is None:
            weights = th.ones_like(feature_ids, dtype=th.float32)
            selected_df = pd.DataFrame(
                {"feature_id": feature_ids.numpy(), "score": weights.numpy(), "abs_score": weights.abs().numpy()}
            )
        else:
            weights = _load_feature_weights(args.feature_weights)
            selected_df = pd.DataFrame(
                {"feature_id": feature_ids.numpy(), "score": weights.numpy(), "abs_score": weights.abs().numpy()}
            )
    else:
        if args.stats_csv is None:
            raise ValueError("Either --stats_csv or --feature_ids must be provided")
        selected_df = compute_feature_scores(
            stats_csv=args.stats_csv,
            method=args.score_method,
            top_k=args.top_k,
            column_regex=args.column_regex,
            aime_accuracies=_parse_float_list(args.aime_accuracies),
            min_abs_score=args.min_abs_score,
        )
        feature_ids = th.tensor(selected_df["feature_id"].to_numpy(), dtype=th.long)
        weights = th.tensor(selected_df["score"].to_numpy(), dtype=th.float32)

    output_norm = args.output_norm if args.output_norm >= 0 else None
    steering_vector = build_steering_vector(
        model_path=args.model_path,
        feature_ids=feature_ids,
        feature_weights=weights,
        base_model_index=args.base_model_index,
        decoder_normalization=args.decoder_normalization,
        normalize_weights=args.normalize_weights,
        output_norm=output_norm,
        device=args.device,
    )

    th.save(steering_vector, output_path)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else output_path.with_suffix(".csv")
    selected_df.to_csv(metadata_csv, index=False)
    logger.info(f"Saved steering vector to {output_path}")
    logger.info(f"Saved selected feature metadata to {metadata_csv}")


if __name__ == "__main__":
    main()
