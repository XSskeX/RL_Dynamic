from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch as th
from loguru import logger
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


def _skip_first_n_tokens(cache: Any, n: int) -> Subset | Any:
    """Skip the first n tokens of each sequence in an activation cache."""
    if n == 0:
        return cache

    sequence_ranges = cache.sequence_ranges
    if sequence_ranges is None:
        raise ValueError("Cannot skip first tokens because cache.sequence_ranges is None")
    if getattr(sequence_ranges, "ndim", 1) > 1:
        sequence_ranges = sequence_ranges[0]

    sequence_starts = sequence_ranges[:-1]
    mask = th.ones(len(cache), dtype=th.bool)
    for offset in range(n):
        token_indices = sequence_starts + offset
        token_indices = token_indices[token_indices < len(cache)]
        mask[token_indices] = False
    return Subset(cache, th.where(mask)[0])


def _safe_column_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if hasattr(container, "get"):
        return container.get(key, default)
    return getattr(container, key, default)


def load_nway_activation_dataset_for_analysis(
    cfg: DictConfig,
    layer: int,
    split: str,
    max_num_samples: int | None = None,
) -> ConcatDataset:
    """Load the n-way activation dataset used for crosscoder activation analysis."""
    model_cfgs = get_nway_model_configurations(cfg)
    dataset_cfgs = get_dataset_configurations(
        cfg,
        use_chat_dataset=cfg.diffing.method.datasets.use_chat_dataset,
        use_pretraining_dataset=cfg.diffing.method.datasets.use_pretraining_dataset,
        use_training_dataset=cfg.diffing.method.datasets.use_training_dataset,
    )

    caches_by_dataset = load_n_activation_datasets_from_config(
        cfg=cfg,
        ds_cfgs=dataset_cfgs,
        model_cfgs=model_cfgs,
        layers=[layer],
        split=split,
    )
    caches = {
        dataset_name: caches_by_dataset[dataset_name][layer]
        for dataset_name in caches_by_dataset
    }

    skip_n = int(_cfg_get(cfg.model, "ignore_first_n_tokens_per_sample_during_training", 0) or 0)
    caches = {
        dataset_name: _skip_first_n_tokens(cache, skip_n)
        for dataset_name, cache in caches.items()
    }

    available = [len(cache) for cache in caches.values()]
    if max_num_samples is None:
        max_num_samples = min(sum(available), cfg.diffing.method.training.num_validation_samples)
    num_samples_per_dataset = calculate_samples_per_dataset(available, max_num_samples)

    logger.info(f"Using {sum(num_samples_per_dataset)} tokens for activation analysis")
    for dataset_name, num_samples in zip(caches.keys(), num_samples_per_dataset):
        logger.info(f"\tUsing {num_samples} tokens for {dataset_name}")

    return ConcatDataset(
        [
            Subset(cache, th.arange(0, num_samples))
            for cache, num_samples in zip(caches.values(), num_samples_per_dataset)
        ]
    )


@th.no_grad()
def analyze_crosscoder_activation_changes(
    cfg: DictConfig,
    layer: int,
    model_path: str | Path | None = None,
    split: str = "validation",
    max_num_samples: int | None = None,
    batch_size: int = 4096,
    num_workers: int = 0,
    device: str | None = None,
    output_dir: str | Path | None = None,
    use_threshold: bool = True,
    compute_per_model_proxy: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Analyze crosscoder latent activations on cached n-way activations.

    This produces:
      1. shared_activation_stats: statistics of the actual shared latent a.
      2. per_model_encoder_contrib_stats: diagnostic per-ckpt encoder contributions
         and model-only activation proxies.

    In a standard n-way crosscoder, a is shared across model slices. The per-model
    contribution table is therefore a proxy, not a separate trained a_i.
    """
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    if model_path is None:
        model_name = _cfg_get(cfg.model, "name", str(cfg.model))
        model_path = (
            Path(cfg.infrastructure.storage.checkpoint_dir)
            / model_name
            / "model_final.pt"
        )
    model_path = Path(model_path)

    crosscoder = load_dictionary_model(model_path).to(device).eval()
    dataset = load_nway_activation_dataset_for_analysis(
        cfg=cfg,
        layer=layer,
        split=split,
        max_num_samples=max_num_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    num_features = crosscoder.dict_size
    num_models = crosscoder.decoder.weight.shape[0]
    model_cfgs = get_nway_model_configurations(cfg)
    model_names = [
        _safe_column_name(getattr(model_cfg, "name", f"model_{idx}"))
        for idx, model_cfg in enumerate(model_cfgs)
    ]

    total_tokens = 0
    sum_a = th.zeros(num_features, device=device)
    sum_sq_a = th.zeros(num_features, device=device)
    nonzero_a = th.zeros(num_features, device=device)
    max_a = th.full((num_features,), -th.inf, device=device)

    contrib_sum = th.zeros(num_models, num_features, device=device)
    contrib_abs_sum = th.zeros(num_models, num_features, device=device)
    model_only_sum = th.zeros(num_models, num_features, device=device)
    model_only_sq_sum = th.zeros(num_models, num_features, device=device)
    model_only_nonzero = th.zeros(num_models, num_features, device=device)
    model_only_max = th.full((num_models, num_features), -th.inf, device=device)

    code_normalization = crosscoder.get_code_normalization().to(device)
    if code_normalization.ndim == 0:
        model_only_scale = code_normalization.reshape(1, 1).expand(num_models, num_features)
    elif code_normalization.shape[0] == num_models:
        model_only_scale = code_normalization
    else:
        model_only_scale = code_normalization.reshape(1, num_features).expand(num_models, num_features)
    threshold = getattr(crosscoder, "threshold", None)

    for batch in dataloader:
        x = batch.to(device=device, dtype=crosscoder.dtype)
        if x.ndim == 4:
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
        if x.ndim != 3:
            raise ValueError(f"Expected batch shape [B, M, D], got {tuple(x.shape)}")

        a = crosscoder.get_activations(x, use_threshold=use_threshold)
        total_tokens += a.shape[0]
        sum_a += a.sum(dim=0)
        sum_sq_a += (a * a).sum(dim=0)
        nonzero_a += (a > 0).sum(dim=0)
        max_a = th.maximum(max_a, a.max(dim=0).values)

        if compute_per_model_proxy:
            x_norm = crosscoder.normalize_activations(x, inplace=False)
            contrib = th.einsum("bmd,mdf->bmf", x_norm, crosscoder.encoder.weight)
            contrib_sum += contrib.sum(dim=0)
            contrib_abs_sum += contrib.abs().sum(dim=0)

            model_only = th.relu(contrib + crosscoder.encoder.bias)
            model_only_scaled = model_only * model_only_scale.unsqueeze(0)
            if use_threshold and threshold is not None:
                model_only_scaled = model_only_scaled * (model_only_scaled > threshold)
            model_only_sum += model_only_scaled.sum(dim=0)
            model_only_sq_sum += (model_only_scaled * model_only_scaled).sum(dim=0)
            model_only_nonzero += (model_only_scaled > 0).sum(dim=0)
            model_only_max = th.maximum(model_only_max, model_only_scaled.max(dim=0).values)

    if total_tokens == 0:
        raise ValueError("No activation tokens were loaded for analysis")

    feature_ids = th.arange(num_features).cpu().numpy()
    active_denominator = nonzero_a.clamp_min(1)
    shared_df = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "activation_freq": (nonzero_a / total_tokens).cpu().numpy(),
            "mean_activation": (sum_a / total_tokens).cpu().numpy(),
            "active_mean_activation": (sum_a / active_denominator).cpu().numpy(),
            "rms_activation": th.sqrt(sum_sq_a / total_tokens).cpu().numpy(),
            "max_activation": max_a.cpu().numpy(),
            "total_activation_mass": sum_a.cpu().numpy(),
        }
    )

    results = {"shared_activation_stats": shared_df}

    if compute_per_model_proxy:
        proxy_data: dict[str, Any] = {"feature_id": feature_ids}
        for model_idx, model_name in enumerate(model_names):
            proxy_data[f"{model_name}_contrib_mean"] = (
                contrib_sum[model_idx] / total_tokens
            ).cpu().numpy()
            proxy_data[f"{model_name}_contrib_abs_mean"] = (
                contrib_abs_sum[model_idx] / total_tokens
            ).cpu().numpy()
            proxy_data[f"{model_name}_model_only_activation_freq"] = (
                model_only_nonzero[model_idx] / total_tokens
            ).cpu().numpy()
            proxy_data[f"{model_name}_model_only_mean_activation"] = (
                model_only_sum[model_idx] / total_tokens
            ).cpu().numpy()
            proxy_data[f"{model_name}_model_only_active_mean_activation"] = (
                model_only_sum[model_idx] / model_only_nonzero[model_idx].clamp_min(1)
            ).cpu().numpy()
            proxy_data[f"{model_name}_model_only_rms_activation"] = th.sqrt(
                model_only_sq_sum[model_idx] / total_tokens
            ).cpu().numpy()
            proxy_data[f"{model_name}_model_only_max_activation"] = (
                model_only_max[model_idx]
            ).cpu().numpy()
            proxy_data[f"{model_name}_model_only_total_activation_mass"] = (
                model_only_sum[model_idx]
            ).cpu().numpy()
        results["per_model_encoder_contrib_stats"] = pd.DataFrame(proxy_data)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        shared_df.to_csv(
            output_path / "shared_activation_stats.csv",
            index=False,
            float_format="%.12f",
        )
        if "per_model_encoder_contrib_stats" in results:
            results["per_model_encoder_contrib_stats"].to_csv(
                output_path / "per_model_encoder_contrib_stats.csv",
                index=False,
                float_format="%.12f",
            )
        logger.info(f"Saved activation analysis to {output_path}")

    return results


def _compute_feature_norm(weight: th.Tensor) -> th.Tensor:
    return th.linalg.vector_norm(weight.float(), ord=2, dim=1)


def compute_feature_norm(crosscoder, save_path: str | Path | None = None) -> pd.DataFrame:
    num_features = crosscoder.decoder.weight.shape[1]
    latent_df = pd.DataFrame(index=range(num_features))
    for model_idx in range(crosscoder.decoder.weight.shape[0]):
        weight = crosscoder.decoder.weight[model_idx]
        norms = _compute_feature_norm(weight).cpu()
        latent_df[f"dec_{model_idx}_norm"] = norms.detach().numpy()

    if save_path is not None:
        latent_df.to_csv(Path(save_path), encoding="utf-8-sig")
    return latent_df


def _compute_feature_drift_cosine(
    weight: th.Tensor, original_weight: th.Tensor
) -> th.Tensor:
    weight = weight.float()
    original_weight = original_weight.float()
    return 1.0 - cosine_similarity(weight, original_weight, dim=1, eps=1e-8)


def compute_feature_drift_cosine(
    crosscoder, save_path: str | Path | None = None
) -> pd.DataFrame:
    num_features = crosscoder.decoder.weight.shape[1]
    latent_df = pd.DataFrame(index=range(num_features))
    original_weight = crosscoder.decoder.weight[0]
    for model_idx in range(1, crosscoder.decoder.weight.shape[0]):
        weight = crosscoder.decoder.weight[model_idx]
        drift = _compute_feature_drift_cosine(weight, original_weight).cpu()
        latent_df[f"dec_{model_idx}_drift_from_origin"] = drift.detach().numpy()

    if save_path is not None:
        latent_df.to_csv(Path(save_path), encoding="utf-8-sig")
    return latent_df


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    model_cfgs = get_nway_model_configurations(cfg)
    layers = _cfg_get(cfg.diffing.method, "layers", None) or cfg.preprocessing.layers
    layer = get_layer_indices(model_cfgs[0].model_id, layers)[0]
    latent_cfg = cfg.diffing.method.analysis.latent_activations
    batch_size = int(_cfg_get(latent_cfg, "batch_size", cfg.diffing.method.training.batch_size))
    max_num_samples = _cfg_get(latent_cfg, "max_num_samples", None)
    split = _cfg_get(latent_cfg, "split", "validation")
    out_dir = (
        Path(cfg.diffing.results_base_dir)
        / "activation_analysis"
        / (cfg.diffing.method.get("nway", {}).get("run_name") or _cfg_get(cfg.model, "name", str(cfg.model)))
        / f"layer_{layer}"
    )
    analyze_crosscoder_activation_changes(
        cfg=cfg,
        layer=layer,
        split=split,
        max_num_samples=max_num_samples,
        batch_size=batch_size,
        num_workers=int(_cfg_get(latent_cfg, "num_workers", 0) or 0),
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
