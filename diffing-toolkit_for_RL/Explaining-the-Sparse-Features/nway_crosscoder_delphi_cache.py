from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch as th
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("nway_crosscoder_delphi_cache")


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "vendor", THIS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

@dataclass
class SelectedCache:
    dataset_name: str
    cache: Any
    indices: th.Tensor


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if hasattr(container, "get"):
        return container.get(key, default)
    return getattr(container, key, default)


def _default_nway_method_override(overrides: list[str]) -> list[str]:
    has_method_override = any(
        override.lstrip("+~").startswith("diffing/method=")
        for override in overrides
    )
    if has_method_override:
        return overrides
    return ["diffing/method=nway_crosscoder", *overrides]


def load_cfg(overrides: list[str]) -> DictConfig:
    from hydra import compose, initialize_config_dir

    from diffing.utils.configs import CONFIGS_DIR

    with initialize_config_dir(
        config_dir=str(CONFIGS_DIR.resolve()),
        version_base=None,
    ):
        return compose(
            config_name="config",
            overrides=_default_nway_method_override(overrides),
        )


def resolve_layer(cfg: DictConfig, layer: int | None) -> int:
    from diffing.utils.activations import get_layer_indices
    from diffing.utils.configs import get_nway_model_configurations

    if layer is not None:
        return layer

    model_cfgs = get_nway_model_configurations(cfg)
    configured_layers = _cfg_get(cfg.diffing.method, "layers", None)
    if configured_layers is None:
        configured_layers = cfg.preprocessing.layers
    return get_layer_indices(model_cfgs[0].model_id, configured_layers)[0]


def default_model_path(cfg: DictConfig) -> Path:
    model_name = _cfg_get(cfg.model, "name", str(cfg.model))
    return (
        Path(cfg.infrastructure.storage.checkpoint_dir)
        / model_name
        / "model_final.pt"
    )


def default_output_dir(cfg: DictConfig, layer: int) -> Path:
    run_name = (
        cfg.diffing.method.get("nway", {}).get("run_name")
        or _cfg_get(cfg.model, "name", str(cfg.model))
    )
    return (
        Path(cfg.diffing.results_base_dir)
        / "nway_crosscoder_delphi_cache"
        / str(run_name)
        / f"layer_{layer}"
        / "raw_latents"
    )


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


def get_cache_tokens(cache: Any, indices: th.Tensor) -> th.Tensor:
    print(f"get cache tokens, tokens dim: {tokens.ndim}")
    tokens = cache.tokens
    if tokens is None:
        raise ValueError("The activation cache does not store tokens")
    if tokens.ndim == 2:
        tokens = tokens[0]
    return tokens[indices].to(dtype=th.long).cpu()


def load_selected_nway_caches(
    cfg: DictConfig,
    layer: int,
    split: str,
    max_num_samples: int | None,
) -> list[SelectedCache]:
    from diffing.utils.activations import (
        calculate_samples_per_dataset,
        load_n_activation_datasets_from_config,
    )
    from diffing.utils.configs import (
        get_dataset_configurations,
        get_nway_model_configurations,
    )

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

    skip_first_n = int(
        _cfg_get(cfg.model, "ignore_first_n_tokens_per_sample_during_training", 0)
        or 0
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


    selected: list[SelectedCache] = []
    logger.info(f"Using {sum(num_samples_per_dataset)} tokens for activation analysis")
    for dataset_name, num_samples in zip(caches.keys(), num_samples_per_dataset):
        logger.info(f"\tUsing {num_samples} tokens for {dataset_name}")
        selected.append(SelectedCache(
            dataset_name=dataset_name,
            cache=cache,
            indices=available[:n_samples],
            )
        )

    return selected


def build_dataset(selected_caches: Iterable[SelectedCache]) -> ConcatDataset:
    subsets = [
        Subset(selected.cache, selected.indices)
        for selected in selected_caches
        if len(selected.indices) > 0
    ]
    if not subsets:
        raise ValueError("No activation samples were selected for caching")
    return ConcatDataset(subsets)


def build_token_matrix(
    selected_caches: Iterable[SelectedCache],
    ctx_len: int,
) -> tuple[np.ndarray, int]:
    flat_tokens = th.cat(
        [
            get_cache_tokens(selected.cache, selected.indices)
            for selected in selected_caches
            if len(selected.indices) > 0
        ],
        dim=0,
    )
    usable_tokens = (flat_tokens.numel() // ctx_len) * ctx_len
    if usable_tokens == 0:
        raise ValueError(
            f"Not enough tokens ({flat_tokens.numel()}) to form ctx_len={ctx_len}"
        )
    if usable_tokens < flat_tokens.numel():
        logger.warning(
            f"Dropping {flat_tokens.numel() - usable_tokens} trailing tokens "
            "so the token matrix is divisible by ctx_len"
        )
    tokens = flat_tokens[:usable_tokens].reshape(-1, ctx_len)
    return tokens.numpy().astype(np.int64), usable_tokens


def split_boundaries(num_features: int, n_splits: int) -> list[tuple[int, int]]:
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    boundaries = th.linspace(0, num_features, steps=n_splits + 1).long().tolist()
    return [
        (int(start), int(end) - 1)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]


def prepare_locations(locations: th.Tensor, start: int) -> np.ndarray:
    if locations.numel() == 0:
        return np.zeros((0, 3), dtype=np.uint16)

    locations = locations.cpu().to(dtype=th.long).numpy()
    locations[:, 2] = locations[:, 2] - start

    if locations[:, 2].max() < 2**16 and locations[:, 0].max() < 2**16:
        return locations.astype(np.uint16)
    return locations.astype(np.uint32)


def encode_crosscoder_activations(
    crosscoder: Any,
    x: th.Tensor,
    use_threshold: bool,
) -> th.Tensor:
    if hasattr(crosscoder, "get_activations"):
        get_parameters = inspect.signature(crosscoder.get_activations).parameters
        get_kwargs = {"normalize_activations": False}
        if "use_threshold" in get_parameters:
            get_kwargs["use_threshold"] = use_threshold
        activations = crosscoder.get_activations(x, **get_kwargs)
        if activations.ndim != 2:
            raise ValueError(
                f"Expected encoded activations [B, F], got {tuple(activations.shape)}"
            )
        return activations

    encode_parameters = inspect.signature(crosscoder.encode).parameters
    encode_kwargs = {
        "normalize_activations": False,
    }
    if "use_threshold" in encode_parameters:
        encode_kwargs["use_threshold"] = use_threshold
    if "return_active" in encode_parameters:
        encoded = crosscoder.encode(x, return_active=True, **encode_kwargs)
        activations = encoded[0] if isinstance(encoded, tuple) else encoded
    else:
        activations = crosscoder.encode(x, **encode_kwargs)

    if activations.ndim == 3:
        activations = activations.sum(dim=1)
    if activations.ndim != 2:
        raise ValueError(
            f"Expected encoded activations [B, F], got {tuple(activations.shape)}"
        )
    return activations


@th.no_grad()
def collect_crosscoder_latents(
    crosscoder: Any,
    dataset: ConcatDataset,
    usable_tokens: int,
    ctx_len: int,
    batch_size: int,
    num_workers: int,
    device: str,
    use_threshold: bool,
    min_activation: float,
    boundaries: list[tuple[int, int]],
) -> tuple[list[list[th.Tensor]], list[list[th.Tensor]]]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    processed = 0
    pbar = tqdm(total=usable_tokens, desc="Collecting latents", unit="token")
    split_locations: list[list[th.Tensor]] = [[] for _ in boundaries]
    split_activations: list[list[th.Tensor]] = [[] for _ in boundaries]

    for batch in dataloader:
        remaining = usable_tokens - processed
        if remaining <= 0:
            break
        x = batch
        if x.shape[0] > remaining:
            x = x[:remaining]
        x = x.to(device=device, dtype=crosscoder.dtype)
        if x.ndim == 4:
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
        if x.ndim != 3:
            raise ValueError(f"Expected batch shape [B, M, D], got {tuple(x.shape)}")

        #x_norm = crosscoder.normalize_activations(x, inplace=False)
        #activations = encode_crosscoder_activations(
        #    crosscoder,
        #    x_norm,
        #    use_threshold=use_threshold,
        #)
        activations = crosscoder.get_activations(x, use_threshold=use_threshold)
        if activations.ndim != 2:
            raise ValueError(
                f"Expected encoded activations [B, F], got {tuple(activations.shape)}"
            )
        active_mask = activations > min_activation
        active_indices = th.nonzero(active_mask, as_tuple=False)
        if active_indices.numel() > 0:
            batch_positions = active_indices[:, 0] + processed
            token_positions = batch_positions % ctx_len
            context_positions = batch_positions // ctx_len
            feature_indices = active_indices[:, 1]
            active_values = activations[active_mask].detach().float().cpu()

            for split_idx, (start, end) in enumerate(boundaries):
                in_split = (feature_indices >= start) & (feature_indices <= end)
                if not in_split.any():
                    continue

                loc = th.stack(
                    [
                        context_positions[in_split],
                        token_positions[in_split],
                        feature_indices[in_split],
                    ],
                    dim=1,
                ).cpu()
                split_locations[split_idx].append(loc)
                split_activations[split_idx].append(active_values[in_split.cpu()])
        
        processed += x.shape[0]
        pbar.update(x.shape[0])
    pbar.close()
    if processed != usable_tokens:
        raise RuntimeError(f"Processed {processed} tokens, expected {usable_tokens}")

    return split_locations, split_activations



def save_delphi_cache(
    output_dir: Path,
    module_name: str,
    tokens: np.ndarray,
    split_locations: list[list[th.Tensor]],
    split_activations: list[list[th.Tensor]],
    boundaries: list[tuple[int, int]],
    model_name: str,
    batch_size: int,
    ctx_len: int,
    n_tokens: int,
    overwrite: bool,
) -> None:
    from safetensors.numpy import save_file

    module_dir = output_dir / module_name
    module_dir.mkdir(parents=True, exist_ok=True)

    existing = list(module_dir.glob("*.safetensors"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{module_dir} already contains safetensors files. "
            "Pass --overwrite to replace matching split files."
        )

    total_activations = 0
    for split_idx, (start, end) in enumerate(boundaries):
        if split_locations[split_idx]:
            locations = th.cat(split_locations[split_idx], dim=0)
            activations = th.cat(split_activations[split_idx], dim=0)
        else:
            locations = th.zeros((0, 3), dtype=th.long)
            activations = th.zeros((0,), dtype=th.float32)

        total_activations += int(activations.numel())
        data = {
            "locations": prepare_locations(locations, start),
            "activations": activations.numpy().astype(np.float16),
            "tokens": tokens,
        }
        save_file(data, module_dir / f"{start}_{end}.safetensors")

    config = {
        "dataset_repo": "local_nway_activation_cache",
        "dataset_split": "nway_crosscoder",
        "dataset_name": "",
        "dataset_column": "text",
        "batch_size": batch_size,
        "cache_ctx_len": ctx_len,
        "n_tokens": n_tokens,
        "n_splits": len(boundaries),
        "model_name": model_name,
    }
    with open(module_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    logger.info(
        f"Saved {total_activations} activations for {module_name} to {module_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert n-way crosscoder feature activations into the Delphi "
            "raw_latents cache format used by Explaining-the-Sparse-Features."
        )
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--module_name", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=1)
    parser.add_argument("--min_activation", type=float, default=0.0)
    parser.add_argument("--no_threshold", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--hydra_overrides",
        nargs="*",
        default=[],
        help="Hydra overrides for the root project config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.hydra_overrides)

    layer = resolve_layer(cfg, args.layer)
    latent_cfg = cfg.diffing.method.analysis.latent_activations

    split = args.split or _cfg_get(latent_cfg, "split", "validation")
    max_num_samples = args.max_num_samples
    if max_num_samples is None:
        max_num_samples = _cfg_get(latent_cfg, "max_num_samples", None)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(
            _cfg_get(
                latent_cfg,
                "batch_size",
                cfg.diffing.method.training.batch_size,
            )
        )

    device = args.device or ("cuda" if th.cuda.is_available() else "cpu")
    model_path = Path(args.model_path) if args.model_path else default_model_path(cfg)
    output_dir = (
        Path(args.output_dir) if args.output_dir else default_output_dir(cfg, layer)
    )
    module_name = args.module_name or f"nway_crosscoder.layer_{layer}"

    from diffing.utils.dictionary import load_dictionary_model

    logger.info(f"Loading nway crosscoder from {model_path}")
    crosscoder = load_dictionary_model(model_path).to(device).eval()
    num_features = int(crosscoder.dict_size)

    selected_caches = load_selected_nway_caches(
        cfg=cfg,
        layer=layer,
        split=split,
        max_num_samples=max_num_samples,
    )
    dataset = build_dataset(selected_caches)
    tokens, usable_tokens = build_token_matrix(selected_caches, args.ctx_len)
    boundaries = split_boundaries(num_features, args.n_splits)
    logger.info(
        f"Caching {num_features} features into {len(boundaries)} split file(s)"
    )

    split_locations, split_activations = collect_crosscoder_latents(
        crosscoder=crosscoder,
        dataset=dataset,
        usable_tokens=usable_tokens,
        ctx_len=args.ctx_len,
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=device,
        use_threshold=not args.no_threshold,
        min_activation=args.min_activation,
        boundaries=boundaries,
    )

    save_delphi_cache(
        output_dir=output_dir,
        module_name=module_name,
        tokens=tokens,
        split_locations=split_locations,
        split_activations=split_activations,
        boundaries=boundaries,
        model_name=str(model_path),
        batch_size=batch_size,
        ctx_len=args.ctx_len,
        n_tokens=usable_tokens,
        overwrite=args.overwrite,
    )

    print(f"Delphi cache saved to: {output_dir / module_name}", flush=True)


if __name__ == "__main__":
    main()
