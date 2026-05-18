"""
Analysis pipeline wrapper for crosscoder evaluation and analysis.

This module provides a wrapper around the comprehensive analysis pipeline from
science-of-finetuning, including evaluation notebooks, scaler computation,
latent statistics, and KL divergence experiments.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from loguru import logger
from transformers import AutoTokenizer
import torch as th
import pandas as pd
from torch.nn.functional import cosine_similarity
from tqdm.auto import trange
import numpy as np
import os
from diffing.utils.dictionary import load_dictionary_model
from diffing.utils.dictionary.utils import push_latent_df, load_latent_df


def build_push_crosscoder_latent_df(
    dictionary_name: str,
    base_layer: int = 0,
    ft_layer: int = 1,
) -> pd.DataFrame:
    crosscoder = load_dictionary_model(dictionary_name)
    print("Successfully loaded crosscoder model for latent df construction")
    try:
        existing_latent_df = load_latent_df(dictionary_name)
        logger.info(
            f"Found existing latent dataframe for {dictionary_name} with {len(existing_latent_df)} latents"
        )
        latent_df = existing_latent_df.T.to_dict()
    except Exception:
        latent_df = {k: {} for k in range(crosscoder.dict_size)}

    # Norms
    norms = crosscoder.decoder.weight.norm(dim=-1)
    norm_diffs = (
        (norms[base_layer] - norms[ft_layer]) / norms.max(dim=0).values + 1
    ) / 2
    norm_diffs = norm_diffs.cpu()
    for f_idx, (base_norm, instruct_norm, norm_diff) in enumerate(
        zip(norms[base_layer], norms[ft_layer], norm_diffs)
    ):
        latent_df[f_idx]["dec_base_norm"] = base_norm.item()
        latent_df[f_idx]["dec_ft_norm"] = instruct_norm.item()
        latent_df[f_idx]["dec_norm_diff"] = norm_diff.item()

    enc_norms = crosscoder.encoder.weight.norm(dim=1)
    enc_norm_diffs = (
        (enc_norms[base_layer] - enc_norms[ft_layer]) / enc_norms.max(dim=0).values + 1
    ) / 2
    enc_norm_diffs = enc_norm_diffs.cpu()

    for f_idx, (base_norm, instruct_norm, norm_diff) in enumerate(
        zip(enc_norms[base_layer], enc_norms[ft_layer], enc_norm_diffs)
    ):
        latent_df[f_idx]["enc_base_norm"] = base_norm.item()
        latent_df[f_idx]["enc_ft_norm"] = instruct_norm.item()
        latent_df[f_idx]["enc_norm_diff"] = norm_diff.item()

    decoder_cos_sims = cosine_similarity(
        crosscoder.decoder.weight[base_layer],  # [num_features, activation_dim]
        crosscoder.decoder.weight[ft_layer],  # [num_features, activation_dim]
        dim=1,
    )
    for f_idx, cos_sim in enumerate(decoder_cos_sims):
        latent_df[f_idx]["dec_cos_sim"] = cos_sim.item()

    # Encoder cos sims
    enc_cos_sims = cosine_similarity(
        crosscoder.encoder.weight[base_layer],  # [activation_dim, num_features]
        crosscoder.encoder.weight[ft_layer],  # [activation_dim, num_features]
        dim=0,
    )
    for f_idx, cos_sim in enumerate(enc_cos_sims):
        latent_df[f_idx]["enc_cos_sim"] = cos_sim.item()

    # Create masks for each category
    # Decoder
    # save ft only and base only feature index
    treshold = 0.1
    only_it_feature_indices = th.nonzero(norm_diffs < treshold, as_tuple=True)[0]
    only_base_feature_indices = th.nonzero(norm_diffs > 1 - treshold, as_tuple=True)[0]
    shared_feature_indices = th.nonzero(
        (norm_diffs - 0.5).abs() < treshold, as_tuple=True
    )[0]
    is_other_feature = th.ones_like(norm_diffs, dtype=bool)
    is_other_feature[only_it_feature_indices] = False
    is_other_feature[only_base_feature_indices] = False
    is_other_feature[shared_feature_indices] = False

    for f_idx in only_it_feature_indices.tolist():
        latent_df[f_idx]["tag"] = "ft_only"
    for f_idx in only_base_feature_indices.tolist():
        latent_df[f_idx]["tag"] = "base_only"
    for f_idx in shared_feature_indices.tolist():
        latent_df[f_idx]["tag"] = "shared"
    for f_idx in is_other_feature.nonzero(as_tuple=True)[0].tolist():
        latent_df[f_idx]["tag"] = "other"

    latent_df = pd.DataFrame(latent_df).T
    logger.info(f"Created latent dataframe with {len(latent_df)} latents")
    push_latent_df(
        latent_df, dictionary_name, confirm=False, create_repo_if_missing=True
    )
    return latent_df


def _self_vs_all_dot_ratio(weight: th.Tensor) -> th.Tensor:
    """
    Unit-normalize each feature, then compute diag(W W^T) divided by the
    row-wise sum of abs(W W^T).

    For unit-normalized feature vectors u_i, this returns:
        ||u_i||^2 / sum_j |<u_i, u_j>|

    The sum includes the self-dot-product term j = i.
    """
    weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(th.finfo(weight.dtype).eps)
    gram = weight @ weight.T
    numerator = th.diag(gram)
    denominator = gram.abs().sum(dim=1)
    eps = th.finfo(gram.dtype).eps
    return numerator / denominator.clamp_min(eps)


def _mean_abs_other_dot(weight: th.Tensor) -> th.Tensor:
    weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(
        th.finfo(weight.dtype).eps
    )
    gram = weight @ weight.T
    n = gram.shape[0]

    off_diag_sum = gram.abs().sum(dim=1) - th.diag(gram).abs()
    return off_diag_sum


def compute_feature_direction_drift(
    dictionary_name: str | Path,
    model_names: Optional[list[str]] = None,
    distance: str = "cosine",
    save_path: Optional[str | Path] = None,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Measure how much each shared crosscoder feature direction moves across models.

    The crosscoder decoder is expected to have shape:
        [num_models, num_features, activation_dim]
    """
    if distance not in {"cosine", "angle"}:
        raise ValueError(f"distance must be 'cosine' or 'angle', got {distance}")

    crosscoder = load_dictionary_model(dictionary_name)
    decoder_weight = crosscoder.decoder.weight.detach().float()
    if decoder_weight.ndim != 3:
        raise ValueError(
            "Expected crosscoder.decoder.weight to have shape "
            f"[num_models, num_features, activation_dim], got {tuple(decoder_weight.shape)}"
        )

    num_models, num_features, _ = decoder_weight.shape
    if model_names is None:
        model_names = [f"model_{i}" for i in range(num_models)]
    elif len(model_names) != num_models:
        raise ValueError(
            f"model_names length ({len(model_names)}) must match num_models ({num_models})"
        )

    norms = decoder_weight.norm(dim=-1)
    valid = norms > eps
    normalized_weight = decoder_weight / norms.clamp_min(eps).unsqueeze(-1)

    cos_to_base = (normalized_weight * normalized_weight[0:1]).sum(dim=-1)
    cos_to_base = cos_to_base.clamp(-1.0, 1.0)
    cos_to_prev = (normalized_weight[1:] * normalized_weight[:-1]).sum(dim=-1)
    cos_to_prev = cos_to_prev.clamp(-1.0, 1.0)

    if distance == "angle":
        drift_to_base = th.arccos(cos_to_base)
        step_drift = th.arccos(cos_to_prev)
    else:
        drift_to_base = 1.0 - cos_to_base
        step_drift = 1.0 - cos_to_prev

    drift_to_base = drift_to_base.masked_fill(~(valid & valid[0:1]), float("nan"))
    step_drift = step_drift.masked_fill(~(valid[1:] & valid[:-1]), float("nan"))

    def _nanmax_with_indices(values: th.Tensor, dim: int) -> tuple[th.Tensor, th.Tensor]:
        all_nan = th.isnan(values).all(dim=dim)
        max_values, max_indices = th.nan_to_num(values, nan=float("-inf")).max(dim=dim)
        max_values = max_values.masked_fill(all_nan, float("nan"))
        max_indices = max_indices.masked_fill(all_nan, -1)
        return max_values, max_indices

    def _nanmean(values: th.Tensor, dim: int) -> th.Tensor:
        valid_values = ~th.isnan(values)
        counts = valid_values.sum(dim=dim).clamp_min(1)
        sums = th.nan_to_num(values, nan=0.0).sum(dim=dim)
        means = sums / counts
        return means.masked_fill(~valid_values.any(dim=dim), float("nan"))

    max_drift_to_base, max_drift_to_base_idx = _nanmax_with_indices(
        drift_to_base, dim=0
    )
    max_step_drift, max_step_drift_idx = _nanmax_with_indices(step_drift, dim=0)
    path_length = th.nan_to_num(step_drift, nan=0.0).sum(dim=0)
    mean_step_drift = _nanmean(step_drift, dim=0)

    max_drift_to_base_idx_cpu = max_drift_to_base_idx.detach().cpu().tolist()
    max_step_drift_idx_cpu = max_step_drift_idx.detach().cpu().tolist()

    data: Dict[str, Any] = {
        "feature_id": np.arange(num_features),
        "base_norm": norms[0].detach().cpu().numpy(),
        "final_norm": norms[-1].detach().cpu().numpy(),
        "final_drift_to_base": drift_to_base[-1].detach().cpu().numpy(),
        "max_drift_to_base": max_drift_to_base.detach().cpu().numpy(),
        "max_drift_to_base_idx": max_drift_to_base_idx.detach().cpu().numpy(),
        "max_drift_to_base_model": [
            model_names[idx] if idx >= 0 else None for idx in max_drift_to_base_idx_cpu
        ],
        "max_step_drift": max_step_drift.detach().cpu().numpy(),
        "max_step_drift_from_idx": max_step_drift_idx.detach().cpu().numpy(),
        "max_step_drift_to_idx": np.array(
            [idx + 1 if idx >= 0 else -1 for idx in max_step_drift_idx_cpu]
        ),
        "max_step_drift_from_model": [
            model_names[idx] if idx >= 0 else None for idx in max_step_drift_idx_cpu
        ],
        "max_step_drift_to_model": [
            model_names[idx + 1] if idx >= 0 else None for idx in max_step_drift_idx_cpu
        ],
        "path_length": path_length.detach().cpu().numpy(),
        "mean_step_drift": mean_step_drift.detach().cpu().numpy(),
        "valid_model_count": valid.sum(dim=0).detach().cpu().numpy(),
    }

    for model_idx, model_name in enumerate(model_names):
        data[f"drift_to_base_{model_name}"] = (
            drift_to_base[model_idx].detach().cpu().numpy()
        )
        data[f"cos_to_base_{model_name}"] = (
            cos_to_base[model_idx].detach().cpu().numpy()
        )

    for model_idx in range(1, num_models):
        model_name = model_names[model_idx]
        data[f"step_drift_to_{model_name}"] = (
            step_drift[model_idx - 1].detach().cpu().numpy()
        )
        data[f"cos_to_prev_{model_name}"] = (
            cos_to_prev[model_idx - 1].detach().cpu().numpy()
        )

    drift_df = pd.DataFrame(data)
    if save_path is not None:
        drift_df.to_csv(Path(save_path), index=False, float_format="%.12f")
    return drift_df

def update_crosscoder_latent_df_with_self_dot_ratio(
    dictionary_name: str,
    base_layer: int = 0,
    ft_layer: int = 1,
    model_name: str = "noname"
) -> pd.DataFrame:
    """
    Add per-feature self-vs-all dot-product ratios to the crosscoder latent df.

    For each feature i and each side (base/ft), computes:
        ||f_i||^2 / sum_j <f_i, f_j>

    using decoder feature vectors.
    """
    crosscoder = load_dictionary_model(dictionary_name)
    print(f"dictionary model has {len(crosscoder.decoder.weight)} layers")
    print(f"dictionary model has {crosscoder.decoder.weight.shape[0]} models per layer")
    num_features = crosscoder.decoder.weight.shape[1]
    latent_df = pd.DataFrame(index=range(num_features))
    for i in range(crosscoder.decoder.weight.shape[0]):
        weight = crosscoder.decoder.weight[i]

        ratios = _self_vs_all_dot_ratio(weight).cpu()

        latent_df[f"dec_{i}_self_dot_ratio_norm"] = ratios.detach().numpy()

    latent_df.to_csv(Path(f"/share/nlp/baijun/shuhan/crosscoder_output_for_8/{model_name}_latent_dimentionality.csv"), encoding='utf-8-sig')
    return latent_df

def build_push_sae_difference_latent_df(
    dictionary_name: str,
    target: str,
) -> pd.DataFrame:
    """
    Build latent dataframe for SAE difference models.

    Args:
        dictionary_name: Name of the SAE model
        target: Training target ("difference_bft" or "difference_ftb")

    Returns:
        DataFrame containing latent statistics for SAE difference model
    """
    logger.info(
        f"Building latent dataframe for SAE difference model: {dictionary_name}"
    )

    sae = load_dictionary_model(dictionary_name)
    try:
        existing_latent_df = load_latent_df(dictionary_name)
        logger.info(
            f"Found existing latent dataframe for {dictionary_name} with {len(existing_latent_df)} latents"
        )
        latent_df = existing_latent_df.T.to_dict()
    except Exception:
        latent_df = {k: {} for k in range(sae.dict_size)}

    # Decoder norms
    decoder_norms = sae.decoder.weight.norm(dim=0)  # [num_features]
    assert decoder_norms.shape == (sae.dict_size,)

    # Encoder norms
    encoder_norms = sae.encoder.weight.norm(dim=1)  # [num_features]
    assert encoder_norms.shape == (sae.dict_size,)

    for f_idx, norm in enumerate(decoder_norms):
        latent_df[f_idx]["dec_norm"] = norm.item()
    for f_idx, norm in enumerate(encoder_norms):
        latent_df[f_idx]["enc_norm"] = norm.item()

    # Convert to DataFrame
    latent_df = pd.DataFrame(latent_df).T

    logger.info(f"Created latent dataframe with {len(latent_df)} latents")
    push_latent_df(
        latent_df, dictionary_name, confirm=False, create_repo_if_missing=True
    )
    return latent_df


def make_plots(
    dictionary_name: str,
    plots_dir: Path,
):
    df = load_latent_df(dictionary_name)
    print("try to mkdir")
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("mkdir successfully")
    print(df.columns)
    os.environ["MPLCONFIGDIR"] = "/data/shuhan/.cache/matplotlib"
    os.environ["TMPDIR"] = "/data/shuhan/.cache/tmp"
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    if (
        "effective_ft_only_latent" in df.columns
        and "shared_baseline_latent" in df.columns
    ):
        target_df = df[df["effective_ft_only_latent"]]
        baseline_df = df[df["shared_baseline_latent"]]
        plot_error_vs_reconstruction(
            target_df, baseline_df, plots_dir, variant="standard"
        )
        plot_error_vs_reconstruction(
            target_df, baseline_df, plots_dir, variant="custom_color"
        )
        plot_error_vs_reconstruction(
            target_df, baseline_df, plots_dir, variant="poster"
        )

        plot_ratio_histogram(target_df, baseline_df, plots_dir, ratio_type="error")
        plot_ratio_histogram(
            target_df, baseline_df, plots_dir, ratio_type="reconstruction"
        )

        plot_beta_distribution_histograms(target_df, plots_dir)
        plot_correlation_with_frequency(df, plots_dir)
        plot_rank_distributions(target_df, plots_dir)
    print("successfully finish 1")
    if "enc_norm" in df.columns:
        plot_histogram(
            df,
            "enc_norm",
            plots_dir,
            title="Encoder Norm",
            xlabel="Norm",
            log_scale=True,
        )

    if "dec_norm" in df.columns:
        plot_histogram(
            df,
            "dec_norm",
            plots_dir,
            title="Decoder Norm",
            xlabel="Norm",
            log_scale=True,
        )

    if "max_act_validation" in df.columns:
        plot_histogram(
            df,
            "max_act_validation",
            plots_dir,
            title="Max Activation Validation",
            xlabel="Max Activation Validation",
            log_scale=True,
        )

    if "max_act_train" in df.columns:
        plot_histogram(
            df,
            "max_act_train",
            plots_dir,
            title="Max Activation Train",
            xlabel="Max Activation Train",
            log_scale=True,
        )

    if "beta_ratio_activation" in df.columns:
        plot_histogram(
            df,
            "beta_ratio_activation",
            plots_dir,
            title="Beta Ratio Activation",
            xlabel="Beta Ratio Activation",
            log_scale=True,
        )

    if "beta_ratio_activation_no_bias" in df.columns:
        plot_histogram(
            df,
            "beta_ratio_activation_no_bias",
            plots_dir,
            title="Beta Ratio Activation No Bias",
            xlabel="Beta Ratio Activation No Bias",
            log_scale=True,
        )

    if "freq_validation" in df.columns:
        plot_histogram(
            df,
            "freq_validation",
            plots_dir,
            title="Frequency Validation",
            xlabel="Frequency Validation",
            log_scale=True,
        )
    print("successfully finish 2")

def plot_histogram(
    df, column, plots_dir, title=None, xlabel=None, filename=None, log_scale=False
):
    """Plot histogram of a column from dataframe

    Args:
        df: DataFrame containing the data
        column: Column name to plot
        plots_dir: Directory to save plots
        title: Plot title (defaults to column name)
        xlabel: X-axis label (defaults to column name)
        filename: Output filename (defaults to column name)
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in dataframe")
        return
    print("successfully enter plot_histogram")
    plt.figure(figsize=(6, 4))
    plt.rcParams["text.usetex"] = False
    plt.rcParams.update({"font.size": 24})

    plt.hist(df[column], bins=100, alpha=0.7, color="blue")
    plt.xlabel(xlabel or column.replace("_", " ").title())
    plt.ylabel("Count")
    plt.title(title or column.replace("_", " ").title())
    if log_scale:
        plt.yscale("log")
    plt.tight_layout()
    output_filename = filename or f"{column}.pdf"
    plt.savefig(plots_dir / output_filename, bbox_inches="tight")
    plt.close()


def plot_beta_ratios_template_perc(target_df, filtered_df, plots_dir):
    """Plot histograms of beta ratios for template percentage

    Args:
        target_df: DataFrame containing all ft-only latents
        filtered_df: DataFrame containing latents with high template percentage
        plots_dir: Directory to save plots
    """
    if (
        "lmsys_ctrl_%" in target_df.columns
        and "beta_ratio_error" in target_df.columns
        and "beta_ratio_reconstruction" in target_df.columns
    ):
        low, high = -0.1, 1.1

        plt.figure(figsize=(8, 4))
        plt.rcParams["text.usetex"] = False
        plt.rcParams.update({"font.size": 24})

        # First subplot for beta_ratio_error
        ax1 = plt.subplot(1, 2, 1)
        # Plot full distribution
        target_df["beta_ratio_error"].hist(
            bins=50,
            range=(low, high),
            alpha=0.5,
            color="gray",
            label="All ft-only latents",
        )
        # Plot filtered distribution on top
        filtered_df["beta_ratio_error"].hist(
            bins=50,
            range=(low, high),
            alpha=0.7,
            color="blue",
            label="High Template Perc.",
        )
        plt.xlabel("$\\nu^\\epsilon$")
        ax1.tick_params(axis="y", rotation=90)
        plt.ylabel("Count")

        # Second subplot for beta_ratio_reconstruction
        ax2 = plt.subplot(1, 2, 2, sharey=ax1)
        # Plot full distribution
        target_df["beta_ratio_reconstruction"].hist(
            bins=50, range=(low, high), alpha=0.5, color="gray"
        )
        # Plot filtered distribution on top
        filtered_df["beta_ratio_reconstruction"].hist(
            bins=50, range=(low, high), alpha=0.7, color="blue"
        )
        plt.xlabel("$\\nu^r$")
        plt.setp(ax2.get_yticklabels(), visible=False)
        # Single legend for both plots
        plt.figlegend(fontsize=18.5, loc="center right", bbox_to_anchor=(0.532, 0.8))

        plt.tight_layout()
        plt.savefig(plots_dir / "beta_ratios_template_perc.pdf", bbox_inches="tight")
        plt.close()


def plot_error_vs_reconstruction(target_df, baseline_df, plots_dir, variant="standard"):
    """Plot scatter plot of error vs reconstruction ratios"""
    if not (
        "beta_ratio_error" in target_df.columns
        and "beta_ratio_reconstruction" in target_df.columns
    ):
        return

    zoom = [0, 1.1] if variant != "zoomed" else [-0.1, 1.1]
    ft_only_color = (0, 0.6, 1) if variant == "custom_color" else None

    # Create figure with a main plot and two side histograms
    fig_size = (
        (6, 3.5)
        if variant == "standard"
        else (4, 3) if variant == "custom_color" else (8, 3)
    )
    fig = plt.figure(figsize=fig_size)

    # Create a grid of subplots
    gs = plt.GridSpec(
        2,
        2,
        width_ratios=[3, 1.3],
        height_ratios=[1, 3],
        left=0.1,
        right=0.85,
        bottom=0.1,
        top=0.9,
        wspace=0.03,
        hspace=0.05,
    )

    # Create the three axes
    ax_scatter = fig.add_subplot(gs[1, 0])  # Main plot
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)  # x-axis histogram
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)  # y-axis histogram

    plt.rcParams["text.usetex"] = False
    plt.rcParams.update({"font.size": 20})

    # Filter out nans and apply zoom
    error_ratio = target_df["beta_ratio_error"]
    reconstruction_ratio = target_df["beta_ratio_reconstruction"]
    valid_mask = ~(np.isnan(error_ratio) | np.isnan(reconstruction_ratio))
    error_ratio_valid = error_ratio[valid_mask]
    reconstruction_ratio_valid = reconstruction_ratio[valid_mask]

    error_ratio_shared = baseline_df["beta_ratio_error"]
    reconstruction_ratio_shared = baseline_df["beta_ratio_reconstruction"]
    valid_mask_shared = ~(
        np.isnan(error_ratio_shared) | np.isnan(reconstruction_ratio_shared)
    )
    error_ratio_shared_valid = error_ratio_shared[valid_mask_shared]
    reconstruction_ratio_shared_valid = reconstruction_ratio_shared[valid_mask_shared]

    # Apply zoom mask to both datasets
    zoom_mask = (
        (error_ratio_valid > zoom[0])
        & (error_ratio_valid < zoom[1])
        & (reconstruction_ratio_valid > zoom[0])
        & (reconstruction_ratio_valid < zoom[1])
    )
    error_ratio_zoomed = error_ratio_valid[zoom_mask]
    reconstruction_ratio_zoomed = reconstruction_ratio_valid[zoom_mask]

    zoom_mask_shared = (
        (error_ratio_shared_valid > zoom[0])
        & (error_ratio_shared_valid < zoom[1])
        & (reconstruction_ratio_shared_valid > zoom[0])
        & (reconstruction_ratio_shared_valid < zoom[1])
    )
    error_ratio_shared_zoomed = error_ratio_shared_valid[zoom_mask_shared]
    reconstruction_ratio_shared_zoomed = reconstruction_ratio_shared_valid[
        zoom_mask_shared
    ]

    # Plot the scatter plots
    scatter_kwargs = {"alpha": 0.2, "s": 5}
    if ft_only_color:
        ax_scatter.scatter(
            error_ratio_zoomed,
            reconstruction_ratio_zoomed,
            label="ft-only",
            color=ft_only_color,
            **scatter_kwargs,
        )
        ax_scatter.scatter(
            error_ratio_shared_zoomed,
            reconstruction_ratio_shared_zoomed,
            label="shared",
            color="C1",
            **scatter_kwargs,
        )
    else:
        ax_scatter.scatter(
            error_ratio_zoomed,
            reconstruction_ratio_zoomed,
            label="ft-only",
            **scatter_kwargs,
        )
        ax_scatter.scatter(
            error_ratio_shared_zoomed,
            reconstruction_ratio_shared_zoomed,
            label="Shared",
            **scatter_kwargs,
        )

    # Plot the histograms
    bins = 50
    hist_kwargs = {"bins": bins, "range": zoom, "alpha": 0.5}

    if ft_only_color:
        ax_histx.hist(
            error_ratio_zoomed, label="ft-only", color=ft_only_color, **hist_kwargs
        )
        ax_histx.hist(
            error_ratio_shared_zoomed, label="shared", color="C1", **hist_kwargs
        )
        ax_histy.hist(
            reconstruction_ratio_zoomed,
            orientation="horizontal",
            color=ft_only_color,
            **hist_kwargs,
        )
        ax_histy.hist(
            reconstruction_ratio_shared_zoomed,
            orientation="horizontal",
            color="C1",
            **hist_kwargs,
        )
    else:
        ax_histx.hist(error_ratio_zoomed, label="ft-only", **hist_kwargs)
        ax_histx.hist(error_ratio_shared_zoomed, label="Shared", **hist_kwargs)
        ax_histy.hist(
            reconstruction_ratio_zoomed, orientation="horizontal", **hist_kwargs
        )
        ax_histy.hist(
            reconstruction_ratio_shared_zoomed, orientation="horizontal", **hist_kwargs
        )

    # Add grid to histograms
    ax_histx.grid(True, alpha=0.15)
    ax_histy.grid(True, alpha=0.15)
    ax_scatter.grid(True, alpha=0.15)

    # Turn off tick labels on histograms
    ax_histx.tick_params(labelbottom=False, bottom=False)
    ax_histy.tick_params(labelleft=False, left=False)

    # Add labels
    if variant == "poster":
        ax_scatter.set_ylabel(
            "$\\uparrow$ \n more \n Latent \n Decoupling ",
            labelpad=40,
            rotation=0,
            y=0.2,
        )
        ax_scatter.set_xlabel("more Complete Shrinkage $\\rightarrow$", labelpad=10)
    else:
        ax_scatter.set_xlabel("$\\nu^\\epsilon$")
        ax_scatter.set_ylabel("$\\nu^r$")

    # Add legend
    if variant == "custom_color":
        ax_histx.legend(
            fontsize=16,
            loc="upper right",
            handletextpad=0.2,
            bbox_to_anchor=(1.65, 1.2),
            handlelength=0.7,
            frameon=False,
        )
    else:
        ax_histx.legend(
            fontsize=16, markerscale=4, loc="lower right", bbox_to_anchor=(1.01, -3.2)
        )

    # Save figure
    suffix = (
        "_43" if variant == "custom_color" else "_poster" if variant == "poster" else ""
    )
    plt.savefig(
        plots_dir / f"error_vs_reconstruction_ratio_with_baseline{suffix}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_ratio_histogram(target_df, baseline_df, plots_dir, ratio_type="error"):
    """Plot histogram of beta ratio values for error or reconstruction"""
    if f"beta_ratio_{ratio_type}" not in target_df.columns:
        return

    zoom = None
    neg_filter_col = f"beta_{ratio_type}_base"
    ratio_col = f"beta_ratio_{ratio_type}"

    neg_mask = target_df[neg_filter_col] >= 0
    ratio_values = target_df[ratio_col][neg_mask]

    # Filter out nans
    ratio_filtered = ratio_values[~np.isnan(ratio_values)]

    if baseline_df is not None:
        baseline_neg_mask = baseline_df[neg_filter_col] >= 0
        ratio_values_shared = baseline_df[ratio_col][baseline_neg_mask]
        ratio_shared_filtered = ratio_values_shared[~np.isnan(ratio_values_shared)]

        # Compute combined range for consistent bins
        all_data = np.concatenate([ratio_filtered, ratio_shared_filtered])
    else:
        # Use only target data for range
        all_data = ratio_filtered

    min_val, max_val = np.min(all_data), np.max(all_data) if zoom is None else zoom
    bins = np.linspace(min_val, max_val, 100)

    plt.figure(figsize=(5, 3))
    plt.rcParams["text.usetex"] = False
    plt.hist(ratio_filtered, bins=bins, alpha=0.5, label="ft-only")

    if baseline_df is not None:
        plt.hist(ratio_shared_filtered, bins=bins, alpha=0.5, label="Shared")

    label = "$\\nu^\\epsilon$" if ratio_type == "error" else "$\\nu^r$"
    plt.xlabel(label)
    plt.ylabel("Count")

    plt.rcParams.update({"font.size": 16})
    plt.rcParams.update({"legend.fontsize": 16})

    if baseline_df is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"{ratio_type}_ratio.pdf", bbox_inches="tight")
    plt.close()


def plot_beta_distribution_histograms(target_df, plots_dir):
    """Plot histograms of beta distribution values"""
    for beta_type in ["error", "reconstruction"]:
        base_col = f"beta_{beta_type}_base"
        ft_col = f"beta_{beta_type}_ft"

        if ft_col in target_df.columns and base_col in target_df.columns:
            try:
                plt.figure(figsize=(10, 6))
                plt.rcParams["text.usetex"] = False
                plt.rcParams.update({"font.size": 16})

                if beta_type == "reconstruction":
                    zoom = [-100, 100]
                    # Apply zoom to focus on a specific range
                    ft_zoomed = target_df[ft_col].clip(zoom[0], zoom[1])
                    base_zoomed = target_df[base_col].clip(zoom[0], zoom[1])

                    # Plot zoomed histograms
                    plt.hist(
                        ft_zoomed,
                        bins=50,
                        alpha=0.7,
                        label=f"ft (zoomed to {zoom})",
                        color="blue",
                        density=True,
                    )
                    plt.hist(
                        base_zoomed,
                        bins=50,
                        alpha=0.7,
                        label=f"Base (zoomed to {zoom})",
                        color="red",
                        density=True,
                    )

                    # Add a note about zooming
                    plt.text(
                        0.05,
                        0.95,
                        f"Values clipped to range {zoom}",
                        transform=plt.gca().transAxes,
                        fontsize=12,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                    )
                else:
                    # Plot regular histograms
                    plt.hist(
                        target_df[ft_col],
                        bins=50,
                        alpha=0.7,
                        label=f"Beta {beta_type.capitalize()} ft",
                        color="blue",
                        density=True,
                    )
                    plt.hist(
                        target_df[base_col],
                        bins=50,
                        alpha=0.7,
                        label=f"Beta {beta_type.capitalize()} Base",
                        color="red",
                        density=True,
                    )
            except Exception as e:
                print(f"Error plotting {beta_type}: {e}")
                continue

            # Add labels and title
            plt.xlabel(f"Beta {beta_type.capitalize()}")
            plt.ylabel("Density")
            plt.title(
                f"Distribution of Beta {beta_type.capitalize()}s for ft and Base Activations"
            )
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                plots_dir / f"beta_{beta_type}_distribution_histogram.pdf",
                bbox_inches="tight",
            )
            plt.close()


def plot_correlation_with_frequency(df, plots_dir):
    """Plot correlation between frequency and beta ratios"""
    if (
        ("freq" in df.columns or "freq_val" in df.columns)
        and "beta_ratio_error" in df.columns
        and "beta_ratio_reconstruction" in df.columns
    ):
        import scipy.stats

        freq = df["freq"] if "freq" in df.columns else df["freq_val"]
        beta_ratio_error = df["beta_ratio_error"]
        beta_ratio_reconstruction = df["beta_ratio_reconstruction"]

        # Remove NaN values
        mask = (
            ~np.isnan(beta_ratio_error)
            & ~np.isnan(beta_ratio_reconstruction)
            & ~np.isnan(freq)
        )
        beta_ratio_error_clean = beta_ratio_error[mask]
        beta_ratio_reconstruction_clean = beta_ratio_reconstruction[mask]
        freq_clean = freq[mask]

        # Compute correlations
        corr_error, p_error = scipy.stats.pearsonr(beta_ratio_error_clean, freq_clean)
        corr_recon, p_recon = scipy.stats.pearsonr(
            beta_ratio_reconstruction_clean, freq_clean
        )

        print(
            f"Correlation between beta_ratio_error and frequency: {corr_error:.3f} (p={p_error:.3e})"
        )
        print(
            f"Correlation between beta_ratio_reconstruction and frequency: {corr_recon:.3f} (p={p_recon:.3e})"
        )

        # Plot scatter for error ratio
        plt.figure(figsize=(8, 4))
        plt.rcParams["text.usetex"] = False
        plt.rcParams.update({"font.size": 16})
        plt.scatter(freq_clean, beta_ratio_error_clean, alpha=0.5)
        plt.xlabel("Frequency")
        plt.ylabel("$\\nu^\\epsilon$ (beta ratio error)")
        plt.text(
            0.05,
            0.95,
            f"Correlation: {corr_error:.3f}\np={p_error:.2e}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
        )
        plt.tight_layout()
        plt.savefig(plots_dir / "freq_vs_beta_ratio_error.pdf", bbox_inches="tight")
        plt.close()

        # Plot scatter for reconstruction ratio
        plt.figure(figsize=(8, 4))
        plt.rcParams["text.usetex"] = False
        plt.rcParams.update({"font.size": 16})
        plt.scatter(freq_clean, beta_ratio_reconstruction_clean, alpha=0.5)
        plt.xlabel("Frequency")
        plt.ylabel("$\\nu^r$ (beta ratio reconstruction)")
        plt.text(
            0.05,
            0.95,
            f"Correlation: {corr_recon:.3f}\np={p_recon:.2e}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
        )
        plt.tight_layout()
        plt.savefig(
            plots_dir / "freq_vs_beta_ratio_reconstruction.pdf", bbox_inches="tight"
        )
        plt.close()


def plot_rank_distributions(target_df, plots_dir):
    """Plot step function of latent rank distributions"""
    for ratio_type in ["error", "reconstruction"]:
        if (
            f"beta_ratio_{ratio_type}" in target_df.columns
            and "dec_norm_diff" in target_df.columns
        ):
            # Get ranks of low nu latents
            low_nu_indices = (
                target_df[f"beta_ratio_{ratio_type}"]
                .sort_values(ascending=True)
                .index[:100]
            )
            all_latent_ranks = target_df["dec_norm_diff"].rank()
            low_nu_ranks = all_latent_ranks[low_nu_indices].sort_values()

            # Calculate fractions
            total_low_nu_latents = len(low_nu_indices)
            fractions = np.arange(1, len(low_nu_ranks) + 1) / total_low_nu_latents

            # Create figure
            plt.figure(figsize=(8, 5))
            plt.rcParams["text.usetex"] = False
            plt.rcParams.update({"font.size": 16})

            # Plot step function
            ratio_str = "$\\nu^\\epsilon$" if ratio_type == "error" else "$\\nu^r$"
            plt.step(
                low_nu_ranks,
                fractions,
                where="post",
                label=f"Fraction of low {ratio_str} latents",
            )

            # Update layout
            plt.xlabel("Rank in ft-only latent set")
            plt.ylabel(f"Fraction of 100 lowest {ratio_str} latents")
            plt.legend(fontsize=20 if ratio_type == "error" else None)
            plt.tight_layout()
            plt.savefig(
                plots_dir / f"low_nu_{ratio_type}_latents_rank_distribution.pdf",
                bbox_inches="tight",
            )
            plt.close()
