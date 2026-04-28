from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_COL = "dec_base_self_dot_ratio_norm"
FT_COL = "dec_ft_self_dot_ratio_norm"


def _looks_like_feature_columns(columns) -> bool:
    values = [str(col) for col in columns]
    if not values:
        return False
    n_numeric = 0
    for value in values:
        try:
            int(value)
            n_numeric += 1
        except ValueError:
            pass
    return n_numeric / len(values) > 0.8


def load_feature_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if len(unnamed) == 1:
        df = df.rename(columns={unnamed[0]: "feature"})
        if df["feature"].is_unique:
            df = df.set_index("feature")
    if _looks_like_feature_columns(df.columns) and not _looks_like_feature_columns(
        df.index
    ):
        df = df.T
    df.index.name = "feature"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def infer_step_label(path: Path) -> str:
    text = str(path)
    match = re.search(r"global_step_(\d+)_to_global_step_(\d+)", text)
    if match:
        return f"{match.group(1)}->{match.group(2)}"
    match = re.search(r"global_step_(\d+)", text)
    if match:
        return match.group(1)
    return path.stem


def robust_z(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return 0.6745 * (values - median) / mad


def summarize_run(df: pd.DataFrame, label: str, epsilon: float, q: float) -> dict:
    delta = df["delta_self_dot"]
    abs_delta = df["abs_delta_self_dot"]
    significant_threshold = float(abs_delta.quantile(q))
    changed = abs_delta > epsilon
    significant = abs_delta >= significant_threshold
    return {
        "run": label,
        "num_features": int(len(df)),
        "epsilon": epsilon,
        "unchanged_count": int((~changed).sum()),
        "unchanged_frac": float((~changed).mean()),
        "changed_count": int(changed.sum()),
        "changed_frac": float(changed.mean()),
        "significant_quantile": q,
        "significant_threshold": significant_threshold,
        "significant_count": int(significant.sum()),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std(ddof=0)),
        "delta_median": float(delta.median()),
        "abs_delta_mean": float(abs_delta.mean()),
        "abs_delta_median": float(abs_delta.median()),
        "base_mean": float(df[BASE_COL].mean()),
        "ft_mean": float(df[FT_COL].mean()),
    }


def build_long_table(paths: list[Path], labels: list[str], epsilon: float, q: float):
    frames = []
    summaries = []
    for path, label in zip(paths, labels):
        df = load_feature_df(path)
        missing = [col for col in (BASE_COL, FT_COL) if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        out = df[[BASE_COL, FT_COL]].copy()
        out["feature"] = out.index.astype(str)
        out["run"] = label
        out["source"] = str(path)
        out["delta_self_dot"] = out[FT_COL] - out[BASE_COL]
        out["abs_delta_self_dot"] = out["delta_self_dot"].abs()
        out["robust_z_abs_delta"] = robust_z(out["abs_delta_self_dot"])
        threshold = float(out["abs_delta_self_dot"].quantile(q))
        out["unchanged"] = out["abs_delta_self_dot"] <= epsilon
        out["significant_by_quantile"] = out["abs_delta_self_dot"] >= threshold
        out["significant_by_robust_z"] = out["robust_z_abs_delta"].abs() >= 3.0
        out["significant"] = (
            out["significant_by_quantile"] | out["significant_by_robust_z"]
        )
        frames.append(out)
        summaries.append(summarize_run(out, label, epsilon, q))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def add_aligned_feature_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, group in long_df.groupby("feature", sort=False):
        y = group["delta_self_dot"].to_numpy(dtype=float)
        x = np.arange(len(y), dtype=float)
        slope = float(np.polyfit(x, y, deg=1)[0]) if len(y) >= 2 else 0.0
        rows.append(
            {
                "feature": feature,
                "num_runs": int(len(group)),
                "delta_min": float(np.min(y)),
                "delta_max": float(np.max(y)),
                "delta_range": float(np.max(y) - np.min(y)),
                "delta_std": float(np.std(y)),
                "delta_slope": slope,
                "significant_runs": int(group["significant"].sum()),
                "unchanged_runs": int(group["unchanged"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["delta_range", "significant_runs"], ascending=[False, False]
    )


def make_plots(long_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(summary_df["run"], summary_df["delta_median"], marker="o", label="median")
    ax.plot(summary_df["run"], summary_df["delta_mean"], marker="o", label="mean")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("run")
    ax.set_ylabel("ft - base self-dot ratio")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "delta_mean_median_by_run.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(summary_df["run"], summary_df["unchanged_frac"], marker="o")
    ax.set_xlabel("run")
    ax.set_ylabel("fraction unchanged")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_dir / "unchanged_fraction_by_run.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    data = [
        long_df.loc[long_df["run"] == run, "delta_self_dot"].to_numpy(dtype=float)
        for run in summary_df["run"]
    ]
    ax.boxplot(data, labels=summary_df["run"], showfliers=False)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("run")
    ax.set_ylabel("ft - base self-dot ratio")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_dir / "delta_distribution_by_run.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze RL evolution of base/ft self-dot ratios across feature CSV files."
    )
    parser.add_argument("feature_csvs", nargs="+", type=Path)
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels, same count and order as feature_csvs.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("self_dot_evolution"))
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Treat abs(ft-base) <= epsilon as unchanged.",
    )
    parser.add_argument(
        "--significant-quantile",
        type=float,
        default=0.95,
        help="Within each run, mark top abs-delta features by this quantile.",
    )
    parser.add_argument(
        "--aligned-features",
        action="store_true",
        help="Also compute per-feature trajectories. Use only if feature ids are comparable across runs.",
    )
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [path.resolve() for path in args.feature_csvs]
    labels = args.labels or [infer_step_label(path) for path in paths]
    if len(labels) != len(paths):
        raise ValueError("--labels must have the same count as feature_csvs")
    if not 0 < args.significant_quantile < 1:
        raise ValueError("--significant-quantile must be between 0 and 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_df, summary_df = build_long_table(
        paths, labels, args.epsilon, args.significant_quantile
    )

    long_path = args.out_dir / "self_dot_long.csv"
    summary_path = args.out_dir / "self_dot_summary_by_run.csv"
    sig_path = args.out_dir / "self_dot_significant_features.csv"
    unchanged_path = args.out_dir / "self_dot_unchanged_features.csv"

    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    long_df[long_df["significant"]].to_csv(sig_path, index=False)
    long_df[long_df["unchanged"]].to_csv(unchanged_path, index=False)

    print(f"Wrote: {long_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {sig_path}")
    print(f"Wrote: {unchanged_path}")
    print(summary_df.to_string(index=False))

    if args.aligned_features:
        aligned_df = add_aligned_feature_stats(long_df)
        aligned_path = args.out_dir / "self_dot_aligned_feature_trajectories.csv"
        aligned_df.to_csv(aligned_path, index=False)
        print(f"Wrote: {aligned_path}")

    if args.plots:
        make_plots(long_df, summary_df, args.out_dir)
        print(f"Wrote plots under: {args.out_dir / 'plots'}")


if __name__ == "__main__":
    main()
