from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_FILE_NAMES = ("feature_df_local.csv", "feature_df.csv", "*_latent_data.csv")


def discover_feature_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in FEATURE_FILE_NAMES:
        files.extend(root.rglob(pattern))
    return sorted(set(path.resolve() for path in files if path.is_file()))


def _looks_like_feature_columns(columns: Iterable[object]) -> bool:
    values = [str(col) for col in columns]
    if not values:
        return False
    numeric = 0
    for value in values:
        try:
            int(value)
            numeric += 1
        except ValueError:
            pass
    return numeric / len(values) > 0.8


def load_feature_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    unnamed_columns = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if len(unnamed_columns) == 1:
        first = unnamed_columns[0]
        df = df.rename(columns={first: "feature"})
        if df["feature"].is_unique:
            df = df.set_index("feature")

    # Some current crosscoder outputs are transposed: metrics as rows,
    # feature ids as columns. Convert to one row per feature.
    if _looks_like_feature_columns(df.columns) and not _looks_like_feature_columns(
        df.index
    ):
        df = df.T

    df.index.name = "feature"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def summarize(df: pd.DataFrame, source: Path) -> dict[str, object]:
    nums = numeric_columns(df)
    summary: dict[str, object] = {
        "source": str(source),
        "num_features": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": list(map(str, df.columns)),
        "numeric_columns": nums,
    }
    if "tag" in df.columns:
        summary["tag_counts"] = df["tag"].fillna("missing").value_counts().to_dict()

    column_stats = {}
    for col in nums:
        series = df[col].dropna()
        if series.empty:
            continue
        column_stats[col] = {
            "min": float(series.min()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "median": float(series.median()),
            "max": float(series.max()),
        }
    summary["column_stats"] = column_stats
    return summary


def top_features(df: pd.DataFrame, column: str, n: int, ascending: bool) -> pd.DataFrame:
    if column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise ValueError(f"Column '{column}' not found. Available columns: {available}")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")
    keep_cols = [column]
    for extra in ("tag", "dec_base_norm", "dec_ft_norm", "dec_norm_diff"):
        if extra in df.columns and extra not in keep_cols:
            keep_cols.append(extra)
    ranked = df.sort_values(column, ascending=ascending).head(n)
    return ranked[keep_cols]


def plot_histograms(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for col in numeric_columns(df):
        series = df[col].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(series.to_numpy(dtype=float), bins=60)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        fig.tight_layout()
        path = out_dir / f"{col}_hist.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_outputs(
    df: pd.DataFrame,
    source: Path,
    out_dir: Path,
    top_column: str | None,
    top_n: int,
    ascending: bool,
    make_plots: bool,
) -> None:
    source_out = out_dir / source.stem
    source_out.mkdir(parents=True, exist_ok=True)

    normalized_csv = source_out / "features_normalized.csv"
    df.to_csv(normalized_csv)

    summary = summarize(df, source)
    summary_path = source_out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSource: {source}")
    print(f"Features: {len(df)}")
    print(f"Columns: {', '.join(map(str, df.columns))}")
    print(f"Wrote: {normalized_csv}")
    print(f"Wrote: {summary_path}")

    if top_column is not None:
        ranked = top_features(df, top_column, top_n, ascending=ascending)
        top_path = source_out / f"top_{top_n}_by_{top_column}.csv"
        ranked.to_csv(top_path)
        print(f"Wrote: {top_path}")
        print(ranked.head(min(top_n, 10)).to_string())

    if make_plots:
        plot_dir = source_out / "plots"
        paths = plot_histograms(df, plot_dir)
        print(f"Wrote {len(paths)} plot(s) under: {plot_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze stored feature_df CSV files from dictionary/crosscoder runs."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Feature CSV file(s), or directory/directories to scan recursively.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("feature_analysis"),
        help="Directory where normalized CSVs, summaries, and plots are written.",
    )
    parser.add_argument(
        "--top-column",
        default=None,
        help="Numeric column used to export top features, e.g. dec_norm_diff.",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort top features from small to large instead of large to small.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save histograms for all numeric columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            feature_files.extend(discover_feature_files(path))
        elif path.is_file():
            feature_files.append(path.resolve())
        else:
            raise FileNotFoundError(path)

    feature_files = sorted(set(feature_files))
    if not feature_files:
        raise FileNotFoundError("No feature CSV files found.")

    for path in feature_files:
        df = load_feature_df(path)
        write_outputs(
            df=df,
            source=path,
            out_dir=args.out_dir,
            top_column=args.top_column,
            top_n=args.top_n,
            ascending=args.ascending,
            make_plots=args.plots,
        )


if __name__ == "__main__":
    main()
