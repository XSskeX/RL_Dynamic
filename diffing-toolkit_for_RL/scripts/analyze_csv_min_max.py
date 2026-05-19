from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def analyze_csv_min_max(
    csv_path: Path,
    axis: str = "rows",
    include_empty: bool = False,
) -> pd.DataFrame:
    """Return per-row or per-column min/max statistics for a CSV file."""
    df = pd.read_csv(csv_path)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    if axis == "rows":
        rows: list[dict[str, object]] = []
        for row_idx, row in numeric_df.iterrows():
            valid = row.dropna()
            if valid.empty and not include_empty:
                continue

            rows.append(
                {
                    "row": int(row_idx),
                    "count": int(valid.shape[0]),
                    "missing_or_non_numeric": int(row.isna().sum()),
                    "min": None if valid.empty else float(valid.min()),
                    "max": None if valid.empty else float(valid.max()),
                }
            )
        return pd.DataFrame(rows)

    rows = []
    for column in numeric_df.columns:
        valid = numeric_df[column].dropna()
        if valid.empty and not include_empty:
            continue

        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "count": int(valid.shape[0]),
                "missing_or_non_numeric": int(numeric_df[column].isna().sum()),
                "min": None if valid.empty else float(valid.min()),
                "max": None if valid.empty else float(valid.max()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze each CSV row or column and report minimum and maximum values."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file to analyze.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to save the min/max summary as CSV.",
    )
    parser.add_argument(
        "--axis",
        choices=("rows", "columns"),
        default="rows",
        help="Analyze rows by default. Use 'columns' for per-column min/max.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include rows/columns with no numeric values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(args.csv_path)

    summary = analyze_csv_min_max(
        args.csv_path,
        axis=args.axis,
        include_empty=args.include_empty,
    )
    print(summary.to_string(index=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False, float_format="%.12f")
        print(f"\nSaved summary to: {args.output}")


if __name__ == "__main__":
    main()
