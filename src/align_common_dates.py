import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MACRO_NUMERIC_COLUMNS = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
SUFFIX_MULTIPLIERS = {"K": 1e3, "M": 1e6, "B": 1e9}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align all CSV files to their common date intersection and "
            "filter a universe parquet file to the same dates."
        )
    )
    parser.add_argument(
        "--csv-dir",
        default="data/raw",
        help="Directory containing the CSV files to align (default: data/raw)",
    )
    parser.add_argument(
        "--csv-glob",
        default="*.csv",
        help="Glob pattern for CSV files inside --csv-dir (default: *.csv)",
    )
    parser.add_argument(
        "--csv-date-column",
        default="Date",
        help='Date column name in CSV files (default: "Date")',
    )
    parser.add_argument(
        "--csv-date-format",
        default="%m/%d/%Y",
        help='Date format for CSV parsing (default: "%%m/%%d/%%Y")',
    )
    parser.add_argument(
        "--universe-input",
        default="data/quality/universe_filtered_2009h2_2026.parquet",
        help=(
            "Path to the filtered universe parquet to align "
            "(default: data/quality/universe_filtered_2009h2_2026.parquet)"
        ),
    )
    parser.add_argument(
        "--universe-start",
        default="2008-01-01",
        help=(
            "Warm-up start date for universe output. Universe rows are kept from this "
            "date until the common CSV end date (default: 2008-01-01)."
        ),
    )
    parser.add_argument(
        "--universe-date-column",
        default="date",
        help='Date column name in universe parquet (default: "date")',
    )
    parser.add_argument(
        "--output-dir",
        default="data/aligned_common_dates",
        help="Output directory where aligned CSVs + aligned parquet are saved",
    )
    parser.add_argument(
        "--sort-order",
        choices=["asc", "desc"],
        default="desc",
        help="Sort order for output files by date (default: desc)",
    )
    return parser.parse_args()


def _load_csv_with_dates(path: Path, date_col: str, date_fmt: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"CSV missing date column '{date_col}': {path}")

    parsed = pd.to_datetime(df[date_col], format=date_fmt, errors="coerce")
    if parsed.isna().any():
        bad_examples = df.loc[parsed.isna(), date_col].head(5).tolist()
        raise ValueError(
            f"Could not parse some dates in {path}. "
            f"Examples: {bad_examples}"
        )

    df = df.copy()
    df["_parsed_date"] = parsed.dt.normalize()
    df = _clean_macro_numeric_columns(df)
    return df


def _date_set(df: pd.DataFrame) -> set[pd.Timestamp]:
    return set(pd.DatetimeIndex(df["_parsed_date"]).unique())


def _parse_numeric_value(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if text in {"", "-", "--", "nan", "NaN", "None"}:
        return np.nan

    if text.endswith("%"):
        text = text[:-1]

    multiplier = 1.0
    suffix = text[-1].upper() if text else ""
    if suffix in SUFFIX_MULTIPLIERS:
        multiplier = SUFFIX_MULTIPLIERS[suffix]
        text = text[:-1]

    text = text.replace(",", "").replace("+", "").strip()
    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def _clean_macro_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MACRO_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = out[col].apply(_parse_numeric_value)
    return out


def _missing_runs(mask: pd.Series) -> list[tuple[int, int, int]]:
    """Return (start_pos, end_pos, run_length) for consecutive True segments."""
    values = mask.to_numpy(dtype=bool)
    runs: list[tuple[int, int, int]] = []
    start = None

    for i, is_missing in enumerate(values):
        if is_missing and start is None:
            start = i
        elif not is_missing and start is not None:
            runs.append((start, i, i - start))
            start = None

    if start is not None:
        runs.append((start, len(values), len(values) - start))

    return runs


def _hybrid_fill_series(s: pd.Series, short_gap_limit: int = 5) -> pd.Series:
    """
    Fill missing values with:
      - forward fill for gaps shorter than short_gap_limit
      - linear interpolation for gaps of short_gap_limit or longer
    """
    out = s.copy()
    missing_mask = out.isna()
    if not missing_mask.any():
        return out

    ffill_vals = out.ffill()
    linear_vals = out.interpolate(method="linear", limit_direction="both")

    for start, end, run_len in _missing_runs(missing_mask):
        if run_len < short_gap_limit:
            out.iloc[start:end] = ffill_vals.iloc[start:end]
        else:
            out.iloc[start:end] = linear_vals.iloc[start:end]

    return out


def _align_and_fill_to_all_dates(
    df: pd.DataFrame,
    date_col: str,
    all_dates: pd.DatetimeIndex,
    sort_order: str,
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("_parsed_date")

    # If a source has duplicate rows on a date, keep the latest one.
    out = out.drop_duplicates(subset=["_parsed_date"], keep="last")
    out = out.set_index("_parsed_date").reindex(all_dates)

    for col in MACRO_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = _hybrid_fill_series(out[col], short_gap_limit=5)

    out = out.reset_index().rename(columns={"index": "_parsed_date"})

    ascending = sort_order == "asc"
    out = out.sort_values("_parsed_date", ascending=ascending)

    out[date_col] = pd.to_datetime(out["_parsed_date"]).dt.strftime("%m/%d/%Y")
    out = out.drop(columns=["_parsed_date"])

    original_cols = [c for c in df.columns if c != "_parsed_date"]
    present_cols = [c for c in original_cols if c in out.columns]
    return out[present_cols]


def main() -> None:
    args = parse_args()

    csv_dir = Path(args.csv_dir)
    universe_input = Path(args.universe_input)
    output_dir = Path(args.output_dir)

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    if not universe_input.exists():
        raise FileNotFoundError(f"Universe parquet not found: {universe_input}")

    csv_paths = sorted(csv_dir.glob(args.csv_glob))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {csv_dir} matching '{args.csv_glob}'"
        )

    csv_frames: dict[Path, pd.DataFrame] = {}
    date_sets: list[set[pd.Timestamp]] = []
    for path in csv_paths:
        frame = _load_csv_with_dates(path, args.csv_date_column, args.csv_date_format)
        csv_frames[path] = frame
        date_sets.append(_date_set(frame))

    all_dates = set.union(*date_sets)
    if not all_dates:
        raise RuntimeError("No dates found across the provided CSV files.")

    all_dates_index = pd.DatetimeIndex(sorted(all_dates))
    all_start = all_dates_index.min().date()
    all_end = all_dates_index.max().date()

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for path, frame in csv_frames.items():
        aligned = _align_and_fill_to_all_dates(
            frame,
            args.csv_date_column,
            all_dates_index,
            args.sort_order,
        )

        out_path = output_dir / f"{path.stem}_aligned.csv"
        aligned.to_csv(out_path, index=False)

        summary_rows.append(
            {
                "dataset": path.name,
                "type": "csv",
                "rows_before": len(frame),
                "rows_after": len(aligned),
                "first_date_after": aligned[args.csv_date_column].iloc[0]
                if len(aligned)
                else "",
                "last_date_after": aligned[args.csv_date_column].iloc[-1]
                if len(aligned)
                else "",
                "output_file": out_path.name,
            }
        )

    universe = pd.read_parquet(universe_input)
    if args.universe_date_column not in universe.columns:
        raise ValueError(
            f"Universe parquet missing date column '{args.universe_date_column}': "
            f"{universe_input}"
        )

    universe = universe.copy()
    universe["_parsed_date"] = pd.to_datetime(
        universe[args.universe_date_column], errors="coerce"
    ).dt.normalize()

    if universe["_parsed_date"].isna().any():
        raise ValueError(
            f"Universe date column '{args.universe_date_column}' contains invalid dates."
        )

    universe_start = pd.Timestamp(args.universe_start).normalize()
    if universe_start > all_dates_index.max():
        raise ValueError(
            "--universe-start is after the latest available CSV date. "
            "Choose an earlier start date."
        )

    universe_aligned = universe[
        (universe["_parsed_date"] >= universe_start)
        & (universe["_parsed_date"] <= all_dates_index.max())
    ].copy()
    universe_aligned = universe_aligned.drop(columns=["_parsed_date"])
    universe_aligned = universe_aligned.sort_values(
        ["ticker", args.universe_date_column]
    ).reset_index(drop=True)

    universe_out = output_dir / "universe_aligned.parquet"
    universe_aligned.to_parquet(universe_out, index=False)

    summary_rows.append(
        {
            "dataset": universe_input.name,
            "type": "parquet",
            "rows_before": len(universe),
            "rows_after": len(universe_aligned),
            "first_date_after": str(universe_aligned[args.universe_date_column].min().date())
            if len(universe_aligned)
            else "",
            "last_date_after": str(universe_aligned[args.universe_date_column].max().date())
            if len(universe_aligned)
            else "",
            "output_file": universe_out.name,
        }
    )

    summary = pd.DataFrame(summary_rows)
    summary_out = output_dir / "alignment_summary.csv"
    summary.to_csv(summary_out, index=False)

    print("Done.")
    print(f"CSV files aligned: {len(csv_paths)}")
    print(f"All-dates union range: {all_start} -> {all_end}")
    print(f"Saved folder: {output_dir}")
    print(f"Saved universe: {universe_out}")
    print(f"Saved summary: {summary_out}")


if __name__ == "__main__":
    main()