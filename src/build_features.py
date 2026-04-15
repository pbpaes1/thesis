import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError as exc:
    raise ImportError(
        "pandas_ta is required. Install it with: pip install pandas-ta"
    ) from exc


ROLLING_WINDOWS = (5, 10, 20, 50)
MACRO_NUMERIC_COLUMNS = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
SUFFIX_MULTIPLIERS = {"K": 1e3, "M": 1e6, "B": 1e9}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build equity + macro engineered features in one parquet file."
    )
    parser.add_argument(
        "--aligned-dir",
        default="data/aligned_common_dates",
        help="Directory containing universe_aligned.parquet and *_aligned.csv files.",
    )
    parser.add_argument(
        "--universe-file",
        default="universe_aligned.parquet",
        help="Universe parquet filename inside --aligned-dir.",
    )
    parser.add_argument(
        "--macro-glob",
        default="*_aligned.csv",
        help="Glob pattern used to discover auxiliary aligned CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/features",
        help="Directory where engineered output parquet is saved.",
    )
    parser.add_argument(
        "--output-file",
        default="engineered_universe.parquet",
        help="Output parquet filename.",
    )
    parser.add_argument(
        "--final-start",
        default="2010-01-01",
        help="Final export start date inclusive (default: 2010-01-01).",
    )
    parser.add_argument(
        "--final-end",
        default="2025-12-31",
        help="Final export end date inclusive (default: 2025-12-31).",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_equities(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Universe parquet not found: {path}")

    df = pd.read_parquet(path)

    required = {"date", "ticker", "adj_close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Universe parquet missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Universe parquet contains invalid dates in column 'date'.")

    if df["adj_close"].isna().any():
        raise ValueError("Universe parquet contains NaN values in required column 'adj_close'.")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def _add_indicators_for_ticker(g: pd.DataFrame) -> pd.DataFrame:
    out = g.sort_values("date").copy()
    close = out["adj_close"].astype(float)

    for window in ROLLING_WINDOWS:
        out[f"sma_{window}"] = ta.sma(close, length=window)
        out[f"ema_{window}"] = ta.ema(close, length=window)

    out["rsi_14"] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["macd"] = macd.iloc[:, 0]
        out["macd_signal"] = macd.iloc[:, 1]
        out["macd_hist"] = macd.iloc[:, 2]
    else:
        out["macd"] = np.nan
        out["macd_signal"] = np.nan
        out["macd_hist"] = np.nan

    bbands = ta.bbands(close, length=20, std=2)
    if bbands is not None and not bbands.empty:
        out["bb_lower"] = bbands.iloc[:, 0]
        out["bb_middle"] = bbands.iloc[:, 1]
        out["bb_upper"] = bbands.iloc[:, 2]
    else:
        out["bb_lower"] = np.nan
        out["bb_middle"] = np.nan
        out["bb_upper"] = np.nan

    return out


def add_equity_indicators(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("ticker", sort=False):
        parts.append(_add_indicators_for_ticker(g))
    return pd.concat(parts, ignore_index=True)


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


def clean_macro_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MACRO_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = out[col].apply(_parse_numeric_value)
    return out


def load_and_clean_macro_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "Date" not in raw.columns:
        raise ValueError(f"CSV missing 'Date' column: {path}")

    raw = raw.copy()
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y", errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    raw = clean_macro_numeric_columns(raw)
    if "Price" not in raw.columns:
        raise ValueError(f"CSV missing 'Price' column: {path}")

    asset_name = path.stem.replace("_aligned", "")
    close_name = f"{asset_name}_Close"

    out = raw[["Date", "Price"]].rename(columns={"Date": "date", "Price": close_name})

    # If duplicate dates exist, keep the latest row after sort.
    out = out.groupby("date", as_index=False).last()
    return out


def build_macro_panel(aligned_dir: Path, macro_glob: str) -> pd.DataFrame:
    csv_paths = sorted(aligned_dir.glob(macro_glob))
    if not csv_paths:
        raise FileNotFoundError(
            f"No macro CSV files found in {aligned_dir} matching '{macro_glob}'"
        )

    macro_frames = [load_and_clean_macro_csv(path) for path in csv_paths]
    merged = macro_frames[0]
    for frame in macro_frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    macro_cols = [c for c in merged.columns if c != "date"]
    merged[macro_cols] = merged[macro_cols].ffill(limit=5)
    return merged


def print_summary(df: pd.DataFrame) -> None:
    print("Final dataframe summary")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    missing_by_col = df.isna().sum().sort_values(ascending=False)
    total_missing = int(missing_by_col.sum())
    print(f"Total missing cells: {total_missing:,}")
    print("Top columns by missing values:")
    print(missing_by_col.head(15).to_string())


def trim_final_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError("--final-end must be on or after --final-start")

    out = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    return out


def main() -> None:
    args = parse_args()

    aligned_dir = Path(args.aligned_dir)
    universe_path = aligned_dir / args.universe_file

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    equities = load_equities(universe_path)
    equities = add_equity_indicators(equities)

    macro_panel = build_macro_panel(aligned_dir, args.macro_glob)

    final_df = equities.merge(macro_panel, on="date", how="left")
    final_df = trim_final_period(final_df, args.final_start, args.final_end)
    final_df = final_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    output_path = output_dir / args.output_file
    final_df.to_parquet(output_path, index=False)

    print(f"Saved engineered dataset: {output_path}")
    print_summary(final_df)


if __name__ == "__main__":
    main()