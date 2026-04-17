import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import pandas_ta as ta
except ImportError as exc:
    raise ImportError(
        "pandas_ta is required. Install it with: pip install pandas-ta"
    ) from exc


ROLLING_WINDOWS = (5, 10, 20, 50)
ZSCORE_WINDOW = 252
PCA_WINDOW = 252
PCA_COMPONENTS = 10
MACRO_NUMERIC_COLUMNS = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
SUFFIX_MULTIPLIERS = {"K": 1e3, "M": 1e6, "B": 1e9}

TECHNICAL_INDICATOR_COLUMNS = [
    *(f"sma_{window}" for window in ROLLING_WINDOWS),
    *(f"ema_{window}" for window in ROLLING_WINDOWS),
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_lower",
    "bb_middle",
    "bb_upper",
]
STOCK_ZSCORE_COLUMNS = ["volume", *TECHNICAL_INDICATOR_COLUMNS]

REQUIRED_MACRO_ASSETS = [
    "Gold",
    "WTI",
    "Semiconductor",
    "EUR_USD",
    "USD_CNY",
    "USD_JPN",
    "Wheat",
    "SPY",
    "VIX",
]
VIX_ASSET = "VIX"
MACRO_ASSET_ALIASES = {
    "SPY": ("SPY", "SP500"),
}


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
    parser.add_argument(
        "--zscore-window",
        type=int,
        default=ZSCORE_WINDOW,
        help="Rolling window used for z-score standardization (default: 252).",
    )
    parser.add_argument(
        "--pca-window",
        type=int,
        default=PCA_WINDOW,
        help="Historical lookback window used for rolling PCA fit (default: 252).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=PCA_COMPONENTS,
        help="Number of rolling PCA factors to export (default: 10).",
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


def calculate_log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))


def rolling_z_score(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std


def pca_component_columns(n_components: int = PCA_COMPONENTS) -> list[str]:
    return [f"PC{i}" for i in range(1, n_components + 1)]


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


def zscore_stock_features_by_ticker(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    missing = [col for col in STOCK_ZSCORE_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Missing stock columns required for z-scoring: {sorted(missing)}")

    for col in STOCK_ZSCORE_COLUMNS:
        out[col] = out.groupby("ticker", sort=False)[col].transform(
            lambda s: rolling_z_score(s.astype(float), window=window)
        )

    return out


def build_universe_log_return_matrix(equities: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "adj_close"}
    missing = required.difference(equities.columns)
    if missing:
        raise ValueError(
            "Cannot build return matrix; equities data is missing columns: "
            f"{sorted(missing)}"
        )

    prices = (
        equities[["date", "ticker", "adj_close"]]
        .pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
    )

    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
    return log_returns


def calculate_rolling_pca(
    df: pd.DataFrame,
    window: int = PCA_WINDOW,
    n_components: int = PCA_COMPONENTS,
) -> pd.DataFrame:
    """
    Compute rolling PCA factors using strict historical fitting:
      - fit window: [t-window, t-1]
      - transform: day t only

    Input `df` is expected to be a date-indexed wide matrix of log-returns
    with columns as tickers.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if n_components < 1:
        raise ValueError("n_components must be at least 1")
    if df.empty:
        return pd.DataFrame(columns=pca_component_columns(n_components))

    matrix = df.copy()
    matrix.index = pd.to_datetime(matrix.index)
    matrix = matrix.sort_index()

    if matrix.index.has_duplicates:
        raise ValueError("Rolling PCA input has duplicate dates in index.")

    pca_cols = pca_component_columns(n_components)
    records: list[dict[str, object]] = []
    dates = matrix.index.to_list()

    for idx in range(window, len(matrix)):
        current_date = dates[idx]
        history = matrix.iloc[idx - window : idx]
        today = matrix.iloc[idx]

        # Complete-data-only tickers for this date (history + current day).
        valid_cols = history.columns[history.notna().all(axis=0) & today.notna()]

        row: dict[str, object] = {"date": current_date}
        row.update({col: np.nan for col in pca_cols})

        if len(valid_cols) >= n_components:
            history_values = history[valid_cols].to_numpy(dtype=float, copy=False)
            today_values = today[valid_cols].to_numpy(dtype=float, copy=False).reshape(1, -1)

            scaler = StandardScaler()
            scaled_history = scaler.fit_transform(history_values)
            scaled_today = scaler.transform(today_values)

            pca = PCA(n_components=n_components)
            pca.fit(scaled_history)
            projected_today = pca.transform(scaled_today).ravel()

            for i, col in enumerate(pca_cols):
                row[col] = float(projected_today[i])

        records.append(row)

    if not records:
        return pd.DataFrame(columns=pca_cols)

    out = pd.DataFrame(records)
    out["date"] = pd.to_datetime(out["date"])
    out = out.set_index("date").sort_index()
    return out[pca_cols]


def build_rolling_pca_factors(
    equities: pd.DataFrame,
    window: int = PCA_WINDOW,
    n_components: int = PCA_COMPONENTS,
) -> pd.DataFrame:
    return_matrix = build_universe_log_return_matrix(equities)
    pca_factors = calculate_rolling_pca(
        return_matrix,
        window=window,
        n_components=n_components,
    )
    return pca_factors.reset_index()


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


def _macro_col_for_asset(asset: str) -> str:
    return f"{asset}_Close"


def _resolve_macro_source_column(df: pd.DataFrame, asset: str) -> str | None:
    candidates = MACRO_ASSET_ALIASES.get(asset, (asset,))
    for candidate in candidates:
        col = _macro_col_for_asset(candidate)
        if col in df.columns:
            return col
    return None


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


def transform_macro_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    transformed = pd.DataFrame({"date": df["date"]})
    missing_assets = []

    for asset in REQUIRED_MACRO_ASSETS:
        source_col = _resolve_macro_source_column(df, asset)
        if source_col is None:
            missing_assets.append(asset)
            continue

        source = df[source_col].astype(float)
        if asset == VIX_ASSET:
            transformed[_macro_col_for_asset(asset)] = rolling_z_score(source, window=window)
        else:
            log_returns = calculate_log_returns(source)
            transformed[_macro_col_for_asset(asset)] = rolling_z_score(
                log_returns,
                window=window,
            )

    if missing_assets:
        available_macro_cols = sorted(c for c in df.columns if c.endswith("_Close"))
        raise ValueError(
            "Missing required macro assets for transformation: "
            f"{sorted(missing_assets)}. "
            f"Available macro close columns: {available_macro_cols}"
        )

    return transformed


def required_transformed_columns(n_pca_components: int = PCA_COMPONENTS) -> list[str]:
    macro_cols = [_macro_col_for_asset(asset) for asset in REQUIRED_MACRO_ASSETS]
    return [*STOCK_ZSCORE_COLUMNS, *macro_cols, *pca_component_columns(n_pca_components)]


def validate_macro_coverage_for_period(
    macro_df: pd.DataFrame,
    start: str,
    end: str,
) -> None:
    period = trim_final_period(macro_df, start=start, end=end)
    macro_cols = [_macro_col_for_asset(asset) for asset in REQUIRED_MACRO_ASSETS]
    missing = [col for col in macro_cols if col not in period.columns]
    if missing:
        raise ValueError(f"Missing transformed macro columns: {sorted(missing)}")

    non_null_counts = period[macro_cols].notna().sum()
    no_coverage = non_null_counts[non_null_counts == 0]
    if not no_coverage.empty:
        raise ValueError(
            "No transformed macro values are available in the final window for: "
            f"{sorted(no_coverage.index.tolist())}. "
            "Check raw macro history; these columns would force an empty output after NaN drop."
        )


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

    if args.zscore_window < 2:
        raise ValueError("--zscore-window must be at least 2")
    if args.pca_window < 2:
        raise ValueError("--pca-window must be at least 2")
    if args.pca_components < 1:
        raise ValueError("--pca-components must be at least 1")

    aligned_dir = Path(args.aligned_dir)
    universe_path = aligned_dir / args.universe_file

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    equities = load_equities(universe_path)
    pca_factors = build_rolling_pca_factors(
        equities,
        window=args.pca_window,
        n_components=args.pca_components,
    )

    equities = add_equity_indicators(equities)
    equities = zscore_stock_features_by_ticker(equities, window=args.zscore_window)
    equities = trim_final_period(equities, args.final_start, args.final_end)

    macro_panel = build_macro_panel(aligned_dir, args.macro_glob)
    macro_panel = transform_macro_features(macro_panel, window=args.zscore_window)
    validate_macro_coverage_for_period(
        macro_panel,
        start=args.final_start,
        end=args.final_end,
    )
    macro_panel = trim_final_period(macro_panel, args.final_start, args.final_end)
    pca_factors = trim_final_period(pca_factors, args.final_start, args.final_end)

    final_df = equities.merge(macro_panel, on="date", how="left")
    final_df = final_df.merge(pca_factors, on="date", how="left")

    required_cols = required_transformed_columns(n_pca_components=args.pca_components)
    missing_final_cols = [col for col in required_cols if col not in final_df.columns]
    if missing_final_cols:
        raise ValueError(
            "Final dataset is missing transformed columns: "
            f"{sorted(missing_final_cols)}"
        )

    before_drop = len(final_df)
    final_df = final_df.dropna(subset=required_cols)
    dropped_rows = before_drop - len(final_df)

    if final_df.empty:
        non_null_counts = final_df[required_cols].notna().sum().to_dict()
        raise ValueError(
            "All rows were dropped after enforcing transformed-feature completeness. "
            "Check transformed feature coverage and rolling-window warm-up settings. "
            f"Non-null counts in final frame: {non_null_counts}"
        )

    final_df = final_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    output_path = output_dir / args.output_file
    final_df.to_parquet(output_path, index=False)

    pca_cols = pca_component_columns(args.pca_components)
    pca_valid = pca_factors.dropna(subset=pca_cols)

    print(f"Saved engineered dataset: {output_path}")
    print(f"Rows dropped due to rolling-window and PCA NaNs: {dropped_rows:,}")
    if not pca_valid.empty:
        print(f"First valid PCA date: {pca_valid['date'].min().date()}")
        print(f"PCA dates with full components: {len(pca_valid):,}")
    if not final_df.empty:
        print(
            "Final date range after drop: "
            f"{final_df['date'].min().date()} -> {final_df['date'].max().date()}"
        )
    print_summary(final_df)


if __name__ == "__main__":
    main()