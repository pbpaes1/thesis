import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DAYS_TO_TAX_COL = "days_until_tax_transition"
RAW_UNREALIZED_GAIN_COL = "unrealized_gains_pct"
NORM_DAYS_TO_TAX_COL = "days_to_tax_transition_norm"
NORM_UNREALIZED_GAIN_COL = "unrealized_gain_pct_norm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DRL episodes from engineered equity panel after a 30% rally "
            "trigger within a 252-trading-day lookback."
        )
    )
    parser.add_argument(
        "--input",
        default="data/features/engineered_universe.parquet",
        help="Path to engineered feature dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/episodes",
        help="Directory where episodic dataset will be saved.",
    )
    parser.add_argument(
        "--output-file",
        default="drl_episodes.parquet",
        help="Output episodic parquet filename.",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=252,
        help="Lookback window (trading days) for rolling minimum.",
    )
    parser.add_argument(
        "--rally-threshold",
        type=float,
        default=0.30,
        help="Trigger threshold. 0.30 means +30%% over rolling minimum.",
    )
    parser.add_argument(
        "--cooldown-days",
        type=int,
        default=252,
        help=(
            "Cooldown in trading rows after a trigger for the same ticker "
            "before a new trigger can fire."
        ),
    )
    parser.add_argument(
        "--tax-horizon-days",
        type=int,
        default=365,
        help="Episode end horizon in calendar days after simulated purchase date.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    df = pd.read_parquet(path)
    required = {"date", "ticker", "adj_close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input dataset missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Input contains invalid dates in 'date'.")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def add_trigger_columns(
    df: pd.DataFrame,
    lookback_window: int,
    rally_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("ticker", sort=False)

    out["ticker_row"] = g.cumcount()

    # Rolling purchase price candidate: minimum adjusted close in last 252 trading days.
    out["simulated_purchase_price"] = g["adj_close"].transform(
        lambda s: s.rolling(lookback_window, min_periods=lookback_window).min()
    )

    # Position of rolling minimum within each lookback window.
    rolling_argmin = (
        g["adj_close"]
        .rolling(lookback_window, min_periods=lookback_window)
        .apply(np.argmin, raw=True)
        .reset_index(level=0, drop=True)
    )

    out["simulated_purchase_pos"] = (
        out["ticker_row"] - (lookback_window - 1) + rolling_argmin
    ).astype("Int64")

    # Map (ticker, simulated_purchase_pos) -> simulated_purchase_date.
    pos_to_date = out[["ticker", "ticker_row", "date"]].rename(
        columns={"ticker_row": "simulated_purchase_pos", "date": "simulated_purchase_date"}
    )
    out = out.merge(
        pos_to_date,
        on=["ticker", "simulated_purchase_pos"],
        how="left",
    )

    out["trigger_candidate"] = (
        out["simulated_purchase_price"].notna()
        & (out["adj_close"] >= (1.0 + rally_threshold) * out["simulated_purchase_price"])
    )
    return out


def _select_trigger_positions(candidate_mask: np.ndarray, cooldown_days: int) -> np.ndarray:
    """Greedy trigger selection with cooldown, operating on numpy arrays only."""
    selected = np.zeros(candidate_mask.shape[0], dtype=bool)
    candidate_pos = np.flatnonzero(candidate_mask)
    if candidate_pos.size == 0:
        return selected

    i = 0
    while i < candidate_pos.size:
        p = candidate_pos[i]
        selected[p] = True
        # Next valid candidate must be strictly after cooldown window.
        i = np.searchsorted(candidate_pos, p + cooldown_days + 1, side="left")

    return selected


def apply_cooldown(df: pd.DataFrame, cooldown_days: int) -> pd.DataFrame:
    out = df.copy()
    out["valid_trigger"] = False

    for ticker, idx in out.groupby("ticker", sort=False).groups.items():
        mask = out.loc[idx, "trigger_candidate"].to_numpy(dtype=bool)
        selected = _select_trigger_positions(mask, cooldown_days=cooldown_days)
        out.loc[idx, "valid_trigger"] = selected

    return out


def build_trigger_table(df: pd.DataFrame, tax_horizon_days: int) -> pd.DataFrame:
    triggers = df.loc[df["valid_trigger"]].copy()
    if triggers.empty:
        return triggers

    triggers = triggers.rename(columns={"date": "trigger_date"})
    triggers["tax_transition_date"] = triggers["simulated_purchase_date"] + pd.to_timedelta(
        tax_horizon_days, unit="D"
    )
    triggers["episode_id"] = (
        triggers["ticker"] + "_" + triggers["simulated_purchase_date"].dt.strftime("%Y-%m-%d")
    )
    keep_cols = [
        "ticker",
        "ticker_row",
        "trigger_date",
        "simulated_purchase_date",
        "simulated_purchase_price",
        "tax_transition_date",
        "episode_id",
    ]
    return triggers[keep_cols].reset_index(drop=True)


def slice_episodes(df: pd.DataFrame, triggers: pd.DataFrame) -> pd.DataFrame:
    if triggers.empty:
        return pd.DataFrame(columns=list(df.columns) + [
            "episode_id",
            "trigger_date",
            "tax_transition_date",
            "holding_period_days",
            "days_until_tax_transition",
            "unrealized_gains_pct",
        ])

    episodes_parts: list[pd.DataFrame] = []

    for ticker, g in df.groupby("ticker", sort=False):
        trig = triggers.loc[triggers["ticker"] == ticker]
        if trig.empty:
            continue

        g = g.reset_index(drop=True)
        dates = g["date"].to_numpy(dtype="datetime64[ns]")

        for row in trig.itertuples(index=False):
            start_pos = int(row.ticker_row)
            end_date = np.datetime64(row.tax_transition_date.to_datetime64())
            end_pos = int(np.searchsorted(dates, end_date, side="right") - 1)

            if end_pos < start_pos:
                continue

            ep = g.iloc[start_pos : end_pos + 1].copy()
            ep["episode_id"] = row.episode_id
            ep["trigger_date"] = row.trigger_date
            ep["simulated_purchase_date"] = row.simulated_purchase_date
            ep["simulated_purchase_price"] = float(row.simulated_purchase_price)
            ep["tax_transition_date"] = row.tax_transition_date

            ep["holding_period_days"] = (
                ep["date"] - ep["simulated_purchase_date"]
            ).dt.days
            ep["days_until_tax_transition"] = (
                ep["tax_transition_date"] - ep["date"]
            ).dt.days
            ep["unrealized_gains_pct"] = (
                ep["adj_close"] / ep["simulated_purchase_price"]
            ) - 1.0

            episodes_parts.append(ep)

    if not episodes_parts:
        return pd.DataFrame(columns=list(df.columns) + [
            "episode_id",
            "trigger_date",
            "tax_transition_date",
            "holding_period_days",
            "days_until_tax_transition",
            "unrealized_gains_pct",
        ])

    episodes = pd.concat(episodes_parts, ignore_index=True)
    return episodes


def drop_rows_with_nan_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Keep only fully observed numeric state features for clean DRL inputs.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cleaned = df.dropna(subset=numeric_cols).reset_index(drop=True)
    return cleaned


def add_normalized_tax_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Keep raw tax columns unchanged and add normalized versions.
    out[NORM_DAYS_TO_TAX_COL] = np.minimum(out[RAW_DAYS_TO_TAX_COL], 365) / 365.0
    out[NORM_UNREALIZED_GAIN_COL] = np.tanh(out[RAW_UNREALIZED_GAIN_COL] / 0.25)
    return out


def validate_tax_normalization(df: pd.DataFrame, tol: float = 1e-9) -> dict[str, str]:
    checks: dict[str, str] = {}

    raw_days_exists = RAW_DAYS_TO_TAX_COL in df.columns
    raw_gain_exists = RAW_UNREALIZED_GAIN_COL in df.columns
    checks["raw_days_until_tax_transition_exists"] = "pass" if raw_days_exists else "fail"
    checks["raw_unrealized_gains_pct_exists"] = "pass" if raw_gain_exists else "fail"

    if NORM_DAYS_TO_TAX_COL in df.columns:
        days_min = float(df[NORM_DAYS_TO_TAX_COL].min())
        days_max = float(df[NORM_DAYS_TO_TAX_COL].max())
        days_ok = days_min >= 0.0 - tol and days_max <= 1.0 + tol
        checks["days_to_tax_transition_norm_in_[0,1]"] = (
            f"{'pass' if days_ok else 'fail'} (min={days_min:.6g}, max={days_max:.6g})"
        )
    else:
        checks["days_to_tax_transition_norm_in_[0,1]"] = "fail (column missing)"

    if NORM_UNREALIZED_GAIN_COL in df.columns:
        gain_min = float(df[NORM_UNREALIZED_GAIN_COL].min())
        gain_max = float(df[NORM_UNREALIZED_GAIN_COL].max())
        gain_ok = gain_min >= -1.0 - tol and gain_max <= 1.0 + tol
        checks["unrealized_gain_pct_norm_in_[-1,1]"] = (
            f"{'pass' if gain_ok else 'fail'} (min={gain_min:.6g}, max={gain_max:.6g})"
        )
    else:
        checks["unrealized_gain_pct_norm_in_[-1,1]"] = "fail (column missing)"

    return checks


def print_summary(df: pd.DataFrame, checks: dict[str, str]) -> None:
    if df.empty:
        print("No episodes generated.")
    else:
        episode_lengths = df.groupby("episode_id").size()
        print(f"Total unique episodes: {episode_lengths.shape[0]:,}")
        print(f"Average episode length (rows): {episode_lengths.mean():.2f}")
    print("")
    print("v1 normalization summary")
    print("- Market variables: existing normalization remains unchanged (rolling/z-score normalization is handled upstream in the current feature pipeline).")
    print("- Tax variables: raw columns are preserved and normalized columns are added with fixed transformations:")
    print("  - `days_to_tax_transition_norm = min(days_to_tax_transition, 365) / 365`")
    print("  - `unrealized_gain_pct_norm = tanh(unrealized_gain_pct / 0.25)`")
    print("- Normalized tax columns were added for modeling use, while raw tax columns were retained for control and later column-selection decisions.")
    print("")
    print("Tax normalization checks")
    for check_name, status in checks.items():
        print(f"- {check_name}: {status}")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.output_file

    df = load_data(input_path)
    df = add_trigger_columns(
        df,
        lookback_window=args.lookback_window,
        rally_threshold=args.rally_threshold,
    )
    df = apply_cooldown(df, cooldown_days=args.cooldown_days)

    triggers = build_trigger_table(df, tax_horizon_days=args.tax_horizon_days)
    episodes = slice_episodes(df, triggers)
    episodes = drop_rows_with_nan_features(episodes)
    episodes = add_normalized_tax_columns(episodes)
    checks = validate_tax_normalization(episodes)

    episodes.to_parquet(output_path, index=False)

    print(f"Saved episodes: {output_path}")
    print_summary(episodes, checks)


if __name__ == "__main__":
    main()
