import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No episodes generated.")
        return

    episode_lengths = df.groupby("episode_id").size()
    print(f"Total unique episodes: {episode_lengths.shape[0]:,}")
    print(f"Average episode length (rows): {episode_lengths.mean():.2f}")


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

    episodes.to_parquet(output_path, index=False)

    print(f"Saved episodes: {output_path}")
    print_summary(episodes)


if __name__ == "__main__":
    main()