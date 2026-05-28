"""Build Step 9 market-cap heterogeneity diagnostics for the frozen final run.

Run from the project root:
    python scripts/quant_analysis/step9_market_cap_heterogeneity.py \
        --run-dir runs/train_reward_c_lite_v5_full \
        --preferred-policy trained_dqn_first_sale_margin_0p070_normal_0p020
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FINAL_RUN_NAME = "train_reward_c_lite_v5_full"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / FINAL_RUN_NAME
DEFAULT_PREFERRED_POLICY = "trained_dqn_first_sale_margin_0p070_normal_0p020"
DEFAULT_EPISODE_MARKET_CAP_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "ticker_market_cap_point_in_time_yfinance.csv"
)
PRIMARY_SPLITS = ["validation", "test"]
POLICIES = [
    DEFAULT_PREFERRED_POLICY,
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "sell_quarters_over_time",
    "random_policy",
]
BENCHMARKS = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "sell_quarters_over_time",
]
STANDARD_BUCKETS = ["micro", "small", "mid", "large", "mega"]
QUINTILE_BUCKETS = ["Q1_smallest", "Q2", "Q3", "Q4", "Q5_largest"]
STANDARD_BUCKET_LABELS = {
    "micro": "micro (<$300m)",
    "small": "small ($300m-$2bn)",
    "mid": "mid ($2bn-$10bn)",
    "large": "large ($10bn-$200bn)",
    "mega": "mega (>=$200bn)",
}
OUTPUTS = {
    "standard_csv": "step9_market_cap_heterogeneity_standard_buckets.csv",
    "standard_md": "step9_market_cap_heterogeneity_standard_buckets.md",
    "quintile_csv": "step9_market_cap_heterogeneity_quintiles.csv",
    "quintile_md": "step9_market_cap_heterogeneity_quintiles.md",
    "notes": "step9_market_cap_heterogeneity_notes.txt",
}


def relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def markdown_table_text(df: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        if value is None or pd.isna(value):
            text = ""
        elif isinstance(value, float):
            text = f"{value:.10g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")

    headers = [str(column) for column in df.columns]
    rows = [[fmt(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    lines = [
        "| "
        + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
        + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(row[index].ljust(widths[index]) for index, header in enumerate(headers))
            + " |"
        )
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table_text(df), encoding="utf-8")


def parse_ticker(episode_id: Any) -> str:
    text = str(episode_id)
    return text.split("_", 1)[0].strip() if "_" in text else text.strip()


def yfinance_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


POINT_IN_TIME_CACHE_COLUMNS = [
    "episode_id",
    "ticker",
    "episode_entry_date",
    "price_date_used",
    "close_price_asof_entry",
    "shares_date_used",
    "shares_outstanding_asof_entry",
    "market_cap_at_entry",
    "market_cap_source",
    "market_cap_missing_reason",
]


def normalize_market_cap_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=POINT_IN_TIME_CACHE_COLUMNS)
    cache = pd.read_csv(path, low_memory=False)
    for column in POINT_IN_TIME_CACHE_COLUMNS:
        if column not in cache.columns:
            cache[column] = np.nan if column not in {"market_cap_source", "market_cap_missing_reason"} else ""
    cache = cache[POINT_IN_TIME_CACHE_COLUMNS].copy()
    cache["episode_id"] = cache["episode_id"].astype(str)
    cache["ticker"] = cache["ticker"].astype(str).str.strip()
    for column in ["episode_entry_date", "price_date_used", "shares_date_used"]:
        cache[column] = pd.to_datetime(cache[column], errors="coerce").dt.tz_localize(None)
    for column in [
        "close_price_asof_entry",
        "shares_outstanding_asof_entry",
        "market_cap_at_entry",
    ]:
        cache[column] = pd.to_numeric(cache[column], errors="coerce")
    cache["market_cap_source"] = cache["market_cap_source"].fillna("").astype(str)
    cache["market_cap_missing_reason"] = cache["market_cap_missing_reason"].fillna("").astype(str)
    return cache.drop_duplicates("episode_id", keep="last")


def _as_naive_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.to_datetime(index, errors="coerce")
    if getattr(dt_index, "tz", None) is not None:
        dt_index = dt_index.tz_convert(None)
    return pd.DatetimeIndex(dt_index).tz_localize(None) if getattr(dt_index, "tz", None) is not None else pd.DatetimeIndex(dt_index)


def _latest_observation_on_or_before(
    series: pd.Series,
    asof_date: pd.Timestamp,
) -> tuple[pd.Timestamp | pd.NaT, float]:
    valid = series.dropna().sort_index()
    if valid.empty or pd.isna(asof_date):
        return pd.NaT, np.nan
    positions = valid.index.searchsorted(asof_date, side="right") - 1
    if positions < 0:
        return pd.NaT, np.nan
    return pd.Timestamp(valid.index[positions]), float(valid.iloc[positions])


def fetch_ticker_point_in_time_market_caps(
    ticker: str,
    episodes: pd.DataFrame,
) -> list[dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError as exc:
        reason = f"yfinance import failed: {exc}"
        return [
            market_cap_missing_row(row, reason)
            for row in episodes.itertuples(index=False)
        ]

    ticker = str(ticker)
    yf_ticker = yf.Ticker(yfinance_symbol(ticker))
    min_entry = pd.to_datetime(episodes["episode_entry_date"]).min()
    max_entry = pd.to_datetime(episodes["episode_entry_date"]).max()
    price_series = pd.Series(dtype=float)
    shares_series = pd.Series(dtype=float)
    price_error = ""
    shares_error = ""

    try:
        price_start = (min_entry - timedelta(days=14)).strftime("%Y-%m-%d")
        price_end = (max_entry + timedelta(days=1)).strftime("%Y-%m-%d")
        history = yf_ticker.history(
            start=price_start,
            end=price_end,
            auto_adjust=False,
            actions=False,
        )
        if isinstance(history, pd.DataFrame) and "Close" in history.columns:
            price_series = pd.Series(
                pd.to_numeric(history["Close"], errors="coerce").to_numpy(),
                index=_as_naive_datetime_index(history.index),
                dtype=float,
            ).dropna()
    except Exception as exc:
        price_error = str(exc)

    try:
        shares_end = (max_entry + timedelta(days=1)).strftime("%Y-%m-%d")
        shares = yf_ticker.get_shares_full(start="1990-01-01", end=shares_end)
        if isinstance(shares, pd.Series):
            shares_series = pd.Series(
                pd.to_numeric(shares, errors="coerce").to_numpy(),
                index=_as_naive_datetime_index(shares.index),
                dtype=float,
            ).dropna()
        elif isinstance(shares, pd.DataFrame) and not shares.empty:
            column = "Shares" if "Shares" in shares.columns else shares.columns[0]
            shares_series = pd.Series(
                pd.to_numeric(shares[column], errors="coerce").to_numpy(),
                index=_as_naive_datetime_index(shares.index),
                dtype=float,
            ).dropna()
    except Exception as exc:
        shares_error = str(exc)

    rows: list[dict[str, Any]] = []
    for row in episodes.sort_values("episode_entry_date").itertuples(index=False):
        entry_date = pd.Timestamp(row.episode_entry_date)
        price_date, close_price = _latest_observation_on_or_before(price_series, entry_date)
        shares_date, shares_outstanding = _latest_observation_on_or_before(shares_series, entry_date)
        missing_reasons: list[str] = []
        if pd.isna(price_date) or not np.isfinite(close_price) or close_price <= 0:
            missing_reasons.append("missing_close_price_on_or_before_entry")
            if price_error:
                missing_reasons.append("price_fetch_error=" + price_error[:180])
        if pd.isna(shares_date) or not np.isfinite(shares_outstanding) or shares_outstanding <= 0:
            missing_reasons.append("missing_shares_outstanding_on_or_before_entry")
            if shares_error:
                missing_reasons.append("shares_fetch_error=" + shares_error[:180])
        market_cap = close_price * shares_outstanding if not missing_reasons else np.nan
        rows.append(
            {
                "episode_id": row.episode_id,
                "ticker": ticker,
                "episode_entry_date": entry_date.date().isoformat(),
                "price_date_used": ""
                if pd.isna(price_date)
                else pd.Timestamp(price_date).date().isoformat(),
                "close_price_asof_entry": close_price,
                "shares_date_used": ""
                if pd.isna(shares_date)
                else pd.Timestamp(shares_date).date().isoformat(),
                "shares_outstanding_asof_entry": shares_outstanding,
                "market_cap_at_entry": market_cap,
                "market_cap_source": "yfinance_point_in_time_proxy",
                "market_cap_missing_reason": ";".join(missing_reasons),
            }
        )
    return rows


def market_cap_missing_row(row: Any, reason: str) -> dict[str, Any]:
    entry_date = pd.Timestamp(row.episode_entry_date)
    return {
        "episode_id": row.episode_id,
        "ticker": row.ticker,
        "episode_entry_date": entry_date.date().isoformat(),
        "price_date_used": "",
        "close_price_asof_entry": np.nan,
        "shares_date_used": "",
        "shares_outstanding_asof_entry": np.nan,
        "market_cap_at_entry": np.nan,
        "market_cap_source": "yfinance_point_in_time_proxy",
        "market_cap_missing_reason": reason,
    }


def ensure_point_in_time_market_cap_cache(
    episode_meta: pd.DataFrame,
    *,
    path: Path,
    max_workers: int,
    force_refresh: bool,
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    episode_meta = episode_meta[["episode_id", "ticker", "episode_entry_date"]].copy()
    episode_meta["episode_id"] = episode_meta["episode_id"].astype(str)
    episode_meta["ticker"] = episode_meta["ticker"].astype(str).str.strip()
    episode_meta["episode_entry_date"] = pd.to_datetime(
        episode_meta["episode_entry_date"], errors="coerce"
    )
    if episode_meta["episode_entry_date"].isna().any():
        raise ValueError("Episode entry dates are required for point-in-time market-cap construction.")
    cache = normalize_market_cap_cache(path)
    required_ids = set(episode_meta["episode_id"])
    if force_refresh:
        cached_ids: set[str] = set()
    else:
        cached_ids = set(cache.loc[cache["episode_id"].isin(required_ids), "episode_id"])
    missing_meta = episode_meta[~episode_meta["episode_id"].isin(cached_ids)].copy()
    if not missing_meta.empty:
        rows: list[dict[str, Any]] = []
        grouped = [(ticker, group.copy()) for ticker, group in missing_meta.groupby("ticker", sort=True)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_ticker_point_in_time_market_caps, ticker, group): ticker
                for ticker, group in grouped
            }
            for future in concurrent.futures.as_completed(futures):
                rows.extend(future.result())
        fetched = pd.DataFrame(rows, columns=POINT_IN_TIME_CACHE_COLUMNS)
        cache = pd.concat(
            [cache[~cache["episode_id"].isin(set(fetched["episode_id"].astype(str)))], fetched],
            ignore_index=True,
        )
        cache = cache.drop_duplicates("episode_id", keep="last").sort_values(["ticker", "episode_entry_date", "episode_id"])
        cache.to_csv(path, index=False)
        cache = normalize_market_cap_cache(path)
    required = cache[cache["episode_id"].isin(required_ids)].copy()
    missing_ids = sorted(required_ids - set(required["episode_id"]))
    if missing_ids:
        raise ValueError(
            "Point-in-time market-cap cache is incomplete for episode_id(s): "
            + ", ".join(missing_ids[:20])
        )
    return required


def load_raw_episode_metadata(run_dir: Path) -> pd.DataFrame:
    config_path = run_dir / "config_used.yaml"
    parquet_path = PROJECT_ROOT / "data" / "episodes" / "drl_episodes.parquet"
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("parquet_path:"):
                parquet_path = resolve_project_path(line.split(":", 1)[1].strip())
                break
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Missing raw episode parquet for entry-date metadata: {relative_project_path(parquet_path)}"
        )
    columns = [
        "episode_id",
        "date",
        "ticker",
        "close",
        "adj_close",
        "simulated_purchase_price",
        "simulated_purchase_date",
        "trigger_date",
    ]
    raw = pd.read_parquet(parquet_path, columns=columns)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for column in ["simulated_purchase_date", "trigger_date"]:
        raw[column] = pd.to_datetime(raw[column], errors="coerce")
    for column in ["close", "adj_close", "simulated_purchase_price"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    summary = (
        raw.sort_values(["episode_id", "date"], kind="mergesort")
        .groupby("episode_id", as_index=False)
        .agg(
            ticker=("ticker", "first"),
            episode_start_date=("date", "min"),
            price_at_entry=("simulated_purchase_price", "first"),
            close_at_entry=("close", "first"),
            adj_close_at_entry=("adj_close", "first"),
            simulated_purchase_date=("simulated_purchase_date", "first"),
            trigger_date=("trigger_date", "first"),
        )
    )
    summary["ticker"] = summary["ticker"].fillna(summary["episode_id"].map(parse_ticker))
    summary["entry_date"] = summary["episode_start_date"]
    summary["episode_entry_date"] = summary["episode_start_date"]
    summary["price_at_entry"] = summary["price_at_entry"].where(
        summary["price_at_entry"].notna(), summary["close_at_entry"]
    )
    return summary


def load_episode_market_cap(
    *,
    run_dir: Path,
    required_episode_ids: set[str],
    episode_market_cap_path: Path,
    max_workers: int,
    force_refresh: bool,
) -> tuple[pd.DataFrame, str, list[str]]:
    notes: list[str] = []
    raw_meta = load_raw_episode_metadata(run_dir)
    raw_meta["episode_id"] = raw_meta["episode_id"].astype(str)
    raw_meta = raw_meta[raw_meta["episode_id"].isin(required_episode_ids)].copy()
    if raw_meta.empty:
        raise ValueError("Market-cap classification cannot be created: no raw metadata matched required validation/test episode_id values.")
    market = ensure_point_in_time_market_cap_cache(
        raw_meta[["episode_id", "ticker", "episode_entry_date"]],
        path=episode_market_cap_path,
        max_workers=max_workers,
        force_refresh=force_refresh,
    )
    merged = raw_meta.merge(
        market[POINT_IN_TIME_CACHE_COLUMNS],
        on=["episode_id", "ticker", "episode_entry_date"],
        how="left",
        validate="one_to_one",
    )
    source = "yfinance_point_in_time_proxy"
    notes.extend(
        [
            "market_cap_source_file: " + relative_project_path(episode_market_cap_path),
            "market_cap_formula: close_price_asof_episode_entry * shares_outstanding_asof_episode_entry",
            "price_source: yfinance Ticker.history(auto_adjust=False), Close",
            "shares_source: yfinance Ticker.get_shares_full",
            "no_future_shares_used: True",
            "no_future_prices_used: True",
            "point_in_time_proxy_caveat: yfinance historical shares may be sparse/revised, so this is a point-in-time proxy rather than an audited fundamentals database.",
            "prior_current_market_cap_analysis_replaced: True",
        ]
    )
    return merged, source, notes


def prepare_episode_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_columns = [
        "episode_final_after_tax_total_value",
        "total_tax_paid",
        "mean_effective_tax_rate_on_sales",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "days_to_first_sale",
        "num_discretionary_sales",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "episode_cut_occurred" in out.columns:
        out["episode_cut_occurred"] = out["episode_cut_occurred"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
            if pd.notna(value)
            else False
        )
    return out


def validate_episode_inputs(df: pd.DataFrame, preferred_policy: str) -> None:
    required_columns = [
        "split",
        "episode_id",
        "policy_name",
        "episode_final_after_tax_total_value",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Step 9 market-cap analysis requires missing episode-level column(s): "
            + ", ".join(missing)
        )
    required_policies = {preferred_policy, *BENCHMARKS}
    if "random_policy" in set(df["policy_name"]):
        required_policies.add("random_policy")
    missing_policies = sorted(required_policies - set(df["policy_name"].unique()))
    if missing_policies:
        raise ValueError(
            "Step 9 market-cap analysis is missing required policy row(s): "
            + ", ".join(missing_policies)
        )
    missing_splits = sorted(set(PRIMARY_SPLITS) - set(df["split"].unique()))
    if missing_splits:
        raise ValueError(
            "Step 9 market-cap analysis is missing split(s): "
            + ", ".join(missing_splits)
        )
    duplicate_count = int(df.duplicated(["split", "policy_name", "episode_id"]).sum())
    if duplicate_count:
        raise ValueError(
            "Episode-level policy rows must be unique by split/policy/episode_id; "
            f"found {duplicate_count} duplicates."
        )


def build_analysis_base(
    episode_df: pd.DataFrame,
    market_meta: pd.DataFrame,
    *,
    preferred_policy: str,
    output_dir: Path,
) -> pd.DataFrame:
    policies = [preferred_policy, *BENCHMARKS]
    if "random_policy" in set(episode_df["policy_name"]):
        policies.append("random_policy")
    metrics = episode_df[
        episode_df["split"].isin(PRIMARY_SPLITS) & episode_df["policy_name"].isin(policies)
    ].copy()
    wide = metrics.pivot(
        index=["split", "episode_id"],
        columns="policy_name",
        values="episode_final_after_tax_total_value",
    ).reset_index()
    rename = {
        preferred_policy: "preferred_final_after_tax_value",
        "hold_to_terminal": "hold_to_terminal_final_after_tax_value",
        "sell_immediately": "sell_immediately_final_after_tax_value",
        "sell_half_then_hold": "sell_half_then_hold_final_after_tax_value",
        "sell_quarters_over_time": "sell_quarters_over_time_final_after_tax_value",
        "random_policy": "random_policy_final_after_tax_value",
    }
    wide = wide.rename(columns=rename)
    required_value_columns = [
        "preferred_final_after_tax_value",
        "hold_to_terminal_final_after_tax_value",
        "sell_immediately_final_after_tax_value",
        "sell_half_then_hold_final_after_tax_value",
        "sell_quarters_over_time_final_after_tax_value",
    ]
    if wide[required_value_columns].isna().any().any():
        raise ValueError(
            "Benchmark comparisons cannot be paired by episode_id for all required policies."
        )
    preferred = metrics[metrics["policy_name"].eq(preferred_policy)].copy()
    preferred = preferred.rename(
        columns={
            "total_tax_paid": "preferred_tax_paid",
            "mean_effective_tax_rate_on_sales": "preferred_effective_tax_rate",
            "pct_episode_position_sold_short_term": "preferred_short_term_sold_fraction",
            "pct_episode_position_sold_long_term": "preferred_long_term_sold_fraction",
            "days_to_first_sale": "preferred_days_to_first_sale",
            "episode_cut_occurred": "preferred_episode_cut_occurred",
        }
    )
    preferred["preferred_no_cut"] = ~preferred["preferred_episode_cut_occurred"].fillna(False)
    keep = [
        "split",
        "episode_id",
        "preferred_tax_paid",
        "preferred_effective_tax_rate",
        "preferred_short_term_sold_fraction",
        "preferred_long_term_sold_fraction",
        "preferred_days_to_first_sale",
        "preferred_episode_cut_occurred",
        "preferred_no_cut",
    ]
    base = wide.merge(preferred[keep], on=["split", "episode_id"], how="inner", validate="one_to_one")
    meta_keep = [
        "episode_id",
        "ticker",
        "entry_date",
        "episode_entry_date",
        "price_at_entry",
        "price_date_used",
        "close_price_asof_entry",
        "shares_date_used",
        "shares_outstanding_asof_entry",
        "market_cap_at_entry",
        "market_cap_source",
        "market_cap_missing_reason",
    ]
    base = base.merge(
        market_meta[meta_keep], on="episode_id", how="left", validate="many_to_one"
    )
    sharpe_path = output_dir / "step3b_episode_eaat_sharpe_metrics.csv"
    if sharpe_path.exists():
        sharpe = pd.read_csv(
            sharpe_path,
            usecols=[
                "split",
                "episode_id",
                "policy_name",
                "EAAT_Sharpe",
                "TA_EAAT_Sharpe",
            ],
            low_memory=False,
        )
        sharpe = sharpe[
            sharpe["split"].isin(PRIMARY_SPLITS)
            & sharpe["policy_name"].eq(preferred_policy)
        ].rename(
            columns={
                "EAAT_Sharpe": "preferred_EAAT_Sharpe",
                "TA_EAAT_Sharpe": "preferred_TA_EAAT_Sharpe",
            }
        )
        base = base.merge(
            sharpe[
                [
                    "split",
                    "episode_id",
                    "preferred_EAAT_Sharpe",
                    "preferred_TA_EAAT_Sharpe",
                ]
            ],
            on=["split", "episode_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        base["preferred_EAAT_Sharpe"] = np.nan
        base["preferred_TA_EAAT_Sharpe"] = np.nan
    base["dqn_minus_hold"] = (
        base["preferred_final_after_tax_value"]
        - base["hold_to_terminal_final_after_tax_value"]
    )
    base["dqn_minus_sell_immediately"] = (
        base["preferred_final_after_tax_value"]
        - base["sell_immediately_final_after_tax_value"]
    )
    base["dqn_minus_sell_half_then_hold"] = (
        base["preferred_final_after_tax_value"]
        - base["sell_half_then_hold_final_after_tax_value"]
    )
    base["dqn_minus_sell_quarters_over_time"] = (
        base["preferred_final_after_tax_value"]
        - base["sell_quarters_over_time_final_after_tax_value"]
    )
    return base


def assign_standard_bucket(market_cap: pd.Series) -> pd.Series:
    values = pd.to_numeric(market_cap, errors="coerce")
    return pd.cut(
        values,
        bins=[-np.inf, 300_000_000, 2_000_000_000, 10_000_000_000, 200_000_000_000, np.inf],
        labels=STANDARD_BUCKETS,
        right=False,
    ).astype(object)


def assign_quintile_bucket(market_cap: pd.Series) -> pd.Series:
    values = pd.to_numeric(market_cap, errors="coerce")
    if values.notna().sum() < 5:
        raise ValueError("Market-cap quintile classification cannot be created with fewer than 5 values.")
    ranks = values.rank(method="first")
    return pd.qcut(ranks, q=5, labels=QUINTILE_BUCKETS).astype(object)


def assign_quintile_bucket_by_split(base: pd.DataFrame) -> pd.Series:
    output = pd.Series(index=base.index, dtype=object)
    for split in PRIMARY_SPLITS:
        split_idx = base.index[base["split"].eq(split)]
        if len(split_idx) < 5:
            raise ValueError(
                f"Market-cap quintile classification cannot be created for {split}; fewer than 5 non-missing values."
            )
        output.loc[split_idx] = assign_quintile_bucket(base.loc[split_idx, "market_cap_at_entry"])
    return output


def validate_market_cap_proxy(base: pd.DataFrame) -> None:
    for column in ["episode_entry_date", "price_date_used", "shares_date_used"]:
        base[column] = pd.to_datetime(base[column], errors="coerce")
    usable = base["market_cap_at_entry"].notna()
    if usable.sum() == 0:
        raise ValueError(
            "Market-cap classification cannot be created: no validation/test episode has "
            "a positive yfinance point-in-time proxy. Required columns are episode_id, ticker, "
            "episode_entry_date, close_price_asof_entry, shares_outstanding_asof_entry, and market_cap_at_entry."
        )
    future_price = usable & base["price_date_used"].gt(base["episode_entry_date"])
    if future_price.any():
        raise ValueError("Point-in-time validation failed: price_date_used is after episode_entry_date.")
    future_shares = usable & base["shares_date_used"].gt(base["episode_entry_date"])
    if future_shares.any():
        raise ValueError("Point-in-time validation failed: shares_date_used is after episode_entry_date.")
    non_positive = usable & base["market_cap_at_entry"].le(0)
    if non_positive.any():
        raise ValueError("Point-in-time validation failed: market_cap_at_entry must be positive when present.")
    bad_source = usable & ~base["market_cap_source"].eq("yfinance_point_in_time_proxy")
    if bad_source.any():
        raise ValueError("Step 9 market-cap outputs must use yfinance_point_in_time_proxy, not current market cap.")


def compute_market_cap_coverage(base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in PRIMARY_SPLITS:
        data = base[base["split"].eq(split)]
        usable = data["market_cap_at_entry"].notna() & data["market_cap_at_entry"].gt(0)
        total = int(len(data))
        rows.append(
            {
                "split": split,
                "total_paired_episodes": total,
                "episodes_with_market_cap_at_entry": int(usable.sum()),
                "coverage_pct": float(usable.mean()) if total else np.nan,
                "missing_market_cap_episode_count": int((~usable).sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_missing_reasons(base: pd.DataFrame) -> list[str]:
    missing = base[
        base["market_cap_at_entry"].isna() | base["market_cap_at_entry"].le(0)
    ].copy()
    if missing.empty:
        return ["missing_market_cap_reasons: none"]
    rows = ["missing_market_cap_reasons:"]
    reason_counts = (
        missing.assign(
            market_cap_missing_reason=missing["market_cap_missing_reason"].replace("", "unknown")
        )
        .groupby(["split", "market_cap_missing_reason"], dropna=False)
        .size()
        .reset_index(name="episode_count")
        .sort_values(["split", "episode_count"], ascending=[True, False])
    )
    for row in reason_counts.itertuples(index=False):
        rows.append(f"{row.split},{row.market_cap_missing_reason},{row.episode_count}")
    return rows


def summarize_bucket(
    base: pd.DataFrame,
    *,
    bucket_column: str,
    bucket_order: list[str],
    bucket_system: str,
    market_cap_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in PRIMARY_SPLITS:
        split_df = base[base["split"].eq(split)]
        for bucket in bucket_order:
            group = split_df[split_df[bucket_column].eq(bucket)]
            rows.append(
                {
                    "split": split,
                    "bucket_system": bucket_system,
                    "market_cap_bucket": bucket,
                    "market_cap_source": market_cap_source,
                    "num_episodes": int(len(group)),
                    "preferred_dqn_mean_final_after_tax_value": group[
                        "preferred_final_after_tax_value"
                    ].mean(),
                    "preferred_dqn_median_final_after_tax_value": group[
                        "preferred_final_after_tax_value"
                    ].median(),
                    "hold_to_terminal_mean_final_after_tax_value": group[
                        "hold_to_terminal_final_after_tax_value"
                    ].mean(),
                    "dqn_minus_hold_to_terminal_mean_difference": group[
                        "dqn_minus_hold"
                    ].mean(),
                    "dqn_minus_hold_to_terminal_median_difference": group[
                        "dqn_minus_hold"
                    ].median(),
                    "dqn_win_rate_vs_hold_to_terminal": group["dqn_minus_hold"].gt(0).mean()
                    if len(group)
                    else np.nan,
                    "dqn_win_rate_vs_sell_immediately": group[
                        "dqn_minus_sell_immediately"
                    ].gt(0).mean()
                    if len(group)
                    else np.nan,
                    "dqn_win_rate_vs_sell_half_then_hold": group[
                        "dqn_minus_sell_half_then_hold"
                    ].gt(0).mean()
                    if len(group)
                    else np.nan,
                    "dqn_win_rate_vs_sell_quarters_over_time": group[
                        "dqn_minus_sell_quarters_over_time"
                    ].gt(0).mean()
                    if len(group)
                    else np.nan,
                    "mean_tax_paid_preferred_dqn": group["preferred_tax_paid"].mean(),
                    "mean_effective_tax_rate_preferred_dqn": group[
                        "preferred_effective_tax_rate"
                    ].mean(),
                    "short_term_sold_fraction_preferred_dqn": group[
                        "preferred_short_term_sold_fraction"
                    ].mean(),
                    "long_term_sold_fraction_preferred_dqn": group[
                        "preferred_long_term_sold_fraction"
                    ].mean(),
                    "no_cut_percentage_preferred_dqn": group["preferred_no_cut"].astype(float).mean()
                    if len(group)
                    else np.nan,
                    "discretionary_sale_percentage_preferred_dqn": group[
                        "preferred_episode_cut_occurred"
                    ].astype(float).mean()
                    if len(group)
                    else np.nan,
                    "average_days_to_first_sale_preferred_dqn": group[
                        "preferred_days_to_first_sale"
                    ].mean(),
                    "median_days_to_first_sale_preferred_dqn": group[
                        "preferred_days_to_first_sale"
                    ].median(),
                    "median_EAAT_Sharpe_preferred_dqn": group[
                        "preferred_EAAT_Sharpe"
                    ].median(),
                    "median_TA_EAAT_Sharpe_preferred_dqn": group[
                        "preferred_TA_EAAT_Sharpe"
                    ].median(),
                    "mean_market_cap_at_entry": group["market_cap_at_entry"].mean(),
                    "median_market_cap_at_entry": group["market_cap_at_entry"].median(),
                    "sparse_bucket_warning": bool(len(group) < 30),
                }
            )
    out = pd.DataFrame(rows)
    numeric = out.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError(f"{bucket_system} market-cap output contains infinite values.")
    return out


def plot_bar(
    df: pd.DataFrame,
    *,
    bucket_order: list[str],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plot_df = df[df["split"].eq("test")].copy()
    plot_df["_order"] = plot_df["market_cap_bucket"].map(
        {bucket: index for index, bucket in enumerate(bucket_order)}
    )
    plot_df = plot_df.sort_values("_order")
    labels = [
        STANDARD_BUCKET_LABELS.get(str(bucket), str(bucket))
        for bucket in plot_df["market_cap_bucket"]
    ]
    colors = np.where(plot_df[metric].fillna(0).ge(0), "#3b6ea8", "#8a4f3d")
    plt.figure(figsize=(8.8, 4.8))
    plt.bar(labels, plot_df[metric], color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_notes(
    path: Path,
    *,
    preferred_policy: str,
    market_cap_source: str,
    market_cap_notes: list[str],
    coverage_df: pd.DataFrame,
    missing_reason_lines: list[str],
    standard_df: pd.DataFrame,
    quintile_df: pd.DataFrame,
) -> None:
    sparse_standard = standard_df[
        standard_df["sparse_bucket_warning"].fillna(False)
    ][["split", "market_cap_bucket", "num_episodes"]]
    lines = [
        "Step 9 market-cap heterogeneity notes",
        "This is a heterogeneity / interpretability analysis, not a policy-selection step.",
        f"The preferred policy remains {preferred_policy}.",
        "No retraining or reward redesign was performed.",
        "Market cap should be measured at episode entry date to avoid look-ahead bias.",
        f"market_cap_source_used: {market_cap_source}",
        *market_cap_notes,
        "This replaces the earlier current-market-cap sensitivity because current market cap can introduce look-ahead bias.",
        "Episodes without a usable close price and historical shares observation on/before entry are excluded from market-cap bucket calculations and reported in coverage.",
        "Standard buckets may be imbalanced because the universe is S&P 500-like.",
        "Quintile buckets are computed separately within each split using only episodes with non-missing market_cap_at_entry.",
        "Final after-tax value remains the primary metric.",
        "EAAT and TA-EAAT Sharpe, if included, are secondary diagnostics.",
        "The test split remains the main thesis evidence; validation is supportive.",
        "Train and all splits are not included in this Step 9 market-cap output.",
        "",
        "coverage:",
    ]
    for row in coverage_df.itertuples(index=False):
        lines.append(
            f"{row.split},{row.episodes_with_market_cap_at_entry}/{row.total_paired_episodes},"
            f"{row.coverage_pct:.6f},missing={row.missing_market_cap_episode_count}"
        )
    low_coverage = coverage_df["coverage_pct"].lt(0.8).any()
    if low_coverage:
        lines.append("coverage_warning: at least one split has market-cap coverage below 80%; interpret bucket results cautiously.")
    lines.extend(
        [
            "",
            *missing_reason_lines,
            "",
            "standard_bucket_counts:",
        ]
    )
    for row in standard_df[["split", "market_cap_bucket", "num_episodes"]].itertuples(index=False):
        lines.append(f"{row.split},{row.market_cap_bucket},{row.num_episodes}")
    lines.append("")
    lines.append("quintile_bucket_counts:")
    for row in quintile_df[["split", "market_cap_bucket", "num_episodes"]].itertuples(index=False):
        lines.append(f"{row.split},{row.market_cap_bucket},{row.num_episodes}")
    lines.append("")
    if sparse_standard.empty:
        lines.append("sparse_standard_buckets: none under num_episodes < 30")
    else:
        lines.append("sparse_standard_buckets:")
        for row in sparse_standard.itertuples(index=False):
            lines.append(f"{row.split},{row.market_cap_bucket},{row.num_episodes}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def update_manifest(output_dir: Path, paths: list[Path]) -> None:
    manifest_path = output_dir / "phase_4_11_outputs_manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a list: {relative_project_path(manifest_path)}")
    existing = {
        row.get("path")
        for row in payload
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    descriptions = {
        "step9_market_cap_heterogeneity_standard_buckets": "Step 9 market-cap heterogeneity by standard size buckets.",
        "step9_market_cap_heterogeneity_quintiles": "Step 9 market-cap heterogeneity by within-sample quintiles.",
        "step9_market_cap_heterogeneity_notes": "Step 9 market-cap heterogeneity notes.",
    }
    for path in paths:
        rel = relative_project_path(path)
        if rel in existing:
            continue
        payload.append(
            {
                "path": rel,
                "type": path.suffix.lstrip("."),
                "step": "9",
                "description": descriptions.get(path.stem, "Step 9 market-cap heterogeneity plot."),
            }
        )
        existing.add(rel)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 9 market-cap heterogeneity diagnostics.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preferred-policy", default=DEFAULT_PREFERRED_POLICY)
    parser.add_argument("--episode-market-cap-path", type=Path, default=DEFAULT_EPISODE_MARKET_CAP_PATH)
    parser.add_argument("--force-refresh-market-cap-cache", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_project_path(args.run_dir)
    if run_dir.name != FINAL_RUN_NAME:
        raise ValueError(
            f"Expected frozen run directory {FINAL_RUN_NAME!r}, got {relative_project_path(run_dir)!r}."
        )
    output_dir = run_dir / "quant_analysis"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    episode_path = run_dir / "baselines" / "baseline_episode_metrics.csv"
    if not episode_path.exists():
        raise FileNotFoundError(f"Missing baseline episode metrics: {relative_project_path(episode_path)}")
    episode_df = prepare_episode_metrics(pd.read_csv(episode_path, low_memory=False))
    validate_episode_inputs(episode_df, args.preferred_policy)
    market_meta, market_cap_source, market_notes = load_episode_market_cap(
        run_dir=run_dir,
        required_episode_ids=set(
            episode_df.loc[episode_df["split"].isin(PRIMARY_SPLITS), "episode_id"]
            .astype(str)
            .unique()
        ),
        episode_market_cap_path=resolve_project_path(args.episode_market_cap_path),
        max_workers=max(1, int(args.max_workers)),
        force_refresh=bool(args.force_refresh_market_cap_cache),
    )
    base_all = build_analysis_base(
        episode_df,
        market_meta,
        preferred_policy=args.preferred_policy,
        output_dir=output_dir,
    )
    validate_market_cap_proxy(base_all)
    coverage = compute_market_cap_coverage(base_all)
    missing_reason_lines = compute_missing_reasons(base_all)
    base = base_all[
        base_all["market_cap_at_entry"].notna() & base_all["market_cap_at_entry"].gt(0)
    ].copy()
    if base.empty:
        raise ValueError("Market-cap classification cannot be created: zero usable market_cap_at_entry rows after filtering.")
    base["standard_market_cap_bucket"] = assign_standard_bucket(base["market_cap_at_entry"])
    base["market_cap_quintile_bucket"] = assign_quintile_bucket_by_split(base)
    if base["standard_market_cap_bucket"].isna().any() or base["market_cap_quintile_bucket"].isna().any():
        raise ValueError("Market-cap classification cannot be created for all paired episodes.")

    standard = summarize_bucket(
        base,
        bucket_column="standard_market_cap_bucket",
        bucket_order=STANDARD_BUCKETS,
        bucket_system="standard",
        market_cap_source=market_cap_source,
    )
    quintile = summarize_bucket(
        base,
        bucket_column="market_cap_quintile_bucket",
        bucket_order=QUINTILE_BUCKETS,
        bucket_system="sample_relative_quintile",
        market_cap_source=market_cap_source,
    )

    standard_csv = output_dir / OUTPUTS["standard_csv"]
    standard_md = output_dir / OUTPUTS["standard_md"]
    quintile_csv = output_dir / OUTPUTS["quintile_csv"]
    quintile_md = output_dir / OUTPUTS["quintile_md"]
    notes_path = output_dir / OUTPUTS["notes"]
    save_table(standard, standard_csv, standard_md)
    save_table(quintile, quintile_csv, quintile_md)
    write_notes(
        notes_path,
        preferred_policy=args.preferred_policy,
        market_cap_source=market_cap_source,
        market_cap_notes=market_notes,
        coverage_df=coverage,
        missing_reason_lines=missing_reason_lines,
        standard_df=standard,
        quintile_df=quintile,
    )

    plot_paths = [
        plots_dir / "step9_market_cap_standard_preferred_minus_hold_test.png",
        plots_dir / "step9_market_cap_standard_win_rate_vs_hold_test.png",
        plots_dir / "step9_market_cap_standard_no_cut_pct_test.png",
        plots_dir / "step9_market_cap_standard_short_term_sold_fraction_test.png",
        plots_dir / "step9_market_cap_quintile_preferred_minus_hold_test.png",
        plots_dir / "step9_market_cap_quintile_win_rate_vs_hold_test.png",
    ]
    plot_bar(
        standard,
        bucket_order=STANDARD_BUCKETS,
        metric="dqn_minus_hold_to_terminal_mean_difference",
        ylabel="preferred minus hold mean final after-tax value",
        title="Preferred DQN minus hold by market-cap bucket - test",
        path=plot_paths[0],
    )
    plot_bar(
        standard,
        bucket_order=STANDARD_BUCKETS,
        metric="dqn_win_rate_vs_hold_to_terminal",
        ylabel="preferred win rate vs hold",
        title="Preferred DQN win rate vs hold by market-cap bucket - test",
        path=plot_paths[1],
    )
    plot_bar(
        standard,
        bucket_order=STANDARD_BUCKETS,
        metric="no_cut_percentage_preferred_dqn",
        ylabel="preferred no-cut percentage",
        title="Preferred DQN no-cut percentage by market-cap bucket - test",
        path=plot_paths[2],
    )
    plot_bar(
        standard,
        bucket_order=STANDARD_BUCKETS,
        metric="short_term_sold_fraction_preferred_dqn",
        ylabel="preferred short-term sold fraction",
        title="Preferred DQN short-term sold fraction by market-cap bucket - test",
        path=plot_paths[3],
    )
    plot_bar(
        quintile,
        bucket_order=QUINTILE_BUCKETS,
        metric="dqn_minus_hold_to_terminal_mean_difference",
        ylabel="preferred minus hold mean final after-tax value",
        title="Preferred DQN minus hold by market-cap quintile - test",
        path=plot_paths[4],
    )
    plot_bar(
        quintile,
        bucket_order=QUINTILE_BUCKETS,
        metric="dqn_win_rate_vs_hold_to_terminal",
        ylabel="preferred win rate vs hold",
        title="Preferred DQN win rate vs hold by market-cap quintile - test",
        path=plot_paths[5],
    )

    output_paths = [
        standard_csv,
        standard_md,
        quintile_csv,
        quintile_md,
        notes_path,
        *plot_paths,
    ]
    for path in output_paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected output was not written: {relative_project_path(path)}")
    update_manifest(output_dir, output_paths)
    sparse = standard[standard["sparse_bucket_warning"].fillna(False)]
    print(f"Wrote Step 9 market-cap heterogeneity outputs to {relative_project_path(output_dir)}")
    print(f"market_cap_source: {market_cap_source}")
    print(f"standard_rows: {len(standard)}")
    print(f"quintile_rows: {len(quintile)}")
    print(f"sparse_standard_bucket_rows: {len(sparse)}")


if __name__ == "__main__":
    main()
