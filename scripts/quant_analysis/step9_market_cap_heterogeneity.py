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
    PROJECT_ROOT / "data" / "metadata" / "episode_market_cap_at_entry.csv"
)
DEFAULT_FALLBACK_MARKET_CAP_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "ticker_market_cap_current_fallback.csv"
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


def fetch_one_market_cap(ticker: str) -> dict[str, Any]:
    try:
        import yfinance as yf
    except ImportError as exc:
        return {
            "ticker": ticker,
            "current_market_cap": np.nan,
            "metadata_source": "yfinance.Ticker.fast_info.market_cap",
            "metadata_fetch_status": "failed",
            "metadata_fetch_error": f"yfinance import failed: {exc}",
        }
    try:
        fast_info = yf.Ticker(yfinance_symbol(ticker)).fast_info
        market_cap = getattr(fast_info, "market_cap", np.nan)
        market_cap = float(market_cap) if pd.notna(market_cap) else np.nan
        if not np.isfinite(market_cap) or market_cap <= 0:
            info = yf.Ticker(yfinance_symbol(ticker)).get_info()
            if not isinstance(info, dict):
                raise ValueError("missing or non-positive market_cap")
            market_cap = info.get("marketCap")
            market_cap = float(market_cap) if pd.notna(market_cap) else np.nan
            if not np.isfinite(market_cap) or market_cap <= 0:
                shares = info.get("sharesOutstanding")
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                if pd.notna(shares) and pd.notna(price):
                    market_cap = float(shares) * float(price)
            if not np.isfinite(market_cap) or market_cap <= 0:
                raise ValueError("missing or non-positive market_cap")
        return {
            "ticker": ticker,
            "current_market_cap": market_cap,
            "metadata_source": "yfinance.Ticker.fast_info.market_cap",
            "metadata_fetch_status": "success",
            "metadata_fetch_error": "",
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "current_market_cap": np.nan,
            "metadata_source": "yfinance.Ticker.fast_info.market_cap",
            "metadata_fetch_status": "failed",
            "metadata_fetch_error": str(exc),
        }


def normalize_fallback_cache(path: Path) -> pd.DataFrame:
    columns = [
        "ticker",
        "current_market_cap",
        "metadata_source",
        "metadata_fetch_status",
        "metadata_fetch_error",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    cache = pd.read_csv(path, low_memory=False)
    for column in columns:
        if column not in cache.columns:
            cache[column] = ""
    cache["ticker"] = cache["ticker"].astype(str).str.strip()
    cache["current_market_cap"] = pd.to_numeric(
        cache["current_market_cap"], errors="coerce"
    )
    cache = cache[cache["ticker"].ne("")]
    cache = cache.drop_duplicates("ticker", keep="last")
    return cache[columns]


def ensure_current_market_cap_fallback(
    tickers: list[str],
    *,
    path: Path,
    max_workers: int,
    force_refresh: bool,
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = normalize_fallback_cache(path)
    if force_refresh:
        cached_ok: set[str] = set()
    else:
        cached_ok = set(
            cache.loc[
                cache["ticker"].isin(tickers)
                & cache["current_market_cap"].notna()
                & cache["current_market_cap"].gt(0),
                "ticker",
            ]
        )
    missing = sorted(set(tickers) - cached_ok)
    if missing:
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one_market_cap, ticker): ticker for ticker in missing}
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
        fetched = pd.DataFrame(rows)
        cache = pd.concat([cache[~cache["ticker"].isin(missing)], fetched], ignore_index=True)
        cache = cache.drop_duplicates("ticker", keep="last").sort_values("ticker")
        cache.to_csv(path, index=False)
    required = cache[cache["ticker"].isin(tickers)].copy()
    if required.empty or required["current_market_cap"].notna().sum() == 0:
        raise ValueError(
            "Market-cap classification cannot be created. Provide a point-in-time "
            f"episode-level file at {relative_project_path(DEFAULT_EPISODE_MARKET_CAP_PATH)} "
            "with columns episode_id, market_cap_at_entry, or provide a fallback file "
            f"at {relative_project_path(path)} with ticker,current_market_cap."
        )
    missing_caps = sorted(
        set(tickers)
        - set(
            required.loc[
                required["current_market_cap"].notna()
                & required["current_market_cap"].gt(0),
                "ticker",
            ]
        )
    )
    if missing_caps:
        raise ValueError(
            "Market-cap classification cannot be created for all required tickers. "
            "Missing/non-positive current_market_cap for ticker(s): "
            + ", ".join(missing_caps[:50])
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
    summary["entry_date"] = summary["simulated_purchase_date"].where(
        summary["simulated_purchase_date"].notna(), summary["episode_start_date"]
    )
    summary["price_at_entry"] = summary["price_at_entry"].where(
        summary["price_at_entry"].notna(), summary["close_at_entry"]
    )
    return summary


def load_episode_market_cap(
    *,
    run_dir: Path,
    required_episode_ids: set[str],
    episode_market_cap_path: Path,
    fallback_market_cap_path: Path,
    max_workers: int,
    force_refresh_fallback: bool,
) -> tuple[pd.DataFrame, str, list[str]]:
    notes: list[str] = []
    raw_meta = load_raw_episode_metadata(run_dir)
    raw_meta["episode_id"] = raw_meta["episode_id"].astype(str)
    raw_meta = raw_meta[raw_meta["episode_id"].isin(required_episode_ids)].copy()
    if raw_meta.empty:
        raise ValueError("Market-cap classification cannot be created: no raw metadata matched required validation/test episode_id values.")
    if episode_market_cap_path.exists():
        market = pd.read_csv(episode_market_cap_path, low_memory=False)
        if "episode_id" in market.columns and "market_cap_at_entry" in market.columns:
            market = market[["episode_id", "market_cap_at_entry"]].copy()
            market["episode_id"] = market["episode_id"].astype(str)
            market["market_cap_at_entry"] = pd.to_numeric(
                market["market_cap_at_entry"], errors="coerce"
            )
            merged = raw_meta.merge(market, on="episode_id", how="left", validate="one_to_one")
            if merged["market_cap_at_entry"].isna().any():
                missing = int(merged["market_cap_at_entry"].isna().sum())
                raise ValueError(
                    f"Point-in-time market-cap file is missing {missing} episode_id rows: "
                    f"{relative_project_path(episode_market_cap_path)}"
                )
            source = "point_in_time_episode_market_cap_at_entry"
            notes.append(
                "market_cap_source_file: "
                + relative_project_path(episode_market_cap_path)
            )
            return merged, source, notes
        raise ValueError(
            "Market-cap file exists but does not contain required columns "
            f"episode_id and market_cap_at_entry: {relative_project_path(episode_market_cap_path)}"
        )

    tickers = sorted(raw_meta["ticker"].dropna().astype(str).unique())
    fallback = ensure_current_market_cap_fallback(
        tickers,
        path=fallback_market_cap_path,
        max_workers=max_workers,
        force_refresh=force_refresh_fallback,
    )
    merged = raw_meta.merge(
        fallback[["ticker", "current_market_cap", "metadata_source"]],
        on="ticker",
        how="left",
        validate="many_to_one",
    )
    merged["market_cap_at_entry"] = merged["current_market_cap"]
    source = "non_point_in_time_current_market_cap_sensitivity"
    notes.extend(
        [
            "point_in_time_market_cap_unavailable: True",
            "fallback_market_cap_source_file: "
            + relative_project_path(fallback_market_cap_path),
            "fallback_market_cap_source: yfinance.Ticker.fast_info.market_cap with get_info marketCap/sharesOutstanding fallback",
            "fallback_label: non-point-in-time current-market-cap sensitivity",
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
        "price_at_entry",
        "market_cap_at_entry",
    ]
    base = base.merge(
        market_meta[meta_keep], on="episode_id", how="left", validate="many_to_one"
    )
    if base["market_cap_at_entry"].isna().any() or base["market_cap_at_entry"].le(0).any():
        raise ValueError("Market-cap classification cannot be created for every paired episode.")
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
                    "mean_market_cap": group["market_cap_at_entry"].mean(),
                    "median_market_cap": group["market_cap_at_entry"].median(),
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
        "If point-in-time market cap was unavailable, this output is labeled as a non-point-in-time sensitivity analysis.",
        "Standard buckets may be imbalanced because the universe is S&P 500-like.",
        "Quintile buckets are included to provide a balanced within-sample size comparison.",
        "Final after-tax value remains the primary metric.",
        "EAAT and TA-EAAT Sharpe, if included, are secondary diagnostics.",
        "The test split remains the main thesis evidence; validation is supportive.",
        "Train and all splits are not included in this Step 9 market-cap output.",
        "",
        "standard_bucket_counts:",
    ]
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
    parser.add_argument("--fallback-market-cap-path", type=Path, default=DEFAULT_FALLBACK_MARKET_CAP_PATH)
    parser.add_argument("--force-refresh-fallback", action="store_true")
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
        fallback_market_cap_path=resolve_project_path(args.fallback_market_cap_path),
        max_workers=max(1, int(args.max_workers)),
        force_refresh_fallback=bool(args.force_refresh_fallback),
    )
    base = build_analysis_base(
        episode_df,
        market_meta,
        preferred_policy=args.preferred_policy,
        output_dir=output_dir,
    )
    base["standard_market_cap_bucket"] = assign_standard_bucket(base["market_cap_at_entry"])
    base["market_cap_quintile_bucket"] = assign_quintile_bucket(base["market_cap_at_entry"])
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
