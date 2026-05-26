"""Build reusable ticker sector metadata for the frozen final run.

Run from the project root:
    python scripts/build_sector_metadata.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "train_reward_c_lite_v5_full"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "metadata" / "ticker_sector_metadata.csv"
DEFAULT_NOTES = PROJECT_ROOT / "data" / "metadata" / "ticker_sector_metadata_notes.txt"

REQUIRED_METADATA_COLUMNS = [
    "ticker",
    "sector",
    "industry",
    "quoteType",
    "longName",
    "shortName",
    "metadata_source",
    "metadata_fetch_status",
    "metadata_fetch_error",
]
TICKER_COLUMN_CANDIDATES = [
    "ticker",
    "symbol",
    "asset",
    "asset_symbol",
    "stock",
    "stock_ticker",
]


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def read_csv_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    return pd.read_csv(path, nrows=0).columns.tolist()


def parse_ticker_from_episode_id(series: pd.Series) -> pd.Series:
    return (
        series.dropna()
        .astype(str)
        .str.strip()
        .str.split("_", n=1)
        .str[0]
        .str.strip()
    )


def extract_tickers_from_artifacts(run_dir: Path) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    candidate_paths = [
        run_dir / "baselines" / "baseline_episode_metrics.csv",
        run_dir / "baselines" / "baseline_step_rollouts.csv",
    ]
    for path in candidate_paths:
        columns = read_csv_columns(path)
        ticker_columns = [
            column
            for column in columns
            if column.lower() in TICKER_COLUMN_CANDIDATES
        ]
        if ticker_columns:
            ticker_column = ticker_columns[0]
            tickers = (
                pd.read_csv(path, usecols=[ticker_column], low_memory=False)[ticker_column]
                .dropna()
                .astype(str)
                .str.strip()
            )
            unique_tickers = sorted(ticker for ticker in tickers.unique() if ticker)
            notes.append(
                "ticker_extraction_method: direct artifact column "
                f"{ticker_column!r} from {relative_path(path)}."
            )
            return unique_tickers, notes

    for path in candidate_paths:
        columns = read_csv_columns(path)
        if "episode_id" not in columns:
            continue
        episode_ids = pd.read_csv(path, usecols=["episode_id"], low_memory=False)[
            "episode_id"
        ]
        tickers = parse_ticker_from_episode_id(episode_ids)
        unique_tickers = sorted(ticker for ticker in tickers.unique() if ticker)
        notes.append(
            "ticker_extraction_method: parsed ticker from episode_id prefix before "
            f"the first underscore in {relative_path(path)}."
        )
        notes.append(
            "episode_id_parsing_convention: examples such as AAPL_2018-01-03 "
            "are interpreted as ticker=AAPL."
        )
        return unique_tickers, notes

    raise FileNotFoundError(
        "Could not find ticker metadata columns or episode_id in baseline artifacts "
        f"under {relative_path(run_dir)}."
    )


def normalize_existing_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_METADATA_COLUMNS)
    cache = pd.read_csv(path, low_memory=False)
    for column in REQUIRED_METADATA_COLUMNS:
        if column not in cache.columns:
            cache[column] = ""
    cache = cache[REQUIRED_METADATA_COLUMNS].copy()
    cache["ticker"] = cache["ticker"].astype(str).str.strip()
    cache = cache[cache["ticker"].ne("")]
    cache = cache.drop_duplicates("ticker", keep="last")
    return cache


def yfinance_symbol_for_ticker(ticker: str) -> str:
    return ticker.replace(".", "-")


def fetch_one_ticker(ticker: str, pause_seconds: float) -> dict[str, Any]:
    if pause_seconds > 0:
        time.sleep(pause_seconds)
    try:
        import yfinance as yf
    except ImportError as exc:
        return {
            "ticker": ticker,
            "sector": "Unknown",
            "industry": "Unknown",
            "quoteType": "",
            "longName": "",
            "shortName": "",
            "metadata_source": "yfinance.Ticker.info",
            "metadata_fetch_status": "failed",
            "metadata_fetch_error": f"yfinance import failed: {exc}",
        }

    try:
        info = yf.Ticker(yfinance_symbol_for_ticker(ticker)).info
        if not isinstance(info, dict) or not info:
            raise ValueError("empty Ticker.info response")
        sector = info.get("sector") or "Unknown"
        industry = info.get("industry") or "Unknown"
        return {
            "ticker": ticker,
            "sector": sector,
            "industry": industry,
            "quoteType": info.get("quoteType") or "",
            "longName": info.get("longName") or "",
            "shortName": info.get("shortName") or "",
            "metadata_source": "yfinance.Ticker.info",
            "metadata_fetch_status": "success",
            "metadata_fetch_error": "",
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "sector": "Unknown",
            "industry": "Unknown",
            "quoteType": "",
            "longName": "",
            "shortName": "",
            "metadata_source": "yfinance.Ticker.info",
            "metadata_fetch_status": "failed",
            "metadata_fetch_error": str(exc),
        }


def fetch_missing_metadata(
    tickers: list[str],
    *,
    max_workers: int,
    pause_seconds: float,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=REQUIRED_METADATA_COLUMNS)
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_one_ticker, ticker, pause_seconds): ticker
            for ticker in tickers
        }
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    out = pd.DataFrame(rows)
    for column in REQUIRED_METADATA_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return out[REQUIRED_METADATA_COLUMNS]


def summarize(metadata: pd.DataFrame, required_tickers: list[str]) -> dict[str, Any]:
    required = metadata[metadata["ticker"].isin(required_tickers)].copy()
    required["sector"] = required["sector"].fillna("Unknown").replace("", "Unknown")
    return {
        "num_tickers_found": len(required_tickers),
        "num_with_sector": int(required["sector"].ne("Unknown").sum()),
        "num_missing_sector": int(required["sector"].eq("Unknown").sum()),
        "num_failed": int(required["metadata_fetch_status"].eq("failed").sum()),
    }


def write_notes(
    path: Path,
    *,
    output_path: Path,
    run_dir: Path,
    extraction_notes: list[str],
    summary: dict[str, Any],
    reused_cache: bool,
    force_refresh: bool,
) -> None:
    lines = [
        "Ticker sector metadata notes",
        f"run_dir: {relative_path(run_dir)}",
        f"metadata_source: yfinance.Ticker.info",
        f"metadata_cache_file: {relative_path(output_path)}",
        f"force_refresh: {force_refresh}",
        f"reused_cache_for_all_required_tickers: {reused_cache}",
        *extraction_notes,
        f"number_of_tickers_found: {summary['num_tickers_found']}",
        f"number_with_sector: {summary['num_with_sector']}",
        f"number_missing_sector: {summary['num_missing_sector']}",
        f"number_failed: {summary['num_failed']}",
        "missing or failed sectors are filled as Unknown.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_metadata(
    *,
    run_dir: Path,
    output_path: Path,
    force_refresh: bool,
    max_workers: int,
    pause_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any], bool, list[str]]:
    tickers, extraction_notes = extract_tickers_from_artifacts(run_dir)
    cache = normalize_existing_cache(output_path)
    cached_tickers = set(cache["ticker"].tolist())
    required_tickers = set(tickers)
    cache_complete = required_tickers.issubset(cached_tickers)
    if cache_complete and not force_refresh:
        metadata = cache.copy()
        reused_cache = True
    else:
        tickers_to_fetch = tickers if force_refresh else sorted(required_tickers - cached_tickers)
        fetched = fetch_missing_metadata(
            tickers_to_fetch,
            max_workers=max_workers,
            pause_seconds=pause_seconds,
        )
        if force_refresh:
            stale_extra = cache[~cache["ticker"].isin(tickers_to_fetch)]
            metadata = pd.concat([stale_extra, fetched], ignore_index=True)
        else:
            metadata = pd.concat([cache, fetched], ignore_index=True)
        metadata = metadata.drop_duplicates("ticker", keep="last")
        reused_cache = False

    for column in ["sector", "industry"]:
        metadata[column] = metadata[column].fillna("Unknown").replace("", "Unknown")
    for column in REQUIRED_METADATA_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = ""
    metadata = metadata[REQUIRED_METADATA_COLUMNS].sort_values("ticker")
    summary = summarize(metadata, tickers)
    return metadata, summary, reused_cache, extraction_notes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cached ticker sector metadata for the frozen final run."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata, summary, reused_cache, extraction_notes = build_metadata(
        run_dir=run_dir,
        output_path=output_path,
        force_refresh=args.force_refresh,
        max_workers=max(1, int(args.max_workers)),
        pause_seconds=max(0.0, float(args.pause_seconds)),
    )
    metadata.to_csv(output_path, index=False)
    notes_path = output_path.with_name(output_path.stem + "_notes.txt")
    if output_path == DEFAULT_OUTPUT.resolve():
        notes_path = DEFAULT_NOTES
    write_notes(
        notes_path,
        output_path=output_path,
        run_dir=run_dir,
        extraction_notes=extraction_notes,
        summary=summary,
        reused_cache=reused_cache,
        force_refresh=args.force_refresh,
    )
    print(f"number of tickers found: {summary['num_tickers_found']}")
    print(f"number with sector: {summary['num_with_sector']}")
    print(f"number missing sector: {summary['num_missing_sector']}")
    print(f"number failed: {summary['num_failed']}")
    print(f"output path: {relative_path(output_path)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
