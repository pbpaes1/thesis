"""
fetch_data.py
-------------
Downloads historical OHLCV price data from Yahoo Finance via yfinance.

Supports:
  - Full S&P 500 constituent list (scraped from Wikipedia)
  - Arbitrary lists of US or international tickers
  - Configurable date range and output format (Parquet or CSV)

International ticker convention (Yahoo Finance suffixes):
  ASML.AS  → Euronext Amsterdam
  SAP.DE   → XETRA / Frankfurt
  MC.PA    → Euronext Paris
  0700.HK  → Hong Kong
  7203.T   → Tokyo
  (no suffix) → US markets (NYSE / NASDAQ)

Usage:
  python src/fetch_data.py
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Max history available on Yahoo Finance is roughly 1993-01-01 for most US stocks
DEFAULT_START = "1993-01-01"
DEFAULT_END = "2025-12-31"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Ticker retrieval helpers
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """Scrape the current S&P 500 constituent list from Wikipedia."""
    log.info("Fetching S&P 500 tickers from Wikipedia …")
    tables = pd.read_html(SP500_WIKI_URL)
    df = tables[0]  # first table = constituents
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    log.info("Found %d S&P 500 tickers.", len(tickers))
    return tickers


def get_international_example_tickers() -> list[str]:
    """
    A small curated set of large-cap international stocks to illustrate
    cross-border data retrieval. Extend as needed.
    """
    return [
        # Europe
        "ASML.AS",   # ASML – Netherlands
        "SAP.DE",    # SAP – Germany
        "MC.PA",     # LVMH – France
        "NESN.SW",   # Nestlé – Switzerland
        "BP.L",      # BP – UK
        # Asia-Pacific
        "7203.T",    # Toyota – Japan
        "0700.HK",   # Tencent – Hong Kong
        "005930.KS", # Samsung – South Korea
    ]


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def download_price_data(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a list of tickers.

    Downloads in batches to avoid hitting Yahoo Finance rate limits.

    Returns a MultiLevel DataFrame with columns (Price field, Ticker).
    """
    all_data: list[pd.DataFrame] = []
    total = len(tickers)

    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        log.info(
            "Downloading batch %d/%d  (%d tickers) …",
            i // batch_size + 1,
            -(-total // batch_size),  # ceil division
            len(batch),
        )
        df = yf.download(
            tickers=batch,
            start=start,
            end=end,
            auto_adjust=True,   # adjusts for splits and dividends
            progress=False,
        )
        all_data.append(df)

    combined = pd.concat(all_data, axis=1)

    # Drop fully-NaN tickers (e.g., newly listed or delisted within range)
    close = combined["Close"]
    valid_tickers = close.columns[close.notna().any()].tolist()
    n_dropped = len(tickers) - len(valid_tickers)
    if n_dropped:
        log.warning("Dropped %d tickers with no data.", n_dropped)

    return combined


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_data(df: pd.DataFrame, name: str, fmt: str = "parquet") -> Path:
    """Save DataFrame to data/raw/<name>.<fmt>."""
    out_dir = DATA_DIR / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{fmt}"

    if fmt == "parquet":
        df.to_parquet(path)
    elif fmt == "csv":
        df.to_csv(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    log.info("Saved → %s  (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical price data.")
    parser.add_argument(
        "--universe",
        choices=["sp500", "international", "both"],
        default="sp500",
        help="Which ticker universe to download (default: sp500).",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--fmt",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output file format (default: parquet).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of tickers per yfinance download call (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tickers_sp500 = get_sp500_tickers() if args.universe in ("sp500", "both") else []
    tickers_intl = get_international_example_tickers() if args.universe in ("international", "both") else []

    if args.universe == "both":
        # Download each universe separately so files are cleanly separated
        if tickers_sp500:
            df_sp500 = download_price_data(tickers_sp500, args.start, args.end, args.batch_size)
            save_data(df_sp500, "sp500", args.fmt)

        if tickers_intl:
            df_intl = download_price_data(tickers_intl, args.start, args.end, args.batch_size)
            save_data(df_intl, "international", args.fmt)

    elif args.universe == "sp500":
        df = download_price_data(tickers_sp500, args.start, args.end, args.batch_size)
        save_data(df, "sp500", args.fmt)

    else:  # international
        df = download_price_data(tickers_intl, args.start, args.end, args.batch_size)
        save_data(df, "international", args.fmt)

    log.info("Done.")


if __name__ == "__main__":
    main()
