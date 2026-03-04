"""
fetch_data.py
-------------
Builds a single long-format dataset (one row per stock × day) with:
  - All S&P 500 constituents            (source = "sp500")
  - Major ADRs traded on US exchanges   (source = "adr")
  - S&P 400 Mid-Cap stocks              (source = "sp400")

Output schema
─────────────
  date (datetime) | ticker (str) | close (float) | volume (int) | source (str)

Usage
─────
  python src/fetch_data.py                      # full run, parquet
  python src/fetch_data.py --start 2010-01-01   # custom date range
  python src/fetch_data.py --fmt csv            # save as CSV
  python src/fetch_data.py --no-sp400           # skip S&P 400
"""

import argparse
import io
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

DEFAULT_START = "1993-01-01"
DEFAULT_END   = date.today().isoformat()   # always up to today
BATCH_SIZE    = 50

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ── Major ADRs traded on US exchanges ─────────────────────────────────────────
# Foreign companies with primary listings abroad but quoted in USD on NYSE/NASDAQ.
MAJOR_ADRS = {
    # Semiconductors & Tech
    "TSM":   "TSMC (Taiwan)",
    "ASML":  "ASML (Netherlands)",
    "SONY":  "Sony (Japan)",
    "SAP":   "SAP (Germany)",
    "INFY":  "Infosys (India)",
    "WIT":   "Wipro (India)",
    "ERIC":  "Ericsson (Sweden)",
    "NOK":   "Nokia (Finland)",
    # Pharma & Healthcare
    "NVO":   "Novo Nordisk (Denmark)",
    "AZN":   "AstraZeneca (UK)",
    "RHHBY": "Roche (Switzerland)",
    "NVS":   "Novartis (Switzerland)",
    "SNY":   "Sanofi (France)",
    "GSK":   "GSK (UK)",
    # Financials
    "HSBC":  "HSBC (UK)",
    "ING":   "ING Group (Netherlands)",
    "SAN":   "Banco Santander (Spain)",
    "BBVA":  "BBVA (Spain)",
    "ITUB":  "Itaú Unibanco (Brazil)",
    "VALE":  "Vale (Brazil)",
    # Energy
    "SHEL":  "Shell (UK/Netherlands)",
    "BP":    "BP (UK)",
    "TTE":   "TotalEnergies (France)",
    "E":     "Eni (Italy)",
    "EQNR":  "Equinor (Norway)",
    # Consumer
    "NSRGY": "Nestlé (Switzerland)",
    "LVMUY": "LVMH (France)",
    "LRLCY": "L'Oréal (France)",
    "UL":    "Unilever (UK/NL)",
    "DEO":   "Diageo (UK)",
    "BUD":   "AB InBev (Belgium)",
    # Auto
    "TM":    "Toyota (Japan)",
    "HMC":   "Honda (Japan)",
    "STLA":  "Stellantis (Netherlands)",
    "VWAGY": "Volkswagen (Germany)",
    # China / Asia
    "BABA":  "Alibaba (China)",
    "BIDU":  "Baidu (China)",
    "JD":    "JD.com (China)",
    "PDD":   "PDD Holdings (China)",
    "SE":    "Sea Ltd (Singapore)",
}


# ── Ticker list helpers ────────────────────────────────────────────────────────

def _scrape_wiki_sp_table(url: str, label: str) -> list[str]:
    """
    Scrape an S&P index constituent table from Wikipedia.
    Uses BeautifulSoup to isolate just the wikitable before handing it to
    pandas — avoids lxml flooding stdout when parsing the full Wikipedia page.
    """
    log.info("Fetching %s tickers from Wikipedia …", label)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the first wikitable that has a "Symbol" column header
    target_table = None
    for table in soup.find_all("table", {"class": "wikitable"}):
        headers_row = table.find("tr")
        if headers_row and "Symbol" in headers_row.get_text():
            target_table = table
            break

    if target_table is None:
        raise ValueError(f"Could not find a 'Symbol' table on {url}")

    # Parse only that one table — avoids lxml stdout noise on the full page
    df = pd.read_html(io.StringIO(str(target_table)))[0]

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    log.info("  → %d %s tickers found.", len(tickers), label)
    return tickers


def get_sp500_tickers() -> list[str]:
    return _scrape_wiki_sp_table(SP500_WIKI, "S&P 500")


def get_sp400_tickers() -> list[str]:
    return _scrape_wiki_sp_table(SP400_WIKI, "S&P 400")


def get_adr_tickers() -> list[str]:
    return list(MAJOR_ADRS.keys())


# ── Universe builder ───────────────────────────────────────────────────────────

def build_universe(include_sp400: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame [ticker, source].
    Priority on deduplication: sp500 > adr > sp400.
    """
    rows: list[dict] = []

    for t in get_sp500_tickers():
        rows.append({"ticker": t, "source": "sp500"})

    seen = {r["ticker"] for r in rows}

    for t in get_adr_tickers():
        if t not in seen:
            rows.append({"ticker": t, "source": "adr"})
            seen.add(t)

    if include_sp400:
        for t in get_sp400_tickers():
            if t not in seen:
                rows.append({"ticker": t, "source": "sp400"})
                seen.add(t)

    universe = pd.DataFrame(rows).reset_index(drop=True)
    log.info(
        "Universe: %d total tickers  (sp500=%d, adr=%d, sp400=%d)",
        len(universe),
        (universe.source == "sp500").sum(),
        (universe.source == "adr").sum(),
        (universe.source == "sp400").sum(),
    )
    return universe


# ── Download ───────────────────────────────────────────────────────────────────

def download_long(
    universe: pd.DataFrame,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Downloads daily adjusted close + volume for the full universe.
    Returns a long-format DataFrame:
      date | ticker | close | volume | source
    """
    tickers = universe["ticker"].tolist()
    source_map = universe.set_index("ticker")["source"].to_dict()

    chunks: list[pd.DataFrame] = []
    total_batches = -(-len(tickers) // batch_size)  # ceiling division

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        log.info("Batch %d/%d — %d tickers …", batch_num, total_batches, len(batch))

        raw = yf.download(
            tickers=batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            log.warning("  Batch %d returned no data.", batch_num)
            continue

        # yfinance returns MultiIndex columns (field, ticker) for multi-ticker,
        # and flat columns for a single ticker — handle both cases
        if isinstance(raw.columns, pd.MultiIndex):
            close_wide  = raw["Close"]
            volume_wide = raw["Volume"]
        else:
            t = batch[0]
            close_wide  = raw[["Close"]].rename(columns={"Close": t})
            volume_wide = raw[["Volume"]].rename(columns={"Volume": t})

        close_long = (
            close_wide.reset_index()
            .melt(id_vars="Date", var_name="ticker", value_name="close")
        )
        volume_long = (
            volume_wide.reset_index()
            .melt(id_vars="Date", var_name="ticker", value_name="volume")
        )

        chunk = close_long.merge(volume_long, on=["Date", "ticker"])
        chunk = chunk.rename(columns={"Date": "date"})
        chunk["source"] = chunk["ticker"].map(source_map)
        chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No data downloaded — check tickers and date range.")

    df = pd.concat(chunks, ignore_index=True)

    # Drop rows where both close and volume are NaN (not listed in this period)
    df = df.dropna(subset=["close", "volume"], how="all")

    df["date"]   = pd.to_datetime(df["date"])
    df["volume"] = df["volume"].astype("Int64")  # nullable int (pandas handles NaN)
    df = df[["date", "ticker", "close", "volume", "source"]]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    log.info(
        "Result: %s rows | %d tickers | %d trading days",
        f"{len(df):,}",
        df["ticker"].nunique(),
        df["date"].nunique(),
    )
    return df


# ── Save ───────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, fmt: str = "parquet") -> Path:
    out_dir = DATA_DIR / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"universe.{fmt}"

    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    log.info("Saved → %s  (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download long-format price dataset.")
    p.add_argument("--start",      default=DEFAULT_START, help="Start date YYYY-MM-DD")
    p.add_argument("--end",        default=DEFAULT_END,   help="End date YYYY-MM-DD")
    p.add_argument("--fmt",        choices=["parquet", "csv"], default="parquet")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--no-sp400",   action="store_true", help="Skip S&P 400 mid-caps")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    universe = build_universe(include_sp400=not args.no_sp400)
    df = download_long(universe, args.start, args.end, args.batch_size)
    save(df, args.fmt)
    print("\nSample (first 3 rows per source):")
    print(df.groupby("source").head(3).to_string(index=False))
    log.info("Done.")


if __name__ == "__main__":
    main()
