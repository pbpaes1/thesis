# Thesis

Deep learning approach to optimizing stock position exit timing under US short-term / long-term capital gains tax rules — Politecnico di Milano.

---

## Setup

### 1. Create the virtual environment (Python 3.12)

```powershell
py -3.12 -m venv .venv
```

### 2. Activate it

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Data pipeline

Raw price data is **not tracked in git** (gitignored). Regenerate it locally by running:

```powershell
# S&P 500 + major ADRs from 2025 onwards (fast, ~1 min)
python src/fetch_data.py --start 2025-01-01 --no-sp400

# Full history since 1993 — S&P 500 + ADRs + S&P 400 mid-caps (~15 min)
python src/fetch_data.py --start 1993-01-01
```

Output: `data/raw/universe.parquet` — long format, one row per stock × trading day.

| Column      | Type      | Description                              |
|-------------|-----------|------------------------------------------|
| `date`      | datetime  | Trading date                             |
| `ticker`    | str       | Yahoo Finance ticker symbol              |
| `open`      | float     | Daily open                               |
| `high`      | float     | Daily high                               |
| `low`       | float     | Daily low                                |
| `close`     | float     | Raw close                                |
| `adj_close` | float     | Adjusted close (splits + dividends)      |
| `volume`    | int       | Daily volume                             |
| `source`    | str       | `sp500` / `adr` / `sp400`                |

### Universe coverage

| Source   | Tickers | Description                                      |
|----------|---------|--------------------------------------------------|
| `sp500`  | ~503    | S&P 500 constituents (scraped from Wikipedia)    |
| `adr`    | ~40     | Major foreign stocks listed on NYSE/NASDAQ       |
| `sp400`  | ~400    | S&P 400 mid-caps ≥ $5B market cap (optional)    |

---

## Project structure

```
.
├── .venv/                  # Virtual environment (not tracked)
├── data/
│   ├── raw/                # Source data (not tracked)
│   ├── quality/            # Quality reports + filtered universe (not tracked)
│   ├── aligned_common_dates/  # Aligned macro + universe files (not tracked)
│   └── features/           # Final engineered dataset (not tracked)
├── notebooks/
│   └── explore_data.ipynb  # Interactive data exploration
├── src/
│   └── fetch_data.py       # Data download pipeline
│   └── check_universe_gaps.py  # Quality checks + date filtering
│   └── align_common_dates.py   # Date alignment + interpolation for macro files
│   └── build_features.py       # Technical indicators + macro merge + export
├── requirements.txt
└── README.md
```

---

## End-to-end workflow

### 1) Download or refresh universe prices

```powershell
python src/fetch_data.py --start 1993-01-01
```

### 2) Run quality checks and create warm-up filtered universe

```powershell
python src/check_universe_gaps.py
```

Main output:
- `data/quality/universe_filtered_2009h2_2026.parquet`

### 3) Align macro/market CSVs and universe dates

```powershell
python src/align_common_dates.py
```

Alignment logic in this step:
- Uses union of all macro CSV dates.
- Cleans text-formatted numeric values (`K/M/B`, `%`, commas).
- For missing gaps in numeric macro columns:
	- gaps `< 5` rows: forward fill
	- gaps `>= 5` rows: linear interpolation

Main outputs:
- `data/aligned_common_dates/*_aligned.csv`
- `data/aligned_common_dates/universe_aligned.parquet`

### 4) Build engineered features

```powershell
python src/build_features.py
```

Main final output:
- `data/features/engineered_universe.parquet`

Final export window defaults to:
- start: `2010-01-01`
- end: `2025-12-31`

Indicator warm-up uses earlier data (from 2009-H2) before final trimming.

---

## Features added in engineered dataset

### Equity technical indicators (computed per ticker on `adj_close`)

- `sma_5`, `sma_10`, `sma_20`, `sma_50`
- `ema_5`, `ema_10`, `ema_20`, `ema_50`
- `rsi_14`
- `macd`
- `macd_signal`
- `macd_hist`
- `bb_lower`, `bb_middle`, `bb_upper`

### Macro/market close features (from aligned CSV files)

One close-price feature per input macro asset, for example:
- `EUR_USD_Close`
- `Gold_Close`
- `Semiconductor_Close`
- `USD_CNY_Close`
- `USD_JPN_Close`
- `VIX_Close`
- `Wheat_Close`
- `WTI_Close`

These are merged into the equity panel by `date`.

---

## Key dependencies

| Package      | Purpose                              |
|--------------|--------------------------------------|
| `yfinance`   | Yahoo Finance price data download    |
| `pandas`     | Data manipulation                    |
| `numpy`      | Numeric ops and interpolation helpers |
| `pyarrow`    | Parquet read/write                   |
| `requests`   | HTTP requests for Wikipedia scraping |
| `beautifulsoup4` | HTML parsing for ticker lists    |
| `lxml`       | HTML parser backend                  |
| `matplotlib` | Plotting                             |
| `plotly`     | Interactive charts                   |
| `ipykernel`  | Jupyter notebook support             |
| `exchange_calendars` | Exchange session calendars for quality checks |
| `pandas-ta`  | Technical analysis indicators        |
