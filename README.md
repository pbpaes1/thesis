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

| Column   | Type      | Description                              |
|----------|-----------|------------------------------------------|
| `date`   | datetime  | Trading date                             |
| `ticker` | str       | Yahoo Finance ticker symbol              |
| `close`  | float     | Adjusted closing price (splits+divs)     |
| `volume` | int       | Daily volume                             |
| `source` | str       | `sp500` / `adr` / `sp400`               |

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
│   └── raw/                # Downloaded parquet files (not tracked)
├── notebooks/
│   └── explore_data.ipynb  # Interactive data exploration
├── src/
│   └── fetch_data.py       # Data download pipeline
├── requirements.txt
└── README.md
```

---

## Key dependencies

| Package      | Purpose                              |
|--------------|--------------------------------------|
| `yfinance`   | Yahoo Finance price data download    |
| `pandas`     | Data manipulation                    |
| `pyarrow`    | Parquet read/write                   |
| `requests`   | HTTP requests for Wikipedia scraping |
| `beautifulsoup4` | HTML parsing for ticker lists    |
| `lxml`       | HTML parser backend                  |
| `matplotlib` | Plotting                             |
| `plotly`     | Interactive charts                   |
| `ipykernel`  | Jupyter notebook support             |
