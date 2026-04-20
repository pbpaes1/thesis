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
│   ├── features/           # Final engineered feature dataset (not tracked)
│   ├── episodes/           # DRL episodic dataset (not tracked)
│   └── episode_validation/ # Episode validation report + diagnostics (mostly not tracked)
├── notebooks/
│   └── explore_data.ipynb  # Interactive data exploration
├── src/
│   └── fetch_data.py       # Data download pipeline
│   └── check_universe_gaps.py  # Quality checks + date filtering
│   └── align_common_dates.py   # Date alignment + interpolation for macro files
│   └── build_features.py       # Technical indicators + macro transforms + rolling PCA factors + export
│   └── generate_episodes.py    # DRL episode generation from engineered features
│   └── validate_parquet.py     # Episode-level parquet validation + report generation
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

This step now includes a rolling PCA factor extraction over the stock-universe log-return matrix:
- Builds daily log returns per ticker from adjusted close prices.
- For each day `t`, fits `StandardScaler` and `PCA(n_components=10)` on the strict historical window `[t-252, t-1]`.
- Transforms only day `t` to produce `PC1` to `PC10` (no look-ahead).

Final export window defaults to:
- start: `2010-01-01`
- end: `2025-12-31`

Indicator warm-up uses earlier data (from 2009-H2) before final trimming.

### 5) Generate DRL episodes for tax-aware exit timing

```powershell
python src/generate_episodes.py
```

Main final output:
- `data/episodes/drl_episodes.parquet`

Episode generation logic in this step:
- Trigger when `adj_close >= 1.30 * rolling_min_252(adj_close)`.
- Rolling minimum defines simulated purchase price/date.
- Cooldown suppresses overlapping triggers for the same ticker (default: 252 trading rows).
- Episode starts on trigger day.
- Episode ends when date reaches `simulated_purchase_date + 365 calendar days`.

### 6) Validate episodic parquet dataset

```powershell
python src/validate_parquet.py --input data/episodes/drl_episodes.parquet --output-dir data/episode_validation
```

Main outputs:
- `data/episode_validation/parquet_data_dictionary.csv`
- `data/episode_validation/parquet_column_classification.csv`
- `data/episode_validation/parquet_validation_report.md`

Additional diagnostic outputs:
- `duplicate_episode_date_rows.csv`
- `episode_date_order_issues.csv`
- `holding_period_issues.csv`
- `critical_missingness_summary.csv`
- `trigger_date_issues.csv`
- `tax_transition_issues.csv`
- `unrealized_gain_diagnostics.csv`
- `unrealized_gain_formula_scores.csv`

Validation logic highlights:
- Profiles schema, date-like columns, and numeric columns.
- Builds heuristic data dictionary and column usage classification.
- Checks key episode constraints (duplicate keys, date ordering, holding period consistency, trigger/tax consistency).
- Validates `unrealized_gains_pct` formula and scaling (fraction vs percentage-points interpretation).

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

### Rolling PCA market-structure factors (from stock log-returns)

- `PC1` to `PC10`

Computation details:
- Universe matrix is built as `date × ticker` from stock `adj_close`.
- Daily log returns are `ln(P_t / P_{t-1})`.
- For each day `t`, PCA is fit only on the previous 252 trading days (`t-252` to `t-1`).
- Day `t` is transformed with that fitted scaler/PCA to generate the ten factor values.
- Ticker set is complete-data-only for each date-level rolling fit.

---

## Columns added in episodic dataset

The following columns are added by `src/generate_episodes.py` in `data/episodes/drl_episodes.parquet`:

- `ticker_row`: Row index within each ticker series (internal indexing helper).
- `simulated_purchase_price`: Rolling 252-day minimum adjusted close used as hypothetical purchase price.
- `simulated_purchase_pos`: Position (within ticker series) where that rolling minimum occurred.
- `simulated_purchase_date`: Date of the rolling minimum price.
- `trigger_candidate`: Boolean flag before cooldown (`adj_close` is at least 30% above simulated purchase price).
- `valid_trigger`: Boolean flag after cooldown filtering; only these rows start episodes.
- `episode_id`: Unique identifier in format `{ticker}_{simulated_purchase_date}`.
- `trigger_date`: Date when the valid trigger fired (episode start).
- `tax_transition_date`: `simulated_purchase_date + 365 days` (LTCG threshold).
- `holding_period_days`: Calendar days between current row `date` and `simulated_purchase_date`.
- `unrealized_gains_pct`: `(adj_close / simulated_purchase_price) - 1`.

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
| `scikit-learn` | Rolling PCA and feature scaling (`StandardScaler`, `PCA`) |
