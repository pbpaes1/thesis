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
│   ├── episode_validation/ # Episode validation report + diagnostics (mostly not tracked)
│   ├── freeze/             # Versioned state-freeze snapshots (v1, v2, ...)
│   └── tax_profiles/       # Reusable individual tax-profile configs
├── notebooks/
│   └── explore_data.ipynb  # Interactive data exploration
├── scripts/
│   └── smoke_test_environment.py  # Manual console smoke runner for environment rollouts
├── src/
│   └── fetch_data.py       # Data download pipeline
│   └── check_universe_gaps.py  # Quality checks + date filtering
│   └── align_common_dates.py   # Date alignment + interpolation for macro files
│   └── build_features.py       # Technical indicators + macro transforms + rolling PCA factors + export
│   └── generate_episodes.py    # DRL episode generation from engineered features
│   └── validate_parquet.py     # Episode-level parquet validation + report generation
│   └── freeze_columns_v1.py    # Conservative v1 state column freeze (allowed vs excluded)
│   └── environment/
│       └── tax_aware_env.py    # Tax-aware liquidation environment (agent-ready interface)
├── tests/
│   └── test_env_smoke.py   # Smoke/unit checks for traversal, actions, tax accounting
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
- Raw tax columns are preserved: `days_until_tax_transition`, `unrealized_gains_pct`.
- Normalized tax columns are added:
  - `days_to_tax_transition_norm = min(days_until_tax_transition, 365) / 365`
  - `unrealized_gain_pct_norm = tanh(unrealized_gains_pct / 0.25)`
- Script logs v1 normalization checks to confirm bounds and raw-column presence.

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

### 7) Freeze state columns (allowed vs excluded)

```powershell
python src/freeze_columns_v1.py --input data/episodes/drl_episodes.parquet --output-dir data/freeze --version v1
```

Main outputs:
- `data/freeze/v1/allowed_state_columns_v1.json`
- `data/freeze/v1/excluded_columns_v1.json`
- `data/freeze/v1/state_freeze_v1_summary.md`

### 8) Configure reusable individual tax profiles

Tax profiles are versioned and reusable across the same episode parquet:

- `data/tax_profiles/individual_tax_profiles_v1.json`
- `data/tax_profiles/README.md`

Currently included individual scenarios:
- `tax_free`
- `low_income_individual`
- `mass_affluent_individual`
- `high_income_individual`
- `top_bracket_individual`

These are thesis simulation profiles, not a full legal/tax engine.

### 9) Run environment smoke/unit checks

```powershell
python -m unittest tests.test_env_smoke -v
```

### 10) Run manual environment smoke runner

```powershell
python scripts/smoke_test_environment.py
```

The manual runner prints:
- short-term and long-term scenario blocks
- fixed action trajectories
- per-step bookkeeping and tax-accounting columns for inspection

---

## Tax-aware environment

Environment implementation:
- `src/environment/tax_aware_env.py`

Core behavior currently implemented:
- episode-level parquet loading and per-episode chronological traversal
- observation extraction from externally supplied frozen `state_columns`
- discrete actions as sell fractions of original position: `(0.0, 0.25, 0.50, 0.75, 1.0)`
- oversell prevention and sold/remaining fraction bookkeeping
- sale-level realized pre-tax and after-tax PnL accounting
- short-term vs long-term regime classification using episode dates
- profile-driven individual tax rates (`short_term_rate`, `long_term_rate`, optional `niit_rate`)

Agent-ready interface (plain Python class, Gym-like return contract):
- `reset(...) -> (obs, info)`
- `step(action) -> (obs, reward, done, truncated, info)`

Current terminal logic:
- `done=True` when final episode row is reached and/or remaining fraction is `0.0`
- `truncated=False` (placeholder; no artificial cutoff yet)

Reward status:
- reward remains a placeholder (`0.0`) pending later reward-design step.

Reset `info` includes at least:
- `episode_id`, `current_row_ptr`, `date`
- `sold_fraction`, `remaining_fraction`
- `cum_realized_pre_tax_pnl`, `cum_realized_after_tax_pnl`
- `tax_profile_name`

Step `info` includes at least:
- action bookkeeping (`action`, requested/executed fraction, sold/remaining)
- tax/accounting fields (`tax_regime`, `applicable_tax_rate`, sale increments, cumulative totals)
- episode pointers (`current_row_ptr`, `sale_row_ptr`) and `date`

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
- `days_until_tax_transition`: Calendar days left until `tax_transition_date` (`tax_transition_date - date`).
- `unrealized_gains_pct`: `(adj_close / simulated_purchase_price) - 1`.
- `days_to_tax_transition_norm`: Normalized tax-transition horizon, `min(days_until_tax_transition, 365) / 365`.
- `unrealized_gain_pct_norm`: Normalized unrealized gain, `tanh(unrealized_gains_pct / 0.25)`.

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
