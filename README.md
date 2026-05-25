# Thesis

Deep learning approach to optimizing stock position exit timing under US short-term / long-term capital gains tax rules — Politecnico di Milano.

---

## Current build/train status

The build-and-training phase now contains Reward A and Reward C-lite
experiments used to test tax-aware liquidation behavior:

- Reward A: `A_after_tax_total_value_change`
- Reward C-lite v1:
  `C_lite_after_tax_value_change_minus_cooldown_penalty`
- Reward C-lite v2:
  `C_lite_v2_after_tax_value_change_minus_transaction_and_cooldown_penalty`
- Reward C-lite v3:
  C-lite v1 reward with thresholded-greedy exploitation during training
- Reward C-lite v4:
  C-lite v1 reward with a larger first-sale threshold during thresholded-greedy training
- Reward C-lite v5:
  C-lite v1 greedy training with first-sale threshold variants applied only during evaluation

Reward C-lite v1 keeps Reward A as the economic reward and subtracts a
cooldown penalty for clustered discretionary sales. Reward C-lite v2 additionally
subtracts a small transaction penalty from every discretionary executed sale,
uses `gamma = 1.0`, and uses more hold-biased exploration.
Reward C-lite v3 and v4 keep the v1 reward unchanged and modify only the
training behavior policy. Reward C-lite v5 keeps greedy C-lite v1 training and
extends post-training evaluation with first-sale-margin policies.

These experiments do not implement Reward B, drawdown penalties, or explicit
tax-saving bonuses. Baseline and behavior inspection scripts are config-driven,
include terminal liquidation tax in effective tax-rate metrics, and exclude
tax-transition baselines because generated episodes already terminate at the
one-year tax threshold, making `sell_at_tax_transition` equivalent to
`hold_to_terminal`.

Latest tracked build/train run artifacts:

| Run | Config | Training policy | Best validation mean final after-tax value | Notes |
|-----|--------|-----------------|--------------------------------------------|-------|
| Reward A v3 | `configs/train_reward_a_v3.yaml` | greedy | `0.5830773451153731` | Reward A reference run |
| Reward C-lite v1 | `configs/train_reward_c_lite_v1.yaml` | greedy | `0.5847384187819512` | Cooldown penalty only |
| Reward C-lite v3 | `configs/train_reward_c_lite_v3.yaml` | thresholded greedy, margin `0.020` | `0.5813745591588575` | Training-time no-sell buffer |
| Reward C-lite v4 | `configs/train_reward_c_lite_v4.yaml` | thresholded greedy, first-sale margin `0.040`, normal margin `0.020` | `0.5821660750908308` | Stronger first-sale hold bias |
| Reward C-lite v5 | `configs/train_reward_c_lite_v5.yaml` | greedy | `0.5847384187819512` | First-sale-margin variants evaluated after training |

The baseline summaries for the latest C-lite runs still rank
`hold_to_terminal` highest by mean final after-tax total value on validation and
test. First-sale-margin evaluation policies delay DQN first sales and reduce
short-term selling, but they remain benchmark diagnostics rather than final
thesis conclusions.

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
├── configs/
│   ├── reward_v1.yaml      # Frozen Reward A config
│   ├── train_reward_a_v1.yaml  # First Reward A DQN training config
│   ├── train_reward_a_v2.yaml  # Reward A full-run config iteration
│   ├── train_reward_a_v3.yaml  # Reward A v3 full-run config
│   ├── train_reward_c_lite_v1.yaml  # Reward A minus cooldown penalty
│   ├── train_reward_c_lite_v2.yaml  # Reward A minus transaction and cooldown penalties
│   ├── train_reward_c_lite_v3.yaml  # C-lite v1 reward with thresholded-greedy training
│   ├── train_reward_c_lite_v4.yaml  # C-lite v1 reward with first-sale thresholded training
│   └── train_reward_c_lite_v5.yaml  # C-lite v1 training with first-sale-margin evaluation variants
├── docs/
│   ├── reward_freeze_v1.md # Reward A freeze document
│   ├── reward_c_lite_design.md # Reward C-lite v1 design note
│   ├── reward_c_lite_v2_design.md # Reward C-lite v2 design note
│   ├── reward_c_lite_v3_thresholded_training_design.md # Thresholded training design note
│   ├── model_freeze_reward_c_lite_v1.md # Pending C-lite v1 freeze template
│   └── training_plan_v1.md # Build/train plan for Reward A
├── notebooks/
│   └── explore_data.ipynb  # Interactive data exploration
├── scripts/
│   ├── smoke_test_environment.py  # Manual console smoke runner for environment rollouts
│   ├── reward_sanity_checks.py    # Reward A identity and scale checks
│   ├── check_reward_scale.py      # Reward scale diagnostics
│   ├── train_dqn_reward_a_debug.py  # Small Reward A DQN debug run
│   ├── train_dqn_reward_a_full.py   # Config-driven full DQN training run
│   ├── evaluate_reward_a_baselines.py  # Config-driven DQN vs baseline evaluation
│   ├── inspect_reward_a_policy_behavior.py  # Config-driven behavior inspection
│   ├── reward_c_lite_sanity_checks.py  # Reward C-lite v1 synthetic checks
│   ├── reward_c_lite_v2_sanity_checks.py  # Reward C-lite v2 synthetic checks
│   └── write_run_manifest.py  # Reproducibility manifest writer
├── runs/
│   ├── train_reward_a_v1/  # Local Reward A v1 model, metrics, baselines, behavior summaries
│   ├── train_reward_a_v2_full/  # Reward A v2 full-run outputs
│   ├── train_reward_a_v3_full/  # Reward A v3 full-run outputs
│   ├── train_reward_c_lite_v1_full/  # Reward C-lite v1 outputs
│   ├── train_reward_c_lite_v2_full/  # Reward C-lite v2 outputs
│   ├── train_reward_c_lite_v3_full/  # C-lite v3 thresholded-training outputs
│   ├── train_reward_c_lite_v4_full/  # C-lite v4 first-sale-thresholded outputs
│   └── train_reward_c_lite_v5_full/  # C-lite v5 first-sale evaluation outputs
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
│   ├── test_env_smoke.py   # Smoke/unit checks for traversal, actions, tax accounting
│   ├── test_reward_c_lite.py  # Reward C-lite v1 checks
│   ├── test_reward_c_lite_v2.py  # Reward C-lite v2 checks
│   ├── test_reward_c_lite_v3_thresholded_training.py  # Thresholded training checks
│   └── test_baseline_terminal_tax_metrics.py  # Terminal liquidation tax metric checks
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
- `configs/individual_tax_profiles_v1.yaml`
- `data/tax_profiles/README.md`

Currently included individual scenarios:
- `tax_free`
- `low_income_individual`
- `mass_affluent_individual`
- `high_income_individual`
- `top_bracket_individual`

These are thesis simulation profiles, not a full legal/tax engine.
Training and evaluation configs can reference the YAML profile file by
`tax_profile.config_path` plus `tax_profile.profile_name`.

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

### 11) Train a configured DQN reward experiment

Debug run:

```powershell
python scripts/train_dqn_reward_a_debug.py
```

Full runs use the config passed by `--config`:

```powershell
python scripts/train_dqn_reward_a_full.py --config configs/train_reward_a_v3.yaml
python scripts/train_dqn_reward_a_full.py --config configs/train_reward_c_lite_v1.yaml
python scripts/train_dqn_reward_a_full.py --config configs/train_reward_c_lite_v2.yaml
python scripts/train_dqn_reward_a_full.py --config configs/train_reward_c_lite_v3.yaml
python scripts/train_dqn_reward_a_full.py --config configs/train_reward_c_lite_v4.yaml
python scripts/train_dqn_reward_a_full.py --config configs/train_reward_c_lite_v5.yaml
```

Main local outputs for each run:
- `final_model.pt`
- `best_validation_model.pt`
- `best_validation_summary.txt`
- `episode_splits.csv`
- `train_metrics.csv`
- `eval_metrics.csv`
- `train_episode_rollouts.csv`
- `validation_episode_rollouts.csv`
- `training_summary.txt`

Best-model selection remains based on validation
`mean_final_after_tax_total_value`, not the penalized training reward.

Training exploitation policy is config-driven:

- default `greedy`: use the raw DQN argmax action during exploitation steps.
- `thresholded_greedy`: sell only when the best non-hold Q-value exceeds the
  hold Q-value by the configured margin.
- optional `training.thresholded_greedy.first_sale_margin`: use a larger margin
  before the first discretionary sale, then revert to the normal margin.

### 12) Evaluate baselines

```powershell
python scripts/evaluate_reward_a_baselines.py --config configs/train_reward_c_lite_v5.yaml
```

This reads the selected training config, loads `best_validation_model.pt` when
available, and evaluates validation and test episodes only.

Policies evaluated:
- `trained_dqn_greedy`
- `trained_dqn_thresholded_margin_0p001`
- `trained_dqn_thresholded_margin_0p005`
- `trained_dqn_thresholded_margin_0p010`
- `trained_dqn_thresholded_margin_0p015`
- `trained_dqn_thresholded_margin_0p020`
- `hold_to_terminal`
- `sell_immediately`
- `sell_half_then_hold`
- `sell_quarters_over_time`
- `random_policy`

Tax-transition baseline policies are intentionally excluded because episodes
already end at the one-year tax threshold.

Configs may also enable first-sale-margin DQN variants under
`evaluation.first_sale_margin_policies`. These policies apply a larger
threshold only before the first discretionary sale, then use the normal
thresholded-greedy margin for later sell decisions.

Episode-level tax metrics include both discretionary sales and automatic
terminal liquidation. This matters for `hold_to_terminal`, where realized tax is
created only by terminal liquidation.

Main local outputs under the run directory:
- `baselines/baseline_config_used.yaml`
- `baselines/baseline_episode_metrics.csv`
- `baselines/baseline_step_rollouts.csv`
- `baselines/baseline_summary_by_policy.csv`
- `baselines/baseline_evaluation_summary.txt`

### 13) Inspect learned policy behavior

```powershell
python scripts/inspect_reward_a_policy_behavior.py --config configs/train_reward_c_lite_v5.yaml
```

This is a build-and-train behavior inspection only. It reads the existing
baseline CSV outputs and does not reload the model, rerun the environment, or
recompute baseline evaluation.

Main local outputs under the run directory:
- `behavior_inspection/behavior_summary.txt`
- `behavior_inspection/dqn_action_distribution.csv`
- `behavior_inspection/dqn_first_cut_summary.csv`
- `behavior_inspection/dqn_episode_behavior.csv`
- `behavior_inspection/policy_behavior_comparison.csv`
- `behavior_inspection/cooldown_penalty_summary.csv`
- `behavior_inspection/transaction_penalty_summary.csv`
- `behavior_inspection/early_selling_summary.csv`

### 14) Write a reproducibility manifest

```powershell
python scripts/write_run_manifest.py --config configs/train_reward_c_lite_v5.yaml
```

Main local output:
- `runs/train_reward_c_lite_v5_full/run_manifest.json`

### 15) Build quantitative analysis outputs

```powershell
python scripts/build_quant_analysis_phase_1_3.py --config configs/train_reward_c_lite_v5.yaml
python scripts/build_quant_analysis_phase_4_11.py --config configs/train_reward_c_lite_v5.yaml
```

Step 3B reports supplementary risk-adjusted diagnostics for the frozen
C-lite v5 policy universe. The main Step 3B Sharpe outputs are after-tax:

- `EAAT_Sharpe` uses terminal after-tax wealth and an exposure-adjusted daily
  stock/cash return path. After a sale, liquidated after-tax proceeds earn the
  daily risk-free rate.
- `TA_EAAT_Sharpe` uses sale-level after-tax tranches. Each tranche is
  annualized from original purchase day to sale day and paired with stock
  volatility over the same holding window.
- Pre-tax values are not policy-table metric columns. They appear only in audit
  notes as tax-transformation checks and counterfactual comparisons.

Main Step 3B outputs:
- `quant_analysis/step3b_policy_eaat_sharpe_metrics.md`
- `quant_analysis/step3b_episode_eaat_sharpe_metrics.csv`
- `quant_analysis/step3b_episode_tranche_records.csv`
- `quant_analysis/step3b_median_episode_eaat_verification.md`
- `quant_analysis/step3b_one_case_eaat_verification.md`
- `quant_analysis/step3b_eaat_sharpe_metrics_notes.txt`

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
- Reward A is implemented as the change in total after-tax position value:
  cumulative realized after-tax PnL plus after-tax liquidation value of
  remaining inventory.
- terminal unsold inventory is liquidated with long-term tax treatment.
- automatic terminal liquidation is not penalized by Reward C-lite transaction
  or cooldown penalties.
- Reward C-lite v1 is Reward A minus a cooldown penalty for repeated
  discretionary sales inside the configured cooldown window.
- Reward C-lite v2 is Reward A minus both a transaction penalty for every
  discretionary executed sale and the cooldown penalty for clustered sales.
- Reward C-lite v2 uses `training.discount_factor_gamma: 1.0` and exploration
  probabilities `[0.65, 0.20, 0.10, 0.04, 0.01]`.
- Reward C-lite v3, v4, and v5 use the C-lite v1 environment reward and vary
  the training or evaluation policy configuration rather than `reward.version`.
- Reward A, Reward C-lite v1, and Reward C-lite v2 are selected through
  `reward.version` in the training config.

Reset `info` includes at least:
- `episode_id`, `current_row_ptr`, `date`
- `sold_fraction`, `remaining_fraction`
- `cum_realized_pre_tax_pnl`, `cum_realized_after_tax_pnl`
- `tax_profile_name`

Step `info` includes at least:
- action bookkeeping (`action`, requested/executed fraction, sold/remaining)
- tax/accounting fields (`tax_regime`, `applicable_tax_rate`, sale increments, cumulative totals)
- Reward A fields (`previous_after_tax_total_value`, `after_tax_total_value`,
  `after_tax_liquidation_value_remaining`, `reward_A`)
- Reward C-lite fields when configured (`reward_C_lite`, `reward_C_lite_v2`,
  `transaction_penalty`, `cooldown_penalty`, `days_since_last_sale`,
  `sale_count`)
- terminal liquidation fields when applicable
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
