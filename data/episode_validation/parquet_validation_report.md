# Parquet Validation Report

## 1. Dataset overview

- Input file: `data\episodes\drl_episodes.parquet`
- Row count: **973,658**
- Column count: **54**
- Date-like columns inferred: ['date', 'simulated_purchase_date', 'trigger_date', 'tax_transition_date']
- Main per-step date column: `date`
- Main date selection rationale: Selected 'date' as the main time-step column (score=130.00; exact name match 'date'; contains 'date'; datetime dtype; non-null ratio=1.000).
- Strict numeric columns: 47
- Numeric-like non-numeric columns (heuristic): None

## 2. Column classification summary

- `observation_feature`: 40 columns
- `tax_variable`: 6 columns
- `identifier`: 3 columns
- `simulator_only`: 2 columns
- `excluded`: 3 columns
- Full classification CSV: `parquet_column_classification.csv`
- Full data dictionary CSV: `parquet_data_dictionary.csv`

## 3. Key validation checks

| Check | Status | Notes | Diagnostics |
|---|---|---|---|
| duplicate_episode_date | pass | No duplicate (episode_id, date) rows found. | `duplicate_episode_date_rows.csv` |
| episode_date_order | pass | All episodes have non-decreasing dates in dataset row order. | `episode_date_order_issues.csv` |
| holding_period_monotonicity | pass | Holding-period checks passed for columns: ['holding_period_days'] | `holding_period_issues.csv` |
| critical_missingness | pass | Critical missingness summary generated for 18 columns; critical=0, minor=0. | `critical_missingness_summary.csv` |
| trigger_date_consistency | pass | Trigger date consistency checks passed. | `trigger_date_issues.csv` |
| tax_transition_consistency | pass | Tax transition consistency checks passed. | `tax_transition_issues.csv` |
| unrealized_gain_consistency | pass | Best unrealized-gain interpretation: fraction_(current-purchase)/purchase using purchase='simulated_purchase_price' and current='adj_close'. MAE=5.83353e-17, valid_rows=973658. This indicates the stored values are fractional returns (for display as percentage terms, multiply by 100). | `unrealized_gain_diagnostics.csv, unrealized_gain_formula_scores.csv` |

## 4. Issues found

- No blocking issues detected by automated checks.

## 5. Candidate fixes or manual review items

- Confirm semantic intent for columns classified as `excluded` before training.
- Confirm that dashboard/notebook display logic matches the detected unrealized-gain scale (fraction vs percentage points).
- If `first_row_after_trigger` issues appear, verify whether rows before first valid feature vector were intentionally dropped.
- If any missingness appears in critical fields, decide whether to impute, drop, or regenerate episodes upstream.

## 6. Overall assessment

- Automated checks passed for the available fields.