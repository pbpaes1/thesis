# Step 3B One-Case EAAT Verification

This audit verifies why the TA-EAAT metric is extremely large for `validation / random_policy / JD_2022-03-14`.

The tranche file contains `after_tax_return`; there is no pre-tax tranche-return metric in the Step 3B EAAT output. The source unrealized return below is shown only to verify the tax transformation.

Core recomputation:

- After-tax tranche return: `0.397839821207`.
- EAAT terminal wealth: `(1 + after_tax_return) * (1 + daily_rf) ** 249` = `1.453074795762`.
- TA-EAAT annualized return: `(1 + after_tax_return) ** (252 / 2) - 1` = `2.126343640598e+18`.

The very large TA-EAAT value is therefore caused by annualizing a 39.784% after-tax return over only 2 trading days, not by using pre-tax returns.

## Audit Values

| Field | Value |
|---|---:|
| split | validation |
| policy_name | random_policy |
| episode_id | JD_2022-03-14 |
| ticker | JD |
| simulated_purchase_date | 2022-03-14 |
| trigger_date | 2022-03-16 |
| sale_date | 2022-03-16 |
| tax_transition_date | 2023-03-14 |
| sale_trading_day | 2 |
| cash_days_to_terminal | 249 |
| source_unrealized_return_for_tax_check | 0.492315086261 |
| applicable_tax_rate | 0.1919 |
| realized_after_tax_increment | 0.397839821207 |
| sale_weight | 1 |
| after_tax_return_from_rollout_increment | 0.397839821207 |
| after_tax_return_in_tranche_file | 0.397839821207 |
| expected_after_tax_return_from_tax_formula | 0.397839821207 |
| expected_EAAT_terminal_wealth | 1.453074795762 |
| EAAT_terminal_after_tax_wealth_in_csv | 1.453074795762 |
| expected_TA_EAAT_annualized_after_tax_return | 2.126343640598e+18 |
| TA_EAAT_annualized_after_tax_return_in_csv | 2.126343640598e+18 |
| TA_EAAT_annualized_volatility_in_csv | 3.624015112422 |
| expected_TA_EAAT_Sharpe | 5.867369684275e+17 |
| TA_EAAT_Sharpe_in_csv | 5.867369684275e+17 |

## Checks

| Check | Result |
|---|---:|
| after_tax_return_matches_tax_formula | PASS |
| after_tax_return_matches_rollout_increment | PASS |
| eaat_terminal_wealth_matches_csv | PASS |
| ta_annualized_return_matches_csv | PASS |
| ta_sharpe_matches_csv | PASS |
