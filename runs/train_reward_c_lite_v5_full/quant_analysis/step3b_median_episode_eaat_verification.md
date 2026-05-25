# Step 3B Median Episode EAAT Verification

Selected case: `test / trained_dqn_first_sale_margin_0p070_normal_0p020 / FTNT_2024-08-05`. It is the episode closest to the median `EAAT_Sharpe` for this split-policy pair: median `1.449197856782`, selected `1.449197856782`.

Only the after-tax side is used by the Step 3B EAAT and TA-EAAT metrics. The pre-tax side below is included only as an audit check of the tax transformation and as a counterfactual comparison.

## Tranche Tax Check

| tranche | type | weight | pre-tax increment | tax rate | tax paid | after-tax increment | after-tax return used |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | terminal_liquidation | 1 | 0.736956131876 | 0.1018 | 0.075022134225 | 0.661933997651 | 0.661933997651 |

Tax transformation:

- `after_tax_increment = pre_tax_increment - max(pre_tax_increment, 0) * tax_rate`.
- `after_tax_return_used = after_tax_increment / tranche_weight`.

## Post-Tax Calculation Used In Step 3B

- Terminal trading day: `250`.
- Daily risk-free rate: `0.000155649863`.
- EAAT terminal after-tax wealth: `1 * (1 + 0.661933997651) * (1 + daily_rf) ** 0` = `1.661933997651`.
- EAAT annualized after-tax return: `terminal_wealth ** (252 / 250) - 1` = `0.66870159984`.
- Daily stock-return sample std: `0.027328551418`.
- EAAT annualized volatility: `daily_std * sqrt(252)` = `0.433827304462`.
- EAAT Sharpe: `(annualized_after_tax_return - 0.04) / annualized_volatility` = `1.449197856782`.
- TA-EAAT annualized after-tax return: `(1 + after_tax_return) ** (252 / 250) - 1` = `0.66870159984`.
- TA-EAAT Sharpe: `(TA_annualized_after_tax_return - 0.04) / TA_annualized_volatility` = `1.449197856782`.

## Pre-Tax Counterfactual Check

- Pre-tax terminal wealth under the same sale timing: `1 * (1 + 0.736956131876) * (1 + daily_rf) ** 0` = `1.736956131876`.
- Pre-tax annualized return under the same sale timing: `pre_tax_terminal_wealth ** (252 / 250) - 1` = `0.744645364846`.
- Pre-tax Sharpe counterfactual using the same exposure volatility: `(pre_tax_annualized_return - 0.04) / annualized_volatility` = `1.624253147736`.
- Pre-tax TA-EAAT annualized return under the same sale timing: `(1 + pre_tax_return) ** (252 / 250) - 1` = `0.744645364846`.
- Pre-tax TA-EAAT Sharpe counterfactual: `(pre_tax_TA_return - 0.04) / TA_annualized_volatility` = `1.624253147736`.

The difference between the pre-tax and post-tax values is the tax paid on the realized gain. The policy tables report the post-tax EAAT and TA-EAAT values.

## Checks

| Check | Result |
|---|---:|
| selected_episode_is_policy_median_EAAT_Sharpe | PASS |
| tax_formula_matches_after_tax_increment | PASS |
| after_tax_return_matches_tranche_file | PASS |
| post_tax_EAAT_terminal_wealth_matches_csv | PASS |
| post_tax_EAAT_annualized_return_matches_csv | PASS |
| post_tax_EAAT_Sharpe_matches_csv | PASS |
| post_tax_TA_EAAT_annualized_return_matches_csv | PASS |
| post_tax_TA_EAAT_Sharpe_matches_csv | PASS |
