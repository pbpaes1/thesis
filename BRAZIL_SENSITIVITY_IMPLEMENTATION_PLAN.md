# Brazil-Inspired Sensitivity Analysis — Implementation Plan

## 1. Objective

Implement a versioned sensitivity-analysis family that asks how the liquidation
policy changes when the original U.S. tax setting is replaced by:

- a flat 15% tax on positive realized equity gains; and
- a fixed gross fixed-income rate earned on full after-tax sale proceeds until
  the existing episode terminal date, with tax charged on the interest earned.

This is a stylized Brazil-inspired economic treatment, not a simulation of the
Brazilian legal or currency environment. The U.S.-listed stock universe, USD
numeraire, historical market paths, chronological data split, action space, and
one-year episode horizon remain unchanged.

The original U.S. experiment must remain reproducible and unchanged. All new
logic, schemas, configurations, runs, and results must be versioned separately.

## 2. Experimental contract to freeze before coding

Create `docs/brazil_sensitivity_design_v1.md` in the code repository and freeze
the following rules.

### 2.1 Equity tax

- Apply a rate of 15% to positive realized gains.
- Apply the same rate on every date; there is no short-/long-term transition.
- Do not grant a tax credit for realized losses.
- Apply the same rule to discretionary sales and mandatory terminal liquidation.
- Allocate cost basis proportionally to the fraction of the original position
  sold.

For a sale at time `t`:

```text
gain_t = gross_sale_proceeds_t - allocated_cost_basis_t
tax_t = 0.15 * max(gain_t, 0)
cash_deposit_t = gross_sale_proceeds_t - tax_t
```

The deposit must include recovered principal as well as the after-tax gain.

### 2.2 Cash account

- Initialize cash at zero.
- Deposit full after-tax equity-sale proceeds immediately after every
  discretionary sale, creating a separate fixed-income lot for each sale.
- Compound every open fixed-income lot over elapsed market days using a
  252-market-day year.
- Leave unsold inventory invested in the stock; it earns no cash interest.
- Tax each lot's positive interest at redemption: 22.5% when held for 180
  calendar days or less and 20% when held for more than 180 calendar days.
- At the terminal date, first value and redeem the existing fixed-income lots,
  including interest tax, then liquidate remaining equity inventory. Terminal
  equity-liquidation proceeds earn no interest and incur no fixed-income tax.
- Do not allow borrowing, withdrawals, or a negative cash balance.

For an annual effective gross rate `r`, a cash lot created with principal `C_i`
and held for `m_i` market-day intervals has:

```text
gross_value_i = C_i * (1 + r)^(m_i / 252)
interest_i = gross_value_i - C_i

fixed_income_tax_rate_i = 22.5%  if calendar_days_held_i <= 180
                          20.0%  otherwise

fixed_income_tax_i = fixed_income_tax_rate_i * max(interest_i, 0)
after_tax_cash_value_i = gross_value_i - fixed_income_tax_i
```

Market-day intervals determine compounding, while calendar days determine the
fixed-income tax bracket. Separate lots are required because proceeds from
different sale dates can fall into different tax brackets at the common terminal
date. Interest compounds gross inside each lot and is taxed when the lot is
valued or redeemed. The convention must be recorded in every run configuration
and manifest.

### 2.3 Wealth and reward

Define normalized total after-tax wealth as:

```text
wealth_t = cash_balance_t
         + after_tax_liquidation_value_of_remaining_inventory_t
```

Here, `cash_balance_t` is the sum of the after-tax liquidation values of all
fixed-income lots at time `t`, using each lot's accrued gross interest and its
current calendar-day tax bracket.

The base economic reward is:

```text
base_reward_t = wealth_t - wealth_(t-1)
```

Continue to apply the selected C-lite behavioral penalty after computing this
base reward. Model selection must continue to use validation final wealth, not
the shaped cumulative reward.

The implementation must satisfy the telescoping identity:

```text
sum(base_reward_t) = final_wealth - initial_wealth
```

### 2.4 Time-state semantics

The episode endpoint is a terminal evaluation horizon, not a tax-transition
date. Replace the agent-visible meaning of:

```text
days_to_tax_transition_norm
```

with:

```text
time_to_terminal_horizon_norm
```

Do this in a new state freeze and a new scenario dataset. Do not rename or edit
the v1 U.S. state contract or its historical data.

## 3. Planned repository changes

### 3.1 Design and documentation

Add:

```text
docs/brazil_sensitivity_design_v1.md
docs/brazil_sensitivity_model_freeze_v1.md
```

The design document should contain the accounting equations, timing convention,
scenario grid, invariants, fixed decision margins, and interpretation limits.
The model-freeze document should be completed after training and validation-based
checkpoint selection.

Update `README.md` with a short section that identifies the sensitivity study,
its entry-point commands, and its run directories. Do not rewrite the historical
Reward A or C-lite design documents.

### 3.2 Scenario configuration

Extend the configuration contract with an explicit economic-scenario block,
rather than representing the treatment only by setting the existing short- and
long-term rates equal:

```yaml
economic_scenario:
  name: brazil_inspired_v1
  equity_tax:
    regime: flat_positive_gains
    rate: 0.15
    loss_credit: false
  cash_account:
    enabled: true
    annual_gross_rate: <scenario value>
    compounding: effective_annual_market_252
    market_days_per_year: 252
    interest_tax:
      regime: holding_period_tiers
      tiers:
        - max_calendar_days: 180
          rate: 0.225
        - max_calendar_days: null
          rate: 0.20
  terminal_horizon: episode_end
  currency: USD
  fx_conversion: false
```

Add one immutable training YAML per cash-rate treatment, for example:

```text
configs/brazil_sensitivity_rate_0700_v1.yaml
configs/brazil_sensitivity_rate_1000_v1.yaml
configs/brazil_sensitivity_rate_1050_v1.yaml
configs/brazil_sensitivity_rate_1200_v1.yaml
configs/brazil_sensitivity_rate_1375_v1.yaml
configs/brazil_sensitivity_rate_1500_v1.yaml
```

The four-digit suffix is the annual gross fixed-income rate in basis points. The
scenario grid and its empirical interpretation are:

| Scenario | Gross annual rate | Basis |
| --- | ---: | --- |
| Hypothetical extreme easing | 7.00% | Deliberately below the current Focus horizon; sensitivity bound, not a Focus forecast |
| Focus longer-term | 10.00% | Focus median for end-2029 |
| Focus medium-term | 10.50% | Focus median for end-2028 |
| Focus gradual easing | 12.00% | Focus median for end-2027 |
| Focus current/high-rate | 13.75% | Current Selic and Focus median for end-2026 |
| Observed pre-easing peak | 15.00% | Selic observed from June 2025 through March 2026 |

The Focus anchors refer to the Banco Central do Brasil Focus survey released on
14 September 2026. The 7% rate must always be described as a hypothetical
extreme-easing scenario, not as a market forecast. The 15% rate is a historical
observed anchor, not a current Focus projection.

Every configuration must have a unique output directory and must reference the
new state freeze.

Retain the existing U.S. C-lite v5 run as the control. None of the six
Brazil-inspired rate treatments should be treated as the control because their
tax and fixed-income accounting differ from the original U.S. specification.

### 3.3 Tax-profile loader

Change `src/config/tax_profiles.py` to:

- validate the new `flat_positive_gains` regime;
- require a flat rate in `[0, 1]`;
- reject contradictory short-/long-term fields in flat mode;
- validate the gross cash rate, 252-market-day compounding convention, and the
  180-calendar-day fixed-income tax tiers;
- preserve all existing tax-profile behavior when the scenario block is absent;
- return a normalized scenario object to training, evaluation, and the
  environment.

If the project keeps tax profiles in a separate YAML/JSON artifact, create a v2
file rather than editing `individual_tax_profiles_v1.*`.

### 3.4 Scenario episode data and state freeze

Change the data pipeline so it can create a new scenario parquet without
changing the original episode parquet:

```text
data/episodes/drl_episodes_brazil_v1.parquet
```

For every row, compute time to the actual episode terminal date and expose its
normalized version as `time_to_terminal_horizon_norm`. Confirm that the value is
monotonically non-increasing and reaches zero at the terminal observation.

Add:

```text
data/freeze/v2/allowed_state_columns_v2.json
data/freeze/v2/excluded_columns_v2.json
data/freeze/v2/state_freeze_v2_summary.md
```

The v2 freeze should preserve the other 35 v1 inputs and substitute only the
time variable unless an explicit state audit finds another necessary change.
Before freezing, confirm how remaining inventory is represented in the current
observation. If it is not agent-visible, document why it is intentionally
omitted or add a normalized remaining-inventory field and treat that as a
separate, explicitly justified state change.

Likely files to modify:

- `src/generate_episodes.py` for terminal-horizon fields and scenario output;
- `src/validate_parquet.py` for terminal-horizon validation;
- `src/freeze_columns_v1.py`, generalized into a version-aware freeze builder,
  or a new v2 freeze script that leaves v1 untouched.

### 3.5 Environment accounting

Extend `src/environment/tax_aware_env.py` behind the new scenario configuration.
Do not fork the complete environment unless backward compatibility becomes
unmanageable.

Add internal state for:

- fixed-income lots, each containing deposit date, principal, and market-day age;
- aggregate gross and after-tax cash balances;
- cumulative gross cash interest;
- cumulative fixed-income interest tax;
- cumulative equity tax;
- cumulative gross proceeds;
- previous observation date;
- normalized total after-tax wealth.

At each step, use this order:

1. Advance every existing fixed-income lot by one market-day interval and
   calculate its gross accrued interest.
2. Read and clamp the requested sale to remaining inventory.
3. Calculate gross proceeds and proportional basis.
4. Apply 15% tax only to a positive gain.
5. Deposit net equity-sale proceeds into a new fixed-income lot with zero accrued
   market days.
6. Value remaining inventory at the current stock price under immediate flat-tax
   liquidation.
7. Value each fixed-income lot after tax, using market-day compounding and its
   calendar-day holding-period tax tier.
8. Compute total after-tax wealth and the base reward change.
9. Apply any C-lite penalty to obtain the training reward.
10. On the terminal row, redeem all existing fixed-income lots after their final
    accrual and tax, liquidate all remaining equity inventory, and do not invest
    or accrue interest on the terminal equity proceeds.

Keep the existing U.S. accounting path byte-for-byte equivalent in outputs for
the same configurations and seeds. Scenario-specific branches must be selected
by explicit configuration, never inferred from equal tax rates.

Expose these values in the environment `info` record and rollout outputs:

```text
cash_balance
cash_balance_gross
cash_interest_gross_step
cash_interest_gross_cumulative
fixed_income_tax_step
fixed_income_tax_cumulative
gross_sale_proceeds
equity_tax_step
equity_tax_cumulative
remaining_inventory_value
after_tax_liquidation_value
total_after_tax_wealth
```

### 3.6 Training and model selection

Generalize `scripts/train_dqn_reward_a_full.py` only where needed to pass the
scenario configuration and new state freeze to the environment. Preserve the
existing DQN architecture and primary hyperparameters so the economic treatment,
not a redesigned model, drives the comparison.

For every cash-rate scenario:

1. Use the same chronological train/validation/test episode assignments as the
   control.
2. Retrain the DQN from scratch.
3. Use the existing reproducibility seed policy; use multiple seeds if compute
   permits and report their dispersion.
4. Select checkpoints on validation mean final after-tax wealth.
5. Apply the margins selected in the original thesis: `0.070` before the first
   sale and `0.020` after the first sale.
6. Do not recalibrate or search over margins separately for the Brazil-inspired
   scenarios.
7. Freeze the validation-selected checkpoint before opening test results.
8. Evaluate only that checkpoint with the fixed `0.070/0.020` decision rule on
   the test split.

The margins modify action selection from the trained Q-values; they are not DQN
training parameters. The model weights are retrained for each cash-rate scenario,
while the decision margins are transferred unchanged from the original thesis.

An optional fixed-policy counterfactual may revalue the original U.S. DQN under
the new accounting, but label it clearly and keep it out of claims about policy
adaptation.

### 3.7 Baselines and rollout schema

Extend `scripts/evaluate_reward_a_baselines.py` so the selected DQN and the
essential economic benchmarks use the same scenario accounting. Retain:

- hold to terminal;
- immediate full sale;
- existing partial/staged-sale policies;
- random policy;
- the single validation-selected DQN checkpoint using a `0.070` first-sale
  margin and `0.020` subsequent-sale margin.

Do not report raw-greedy or alternative-margin DQN variants in the main test
comparison. The transferred thesis margins are fixed before the Brazil-inspired
runs and are not re-optimized from those scenarios.

The hold baseline should earn no cash interest before terminal liquidation. The
immediate-sale baseline should receive the longest cash holding period. These
two cases are important accounting anchors.

Add episode-level output fields for:

- final total after-tax wealth;
- excess wealth versus hold to terminal;
- ending cash balance;
- cumulative gross cash interest;
- cumulative fixed-income interest tax;
- cumulative net cash interest;
- cumulative equity tax;
- number and fraction of discretionary sales;
- first-sale date and days from episode start;
- remaining fraction immediately before terminal liquidation;
- scenario name and annual gross fixed-income rate;
- seed, checkpoint, and fixed-margin identifiers.

### 3.8 Sensitivity-analysis builder

Add:

```text
scripts/build_brazil_sensitivity_analysis.py
```

It should consume frozen run directories and create a separate analysis tree,
not modify the original U.S. quantitative-analysis outputs. Required tables and
figures:

- performance by cash rate and policy;
- mean and median excess wealth versus hold;
- paired episode-level win/tie/loss counts;
- paired bootstrap confidence intervals;
- first-sale timing and sale-frequency distributions;
- gross cash interest, fixed-income tax, net cash interest, and equity-tax
  decomposition;
- policy action shares by cash-rate scenario;
- performance and behavior changes across the rate grid;
- optional sector, market-cap, and episode-gain heterogeneity using the existing
  analysis utilities.

The primary comparison should be each retrained scenario DQN, using the fixed
`0.070/0.020` decision margins, against hold-to-terminal under the same scenario.
Cross-rate comparisons should use the same test episodes and report paired
differences.

Suggested output structure:

```text
runs/brazil_sensitivity_v1/
├── analysis_manifest.json
├── tables/
├── figures/
└── summary.md
```

Individual training runs should remain separate, such as
`runs/brazil_sensitivity_rate_1200_v1/`.

### 3.9 Reproducibility manifest

Extend `scripts/write_run_manifest.py` to record:

- scenario name and version;
- flat equity-tax rate;
- annual gross fixed-income rate;
- 252-market-day compounding convention;
- fixed-income tax tiers and calendar-day holding-period convention;
- state-freeze version and hash;
- scenario episode-parquet hash;
- train/validation/test episode-list hashes;
- random seeds;
- selected checkpoint and fixed transferred margins;
- repository commit and working-tree status.

## 4. Test plan

Add focused tests before any full training run, preferably in:

```text
tests/test_brazil_flat_tax_cash_account.py
tests/test_brazil_scenario_config.py
tests/test_brazil_terminal_horizon_state.py
```

### 4.1 Unit accounting tests

- Same positive gain before and after the former U.S. transition date produces
  the same 15% tax.
- A zero or negative gain produces zero tax and no tax credit.
- A sale deposits gross proceeds less equity tax, including recovered basis.
- Cash compounds correctly over known market-day intervals using 252 periods per
  year.
- Interest earned by a lot held for 180 calendar days is taxed at 22.5%.
- Interest earned by a lot held for 181 calendar days is taxed at 20%.
- Fixed-income tax applies only to positive interest, never to deposited
  principal.
- A partial sale earns interest only on the sold portion's proceeds.
- Unsold stock earns no cash interest.
- Multiple partial sales preserve separate effective holding periods through
  separate lots, including cases where the lots fall into different tax tiers.
- Immediate full sale matches an independently calculated closed-form terminal
  cash value.
- Hold-to-terminal receives no pre-terminal cash interest.
- Terminal liquidation proceeds receive no post-terminal interest.
- Selling all inventory early prevents any later terminal inventory sale.
- The base reward telescopes exactly to final wealth minus initial wealth within
  numerical tolerance.
- C-lite penalties change shaped reward but not final economic wealth.

### 4.2 Data and state tests

- Terminal-horizon values derive from episode dates, not the old tax-transition
  field.
- Time to terminal is non-increasing and is zero on the last row.
- The v2 allowed-state schema contains no U.S. tax-transition variable.
- No simulator-only cash, tax, or future terminal-price fields leak into the
  observation unless deliberately included and documented.
- Scenario episode IDs and split assignments match the control universe.

### 4.3 Regression tests

- Existing U.S. environment tests pass unchanged.
- A fixed existing U.S. config produces unchanged accounting outputs.
- Reward A and C-lite identities remain valid in U.S. mode.
- Baseline evaluator results are unchanged for a frozen U.S. fixture.
- Missing scenario fields fail with clear validation errors rather than silently
  falling back to another regime.

## 5. Execution phases and gates

### Phase 0 — Freeze the design

Deliverables:

- approved accounting/timing convention;
- frozen six-rate grid and documented provenance for each rate;
- frozen interpretation and exclusions;
- documented primary hypotheses and metrics.

Gate: no environment coding until ambiguous timing rules and rate values are
resolved.

### Phase 1 — Configuration and pure accounting

Implement scenario parsing and small pure functions for equity tax, 252-day cash
growth, holding-period fixed-income tax, and sale proceeds. Add unit tests with
hand-calculated fixtures.

Gate: all new pure-function tests and all existing tests pass.

### Phase 2 — Environment integration

Integrate lot-level fixed-income accounting and the flat-equity-tax path into
`TaxAwareEnv`; add info fields, terminal handling, reward telescoping checks, and
U.S. regression fixtures.

Gate: accounting tests, terminal invariants, and backward-compatibility tests
pass.

### Phase 3 — Dataset and state v2

Build the scenario parquet and v2 state freeze, validate terminal-horizon
semantics, and confirm identical episode/split membership.

Gate: validation report has no unresolved schema, ordering, missingness, or
look-ahead failures.

### Phase 4 — Smoke runs

Run manual trajectories and a small debug DQN at the lowest, middle, and highest
cash rates. Inspect hold, immediate sale, and staggered partial-sale examples by
hand.

Gate: accounting anchors match independent calculations and training remains
numerically stable.

### Phase 5 — Full training and checkpoint selection

Train one model family per rate, select checkpoints using validation episodes,
apply the fixed `0.070/0.020` margins inherited from the original thesis, and
write manifests. Do not inspect test metrics while making checkpoint choices.

Gate: every scenario has a frozen config, checkpoint, validation summary,
fixed-margin record, and reproducibility manifest.

### Phase 6 — Locked test evaluation

Evaluate all frozen policies and baselines on the common test set. Generate
rollouts with the expanded accounting fields.

Gate: run completeness checks pass and every scenario contains the same test
episode IDs.

### Phase 7 — Analysis and thesis integration

Build the cross-rate tables and figures, write the sensitivity-results section,
and update limitations to state that this is a USD Brazil-inspired parameter
experiment with stylized equity and fixed-income taxes, but without broader
Brazilian legal detail or FX risk.

Gate: all reported numbers trace to a run manifest and a generated table; the
original U.S. results remain unchanged.

## 6. Recommended implementation order

```text
design freeze
  -> config validation
  -> pure accounting functions
  -> environment integration
  -> state/data v2
  -> unit and regression tests
  -> smoke trajectories
  -> debug training
  -> full per-rate training
  -> validation checkpoint selection
  -> apply fixed 0.070/0.020 margins
  -> locked test evaluation
  -> cross-rate analysis
  -> thesis text and figures
```

## 7. Key risks and controls

| Risk | Control |
| --- | --- |
| Interest is applied only to gain instead of full proceeds | Test deposits against hand-calculated principal-plus-gain examples. |
| Fixed-income tax is applied to principal | Track principal and interest separately for every lot and test both components. |
| Multiple deposits receive the wrong tax tier | Keep sale-date lots separate and use calendar days only for tax-tier selection. |
| Market-day compounding is confused with tax holding days | Store market-day age and calendar deposit date separately. |
| Terminal proceeds incorrectly earn interest | Enforce and test the step-order convention. |
| Old tax-transition meaning leaks into the new policy | Use a separate parquet and v2 state freeze. |
| Equal short-/long-term rates silently select flat mode | Require an explicit scenario regime. |
| U.S. results change after environment edits | Add frozen-fixture regression tests and run the full existing suite. |
| Brazil-scenario results influence margin choice | Freeze the transferred `0.070/0.020` margins in every config before training and do not search over alternatives. |
| Different scenarios use different episodes | Hash and compare episode and split ID lists in manifests. |
| Results are described as literal Brazilian evidence | Use “Brazil-inspired sensitivity” consistently and state exclusions. |
| Cash-rate effects are confused with tax-only effects | State that every treatment jointly applies flat tax and its assigned cash rate; retain the U.S. run only as the control. |

## 8. Definition of done

The implementation is complete when:

- the U.S. pipeline and historical artifacts remain reproducible;
- the design and final rate grid are frozen and versioned;
- all equity-tax, fixed-income-lot, and cash-account invariants have executable
  tests;
- the v2 terminal-horizon state has no tax-transition interpretation;
- one separately trained, validation-selected DQN using the fixed
  `0.070/0.020` margins exists for every rate;
- every scenario is evaluated against identical baselines and test episodes;
- manifests capture all economic, data, model, checkpoint-selection, and fixed
  margin settings;
- generated tables report wealth, excess versus hold, selling behavior, cash
  interest, taxes, and paired uncertainty;
- the thesis describes the exercise as stylized, USD-denominated, and
  Brazil-inspired rather than a full Brazilian-market simulation.

## 9. Decision still to finalize

One design input remains intentionally open in this plan:

1. Whether to run multiple DQN seeds per rate. Multiple seeds are preferable for
   robustness; a single seed is acceptable only if compute constraints are
   disclosed and the episode-level paired uncertainty analysis is retained.

All other implementation choices above are intended to be the v1 contract.
