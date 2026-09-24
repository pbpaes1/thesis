# Brazil-Inspired Sensitivity Analysis — Execution Checklist

This checklist translates [`BRAZIL_SENSITIVITY_IMPLEMENTATION_PLAN.md`](BRAZIL_SENSITIVITY_IMPLEMENTATION_PLAN.md) into a sequence of reviewable Codex tasks. Use it alongside the implementation plan: that document defines the intended experiment; this one defines how to build, calculate, and verify it without changing the original U.S. experiment.

## Goal and scope

Estimate how a liquidation policy behaves when the original U.S. tax specification is replaced by a flat 15% tax on positive equity gains and sale proceeds can earn fixed-income interest. Use the existing U.S.-listed equity episodes, USD units, and episode terminal dates. Treat the results as a **Brazil-inspired sensitivity analysis**, not as a simulation of Brazilian law, investors, or currency exposure.

The six gross annual fixed-income rates are 7%, 10%, 10.5%, 12%, 13.75%, and 15%. Every sensitivity treatment combines its assigned cash rate with the same flat equity tax. The original U.S. experiment remains the control. Do not edit or overwrite its state freeze, configs, run artifacts, or reported results.

## Working rules

- Work in the code repository under `.repo-docs-snapshot/`. Before editing, inspect its Git status and repository instructions. Keep the existing workspace documents as design references.
- Implement in small phases. Stop at each gate and summarize files changed, calculations made, outputs produced, and any unresolved issue before beginning the next phase.
- Read actual source/config/run artifacts before choosing integration points. Do not assume a path or schema from this checklist if the repository differs.
- Keep scenario code and outputs versioned and isolated from the U.S. v1 path. Select Brazil mode only through explicit configuration; equal U.S. tax rates must not implicitly activate it.
- Use deterministic, hand-calculable fixtures before full datasets or model training.
- Do not inspect test-set outcomes to choose a model, margin, configuration, or implementation variant. Lock those choices using the design and validation data first.
- If the required episode data or model artifacts are absent, report exactly which input is missing and how the established pipeline creates it. Do not silently substitute a smaller or different universe.

## Economic calculation contract

Before coding, record the precise units and timing conventions in a versioned design document. In particular, reconcile the existing environment's value convention with the new cash-account formulation so comparisons are on one basis.

For an equity sale on date `t`:

```text
gross_sale_proceeds_t = sold_shares_t * price_t
allocated_basis_t = sold_fraction_t * original_position_basis
positive_gain_t = max(gross_sale_proceeds_t - allocated_basis_t, 0)
equity_tax_t = 0.15 * positive_gain_t
cash_deposit_t = gross_sale_proceeds_t - equity_tax_t
```

For a cash lot `i` with principal `C_i`, gross annual effective rate `r`, `m_i` market-day intervals, and `d_i` calendar days held:

```text
gross_value_i = C_i * (1 + r) ** (m_i / 252)
gross_interest_i = gross_value_i - C_i
interest_tax_rate_i = 0.225 if d_i <= 180 else 0.20
interest_tax_i = interest_tax_rate_i * max(gross_interest_i, 0)
after_tax_value_i = gross_value_i - interest_tax_i
```

Keep each deposit as a separate lot. The cash balance is the sum of lot values. Unsold shares remain invested in stock and receive no cash interest. At episode end, value/redeem existing cash lots and tax their interest, then liquidate unsold equity. Terminal equity proceeds do not earn interest.

Use the implementation plan's reward definition: economic reward is the change in total after-tax wealth, with any selected C-lite penalty applied only after the base reward is computed. State explicitly how wealth is normalized and how its initial value relates to the original position basis. Report enough columns to distinguish principal, gross interest, tax liability, and net cash interest; do not label after-tax PnL as full account wealth.

Resolve and document these timing points before the environment work:

1. A sale deposit is created after that date's existing lots accrue. It earns zero interest on its creation date and starts accruing on the next market-day interval.
2. Market-day intervals control compounding; actual date differences control the 180-day interest-tax tier.
3. Specify whether nonterminal reported cash wealth includes the estimated interest-tax liability at that date, while actual interest tax is settled only at redemption. Make the reward and rollout definitions consistent with this choice, including the 180-to-181-calendar-day boundary.
4. On the terminal observation, accrue existing lots through the final interval exactly once, redeem them, and exclude terminal equity proceeds from interest.
5. Define behavior for nonpositive rates or deposits only if the configuration is intended to allow them; otherwise reject them clearly.

## Phases, tasks, and gates

### Phase 0 — Repository and input audit

**Tasks**

- Inspect repository instructions, Git status, source tree, configs, tests, current U.S. run manifest, environment implementation, and current state freeze.
- Locate the episode parquet and split artifacts. Check whether the complete original episode IDs and required date/price/basis fields are locally available.
- Identify how the original trainer, evaluator, manifest writer, and quantitative-analysis scripts are invoked.
- Compare the code snapshot's actual state with both project-level planning documents.

**Deliverable**

An audit note listing the existing entry points, exact reusable inputs, absent/generated inputs, current data and split counts, proposed files to add/change, and risks or contradictions. No model training.

**Gate**

The source of each required dataset and the common episode/split universe are known. Any discrepancy between the snapshot and documented plan is resolved before implementation.

**Codex task prompt**

> Audit the research repository for the Brazil-inspired sensitivity analysis. Read repository instructions and the current U.S. config, environment, data/state contracts, and run manifest. Do not edit files or train models. Report data availability, code entry points, repository status, and discrepancies with the project plan.

### Phase 1 — Freeze design and numerical fixtures

**Tasks**

- Add the versioned scenario design document under repository `docs/`.
- Freeze the six rates, tax tiers, lot timing, terminal ordering, state semantics, normalization, fixed margins, metrics, and interpretation limits.
- Record the source/date and status of rate anchors in the repository. Preserve 7% as a hypothetical bound and 15% as the specified historical anchor, not as a forecast.
- Derive a small set of independent expected-value examples by hand (or in a tiny standalone calculation) for each economic component.

**Minimum fixtures**

1. A position with basis 100 sold for 120: sale tax 3, deposit 117.
2. A sale below basis: no equity tax credit and proceeds still enter cash.
3. A 117 deposit held 252 market intervals: gross value `117 * (1+r)` before interest tax.
4. Lots held 180 and 181 calendar days: apply the respective 22.5% and 20% tiers to positive interest only.
5. A partial sale: only the sold proceeds earn cash interest; remaining shares follow the stock path.
6. A terminal sale: proceeds are present in terminal wealth but earn no post-terminal interest.

**Deliverable**

Frozen design, fixture inputs, expected outputs, and a short assumptions ledger. No environment changes.

**Gate**

All definitions are unambiguous, values use a consistent wealth convention, and each fixture has an independently checkable answer.

### Phase 2 — Scenario configuration and pure accounting

**Tasks**

- Add an explicit `brazil_inspired_v1` economic-scenario configuration contract.
- Validate flat positive-gains equity taxation, no loss credit, cash rate, 252-day compounding, and the fixed-income tax tiers.
- Add pure accounting helpers for equity tax/proceeds, lot growth, holding-period tier selection, and after-tax lot value.
- Add one immutable scenario config per rate with a unique run directory, state-freeze version, seed policy, and fixed decision margins `0.070/0.020`.
- Fail clearly on missing or contradictory scenario fields. Preserve existing tax profile parsing when no scenario block is supplied.

**Deliverable**

Validated configs and reusable pure functions. Keep these functions separate from model training so their arithmetic can be reviewed directly.

**Gate**

Hand fixtures match within a declared floating-point tolerance; invalid configs fail as expected; existing U.S. configs still resolve to their prior normalized settings.

### Phase 3 — Environment and accounting integration

**Tasks**

- Extend the current environment with a scenario-specific branch, preserving the existing U.S. path.
- Track lot principal, deposit date, market-day age, gross interest, interest tax, equity tax, and cumulative proceeds separately.
- Implement the frozen accrual/sale/wealth/terminal ordering.
- Expose step and cumulative accounting fields in `info` and rollout records.
- Compute economic wealth and base reward consistently; apply C-lite shaping afterward.
- Add small synthetic trajectories for hold, immediate sale, partial sale, repeated sales, loss realization, and terminal liquidation.

**Gate**

- Base rewards telescope to final wealth minus initial wealth.
- Hold earns no cash interest before terminal liquidation.
- Immediate sale receives the longest eligible investment period.
- Unsold shares receive no cash interest.
- Separate lots can correctly receive different interest-tax tiers.
- A frozen U.S. fixture and existing U.S. tests retain their prior outputs.

### Phase 4 — Episode and state-schema v2

**Tasks**

- Build a separate scenario episode dataset using actual episode dates and terminal dates.
- Derive `time_to_terminal_horizon_norm`; do not derive it by renaming the old tax-transition field.
- Create v2 allowed/excluded columns and a state-freeze summary. Preserve all other v1 inputs unless an audit justifies a separately documented change.
- Validate chronological ordering, monotonic time-to-terminal, zero on the terminal observation, missingness, duplicate IDs, and leakage exclusions.
- Compare episode ID lists and chronological train/validation/test membership against the U.S. control; record hashes.

**Gate**

The v2 state contains no tax-transition input, all six configs point to v2, and the episode/split IDs match the control exactly (or any documented unavoidable discrepancy is resolved before training).

### Phase 5 — Benchmark accounting and smoke runs

**Tasks**

- Run manual/synthetic accounting at all six rates for hold, immediate sale, and staggered partial sales.
- Run a small debug training job at the low, middle, and high rates to check state shape, finite observations/rewards, and checkpoint selection plumbing.
- Independently recalculate selected episode outcomes outside the environment and compare them with rollout output.
- Inspect the cash and tax ledgers, not only the final wealth summary.

**Gate**

All accounting anchors agree with independent calculations; outputs are finite; state and reward dimensions are stable; and no test-set results have been used to tune design or training.

### Phase 6 — Full per-rate training and checkpoint freeze

**Tasks**

- Train from scratch for each of the six rate scenarios using identical episode assignments, architecture, primary hyperparameters, and the declared seed policy.
- Select each checkpoint using validation mean final after-tax wealth—not shaped training reward.
- Apply the transferred first-sale and subsequent-sale margins `0.070/0.020` without searching scenario-specific margins.
- Save effective config, split hashes, seed, best checkpoint, validation metric, and manifest for each run.
- If using multiple seeds, retain per-seed results and define in advance how the policy/checkpoint is selected. Do not pick the seed based on test performance.

**Compute control**

Start with one smoke/debug job. Estimate time and storage from that run before launching the six full jobs. Launch the full matrix only after Phase 5 passes. Keep run logs and checkpoint paths in each run directory. If resource limits require one seed, record the limitation and retain paired episode-level uncertainty estimates.

**Gate**

Every rate has a completed training record, validation-selected checkpoint, fixed-margin record, and complete manifest. Test outcomes remain unopened.

### Phase 7 — Locked test evaluation

**Tasks**

- Freeze configs/checkpoints and record a test-access checkpoint in the run manifest before evaluation.
- Evaluate the selected DQN at fixed `0.070/0.020` margins and the same required baselines under each scenario's accounting.
- Use identical test episode IDs for every rate and policy.
- Save episode-level outcomes and step-level rolls with scenario, checkpoint, seed, sales, tax, interest, remaining inventory, and final wealth fields.
- Check row counts, duplicate/missing episode IDs, terminal completion, and accounting identities.

**Primary comparisons**

- Each scenario's DQN versus hold-to-terminal under that same scenario.
- Each policy across rates, paired by the common test episode IDs.
- The original U.S. control reported separately; do not present it as if it shared the sensitivity accounting.

**Gate**

Every scenario has complete and comparable test outputs, all identities pass, and model choices were fixed before test evaluation.

### Phase 8 — Sensitivity calculations and report

**Tasks**

- Create a separate sensitivity analysis builder and output directory; do not overwrite original quantitative-analysis outputs.
- Produce per-rate/policy mean and median final wealth, excess versus hold, paired win/tie/loss, and paired bootstrap intervals.
- Decompose sale behavior, first-sale timing, sale frequency, remaining inventory, equity tax, gross interest, interest tax, and net interest.
- Plot performance and behavior across rates with episode pairing preserved.
- Link every reported number to source run manifests and generated tables.
- Write a concise interpretation with the USD/Brazil-inspired scope, combined treatment (flat tax plus cash rate), rate-anchor status, and limitations.

**Gate**

The report can be regenerated from saved run artifacts; all summary values reconcile with episode-level files; and U.S. historical outputs are unchanged.

## Required run and analysis records

Each rate run should retain at least:

- effective config and scenario identifier;
- data, state-freeze, split, and source-code hashes;
- seed(s), training summary, selected checkpoint, and validation criterion;
- fixed-margin identifiers (`0.070` first sale, `0.020` thereafter);
- baseline summaries, episode-level metrics, and step-level rollouts;
- accounting summaries for cash interest, fixed-income tax, equity tax, and final wealth;
- command/log record sufficient to reproduce the run.

The cross-rate analysis should include an input manifest listing all six run directories and their hashes, plus generated tables, figures, and a summary document.

## Final completion checklist

- [ ] Original U.S. configs, state freeze, environment behavior, and run artifacts remain intact.
- [ ] Flat equity-tax and cash-lot accounting match hand fixtures.
- [ ] New environment regression preserves U.S. outputs.
- [ ] Scenario v2 data/state and split IDs are validated and hashed.
- [ ] Smoke trajectories and debug runs pass before full training.
- [ ] Six rate-specific models are retrained and selected on validation data.
- [ ] The fixed `0.070/0.020` margins are used without scenario recalibration.
- [ ] Test evaluation uses the same locked episodes and frozen policies.
- [ ] The analysis includes uncertainty, paired comparisons, and tax/interest decomposition.
- [ ] The final report traces results to manifests and describes the study as Brazil-inspired and USD-denominated.
