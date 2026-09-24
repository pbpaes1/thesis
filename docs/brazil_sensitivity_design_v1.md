# Brazil-inspired sensitivity: design freeze v1

This document freezes the economic and measurement contract for the six
`brazil_inspired_v1` treatments. It governs later implementation phases; it does
not change the original U.S. experiment. The implementation repository is
`C:\Users\pbpae\OneDrive\Documentos\Polimi\Thesis\Code`. The original
`data/episodes/drl_episodes.parquet` contains 10,119 episode IDs, and the C-lite
v5 train/validation/test split file covers exactly those IDs. All treatments
reuse that episode and split universe.

## Units and equity accounting

Let `B = Q * P_basis > 0` be the original position basis, where `Q` is the
original share count and `P_basis` is the episode's
`simulated_purchase_price`. Divide every account amount by `B`; the original
basis is therefore `1`. Use the existing episode `adj_close` as the equity
price, consistently with `unrealized_gains_pct = adj_close / P_basis - 1`.
Actions request fractions of the **original** position and are clamped to
remaining inventory. For a sale of original fraction `f` at price `P_t`:

```text
gross_proceeds_t / B = f * P_t / P_basis
allocated_basis_t / B = f
realized_gain_t / B = f * (P_t / P_basis - 1)
equity_tax_t / B = 0.15 * max(realized_gain_t / B, 0)
cash_deposit_t / B = gross_proceeds_t / B - equity_tax_t / B
```

The same flat 15% positive-gains rule applies on every sale date, including
mandatory terminal liquidation. Losses create no credit. Proceeds include
recovered basis. No leverage, withdrawals, transaction costs, or negative cash
deposits are part of this scenario. Reject a missing, nonfinite, or nonpositive
position basis or equity price; the configured annual gross cash rate must be
one of the six positive rates below.

The existing U.S. `TaxAwareEnv` computes `after_tax_total_value` from realized
after-tax **PnL** plus after-tax unrealized **PnL** on remaining inventory. It
does not include recovered basis or earn interest on sale proceeds. For the same
tax assumptions and date, full normalized wealth without interest would be
`1 + after_tax_total_value`; the scenario has different tax assumptions and
cash growth, so its wealth is a separate metric. Keep all historical U.S.
values and their field names unchanged. Never label the old PnL field as full
account wealth or directly compare it with scenario wealth.

## Cash lots, valuation, and reward

Each discretionary sale creates a separate lot with normalized principal `C_i`,
deposit date `s_i`, and market-day age `m_i = 0`. The annual effective gross rate
`r` uses 252 market-day intervals per year. At observation date `t`, let
`d_i = (t - s_i).days` be actual calendar days held:

```text
gross_i(t) = C_i * (1 + r) ** (m_i / 252)
interest_i(t) = gross_i(t) - C_i
tax_rate_i(t) = 0.225 if d_i <= 180 else 0.20
estimated_interest_tax_i(t) = tax_rate_i(t) * max(interest_i(t), 0)
after_tax_lot_i(t) = gross_i(t) - estimated_interest_tax_i(t)
```

Interest compounds gross within a lot; no interest tax is deducted from its
principal during the episode. Before terminal redemption,
`estimated_interest_tax_i` is a mark-to-redemption liability, **not** tax
already paid. The nonterminal reported cash balance is the sum of after-tax lot
values. Revalue the liability at every observation using that day's calendar
bracket. Crossing from day 180 to day 181 can increase marked cash wealth and
base reward when the rate falls from 22.5% to 20%, even if accrued interest is
unchanged. At redemption, record the then-current liability as settled tax;
do not count earlier estimated liabilities as payments.

If `u_t` is the unsold fraction of the original position, its immediate
after-tax liquidation value is:

```text
inventory_value_t = u_t * P_t / P_basis
inventory_equity_tax_t = 0.15 * max(inventory_value_t - u_t, 0)
after_tax_inventory_t = inventory_value_t - inventory_equity_tax_t
wealth_t = sum_i(after_tax_lot_i(t)) + after_tax_inventory_t
```

All four terms are normalized by `B`. Initial cash is zero. At reset, before
any action on the first episode date, `initial_wealth` equals that date's
after-tax liquidation value of the entire position; it is generally **not**
exactly `1`, because the episode starts after the simulated purchase. Economic
base reward for each completed decision/valuation event is the change from the
previous marked wealth, starting at this reset value. Thus
`sum(base_reward) = final_wealth - initial_wealth` within floating-point
tolerance for both an early full sale and a hold-to-terminal episode. Use
`training.discount_factor_gamma = 1.0` in all six Brazil DQN configurations so
the training return does not discount the terminal-date valuation merely
because one policy takes more action steps. The DQN receives the selected
C-lite shaped reward: each cooldown penalty is recorded separately and
subtracted after the base reward. Thus its undiscounted shaped return equals
the final-minus-initial wealth change less cumulative cooldown penalties;
the penalties do not change economic wealth. Select checkpoints by validation
mean final total after-tax wealth, never shaped cumulative reward.

## Event timeline and terminal horizon

The final dated row of each original episode is its fixed evaluation horizon.
On the first observation there is no prior cash interval. On each subsequent
market observation, advance every previously open lot by exactly one market-day
interval before processing that date's decision. Then clamp and execute a
discretionary sale, allocate basis, charge equity tax, and deposit full net
proceeds into a new age-zero lot. Mark cash and remaining inventory at the
current date and calculate the base reward. A new deposit has no interest on
its sale date and first accrues on the next market-day interval. Calendar gaps
between observations affect the interest-tax tier, not the count of compound
intervals.

An early full discretionary sale **ends the agent episode on that action**.
Before returning `done=True`, advance every open lot through the remaining
market-day intervals to the original final observation, then redeem it there.
Use calendar days from each lot's deposit to that original final date to select
its interest-tax tier. Include all remaining cash growth and settled interest
tax in the terminating wealth and base reward. Do not accept further actions.
The terminating step's gross-interest increment includes the skipped intervals;
its fixed-income tax is settled rather than left as an estimated liability.
The action date and row pointer remain those of the executed sale; report the
original final date separately as the terminal valuation date. The original
final observation is the cash valuation horizon, even if the policy stops
acting earlier.

At the final observation, advance previously open lots through the final
interval exactly once, then execute and clamp the requested discretionary sale
at that row's price. Apply flat equity tax and deposit its net proceeds into a
new age-zero lot, even though it has no time to earn interest. Redeem all lots
at the final date, taxing only positive accrued interest, and mandatorily
liquidate any remaining shares at that same price and flat equity-tax rate.
Terminal liquidation proceeds enter final wealth directly; they create no lot,
earn no interest, and incur no discretionary-sale penalty. Compute the final
base reward from wealth after both sales, then apply the C-lite penalty, if
applicable, to the **executed discretionary** terminal-row sale only. If there
is only one row, there is no cash accrual interval. The scenario must use this
final-row settlement instead of the U.S. path's next-row terminal shortcut.

Count any positive executed sale requested on the final row as a discretionary
sale, including for sale fraction and cooldown metrics. If it is the first
executed discretionary sale, its first-sale date is the terminal date and its
days from episode start equal the elapsed calendar days; use the first-sale
`0.070` decision margin for its action selection. A terminal request clamped to
zero is not a sale. A full sale on an earlier row terminates immediately, so
there are no later zero-execution actions. Mandatory liquidation never
increments discretionary sale count and never supplies a first-sale date; a
hold-to-terminal policy therefore has no discretionary first sale. A full
discretionary sale on the final row leaves zero shares for mandatory
liquidation. Both routes produce the same final economic wealth at that price,
but their sale-behavior records and any discretionary C-lite penalty differ.

## Scenarios, state, and comparisons

| Gross annual rate | Identifier suffix | Anchor and status |
| ---: | --- | --- |
| 7.00% | `0700` | Hypothetical extreme-easing bound; not a Focus forecast. |
| 10.00% | `1000` | End-2029 Focus median stated in the implementation plan. |
| 10.50% | `1050` | End-2028 Focus median stated in the implementation plan. |
| 12.00% | `1200` | End-2027 Focus median stated in the implementation plan. |
| 13.75% | `1375` | End-2026 Focus median stated in the implementation plan; also the September 2026 Copom rate. |
| 15.00% | `1500` | Historical pre-easing peak specified for June 2025–March 2026; not a current forecast. |

The plan identifies the Focus survey released **14 September 2026** as the
forecast anchor. Its individual figures are recorded above as plan assertions,
not as independently verified transcriptions of the dated primary report.
The [BCB Focus publication page](https://www.bcb.gov.br/controleinflacao/relatoriofocus)
describes the survey and release convention; the [BCB September 2026 Copom
minutes](https://www.bcb.gov.br/publicacoes/atascopom/16092026) document the
13.75% policy decision. Preserve these provenance qualifications in reports;
do not re-label 7% or 15% as Focus projections. The cash rate is fixed within
each episode and applies to full net proceeds, not only realized gain.

Use the original U.S. C-lite v5 run as the separately reported control. All
six sensitivity treatments change equity tax and assign a cash rate together;
none is a tax-only control. Retrain each treatment later on the same original
episode IDs and chronological train/validation/test assignments. Fix the
transferred Q-value decision margins at `0.070` before the first discretionary
sale and `0.020` thereafter; do not tune margins on Brazil-scenario outcomes.

In a separate episode parquet and v2 state freeze, replace the v1 agent input
`days_to_tax_transition_norm` with
`time_to_terminal_horizon_norm = min(max((terminal_date - date).days, 0), 365) / 365`,
where `terminal_date` is the **actual last observation date** of that episode.
It is non-increasing and zero on the final row. Preserve the other 35 v1 inputs.
The Phase 4 inventory audit found that the v1 observation does not expose
remaining inventory, while the Brazil environment changes that inventory after
sales. The chosen v2 freeze keeps the same 36-input width: remaining fraction
and cash-lot history stay outside the observation to preserve the intended
input comparison with the U.S. control. This makes the Brazil observation
partially observable to the feedforward DQN: identical market/time features can
have different remaining sale capacity and cash-lot tax exposure. Record this
limitation in analysis; do not add a static inventory column to the episode
parquet. Do not expose future terminal prices, cash/tax ledger internals, or
the old tax-transition countdown as a renamed input.

Primary episode metrics are final normalized total after-tax wealth **at the
original final observation date**, even for an episode terminated by an earlier
full discretionary sale, and its
paired difference from hold-to-terminal under the **same** rate. Report means,
medians, paired win/tie/loss counts, and paired uncertainty by common episode
ID. Also report first discretionary-sale date and elapsed days, discretionary
sale count and original-position fraction sold, remaining fraction immediately
before terminal liquidation, gross equity proceeds, equity tax, ending cash
principal, gross cash interest, estimated interest-tax liability while open,
settled terminal interest tax, and net cash interest (`gross interest - settled
interest tax`). The final account wealth includes recovered equity basis;
after-tax PnL is a distinct derived metric. Keep policy, rate, episode, seed,
checkpoint, and fixed-margin identifiers with results. First-sale timing and
discretionary sale counts refer only to executed actions and their actual dates;
early termination can shorten the environment rollout without shortening the
cash-growth horizon.

This is a stylized, USD-denominated sensitivity using historical U.S.-listed
equity paths and fixed gross cash rates. It does not model Brazilian tax-law
detail, Brazilian assets, FX conversion or risk, changing rates, inflation,
fees, or external cash flows. The original U.S. results remain untouched.

## Independent numerical fixtures

The figures below use original basis `B = 100`, one share with basis price
`100`, and `r = 0.10` where cash grows. Dollar amounts divide by `100` to
obtain the normalized values used by the environment. Hold the terminal share
price at `120` in the partial-sale and terminal examples.

| Case | Hand calculation and expected result |
| --- | --- |
| Positive-gain full sale | At price `120`, gain `120 - 100 = 20`, equity tax `0.15 × 20 = 3`, deposit `120 - 3 = 117` (`1.17` normalized). |
| Loss sale | At price `80`, gain `-20`, equity tax `0`, deposit `80` (`0.80` normalized); there is no loss credit. |
| 252 cash intervals | Principal `117` grows to `117 × 1.10 = 128.70`; gross interest `11.70`. If held over 180 calendar days, tax is `0.20 × 11.70 = 2.34`, so final cash is `126.36` (`1.2636` normalized). |
| Calendar-tier boundary | With principal `117` and 126 market intervals held fixed, gross value is `117 × sqrt(1.10) = 122.7106352359`; interest is `5.7106352359`. At 180 calendar days, tax is `1.2848929281` and net value `121.4257423078`. At 181 days, tax is `1.1421270472` and net value `121.5685081887`. The `0.1427658809` increase comes solely from the bracket change. |
| Half sale | Sell `0.5` original share at `120`: proceeds `60`, allocated basis `50`, tax `1.50`, lot principal `58.50`. The other half has terminal proceeds `60`, tax `1.50`, net `58.50` and no cash interest. If the first lot earns 252 intervals and is held over 180 calendar days, it ends at `58.50 × 1.10 - 0.20 × 5.85 = 63.18`; total final wealth is `63.18 + 58.50 = 121.68` (`1.2168` normalized). |
| Terminal sale versus early full sale | Holding the whole share to terminal at `120` yields `117` (`1.17`) with zero interest and no discretionary sale. Selling it earlier at `120` and then holding the `117` lot for 252 eligible market intervals and over 180 calendar days yields `126.36`; the agent episode ends on the sale action, with its cash valued at the original terminal date. A requested full discretionary sale on the terminal row also yields `117` with no subsequent accrual, but counts as one discretionary sale dated at the terminal observation. |

The 180/181 fixture holds market-day age fixed to isolate the tax-bracket
effect; an actual episode can cross that calendar boundary over a non-trading
gap. Other fixtures assume no price movement after the stated sale unless
specified. Compare later implementations with these results at a declared
floating-point tolerance, without sourcing expected values from the
environment under test.

## Assumptions and later decisions

- Scenario configurations explicitly select `brazil_inspired_v1`; equal U.S.
  short- and long-term tax rates never activate this path.
- The terminal horizon is the last **available** episode row, which may precede
  the nominal tax-transition date. Its date is derived per episode, never by
  renaming the old countdown.
- Use only the six strictly positive specified rates in v1. Zero or negative
  rates and deposits are outside this scenario contract.
- Phase 2 fixes one DQN training seed, `42`, for each rate, matching C-lite v5.
  This choice does not alter the accounting or numerical fixtures. No Phase 1
  accounting decision remains open.
