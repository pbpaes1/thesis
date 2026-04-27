# Reward Function Design v1

## Purpose

This document freezes the first reward-function specification for the tax-aware optimal stopping / partial liquidation environment.

The thesis environment represents a single-entry winning position. The agent observes one chronological row at a time and chooses whether to hold or sell a fraction of the original position. The reward must therefore value both:

1. realized after-tax PnL from executed sales, and
2. the after-tax economic value of any remaining unsold inventory.

The main design principle is to avoid rewarding only realized sales, because that could make holding economically invisible. At the same time, the model should not pretend that taxes are actually paid on unrealized gains every day. The solution is to use a hypothetical after-tax liquidation value for the remaining position.

## Literature motivation

The reward should represent the investor's true economic objective rather than a generic prediction metric. This follows the reinforcement-learning framing in quantitative finance, where the agent observes the financial state, acts, receives a numerical reward, and learns a policy that maximizes cumulative reward.

This design is also consistent with the direct reinforcement learning literature. Moody and Saffell argue that investment performance is path-dependent and that trading decisions must account for current positions and market frictions, including transaction costs and taxes.

Recent deep portfolio optimization literature also supports embedding practical frictions directly into the reward. Cost-sensitive and risk-aware DRL portfolio papers show that ignoring transaction costs and risk can produce unrealistic or overly aggressive trading behavior. This thesis therefore starts from an after-tax value reward and later adds drawdown and short-interval trading penalties.

## Environment assumptions

This specification assumes the current simplified environment design:

- each episode is one `episode_id`
- each episode represents one single entry lot
- cost basis is fixed within the episode
- no rebuying occurs within the episode
- actions are fractions of the original position
- execution is clamped by remaining inventory
- `unrealized_gains_pct` is the full-position unrealized PnL relative to original entry basis
- tax is paid only on executed realized sales
- unrealized inventory is valued using a hypothetical liquidation value
- if the episode ends with unsold inventory, the remaining fraction is terminally liquidated using long-term tax treatment

## Tax regime convention

For regular step-level sale accounting:

```text
if date < tax_transition_date:
    tax_regime = short_term
else:
    tax_regime = long_term
```

For remaining inventory valuation:

```text
if date < tax_transition_date:
    remaining inventory is valued at short-term liquidation value
else:
    remaining inventory is valued at long-term liquidation value
```

For terminal liquidation:

```text
remaining inventory is liquidated using long-term tax treatment
```

This terminal convention prevents the agent from hiding value or risk in unsold inventory at the end of the episode.

## Core variables

Let:

```text
u_t = full-position unrealized PnL at step t
x_t = executed sale fraction of original position at step t
q_t = remaining fraction of original position after action at step t
C_t^AT = cumulative realized after-tax PnL after action at step t
L_t^AT = after-tax liquidation value of remaining inventory after action at step t
V_t^AT = total after-tax position value after action at step t
```

The full-position unrealized PnL comes from:

```text
u_t = unrealized_gains_pct_t
```

## Sale-level accounting

For an executed sale at step `t`:

```text
realized_pre_tax_increment_t = x_t * u_t
```

Tax is applied only to positive realized gains:

```text
if realized_pre_tax_increment_t > 0:
    tax_paid_t = realized_pre_tax_increment_t * applicable_tax_rate_t
else:
    tax_paid_t = 0
```

Then:

```text
realized_after_tax_increment_t = realized_pre_tax_increment_t - tax_paid_t
```

And cumulative realized after-tax PnL becomes:

```text
C_t^AT = C_{t-1}^AT + realized_after_tax_increment_t
```

## Remaining inventory valuation

The remaining inventory is not taxed as an actual event. It is valued as a hypothetical after-tax liquidation value.

First define the applicable liquidation tax rate for the remaining inventory:

```text
tau_t^liq = short_term_rate, if date_t < tax_transition_date_t
tau_t^liq = long_term_rate, otherwise
```

For positive remaining unrealized gain:

```text
L_t^AT = q_t * u_t * (1 - tau_t^liq)
```

For zero or negative remaining unrealized gain, no tax credit is assumed in this simplified simulator:

```text
L_t^AT = q_t * u_t
```

Equivalent compact form:

```text
L_t^AT = q_t * u_t - max(q_t * u_t, 0) * tau_t^liq
```

## Primary total after-tax value

The total after-tax value of the episode state after the action is:

```text
V_t^AT = C_t^AT + L_t^AT
```

This is the central state-value quantity used for Reward A.

## Reward A: change in after-tax total position value

Reward A is the primary reward for the first implementation:

```text
r_t^A = V_t^AT - V_{t-1}^AT
```

This rewards economic improvement in total after-tax value, not merely the act of selling.

Important implications:

- holding while the price rises produces positive reward
- holding while the price falls produces negative reward
- selling moves value from hypothetical liquidation value into realized after-tax PnL without double-counting
- crossing the tax transition can increase after-tax liquidation value even if price is unchanged
- terminal unsold inventory is valued through forced long-term liquidation

## Reward B: add drawdown penalty

Reward B extends Reward A with a drawdown penalty:

```text
r_t^B = r_t^A - lambda_DD * drawdown_penalty_t
```

The preferred first drawdown penalty is based on increases in drawdown of total after-tax value:

```text
peak_t = max(peak_{t-1}, V_t^AT)
drawdown_t = peak_t - V_t^AT
drawdown_penalty_t = max(0, drawdown_t - drawdown_{t-1})
```

This penalizes new deterioration rather than repeatedly penalizing the same drawdown.

## Reward C: add short-interval trading penalty

Reward C extends Reward B with a penalty for repeated sales too close together:

```text
r_t^C = r_t^B - lambda_trade * short_interval_trade_penalty_t
```

The recommended smooth penalty is:

```text
short_interval_trade_penalty_t =
    x_t * max(0, cooldown_days - days_since_last_sale_t) / cooldown_days
```

where the penalty is zero when no sale is executed.

This discourages excessive partial liquidation or repeated small sales in a short period, while preserving flexibility when multiple exits are economically justified.

## Do not add a separate tax-saving bonus yet

A separate tax-saving reward term should not be included in the first training version.

Reason: the benefit of long-term tax treatment is already embedded in `L_t^AT` and in realized after-tax PnL after the tax transition. Adding a separate tax-saving bonus would risk double-counting the tax advantage and could make the agent wait for the tax date even when market risk makes waiting unattractive.

A tax-saving bonus can be revisited only if sanity checks show that the agent fails to respond to the tax transition despite the after-tax value reward.

## Required implementation order

1. Implement Reward A only.
2. Expose all reward components in `info`.
3. Run synthetic sanity checks.
4. Inspect reward scale and component distributions.
5. Add Reward B only after Reward A behaves correctly.
6. Add Reward C only after Reward B behaves correctly.
7. Freeze one reward version for the first full training run.

## Reward sanity checks

The first implementation should pass the following checks:

| Scenario | Expected behavior |
|---|---|
| Hold while price rises | positive reward |
| Hold while price falls | negative reward |
| Sell before tax transition | realized and remaining values use short-term tax treatment |
| Sell after tax transition | realized and remaining values use long-term tax treatment |
| Cross tax transition with unchanged price | positive after-tax value change from lower tax burden |
| End with unsold inventory | terminal liquidation applies long-term tax |
| Partial sale followed by later sale | no double-counting |
| Oversell attempt | reward uses only executable remaining fraction |
| Loss scenario | no tax credit is assumed |

## Reward scale checks

Before training, inspect distributions for:

```text
reward_A
realized_after_tax_increment
after_tax_liquidation_value_remaining
after_tax_total_value
drawdown_penalty
short_interval_trade_penalty
terminal_liquidation_after_tax_increment
```

The goal is to ensure no single component dominates mechanically. If necessary, scale rewards by a constant factor, but do not change the economic meaning of the reward.
