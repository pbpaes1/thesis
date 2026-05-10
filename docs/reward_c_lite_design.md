# Reward C-lite Design

## Reward version

`C_lite_after_tax_value_change_minus_cooldown_penalty`

Reward C-lite preserves Reward A as the economic base reward:

```text
Reward_A_t = after_tax_total_value_t - previous_after_tax_total_value_t
```

It then subtracts a cooldown penalty when the agent executes another sale too soon after a prior sale in the same episode:

```text
Reward_C_lite_t =
    Reward_A_t
    - lambda_cooldown
      * executed_fraction_t
      * max(0, cooldown_days - days_since_last_sale_t) / cooldown_days
```

The first sale in an episode is not penalized. The penalty applies only to repeated, explicit executed sales inside the cooldown window.

## Rationale

Financial trading decisions are path-dependent: the current position, remaining inventory, realized gains, and previous trades all affect the next decision. A liquidation policy should therefore consider not only the current market state but also the trade path that led to the current position.

Market frictions such as taxes and transaction costs mean repeated position changes should not be treated as free. In trading reinforcement learning, omitting transaction or turnover costs can encourage aggressive, fragmented, and unrealistic trading behavior.

Reward C-lite is a simplified transaction-discipline proxy. It does not add a drawdown penalty and does not replace Reward B. It also does not implement the full Reward C tax-saving bonus. It is an ablation designed to test whether penalizing clustered sales improves learned liquidation behavior while preserving final after-tax value as the economic evaluation metric.

## Literature notes

- Moody and Saffell, "Learning to Trade via Direct Reinforcement," motivate path-dependent trading rewards where position state, taxes, and transaction costs affect decisions.
- Cost-sensitive portfolio-selection DRL work motivates reward shaping that accounts for transaction and risk costs rather than treating portfolio changes as frictionless.
- Deep portfolio optimization literature commonly studies reward functions augmented with risk and cost terms to better align training objectives with investable behavior.
- Attention-enhanced portfolio RL literature also uses turnover, friction, or transaction-cost penalties to improve economic realism.

## Scope

Reward C-lite includes:

- Reward A after-tax total value change.
- A repeated-sale cooldown penalty.
- Scaling by the fraction of the original position actually executed after oversell clamping.

Reward C-lite excludes:

- Reward B.
- Drawdown penalties.
- Explicit tax-saving bonuses.
- Reward clipping or normalization.
- Any dataset, state schema, tax accounting, or action-semantics changes.
