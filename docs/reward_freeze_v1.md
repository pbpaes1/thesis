# Reward Freeze v1: First Training Run

## Frozen reward version

```text
reward_version = "A_after_tax_total_value_change"
```

## Decision

Reward A is selected as the only reward version for the first training run.

## Mathematical definition

```text
V_t^AT = C_t^AT + L_t^AT
r_t = V_t^AT - V_{t-1}^AT
```

Terms:

- `C_t^AT`: cumulative realized after-tax PnL after all sales executed through step `t`.
- `L_t^AT`: hypothetical after-tax liquidation value of the remaining inventory after step `t`.
- `V_t^AT`: total after-tax position value after step `t`, equal to realized after-tax PnL plus the hypothetical after-tax liquidation value of remaining inventory.
- `r_t`: step reward, equal to the change in total after-tax position value from step `t-1` to step `t`.

## Economic interpretation

Reward A measures the change in total after-tax economic value of the position. It includes:

- realized after-tax PnL from executed sales
- hypothetical after-tax liquidation value of remaining inventory
- terminal liquidation of remaining inventory using long-term tax treatment

Unrealized gains are not actually taxed each step. Remaining inventory is only valued under a hypothetical liquidation convention so that the reward reflects the economic value still held by the agent.

## Why Reward A is frozen first

Reward A is the cleanest first training reward because:

- it directly captures the tax-aware liquidation objective
- it avoids rewarding only realized sales
- it avoids double-counting realized and unrealized value
- synthetic sanity checks passed
- reward-scale checks showed stable magnitudes
- it provides a baseline before adding preference-based penalties

## Explicit exclusions

The first training run does not include:

- drawdown penalty
- cooldown / short-interval trading penalty
- explicit tax-saving bonus
- reward clipping
- reward normalization
- corporate tax logic
- multiple-lot accounting
- wash-sale logic
- capital-loss carry rules

## Deferred reward variants

These reward variants are deferred for later ablations:

```text
Reward B = Reward A - drawdown penalty
Reward C = Reward B - short-interval trading penalty
```

Do not implement Reward B or Reward C for the first training run.

## Validation evidence

Reward A validation is based on the completed synthetic reward sanity checks, terminal liquidation final-row PnL check, and reward-scale stability checks.

Validation scripts:

- `scripts/reward_sanity_checks.py`
- `scripts/check_reward_scale.py`

Generated reward-scale outputs:

- `reports/reward_scale/reward_scale_steps.csv`
- `reports/reward_scale/reward_scale_episodes.csv`
- `reports/reward_scale/reward_scale_summary.txt`

Reward A is considered ready for first training if:

- reward identities pass
- episode telescoping identity passes
- no NaN or infinite rewards are found
- no scale-stability warnings are triggered

## Training implication

The first training environment should use whatever reward is returned by `TaxAwareEnv.step()`. For this training run, `info["reward_version"]` must equal:

```text
A_after_tax_total_value_change
```

Any later training run with Reward B or Reward C should use a different reward version name.
