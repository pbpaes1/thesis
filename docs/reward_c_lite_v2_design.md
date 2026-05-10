# Reward C-lite v2 Design

Reward version:

`C_lite_v2_after_tax_value_change_minus_transaction_and_cooldown_penalty`

Reward C-lite v2 keeps Reward A as the economic base:

```text
Reward_A_t = after_tax_total_value_t - previous_after_tax_total_value_t
```

It subtracts two trade-discipline terms:

```text
transaction_penalty_t =
    lambda_transaction * action_fraction_executed_t

cooldown_penalty_t =
    lambda_cooldown
    * action_fraction_executed_t
    * max(0, cooldown_days - days_since_last_sale_t) / cooldown_days

Reward_C_lite_v2_t =
    Reward_A_t - transaction_penalty_t - cooldown_penalty_t
```

The transaction penalty applies to every discretionary executed sale, including the first sale. It does not apply to automatic terminal liquidation.

The cooldown penalty applies only to repeated discretionary sales inside the cooldown window. The first sale remains free of the cooldown penalty. Automatic terminal liquidation is not penalized because it is not an agent discretionary clustered sale.

## Rationale

In financial reinforcement learning, the reward function shapes the agent's behavior and must reflect the actual economic objective. In this project, the investor objective remains final after-tax value. Transaction and cooldown penalties are training reward-shaping terms intended to discourage low-discipline selling, not replacement economic objectives.

DQN estimates discounted future rewards, so `gamma` controls how much future value changes matter. For an episodic final-value liquidation problem, `gamma = 1.0` is appropriate because the objective is final after-tax value rather than early realized gains.

Trading is path-dependent: current holdings and past trades affect future decisions, tax treatment, and remaining option value. Market frictions such as transaction costs, taxes, slippage, bid/ask spread, and market impact mean repeated or casual selling should not be treated as free.

Moody and Saffell, "Learning to Trade via Direct Reinforcement," motivate trading rewards that account for position state, path-dependence, taxes, and transaction costs. Cost-sensitive DRL portfolio literature similarly warns that ignoring transaction costs can produce aggressive trading and biased return estimates. Deep portfolio optimization and attention-enhanced portfolio RL work also motivate reward functions that include risk, cost, turnover, or friction terms for economic realism.

Reward C-lite v2 is not a drawdown reward. It does not implement Reward B and it does not add an explicit tax-saving bonus. It is a trade-discipline ablation designed to test whether penalizing discretionary sales improves early-selling behavior while preserving final after-tax value as the model-selection and economic evaluation metric.

## V2 Changes

- Discount factor `gamma` is set to `1.0`.
- A small transaction penalty is applied to every discretionary executed sale.
- The existing cooldown penalty remains focused on repeated fragmented sales.
- Random exploration is more hold-biased: `[0.65, 0.20, 0.10, 0.04, 0.01]`.
- Early-selling diagnostics are added to baseline evaluation and behavior inspection outputs.
