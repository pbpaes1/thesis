# Reward C-lite v3 Thresholded Training Design

Reward C-lite v3 keeps the Reward C-lite v1 reward unchanged:

```text
Reward C-lite = Reward A - cooldown_penalty
```

The change is only in the DQN training behavior policy. During random
exploration, the agent still uses the configured hold-biased random action
probabilities. During exploitation, the agent uses thresholded greedy action
selection instead of pure greedy action selection.

For each exploitation step, the policy computes Q-values for all actions,
compares the hold action (`action_fraction = 0.0`) with the best non-hold sell
action, and sells only when:

```text
Q(best_sell_action) > Q(hold_action) + 0.020
```

The comparison is strict. Equality returns hold.

This creates a no-sell buffer in the training trajectories. The experiment tests
whether reducing marginal sell decisions in replay data improves the learned
policy's early-selling behavior before the tax transition.

This is not a reward change, not a tax logic change, and not an environment
action-semantics change. Reward B, drawdown penalties, transaction penalties,
and explicit tax-saving bonuses remain disabled. Model selection still uses
validation mean final after-tax total value.
