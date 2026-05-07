# Training Plan v1: Reward A First Run

## Current phase

The project is in:

```text
Build and Train the Model
```

The project is not yet in:

```text
Quantitative Analysis and Simulation
```

## Objective

The objective is to train the first reinforcement learning agent using frozen Reward A, with reward version `A_after_tax_total_value_change`.

## Why DQN first

DQN is a reasonable first algorithm because the environment uses a discrete action space with five fixed actions:

- hold
- sell 25%
- sell 50%
- sell 75%
- sell 100%

## First run sequence

1. Load `configs/train_reward_a_v1.yaml`
2. Instantiate `TaxAwareEnv`
3. Verify `info["reward_version"] == "A_after_tax_total_value_change"`
4. Run tiny debug training on a small episode subset
5. Check reward/loss/action logging
6. Compare debug policy against benchmark policies
7. Only then run full Reward A training

## Baselines to compare later

- hold_to_terminal
- sell_immediately
- sell_half_then_hold
- sell_quarters_over_time
- random_policy

## Not in scope yet

The following are not part of this step:

- Reward B
- Reward C
- drawdown penalty
- cooldown penalty
- quantitative analysis
- simulation study
- final thesis results
