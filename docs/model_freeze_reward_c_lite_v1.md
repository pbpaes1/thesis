# Reward C-lite Model Freeze v1

## 1. Purpose

Prepare a freeze record for the Reward C-lite v1 training run after the build-and-train pipeline completes.

## 2. Reward definition

Reward version:

```text
C_lite_after_tax_value_change_minus_cooldown_penalty
```

Base reward:

```text
A_after_tax_total_value_change
```

## 3. Cooldown penalty formula

```text
Reward_C_lite_t =
    Reward_A_t
    - lambda_cooldown
      * executed_fraction_t
      * max(0, cooldown_days - days_since_last_sale_t) / cooldown_days
```

Configured values:

- `lambda_cooldown: 0.0025`
- `cooldown_days: 20`
- `scale_by_executed_fraction: true`
- `penalize_first_sale: false`

Automatic terminal liquidation is not penalized because it is not an agent-requested sale.

## 4. Tax profile

Tax profile is unchanged from Reward A v3 for comparability:

- `profile_name: mass_affluent_individual`
- `config_path: configs/individual_tax_profiles_v1.yaml`

## 5. Training configuration

Primary config:

- `configs/train_reward_c_lite_v1.yaml`

Run directory:

- `runs/train_reward_c_lite_v1_full`

## 6. Dataset and split

Dataset and state schema are unchanged:

- `data/episodes/drl_episodes.parquet`
- `data/freeze/v1/allowed_state_columns_v1.json`

Split file to record after training:

- `runs/train_reward_c_lite_v1_full/episode_splits.csv`

## 7. Model artifact selected

Pending run completion.

Expected artifacts:

- `runs/train_reward_c_lite_v1_full/best_validation_model.pt`
- `runs/train_reward_c_lite_v1_full/best_validation_summary.txt`
- `runs/train_reward_c_lite_v1_full/final_model.pt`

Selection metric:

- `mean_final_after_tax_total_value`

## 8. Baseline evaluation files

Pending run completion.

Expected files:

- `runs/train_reward_c_lite_v1_full/baselines/baseline_episode_metrics.csv`
- `runs/train_reward_c_lite_v1_full/baselines/baseline_step_rollouts.csv`
- `runs/train_reward_c_lite_v1_full/baselines/baseline_summary_by_policy.csv`
- `runs/train_reward_c_lite_v1_full/baselines/baseline_evaluation_summary.txt`

## 9. Behavior inspection files

Pending run completion.

Expected files:

- `runs/train_reward_c_lite_v1_full/behavior_inspection/dqn_action_distribution.csv`
- `runs/train_reward_c_lite_v1_full/behavior_inspection/dqn_first_cut_summary.csv`
- `runs/train_reward_c_lite_v1_full/behavior_inspection/dqn_episode_behavior.csv`
- `runs/train_reward_c_lite_v1_full/behavior_inspection/policy_behavior_comparison.csv`
- `runs/train_reward_c_lite_v1_full/behavior_inspection/cooldown_penalty_summary.csv`
- `runs/train_reward_c_lite_v1_full/behavior_inspection/behavior_summary.txt`

## 10. Reproducibility manifest

Pending run completion.

Expected file:

- `runs/train_reward_c_lite_v1_full/run_manifest.json`

## 11. Known limitations

- Reward C-lite is a reward-shaping ablation, not a final thesis conclusion.
- The cooldown penalty is a simplified proxy for trading discipline, not a calibrated transaction-cost model.
- The first sale remains unpenalized by design.
- Final after-tax total value remains the main economic comparison metric.

## 12. Decision status

`pending_run_completion`
