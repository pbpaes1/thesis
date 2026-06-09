# Chapter 4 Section 5 Figure Notes

## Figure 4.5: Representative preferred-DQN liquidation paths across selected test episodes
- Thesis caption: Figure 4.5: Representative preferred-DQN liquidation paths across selected test episodes
- Source files used:
  - `runs/train_reward_c_lite_v5_full/baselines/baseline_step_rollouts.csv`
  - `data/episodes/drl_episodes.parquet`
- Original Step 11 selected-case table: `runs/train_reward_c_lite_v5_full/quant_analysis/step11_selected_case_studies.csv`
- Replacement panel-C classification source: `runs/train_reward_c_lite_v5_full/quant_analysis/step11_case_study_scenario_classification.csv`
- Selected episodes:
  - `ABNB_2023-05-25`
  - `AEP_2023-10-05`
  - `HMC_2024-12-19`
- Figure construction: the panels are regenerated from the rollout data using only the value-path axis from the Step 11 diagnostics; the position-fraction subplot is omitted.
- Panel-C replacement note: `HMC_2024-12-19` is a test episode classified as both `dqn_waits_correctly_until_long_term` and `dqn_behaves_like_hold_to_terminal`; it reaches the tax-transition date on 2025-12-19 and has no preferred-DQN discretionary sale.
- Panel illustrations:
  - A. Early sale avoids later loss: ABNB_2023-05-25: The preferred DQN sells early enough to reduce exposure before the later drawdown, finishing above hold-to-terminal in this episode.
  - B. Early sale underperforms hold: AEP_2023-10-05: The preferred DQN liquidates before a subsequent recovery, so the early sale underperforms the hold-to-terminal benchmark.
  - C. No discretionary sale: HMC_2024-12-19: The preferred DQN makes no discretionary sale and remains aligned with hold-to-terminal through the tax-transition date.
- One-sentence interpretation: The three representative paths show that the preferred DQN's liquidation rule can protect against later losses, can also sell too early and miss upside, and can choose a hold-like path when no discretionary sale is favored.
