# Chapter 4 Section 4 Figure Notes

## Figure 4.2: Paired robustness against benchmarks
- Source file used: `runs/train_reward_c_lite_v5_full/quant_analysis/step10_robustness_statistical_significance.csv`
- Filters applied: `split == "test"`; `policy_A == "trained_dqn_first_sale_margin_0p070_normal_0p020"`; `policy_B` in hold-to-terminal, sell-immediately, sell-half-then-hold, sell-quarters-over-time, and random-policy benchmarks.
- Columns used: `split, policy_A, policy_B, mean_difference, mean_difference_ci_low, mean_difference_ci_high`
- Interpretation: On the test split, the preferred DQN is significantly below hold-to-terminal (mean difference -0.044) but has positive mean paired differences against Sell immediately, Sell half, then hold, Sell quarters over time, and Random policy.

## Figure 4.3: Preferred DQN behaviour by economic period
- Source file used: `runs/train_reward_c_lite_v5_full/quant_analysis/step7_preferred_policy_behavior_by_economic_period.csv`
- Filters applied: `split == "all"`; `diagnostic_scope == "all_episode_descriptive"`.
- Columns used: `split, diagnostic_scope, economic_period, calendar_year_min, calendar_year_max, num_episodes, no_cut_episode_pct, discretionary_sale_episode_pct, mean_pct_position_sold_short_term, mean_pct_position_sold_long_term, mean_excess_value_vs_hold_to_terminal, mean_excess_value_vs_sell_immediately`
- Interpretation: Across the full period summary, behaviour is most liquidation-heavy during Inflation tightening (59.8% discretionary-sale episodes) and more hold-like in Post-crisis early recovery (4.2%).

## Figure 4.4: Sector heterogeneity versus hold-to-terminal
- Source file used: `runs/train_reward_c_lite_v5_full/quant_analysis/sector_analysis/sector_policy_performance.csv`
- Filters applied: `split == "test"`; `sector != "Unknown"`.
- Columns used: `split, sector, num_episodes, preferred_minus_hold_mean, preferred_win_rate_vs_hold, preferred_minus_sell_immediately_mean`
- Interpretation: Across classified test sectors, the preferred DQN underperforms hold-to-terminal on mean final after-tax value in every sector, with the largest shortfall in Basic Materials (-0.095) and the least negative gap in Consumer Defensive (-0.014).
