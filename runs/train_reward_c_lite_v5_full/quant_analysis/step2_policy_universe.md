| policy_name                                      | policy_family              | role            | included_in_main_table | notes                                                                  |
| ------------------------------------------------ | -------------------------- | --------------- | ---------------------- | ---------------------------------------------------------------------- |
| hold_to_terminal                                 | benchmark                  | benchmark       | true                   | Never sells before automatic terminal liquidation.                     |
| sell_immediately                                 | benchmark                  | benchmark       | true                   | Sells the full position at the first available decision step.          |
| sell_half_then_hold                              | benchmark                  | benchmark       | true                   | Sells half of the original position, then holds the remainder.         |
| sell_quarters_over_time                          | benchmark                  | benchmark       | true                   | Sells quarter-sized portions over the episode.                         |
| random_policy                                    | benchmark                  | benchmark       | true                   | Uses the evaluator random policy with deterministic seeded evaluation. |
| trained_dqn_greedy                               | raw_dqn                    | raw DQN         | true                   | Greedy policy from the frozen C-lite v5 trained DQN.                   |
| trained_dqn_thresholded_margin_0p020             | thresholded_dqn            | thresholded DQN | true                   | DQN action is filtered by a 0.020 margin threshold.                    |
| trained_dqn_first_sale_margin_0p060_normal_0p020 | first_sale_thresholded_dqn | sensitivity DQN | true                   | First sale requires a 0.060 margin; subsequent sales use 0.020.        |
| trained_dqn_first_sale_margin_0p070_normal_0p020 | first_sale_thresholded_dqn | preferred DQN   | true                   | Preferred final interpretation policy.                                 |
| trained_dqn_first_sale_margin_0p080_normal_0p020 | first_sale_thresholded_dqn | sensitivity DQN | true                   | First sale requires a 0.080 margin; subsequent sales use 0.020.        |
| trained_dqn_first_sale_margin_0p090_normal_0p020 | first_sale_thresholded_dqn | sensitivity DQN | true                   | First sale requires a 0.090 margin; subsequent sales use 0.020.        |
