# Step 13 Tax-Transition Timing Summary

| split | policy_label | num_episodes | mean_pct_original_sold_before_long_term | mean_pct_original_sold_at_or_after_long_term | pct_episodes_with_discretionary_sale_before_long_term | pct_episodes_with_no_discretionary_sale | median_days_remaining_at_first_realization | median_days_remaining_at_first_discretionary_sale | mean_final_after_tax_total_value | mean_excess_final_after_tax_value_vs_hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | Preferred DQN | 1517 | 44.5452 | 55.4548 | 47.0666 | 52.9334 | 0 | 122 | 0.471991 | -0.0412442 |
| validation | Passive hold-to-terminal | 1517 | 0 | 100 | 0 | 100 | 0 |  | 0.513235 | 0 |
| validation | Naive active: sell immediately | 1517 | 99.6704 | 0.329598 | 99.6704 | 0 | 126 | 126 | 0.423216 | -0.0900195 |
| validation | Naive active: sell half then hold | 1517 | 49.8352 | 50.1648 | 99.6704 | 0 | 126 | 134 | 0.468225 | -0.0450098 |
| validation | Naive active: sell quarters | 1517 | 94.8583 | 5.14173 | 99.6704 | 0 | 126 | 134 | 0.425059 | -0.0881763 |
| validation | Random benchmark | 1517 | 97.2643 | 2.73566 | 98.8794 | 0.856955 | 126 | 133 | 0.425027 | -0.0882086 |
| test | Preferred DQN | 1519 | 43.1369 | 56.8631 | 46.6096 | 53.3904 | 0 | 97.5 | 0.460117 | -0.0442281 |
| test | Passive hold-to-terminal | 1519 | 0 | 100 | 0 | 100 | 0 |  | 0.504345 | 0 |
| test | Naive active: sell immediately | 1519 | 97.7617 | 2.23831 | 97.7617 | 0 | 127 | 127 | 0.394303 | -0.110042 |
| test | Naive active: sell half then hold | 1519 | 48.8808 | 51.1192 | 97.7617 | 0 | 127 | 138 | 0.449324 | -0.0550211 |
| test | Naive active: sell quarters | 1519 | 94.7663 | 5.23371 | 97.7617 | 0 | 127 | 138 | 0.397274 | -0.107072 |
| test | Random benchmark | 1519 | 95.6715 | 4.32851 | 97.1034 | 1.05332 | 126 | 135 | 0.396589 | -0.107756 |

