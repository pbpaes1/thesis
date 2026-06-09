# Step 13 DQN Early-Sale Ex-Post Diagnostic

| split | first_discretionary_sale_timing_bucket | bucket_order | num_episodes | mean_dqn_minus_hold_final_after_tax_value | mean_hold_terminal_after_tax_value | mean_hold_return_after_first_dqn_sale_expost | mean_hold_max_drawdown_after_first_dqn_sale_expost | num_valid_expost_path_episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | over_180_days_before | 0 | 131 | -0.143726 | 0.423975 | 0.710174 | -0.793021 | 123 |
| validation | 91_to_180_days_before | 1 | 310 | -0.0771563 | 0.390217 | 0.461176 | -0.465187 | 297 |
| validation | 31_to_90_days_before | 2 | 110 | -0.0867906 | 0.507694 | 0.527347 | -0.243456 | 108 |
| validation | 8_to_30_days_before | 3 | 133 | -0.0717669 | 0.452375 | 0.464709 | -0.0889475 | 126 |
| validation | 1_to_7_days_before | 4 | 23 | -0.0208145 | 0.311184 | 0.128041 | -0.0622215 | 21 |
| validation | no_discretionary_sale | 99 | 810 | -0.000308921 | 0.591235 |  |  | 0 |
| test | over_180_days_before | 0 | 180 | -0.101557 | 0.412522 | 0.355681 | -0.665956 | 178 |
| test | 91_to_180_days_before | 1 | 196 | -0.118566 | 0.437227 | 0.634182 | -0.440712 | 187 |
| test | 31_to_90_days_before | 2 | 140 | -0.104161 | 0.491792 | -5.91651 | -7.53845 | 138 |
| test | 8_to_30_days_before | 3 | 167 | -0.0610318 | 0.544913 | 0.147896 | -0.128632 | 167 |
| test | 1_to_7_days_before | 4 | 21 | -0.0377973 | 0.756733 | 0.0760815 | -0.0503187 | 21 |
| test | no_discretionary_sale | 99 | 815 | -0.000116221 | 0.528108 |  |  | 0 |

