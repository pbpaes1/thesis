# Step 13 Long-Term Sold Percentage By Starting Tax-Transition Distance

| split | policy_label | start_tax_transition_bucket | num_episodes | mean_start_days_to_tax_transition | mean_pct_original_sold_before_long_term | mean_pct_original_sold_at_or_after_long_term | pct_episodes_with_no_discretionary_sale | mean_final_after_tax_total_value | mean_excess_final_after_tax_value_vs_hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | Preferred DQN | 0_to_30_days | 335 | 10 | 3.8806 | 96.1194 | 94.3284 | 0.689169 | -0.00160252 |
| validation | Preferred DQN | 31_to_60_days | 164 | 45.878 | 15.7012 | 84.2988 | 81.0976 | 0.59735 | -0.0155608 |
| validation | Preferred DQN | 61_to_90_days | 134 | 74.4627 | 18.8433 | 81.1567 | 79.8507 | 0.486259 | -0.0144446 |
| validation | Preferred DQN | 91_to_120_days | 110 | 104.655 | 34.0909 | 65.9091 | 63.6364 | 0.435019 | -0.0296533 |
| validation | Preferred DQN | 121_to_180_days | 152 | 149.658 | 61.3487 | 38.6513 | 32.8947 | 0.505942 | -0.0544091 |
| validation | Preferred DQN | over_180_days | 622 | 273.265 | 77.3312 | 22.6688 | 20.418 | 0.317137 | -0.0739726 |
| validation | Passive hold-to-terminal | 0_to_30_days | 335 | 10 | 0 | 100 | 100 | 0.690771 | 0 |
| validation | Passive hold-to-terminal | 31_to_60_days | 164 | 45.878 | 0 | 100 | 100 | 0.61291 | 0 |
| validation | Passive hold-to-terminal | 61_to_90_days | 134 | 74.4627 | 0 | 100 | 100 | 0.500704 | 0 |
| validation | Passive hold-to-terminal | 91_to_120_days | 110 | 104.655 | 0 | 100 | 100 | 0.464672 | 0 |
| validation | Passive hold-to-terminal | 121_to_180_days | 152 | 149.658 | 0 | 100 | 100 | 0.560351 | 0 |
| validation | Passive hold-to-terminal | over_180_days | 622 | 273.265 | 0 | 100 | 100 | 0.39111 | 0 |
| validation | Naive active: sell immediately | 0_to_30_days | 335 | 10 | 98.5075 | 1.49254 | 0 | 0.628959 | -0.0618121 |
| validation | Naive active: sell immediately | 31_to_60_days | 164 | 45.878 | 100 | 0 | 0 | 0.507782 | -0.105128 |
| validation | Naive active: sell immediately | 61_to_90_days | 134 | 74.4627 | 100 | 0 | 0 | 0.45715 | -0.043554 |
| validation | Naive active: sell immediately | 91_to_120_days | 110 | 104.655 | 100 | 0 | 0 | 0.407731 | -0.0569415 |
| validation | Naive active: sell immediately | 121_to_180_days | 152 | 149.658 | 100 | 0 | 0 | 0.438067 | -0.122283 |
| validation | Naive active: sell immediately | over_180_days | 622 | 273.265 | 100 | 0 | 0 | 0.281906 | -0.109204 |
| validation | Naive active: sell half then hold | 0_to_30_days | 335 | 10 | 49.2537 | 50.7463 | 0 | 0.659865 | -0.0309061 |
| validation | Naive active: sell half then hold | 31_to_60_days | 164 | 45.878 | 50 | 50 | 0 | 0.560346 | -0.0525641 |
| validation | Naive active: sell half then hold | 61_to_90_days | 134 | 74.4627 | 50 | 50 | 0 | 0.478927 | -0.021777 |
| validation | Naive active: sell half then hold | 91_to_120_days | 110 | 104.655 | 50 | 50 | 0 | 0.436202 | -0.0284708 |
| validation | Naive active: sell half then hold | 121_to_180_days | 152 | 149.658 | 50 | 50 | 0 | 0.499209 | -0.0611417 |
| validation | Naive active: sell half then hold | over_180_days | 622 | 273.265 | 50 | 50 | 0 | 0.336508 | -0.0546018 |
| validation | Naive active: sell quarters | 0_to_30_days | 335 | 10 | 76.7164 | 23.2836 | 0 | 0.644494 | -0.0462775 |
| validation | Naive active: sell quarters | 31_to_60_days | 164 | 45.878 | 100 | 0 | 0 | 0.506655 | -0.106256 |
| validation | Naive active: sell quarters | 61_to_90_days | 134 | 74.4627 | 100 | 0 | 0 | 0.453289 | -0.0474146 |
| validation | Naive active: sell quarters | 91_to_120_days | 110 | 104.655 | 100 | 0 | 0 | 0.404855 | -0.0598175 |
| validation | Naive active: sell quarters | 121_to_180_days | 152 | 149.658 | 100 | 0 | 0 | 0.434182 | -0.126169 |
| validation | Naive active: sell quarters | over_180_days | 622 | 273.265 | 100 | 0 | 0 | 0.280622 | -0.110488 |
| validation | Random benchmark | 0_to_30_days | 335 | 10 | 87.6119 | 12.3881 | 3.8806 | 0.63836 | -0.0524111 |
| validation | Random benchmark | 31_to_60_days | 164 | 45.878 | 100 | 0 | 0 | 0.507816 | -0.105095 |
| validation | Random benchmark | 61_to_90_days | 134 | 74.4627 | 100 | 0 | 0 | 0.456925 | -0.0437792 |
| validation | Random benchmark | 91_to_120_days | 110 | 104.655 | 100 | 0 | 0 | 0.40629 | -0.0583823 |
| validation | Random benchmark | 121_to_180_days | 152 | 149.658 | 100 | 0 | 0 | 0.435855 | -0.124496 |
| validation | Random benchmark | over_180_days | 622 | 273.265 | 100 | 0 | 0 | 0.282095 | -0.109015 |
| test | Preferred DQN | 0_to_30_days | 299 | 11.1171 | 9.699 | 90.301 | 86.2876 | 0.580393 | -0.00821469 |
| test | Preferred DQN | 31_to_60_days | 145 | 43.9241 | 28.1034 | 71.8966 | 63.4483 | 0.514015 | -0.0237281 |
| test | Preferred DQN | 61_to_90_days | 157 | 74.4522 | 54.4586 | 45.5414 | 33.758 | 0.448633 | -0.0477845 |
| test | Preferred DQN | 91_to_120_days | 138 | 105.486 | 48.0072 | 51.9928 | 50 | 0.505431 | -0.0507107 |
| test | Preferred DQN | 121_to_180_days | 187 | 148.422 | 64.5722 | 35.4278 | 33.1551 | 0.42017 | -0.0618503 |
| test | Preferred DQN | over_180_days | 593 | 269.678 | 52.7825 | 47.2175 | 46.7116 | 0.391386 | -0.059392 |
| test | Passive hold-to-terminal | 0_to_30_days | 299 | 11.1171 | 0 | 100 | 100 | 0.588608 | 0 |
| test | Passive hold-to-terminal | 31_to_60_days | 145 | 43.9241 | 0 | 100 | 100 | 0.537743 | 0 |
| test | Passive hold-to-terminal | 61_to_90_days | 157 | 74.4522 | 0 | 100 | 100 | 0.496418 | 0 |
| test | Passive hold-to-terminal | 91_to_120_days | 138 | 105.486 | 0 | 100 | 100 | 0.556142 | 0 |
| test | Passive hold-to-terminal | 121_to_180_days | 187 | 148.422 | 0 | 100 | 100 | 0.48202 | 0 |
| test | Passive hold-to-terminal | over_180_days | 593 | 269.678 | 0 | 100 | 100 | 0.450778 | 0 |
| test | Naive active: sell immediately | 0_to_30_days | 299 | 11.1171 | 88.6288 | 11.3712 | 0 | 0.529406 | -0.0592017 |
| test | Naive active: sell immediately | 31_to_60_days | 145 | 43.9241 | 100 | 0 | 0 | 0.447236 | -0.0905069 |
| test | Naive active: sell immediately | 61_to_90_days | 157 | 74.4522 | 100 | 0 | 0 | 0.408145 | -0.0882731 |
| test | Naive active: sell immediately | 91_to_120_days | 138 | 105.486 | 100 | 0 | 0 | 0.441031 | -0.115111 |
| test | Naive active: sell immediately | 121_to_180_days | 187 | 148.422 | 100 | 0 | 0 | 0.392475 | -0.0895454 |
| test | Naive active: sell immediately | over_180_days | 593 | 269.678 | 100 | 0 | 0 | 0.299277 | -0.151501 |
| test | Naive active: sell half then hold | 0_to_30_days | 299 | 11.1171 | 44.3144 | 55.6856 | 0 | 0.559007 | -0.0296009 |
| test | Naive active: sell half then hold | 31_to_60_days | 145 | 43.9241 | 50 | 50 | 0 | 0.49249 | -0.0452535 |
| test | Naive active: sell half then hold | 61_to_90_days | 157 | 74.4522 | 50 | 50 | 0 | 0.452281 | -0.0441365 |
| test | Naive active: sell half then hold | 91_to_120_days | 138 | 105.486 | 50 | 50 | 0 | 0.498586 | -0.0575555 |
| test | Naive active: sell half then hold | 121_to_180_days | 187 | 148.422 | 50 | 50 | 0 | 0.437248 | -0.0447727 |
| test | Naive active: sell half then hold | over_180_days | 593 | 269.678 | 50 | 50 | 0 | 0.375027 | -0.0757505 |
| test | Naive active: sell quarters | 0_to_30_days | 299 | 11.1171 | 73.913 | 26.087 | 0 | 0.539027 | -0.0495809 |
| test | Naive active: sell quarters | 31_to_60_days | 145 | 43.9241 | 100 | 0 | 0 | 0.450822 | -0.0869213 |
| test | Naive active: sell quarters | 61_to_90_days | 157 | 74.4522 | 100 | 0 | 0 | 0.410621 | -0.085797 |
| test | Naive active: sell quarters | 91_to_120_days | 138 | 105.486 | 98.913 | 1.08696 | 0 | 0.447098 | -0.109043 |
| test | Naive active: sell quarters | 121_to_180_days | 187 | 148.422 | 100 | 0 | 0 | 0.395906 | -0.0861147 |
| test | Naive active: sell quarters | over_180_days | 593 | 269.678 | 100 | 0 | 0 | 0.298009 | -0.152769 |
| test | Random benchmark | 0_to_30_days | 299 | 11.1171 | 78.0936 | 21.9064 | 5.35117 | 0.537822 | -0.0507861 |
| test | Random benchmark | 31_to_60_days | 145 | 43.9241 | 100 | 0 | 0 | 0.451094 | -0.086649 |
| test | Random benchmark | 61_to_90_days | 157 | 74.4522 | 100 | 0 | 0 | 0.41048 | -0.085938 |
| test | Random benchmark | 91_to_120_days | 138 | 105.486 | 99.8188 | 0.181159 | 0 | 0.44408 | -0.112062 |
| test | Random benchmark | 121_to_180_days | 187 | 148.422 | 100 | 0 | 0 | 0.395716 | -0.0863047 |
| test | Random benchmark | over_180_days | 593 | 269.678 | 100 | 0 | 0 | 0.297596 | -0.153182 |

