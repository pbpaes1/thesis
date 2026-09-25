# Brazil sensitivity model freeze v1

Six seed-42 DQNs were trained from scratch on the saved C-lite v5 episode assignments.
Checkpoints maximize validation mean final normalized after-tax wealth under fixed first/subsequent-sale margins 0.070/0.020. Shaped reward and test outcomes were not selection metrics.

| Annual cash rate | Validation wealth | Selected checkpoint | SHA-256 | Run manifest |
| ---: | ---: | --- | --- | --- |
| 7.00% | 1.602603617 | `runs/brazil_sensitivity_rate_0700_v1/best_validation_model.pt` | `7be955459e2ceedee3320dc930726a86f504b8926a71a8ca2e6b5d952dc423d1` | `runs/brazil_sensitivity_rate_0700_v1/reproducibility_manifest.json` |
| 10.00% | 1.611705466 | `runs/brazil_sensitivity_rate_1000_v1/best_validation_model.pt` | `5646540293c25cb5da4a1855d29a4da669049c1cbceac1a11be05e9f2451ff9a` | `runs/brazil_sensitivity_rate_1000_v1/reproducibility_manifest.json` |
| 10.50% | 1.613151928 | `runs/brazil_sensitivity_rate_1050_v1/best_validation_model.pt` | `ac1a836591da6c498cb929e801162571e99df7ccf90855ae2291fc7c0fa4be1f` | `runs/brazil_sensitivity_rate_1050_v1/reproducibility_manifest.json` |
| 12.00% | 1.617488079 | `runs/brazil_sensitivity_rate_1200_v1/best_validation_model.pt` | `8a59a6e7f23d22b5eb85d1eddfd16c2d232b77d92c910013e4a2d667dcd7be9e` | `runs/brazil_sensitivity_rate_1200_v1/reproducibility_manifest.json` |
| 13.75% | 1.618704807 | `runs/brazil_sensitivity_rate_1375_v1/best_validation_model.pt` | `b7bce3c161f8b758534fb1c19fc6e41bec73d6dec3b18c01eb359d5f53d7fb80` | `runs/brazil_sensitivity_rate_1375_v1/reproducibility_manifest.json` |
| 15.00% | 1.621043475 | `runs/brazil_sensitivity_rate_1500_v1/best_validation_model.pt` | `2854e015f1beffbf8a04961176417116e821ed76e3293c1d0beb57d63c46c183` | `runs/brazil_sensitivity_rate_1500_v1/reproducibility_manifest.json` |

The scenario parquet and v2 state freeze hashes, ordered split hashes, effective configs, repository commit and working-tree status are recorded in each run manifest.
Each run used seed 42, gamma 1.0, the 36-input v2 state, 7,083 saved training IDs for three epochs (21,249 passes), and the same 1,517-ID saved validation assignment. The full trainer evaluated the first 500 validation IDs at each of 43 checkpoints, as specified by its frozen `evaluation.max_eval_episodes` setting; the listed scores are means over those 500 IDs. The saved test IDs were used only for split identity and hashing.
The C-lite cooldown penalty shaped training rewards but did not reduce economic wealth or select checkpoints. The fixed Q-value margins were 0.070 before the first sale and 0.020 after it at every rate.
The 12% initial attempt reached its final training pass but failed while appending a large validation CSV on the synced Windows drive. That attempt is archived under `runs/brazil_phase6_failed_1200_first_attempt/`. The successful 12% run was restarted from scratch after validation rollout logging was segmented into smaller CSV parts; the logging change did not alter model actions, accounting, or selection. The first three rates used the prior single-file logger, and the later three used segmented logging. Their trainer hashes and working-tree states are recorded per run.
All selection used validation episodes only. Test outcomes remain unopened; Phase 7 must record the test-access checkpoint before evaluation.
