# Brazil-inspired sensitivity: locked test analysis

## Scope and method

All numbers below use only the six hashed Phase 7 test outputs, the same 1,519 ordered episode IDs, and the six validation-selected seed-42 DQNs. Wealth is normalized by original position basis and measured at each episode's original terminal date. The first/subsequent-sale Q-value margins remain 0.070/0.020. Within each rate, DQN excess is paired with hold on the same episode ID. Percentile bootstrap intervals resample the 1,519 paired IDs 4,000 times with seed 20260925; they measure episode sampling uncertainty conditional on the frozen models and historical paths.

## Primary comparison: frozen DQN versus same-rate hold

| Cash rate | DQN mean wealth | Hold mean wealth | Mean excess (95% paired CI) | Median excess | Wins / ties / losses |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7.00% | 1.4379 | 1.4771 | -0.0392 [-0.0545, -0.0255] | +0.0000 | 574 / 247 / 698 |
| 10.00% | 1.4513 | 1.4771 | -0.0258 [-0.0407, -0.0121] | +0.0000 | 684 / 133 / 702 |
| 10.50% | 1.4526 | 1.4771 | -0.0245 [-0.0392, -0.0112] | +0.0000 | 683 / 143 / 693 |
| 12.00% | 1.4560 | 1.4771 | -0.0211 [-0.0355, -0.0082] | +0.0000 | 606 / 263 / 650 |
| 13.75% | 1.4536 | 1.4771 | -0.0235 [-0.0377, -0.0102] | +0.0000 | 601 / 257 / 661 |
| 15.00% | 1.4696 | 1.4771 | -0.0075 [-0.0218, +0.0055] | +0.0000 | 710 / 163 / 646 |

The paired median-excess intervals are in `tables/policy_summary.csv`. Figures `figures/final_wealth_by_rate.*` and `figures/dqn_excess_vs_hold.*` show the performance paths and paired uncertainty.

## Final normalized wealth for every policy

Each cell is mean / median at the original terminal date. `tables/policy_summary.csv` retains unrounded values and paired comparisons.

| Cash rate | Frozen DQN | Hold | Immediate | Half then hold | Quarters | Random |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7.00% | 1.4379 / 1.3381 | 1.4771 / 1.3835 | 1.4369 / 1.3274 | 1.4570 / 1.3605 | 1.4377 / 1.3302 | 1.4374 / 1.3290 |
| 10.00% | 1.4513 / 1.3539 | 1.4771 / 1.3835 | 1.4469 / 1.3443 | 1.4620 / 1.3666 | 1.4475 / 1.3458 | 1.4473 / 1.3453 |
| 10.50% | 1.4526 / 1.3548 | 1.4771 / 1.3835 | 1.4485 / 1.3472 | 1.4628 / 1.3681 | 1.4491 / 1.3480 | 1.4489 / 1.3486 |
| 12.00% | 1.4560 / 1.3663 | 1.4771 / 1.3835 | 1.4535 / 1.3564 | 1.4653 / 1.3708 | 1.4539 / 1.3554 | 1.4538 / 1.3574 |
| 13.75% | 1.4536 / 1.3634 | 1.4771 / 1.3835 | 1.4592 / 1.3659 | 1.4681 / 1.3741 | 1.4596 / 1.3640 | 1.4595 / 1.3665 |
| 15.00% | 1.4696 / 1.3769 | 1.4771 / 1.3835 | 1.4633 / 1.3717 | 1.4702 / 1.3754 | 1.4635 / 1.3707 | 1.4635 / 1.3725 |

## Across-rate and behavior checks

`tables/cross_rate_paired.csv` contains all 15 rate pairs for each of the six policies. Each difference is higher-rate wealth minus lower-rate wealth on the same episode ID, with wins, ties, losses and a paired 95% bootstrap interval for the mean. These are comparisons among separately trained policies under a combined scenario change; they do not isolate a tax-only effect.

| Policy | Paired mean wealth difference, 15% minus 7% (95% CI) | Wins / ties / losses at 15% |
| --- | ---: | ---: |
| trained dqn fixed margin | +0.0317 [+0.0278, +0.0357] | 1067 / 128 / 324 |
| hold to terminal | +0.0000 [+0.0000, +0.0000] | 0 / 1519 / 0 |
| sell immediately | +0.0264 [+0.0253, +0.0274] | 1477 / 42 / 0 |
| sell half then hold | +0.0132 [+0.0126, +0.0137] | 1477 / 42 / 0 |
| sell quarters over time | +0.0259 [+0.0248, +0.0269] | 1477 / 42 / 0 |
| random policy | +0.0261 [+0.0250, +0.0272] | 1472 / 47 / 0 |

| Cash rate | DQN sale share | Mean first sale day if sold | Mean discretionary fraction | Mean remaining fraction before terminal | Mean gross interest | Mean interest tax | Mean net interest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7.00% | 85.1% | 11.0 | 0.819 | 0.181 | 0.0263 | 0.0056 | 0.0208 |
| 10.00% | 92.0% | 12.2 | 0.903 | 0.097 | 0.0379 | 0.0080 | 0.0298 |
| 10.50% | 92.4% | 15.2 | 0.896 | 0.104 | 0.0371 | 0.0079 | 0.0292 |
| 12.00% | 83.6% | 15.0 | 0.796 | 0.204 | 0.0421 | 0.0089 | 0.0332 |
| 13.75% | 84.1% | 17.7 | 0.810 | 0.190 | 0.0467 | 0.0099 | 0.0369 |
| 15.00% | 90.2% | 12.0 | 0.876 | 0.124 | 0.0553 | 0.0117 | 0.0436 |

Across the 7%–15% endpoints, the frozen DQN's paired mean wealth difference is +0.0317 (95% CI [+0.0278, +0.0357]). Its mean net interest changes from 0.0208 to 0.0436; mean discretionary fraction sold changes from 0.819 to 0.876. The gross-interest, settled-interest-tax, net-interest, gross-sale-proceeds, equity-tax, sale-count, first-sale, inventory, early-termination and action-share breakdowns for every policy are in `tables/behavior_summary.csv` and `tables/action_shares.csv`.

`figures/sale_and_cash_behavior.*` shows selected behavior measures, and `figures/dqn_action_shares.*` shows the five DQN action shares. Each figure has PNG and SVG versions.

## Interpretation and limits

The treatments combine a flat 15% tax on positive equity gains with a rate-specific cash account. Higher cash rates can increase the value of earlier sale proceeds; the observed policy and wealth differences also reflect separate training at each rate. These results support a sensitivity comparison for the combined treatment, not a tax-only or causal estimate. The cash rate is fixed within an episode, and the study uses historical paths of U.S.-listed stocks in USD without BRL conversion.

The [Banco Central Focus report with reference date 11 September 2026](https://www.bcb.gov.br/content/focus/focus/R20260911.pdf) lists end-year Selic medians of 13.75% (2026), 12% (2027), 10.5% (2028), and 10% (2029). The 7% scenario is a hypothetical lower bound, not a Focus forecast. The [Banco Central June 2025 Copom communication](https://www.bcb.gov.br/controleinflacao/comunicadoscopom/20733) documents 15% as a historical policy-rate anchor, not a current Focus projection. These rates are scenario inputs, not modeled Brazilian cash instruments.

There is one training seed per rate. The 36-input feedforward policy omits remaining inventory and cash-lot history, so its observation is partially observable. Checkpoints were selected using the predetermined first 500 validation IDs, not the 1,519 test IDs. Brazil training used gamma = 1.0; the historical U.S. C-lite v5 run used 0.99. The paired episode bootstrap does not account for overlapping ticker/time paths, seed variation, or market-regime uncertainty.

## Original U.S. control (separate metric)

The original U.S. C-lite v5 control remains in `runs/train_reward_c_lite_v5_full/`. Its reported `after_tax_total_value` is a tax-adjusted PnL measure; Brazil `final_total_after_tax_wealth` includes recovered basis and cash interest. No U.S. output was used as a Phase 8 numerical input, and the two metrics are not plotted on one scale.

## Reproducibility and reconciliation

Run `py -3.13 scripts/build_brazil_sensitivity_phase8.py` from the repository root. The builder verifies the access record, checkpoint/config/data/state hashes, all six evaluation manifests and output hashes, ordered episode coverage, terminal step coverage, and accounting identities before writing this analysis. `analysis_manifest.json` records every input and generated output hash. `tables/reconciliation.csv` contains the maximum row-level residuals for net interest, wealth, sale proceeds, base reward, sale fractions and same-rate hold excess. All residuals must be at most 1e-8. No model was retrained or evaluated by this builder.
