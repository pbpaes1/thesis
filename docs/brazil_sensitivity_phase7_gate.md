# Brazil sensitivity Phase 7 gate

Completed 2026-09-25. This record concerns locked test evaluation only; it contains no Phase 8 sensitivity analysis.

## Frozen access

- Before test evaluation, `scripts/freeze_brazil_phase7_access.py` verified all six selected checkpoint SHA-256 values against both `docs/brazil_sensitivity_model_freeze_v1.md` and the corresponding training manifests.
- The dated, preserved access record is `runs/brazil_phase7_test_access/test_access_checkpoint.json` (2026-09-25T20:07:58.662287+00:00). It records checkpoint, effective config, scenario parquet, v2 state, ordered test-ID hashes, seed 42, and fixed first/subsequent-sale margins 0.070/0.020. Each training manifest links to its hash.
- The ordered 1,519-ID test hash is `273d2547e2cba607dbbb87a2f5125a304516caf70fd7269859968b30c2b6f98c` for every rate.

## Outputs and gate checks

Each `runs/brazil_sensitivity_rate_<rate>_v1/phase7_test/` directory contains `test_episode_metrics.csv`, six policy-specific `*_step_rollouts.parquet` files, and `evaluation_manifest.json`. Rates are `0700`, `1000`, `1050`, `1200`, `1375`, and `1500`. The manifests link output hashes to the access record, checkpoint, effective config, scenario dataset, and state freeze.

Each rate has 9,114 episode rows: 1,519 ordered IDs for each of the frozen DQN with transferred margins, hold-to-terminal, immediate full sale, half-then-hold, quarters-over-time, and random policy. Per-episode rollout checks passed for finite economic values, terminal completion at the original valuation horizon, sale fractions, tax/cash/proceeds identities, and base-reward telescoping. All 1,519 hold episodes per rate had zero cash principal and interest; all 1,519 immediate-sale episodes per rate matched gross interest for the maximum available market-day holding period. The saved-output integration test verifies output hashes, policy/ID order, duplicate absence, terminal step coverage, and hold excess wealth of zero.

| Annual cash rate | Episode rows | Step rows |
| ---: | ---: | ---: |
| 7.00% | 9,114 | 285,464 |
| 10.00% | 9,114 | 283,277 |
| 10.50% | 9,114 | 294,563 |
| 12.00% | 9,114 | 294,905 |
| 13.75% | 9,114 | 296,365 |
| 15.00% | 9,114 | 287,552 |

Focused Phase 7 integration tests: 2 passed. Full repository suite: 75 passed, 2 skipped, 39 subtests passed. No checkpoint, margin, or policy was changed after access was recorded. The U.S. control was not evaluated in this phase.
