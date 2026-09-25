"""Pairing, bootstrap, completeness and provenance for Phase 8."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.build_brazil_sensitivity_phase8 import (
    BOOTSTRAP_REPS, BOOTSTRAP_SEED, OUT, POLICIES, RATES, ROOT,
    bootstrap_ci, paired_counts, paired_difference, sha256,
)


def test_pairing_reorders_by_id_and_rejects_missing_or_duplicate_ids() -> None:
    ids = ["a", "b", "c"]
    a = pd.Series([20., 10., 30.], index=["b", "a", "c"])
    b = pd.Series([2., 3., 1.], index=["b", "c", "a"])
    assert paired_difference(a, b, ids).tolist() == [9., 18., 27.]
    assert paired_counts(np.array([1., 0., -1.])) == (1, 1, 1)
    with pytest.raises(AssertionError, match="missing or extra"):
        paired_difference(a.drop("c"), b, ids)
    with pytest.raises(AssertionError, match="duplicate"):
        paired_difference(pd.concat([a, a.iloc[:1]]), b, ids)


def test_paired_bootstrap_is_reproducible_with_frozen_seed() -> None:
    diff = np.array([0.1, -0.2, 0.3, 0.4, -0.1])
    a = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(diff), size=(BOOTSTRAP_REPS, len(diff)))
    b = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(diff), size=(BOOTSTRAP_REPS, len(diff)))
    assert np.array_equal(a, b)
    assert bootstrap_ci(diff, a) == bootstrap_ci(diff, b)
    assert bootstrap_ci(diff, a, median=True) == bootstrap_ci(diff, b, median=True)


def test_analysis_outputs_and_source_manifests_are_complete_and_hashed() -> None:
    manifest = json.loads((OUT / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == 8
    assert manifest["source"] == "locked_phase7_test_outputs_only"
    assert manifest["episodes_per_policy_rate"] == 1519
    assert tuple(manifest["rates"]) == RATES
    assert tuple(manifest["policies"]) == POLICIES
    assert manifest["reconciliation_max_abs_residual"] <= 1e-8
    assert len(manifest["source_evaluation_manifests"]) == 6
    assert sha256(ROOT / manifest["test_access_path"]) == manifest["test_access_sha256"]
    for source in manifest["source_evaluation_manifests"]:
        assert sha256(ROOT / source["path"]) == source["sha256"]
        assert sha256(ROOT / source["episode_metrics"]["path"]) == source["episode_metrics"]["sha256"]
        for step in source["step_rollouts"]:
            assert sha256(ROOT / step["path"]) == step["sha256"]
    expected = {"policy_summary.csv", "dqn_vs_hold.csv", "cross_rate_paired.csv",
                "behavior_summary.csv", "action_shares.csv", "reconciliation.csv",
                "final_wealth_by_rate.png", "final_wealth_by_rate.svg",
                "dqn_excess_vs_hold.png", "dqn_excess_vs_hold.svg",
                "sale_and_cash_behavior.png", "sale_and_cash_behavior.svg",
                "dqn_action_shares.png", "dqn_action_shares.svg", "summary.md"}
    assert {entry["path"].split("/")[-1] for entry in manifest["outputs"]} == expected
    for output in manifest["outputs"]:
        assert sha256(ROOT / output["path"]) == output["sha256"]
    tables = OUT / "tables"
    assert len(pd.read_csv(tables / "policy_summary.csv")) == 36
    assert len(pd.read_csv(tables / "dqn_vs_hold.csv")) == 6
    assert len(pd.read_csv(tables / "cross_rate_paired.csv")) == 90
    assert len(pd.read_csv(tables / "behavior_summary.csv")) == 36
    assert len(pd.read_csv(tables / "action_shares.csv")) == 36
    reconcile = pd.read_csv(tables / "reconciliation.csv")
    assert len(reconcile) == 36
    assert reconcile.filter(like="max_").to_numpy().max() <= 1e-8
    policy_summary = pd.read_csv(tables / "policy_summary.csv", dtype={"rate_suffix": str})
    behavior = pd.read_csv(tables / "behavior_summary.csv", dtype={"rate_suffix": str})
    for source in manifest["source_evaluation_manifests"]:
        episodes = pd.read_csv(ROOT / source["episode_metrics"]["path"])
        rate = source["rate_suffix"]
        for policy in POLICIES:
            x = episodes.loc[episodes.policy_name == policy]
            p = policy_summary.loc[(policy_summary.rate_suffix == rate) & (policy_summary.policy_name == policy)].iloc[0]
            b = behavior.loc[(behavior.rate_suffix == rate) & (behavior.policy_name == policy)].iloc[0]
            assert np.isclose(p.mean_final_total_after_tax_wealth, x.final_total_after_tax_wealth.mean())
            assert np.isclose(p.median_final_total_after_tax_wealth, x.final_total_after_tax_wealth.median())
            assert np.isclose(p.mean_excess_vs_same_rate_hold, x.excess_wealth_vs_same_rate_hold.mean())
            assert np.isclose(b.mean_gross_interest, x.gross_interest.mean())
            assert np.isclose(b.mean_settled_interest_tax, x.settled_interest_tax.mean())
            assert np.isclose(b.mean_net_interest, x.net_interest.mean())
            assert np.isclose(b.mean_gross_interest - b.mean_settled_interest_tax, b.mean_net_interest)
