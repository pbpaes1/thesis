"""Phase 7 freeze, coverage, and saved rollout integration checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.brazil_phase7_evaluate import POLICIES, locked_record, validate_coverage
from scripts.freeze_brazil_phase7_access import ROOT, RATES, id_hash, sha256
from scripts.evaluate_reward_a_baselines import load_episode_splits


def test_test_access_precedes_outcomes_and_preserves_all_six_frozen_inputs() -> None:
    access, digest = locked_record()
    assert access["saved_test_id_count"] == 1519
    assert len(access["models"]) == 6
    assert tuple(access["policies"]) == POLICIES
    ids = load_episode_splits(ROOT / access["saved_test_ids_path"])["test"]
    assert id_hash(ids) == access["saved_test_id_ordered_sha256"]
    for model in access["models"]:
        manifest = json.loads((ROOT / model["training_manifest_path"]).read_text(encoding="utf-8"))
        assert manifest["phase7_test_access_checkpoint_sha256"] == digest
        assert sha256(ROOT / model["checkpoint_path"]) == model["checkpoint_sha256"]


def test_saved_rate_outputs_cover_exact_paired_test_ids_and_terminal_steps() -> None:
    access, digest = locked_record()
    ids = load_episode_splits(ROOT / access["saved_test_ids_path"])["test"]
    for rate in RATES:
        output = ROOT / "runs" / f"brazil_sensitivity_rate_{rate}_v1" / "phase7_test"
        manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["test_access_checkpoint_sha256"] == digest
        assert manifest["test_id_ordered_sha256"] == id_hash(ids)
        assert manifest["test_id_count"] == 1519
        assert manifest["checks"]["episodes"] == 1519 * len(POLICIES)
        assert manifest["checks"]["hold_zero_interest"] == 1519
        assert manifest["checks"]["immediate_max_holding"] == 1519
        episode_file = ROOT / manifest["episode_metrics"]["path"]
        assert sha256(episode_file) == manifest["episode_metrics"]["sha256"]
        episodes = pd.read_csv(episode_file)
        validate_coverage(episodes, ids)
        assert episodes["done"].all()
        assert (episodes["remaining_fraction_after_terminal"] == 0).all()
        assert (episodes["terminal_valuation_date"] >= episodes["last_action_date"]).all()
        assert np.allclose(episodes["discretionary_sale_fraction"] + episodes["mandatory_sale_fraction"], 1)
        assert np.allclose(episodes["base_reward_sum"],
                           episodes["final_total_after_tax_wealth"] - episodes["initial_total_after_tax_wealth"])
        assert np.allclose(episodes["final_total_after_tax_wealth"],
                           episodes["cash_principal"] + episodes["gross_interest"]
                           - episodes["settled_interest_tax"] + episodes["mandatory_after_tax_proceeds"])
        assert np.allclose(episodes["gross_sale_proceeds_cumulative"],
                           episodes["cash_principal"] + episodes["equity_tax"]
                           + episodes["mandatory_after_tax_proceeds"])
        hold = episodes.loc[episodes.policy_name == "hold_to_terminal"]
        assert (hold["gross_interest"] == 0).all()
        assert (hold["excess_wealth_vs_same_rate_hold"] == 0).all()
        hold_wealth = hold.set_index("episode_id")["final_total_after_tax_wealth"]
        assert np.allclose(episodes["excess_wealth_vs_same_rate_hold"],
                           episodes["final_total_after_tax_wealth"]
                           - episodes["episode_id"].map(hold_wealth))
        for step in manifest["step_rollouts"]:
            path = ROOT / step["path"]
            assert sha256(path) == step["sha256"]
            table = pq.read_table(path, columns=["episode_id", "done", "terminal_valuation_date"])
            frame = table.to_pandas()
            assert len(frame) == step["rows"]
            completed = frame.loc[frame.done]
            assert completed.episode_id.tolist() == ids
            assert completed.terminal_valuation_date.notna().all()
