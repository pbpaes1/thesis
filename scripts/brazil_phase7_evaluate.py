"""Locked, rate-specific Brazil test rollouts. No model selection or analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_reward_a_baselines import (
    action_index_for_fraction, baseline_action_fraction, load_episode_splits,
    load_trained_q_network, make_env, select_dqn_thresholded_greedy_action,
    _resolve_device,
)
from scripts.freeze_brazil_phase7_access import RECORD, ROOT, RATES, id_hash, sha256, verify_frozen_inputs
from scripts.train_dqn_reward_a_full import load_yaml
from src.accounting.brazil_v1 import gross_lot_value

POLICIES = ("trained_dqn_fixed_margin", "hold_to_terminal", "sell_immediately",
            "sell_half_then_hold", "sell_quarters_over_time", "random_policy")
TOL = 1e-8


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=TOL, abs_tol=TOL):
        raise AssertionError(f"{label}: {actual} != {expected}")


def locked_record() -> tuple[dict, str]:
    if not RECORD.exists():
        raise FileNotFoundError("Test-access checkpoint must exist before any test evaluation.")
    access_sha = sha256(RECORD)
    access = json.loads(RECORD.read_text(encoding="utf-8"))
    if access["record_type"] != "brazil_phase7_test_access_checkpoint":
        raise AssertionError("Invalid test-access record.")
    current = verify_frozen_inputs(require_unopened=False)
    if access["saved_test_id_ordered_sha256"] != current["saved_test_id_ordered_sha256"]:
        raise AssertionError("Test ID hash changed after access freeze.")
    if access["models"] != current["models"]:
        # The pre-access training-manifest hashes are deliberately preserved in the record.
        for a, b in zip(access["models"], current["models"]):
            a = {k: v for k, v in a.items() if k != "training_manifest_pre_access_sha256"}
            b = {k: v for k, v in b.items() if k != "training_manifest_pre_access_sha256"}
            if a != b:
                raise AssertionError("Frozen model/input metadata changed after test-access record.")
    if tuple(access["policies"]) != POLICIES or len(access["models"]) != 6:
        raise AssertionError("Access record policy/model set differs from Phase 7 lock.")
    for model in access["models"]:
        manifest = json.loads((ROOT / model["training_manifest_path"]).read_text(encoding="utf-8"))
        if manifest.get("phase7_test_access_checkpoint_sha256") != access_sha:
            raise AssertionError("Training manifest lacks the saved test-access link.")
    return access, access_sha


def validate_coverage(episodes: pd.DataFrame, ids: list[str]) -> None:
    if len(episodes) != len(ids) * len(POLICIES):
        raise AssertionError("Wrong policy/episode row count.")
    for policy in POLICIES:
        actual = episodes.loc[episodes.policy_name == policy, "episode_id"].tolist()
        if actual != ids or len(set(actual)) != len(ids):
            raise AssertionError(f"Missing, duplicate, or out-of-order test IDs for {policy}.")
    if episodes.duplicated(["policy_name", "episode_id"]).any():
        raise AssertionError("Duplicate policy/episode key.")


def evaluate_rate(model: dict, access_sha: str, expected_ids: list[str]) -> dict:
    rate = model["rate_suffix"]
    config = load_yaml(ROOT / model["effective_config_path"])
    run = ROOT / "runs" / f"brazil_sensitivity_rate_{rate}_v1"
    test_ids = load_episode_splits(run / "episode_splits.csv")["test"]
    if test_ids != expected_ids or id_hash(test_ids) != model.get("test_id_sha256", id_hash(expected_ids)):
        raise AssertionError(f"{rate}: saved test IDs differ in order or membership.")
    env = make_env(config)
    device = _resolve_device(str(config["training"]["device"]))
    q_net = load_trained_q_network(config, ROOT / model["checkpoint_path"],
                                   len(env.state_columns), len(env.action_fractions), device)
    margins = model["fixed_decision_margins"]
    scenario_rate = model["annual_cash_rate"]
    seed = model["seed"]
    output = run / "phase7_test"
    output.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict] = []
    step_files: list[dict] = []
    hold_wealth: dict[str, float] = {}
    checks = {"episodes": 0, "steps": 0, "base_reward_telescoping": 0,
              "tax_cash_identities": 0, "terminal_completion": 0,
              "hold_zero_interest": 0, "immediate_max_holding": 0}
    for policy in POLICIES:
        step_rows: list[dict] = []
        for episode_id in test_ids:
            obs, reset = env.reset(episode_id)
            initial = float(reset["total_after_tax_wealth"])
            episode_start = pd.Timestamp(reset["date"])
            original_terminal = pd.Timestamp(env._current_episode_df.iloc[-1]["date"])
            original_intervals = len(env._current_episode_df) - 1
            rng_seed = int.from_bytes(hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()[:8], "big")
            rng = np.random.default_rng(rng_seed)
            base_sum = penalty_sum = reward_sum = 0.0
            discretionary_fraction = mandatory_fraction = 0.0
            step_index = 0
            while True:
                if policy == "trained_dqn_fixed_margin":
                    action = select_dqn_thresholded_greedy_action(
                        q_net, obs, len(env.action_fractions), device, env,
                        margins["subsequent_sale"], margins["first_sale"])
                else:
                    action = action_index_for_fraction(env, baseline_action_fraction(policy, step_index, rng, env))
                obs, reward, done, truncated, info = env.step(action)
                if truncated or not np.isfinite(obs).all() or not math.isfinite(reward):
                    raise AssertionError(f"{rate}/{policy}/{episode_id}: nonfinite or truncated step")
                for key in ("total_after_tax_wealth", "cash_balance", "cash_principal_cumulative",
                            "cash_interest_gross_cumulative", "fixed_income_tax_cumulative",
                            "equity_tax_cumulative", "gross_sale_proceeds_cumulative"):
                    if not math.isfinite(float(info[key])):
                        raise AssertionError(f"Nonfinite {key}")
                close(info["total_after_tax_wealth"], info["cash_balance"] + info["after_tax_liquidation_value"], "marked wealth")
                interest_tax_liability = (info["fixed_income_tax_cumulative"] if done
                                          else info["fixed_income_tax_estimated_liability"])
                close(info["cash_balance"], info["cash_balance_gross"] - interest_tax_liability, "cash mark")
                close(info["base_reward"], info["total_after_tax_wealth"] - info["previous_total_after_tax_wealth"], "base step")
                close(reward, info["base_reward"] - info["cooldown_penalty"], "shaped step")
                base_sum += float(info["base_reward"])
                penalty_sum += float(info["cooldown_penalty"])
                reward_sum += float(reward)
                discretionary_fraction += float(info["discretionary_sale_fraction_executed"])
                mandatory_fraction += float(info["mandatory_sale_fraction_executed"])
                row = {
                    "split": "test", "policy_name": policy, "episode_id": episode_id,
                    "step": step_index, "action_date": info["date"],
                    "terminal_valuation_date": info["terminal_valuation_date"],
                    "scenario_name": info["scenario_name"], "annual_cash_rate": scenario_rate,
                    "seed": seed, "checkpoint_sha256": model["checkpoint_sha256"],
                    "first_sale_margin": margins["first_sale"],
                    "subsequent_sale_margin": margins["subsequent_sale"],
                    "action": action, "action_fraction_requested": info["action_fraction_requested"],
                    "discretionary_sale_fraction_executed": info["discretionary_sale_fraction_executed"],
                    "mandatory_sale_fraction_executed": info["mandatory_sale_fraction_executed"],
                    "remaining_fraction_before_terminal": info["remaining_fraction_before_terminal"],
                    "remaining_fraction": info["remaining_fraction"],
                    "discretionary_sale_count": info["discretionary_sale_count"],
                    "mandatory_sale_count": info["mandatory_sale_count"],
                    "first_sale_date": info["first_sale_date"],
                    "first_sale_days_from_start": info["first_sale_days_from_start"],
                    "gross_sale_proceeds": info["gross_sale_proceeds"],
                    "gross_sale_proceeds_cumulative": info["gross_sale_proceeds_cumulative"],
                    "discretionary_cash_deposit": info["discretionary_cash_deposit"],
                    "mandatory_after_tax_proceeds": info["mandatory_after_tax_proceeds"],
                    "equity_tax_step": info["equity_tax_step"],
                    "equity_tax_cumulative": info["equity_tax_cumulative"],
                    "cash_principal_cumulative": info["cash_principal_cumulative"],
                    "cash_interest_gross_step": info["cash_interest_gross_step"],
                    "cash_interest_gross_cumulative": info["cash_interest_gross_cumulative"],
                    "fixed_income_tax_estimated_liability": info["fixed_income_tax_estimated_liability"],
                    "fixed_income_tax_step": info["fixed_income_tax_step"],
                    "fixed_income_tax_cumulative": info["fixed_income_tax_cumulative"],
                    "net_interest_if_redeemed": info["cash_interest_gross_cumulative"] - (info["fixed_income_tax_cumulative"] if done else info["fixed_income_tax_estimated_liability"]),
                    "cash_balance": info["cash_balance"],
                    "total_after_tax_wealth": info["total_after_tax_wealth"],
                    "base_reward": info["base_reward"], "cooldown_penalty": info["cooldown_penalty"],
                    "reward": reward, "done": done, "truncated": truncated,
                    "early_episode_termination": bool(done and pd.Timestamp(info["date"]) < original_terminal),
                }
                step_rows.append(row)
                step_index += 1
                checks["steps"] += 1
                if done:
                    break
            final = float(info["total_after_tax_wealth"])
            close(base_sum, final - initial, "base telescoping")
            close(reward_sum, base_sum - penalty_sum, "shaped telescoping")
            close(discretionary_fraction + mandatory_fraction, 1.0, "full position sale")
            close(info["cash_principal_cumulative"] + info["equity_tax_cumulative"] + info["mandatory_after_tax_proceeds"],
                  info["gross_sale_proceeds_cumulative"], "sale proceeds")
            close(final, info["cash_principal_cumulative"] + info["cash_interest_gross_cumulative"]
                  - info["fixed_income_tax_cumulative"] + info["mandatory_after_tax_proceeds"], "terminal wealth")
            close(info["fixed_income_tax_step"], info["fixed_income_tax_cumulative"], "settled interest tax")
            if info["terminal_valuation_date"] != original_terminal or info["remaining_fraction"] != 0:
                raise AssertionError("Terminal completion or original valuation horizon failed.")
            if policy == "hold_to_terminal":
                close(info["cash_interest_gross_cumulative"], 0, "hold interest")
                close(info["cash_principal_cumulative"], 0, "hold cash")
                hold_wealth[episode_id] = final
                checks["hold_zero_interest"] += 1
            if policy == "sell_immediately":
                close(info["cash_interest_gross_cumulative"],
                      gross_lot_value(info["cash_principal_cumulative"], scenario_rate, original_intervals)
                      - info["cash_principal_cumulative"], "immediate full-horizon interest")
                if pd.Timestamp(info["first_sale_date"]) != episode_start:
                    raise AssertionError("Immediate sale did not occur at episode start.")
                checks["immediate_max_holding"] += 1
            checks["episodes"] += 1
            checks["base_reward_telescoping"] += 1
            checks["tax_cash_identities"] += 1
            checks["terminal_completion"] += 1
            episode_rows.append({
                "split": "test", "policy_name": policy, "episode_id": episode_id,
                "scenario_name": info["scenario_name"], "annual_cash_rate": scenario_rate,
                "seed": seed, "checkpoint_sha256": model["checkpoint_sha256"],
                "first_sale_margin": margins["first_sale"], "subsequent_sale_margin": margins["subsequent_sale"],
                "episode_start_date": episode_start, "last_action_date": info["date"],
                "terminal_valuation_date": info["terminal_valuation_date"],
                "early_episode_termination": bool(pd.Timestamp(info["date"]) < original_terminal),
                "steps": step_index, "initial_total_after_tax_wealth": initial,
                "final_total_after_tax_wealth": final, "base_reward_sum": base_sum,
                "cooldown_penalty_sum": penalty_sum, "shaped_reward_sum": reward_sum,
                "discretionary_sale_count": info["discretionary_sale_count"],
                "mandatory_sale_count": info["mandatory_sale_count"],
                "discretionary_sale_fraction": discretionary_fraction,
                "mandatory_sale_fraction": mandatory_fraction,
                "remaining_fraction_before_terminal": info["remaining_fraction_before_terminal"],
                "remaining_fraction_after_terminal": info["remaining_fraction"],
                "first_sale_date": info["first_sale_date"],
                "first_sale_days_from_start": info["first_sale_days_from_start"],
                "gross_sale_proceeds_cumulative": info["gross_sale_proceeds_cumulative"],
                "cash_principal": info["cash_principal_cumulative"],
                "gross_interest": info["cash_interest_gross_cumulative"],
                "settled_interest_tax": info["fixed_income_tax_cumulative"],
                "net_interest": info["cash_interest_gross_cumulative"] - info["fixed_income_tax_cumulative"],
                "equity_tax": info["equity_tax_cumulative"],
                "mandatory_after_tax_proceeds": info["mandatory_after_tax_proceeds"],
                "done": done,
            })
        step_path = output / f"{policy}_step_rollouts.parquet"
        pd.DataFrame(step_rows).to_parquet(step_path, index=False)
        step_files.append({"policy_name": policy, "path": step_path.relative_to(ROOT).as_posix(),
                           "sha256": sha256(step_path), "rows": len(step_rows)})
        print(f"{rate} {policy}: {len(test_ids)} episodes, {len(step_rows)} steps", flush=True)
    episodes = pd.DataFrame(episode_rows)
    validate_coverage(episodes, test_ids)
    episodes["excess_wealth_vs_same_rate_hold"] = episodes.apply(
        lambda row: row["final_total_after_tax_wealth"] - hold_wealth[row["episode_id"]], axis=1)
    required_numeric = ("final_total_after_tax_wealth", "base_reward_sum", "shaped_reward_sum",
                        "cash_principal", "gross_interest", "settled_interest_tax",
                        "net_interest", "equity_tax", "excess_wealth_vs_same_rate_hold")
    if not episodes["done"].all() or not np.isfinite(episodes[list(required_numeric)]).all().all():
        raise AssertionError("Incomplete or nonfinite episode result.")
    episode_path = output / "test_episode_metrics.csv"
    episodes.to_csv(episode_path, index=False)
    manifest = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "phase": 7,
        "rate_suffix": rate, "annual_cash_rate": scenario_rate,
        "test_access_checkpoint": RECORD.relative_to(ROOT).as_posix(),
        "test_access_checkpoint_sha256": access_sha,
        "frozen_checkpoint": model["checkpoint_path"], "frozen_checkpoint_sha256": model["checkpoint_sha256"],
        "effective_config": model["effective_config_path"], "effective_config_sha256": model["effective_config_sha256"],
        "scenario_parquet": model["scenario_parquet_path"], "scenario_parquet_sha256": model["scenario_parquet_sha256"],
        "state_freeze": model["state_freeze_path"], "state_freeze_sha256": model["state_freeze_sha256"],
        "test_id_ordered_sha256": id_hash(test_ids), "test_id_count": len(test_ids),
        "seed": seed, "fixed_decision_margins": margins, "policies": list(POLICIES),
        "episode_metrics": {"path": episode_path.relative_to(ROOT).as_posix(), "sha256": sha256(episode_path), "rows": len(episodes)},
        "step_rollouts": step_files, "checks": checks,
    }
    manifest_path = output / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    training_manifest_path = ROOT / model["training_manifest_path"]
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    training_manifest["test_outcomes_opened"] = True
    training_manifest["phase7_evaluation_manifest"] = manifest_path.relative_to(ROOT).as_posix()
    training_manifest["phase7_evaluation_manifest_sha256"] = sha256(manifest_path)
    training_manifest_path.write_text(json.dumps(training_manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", choices=RATES, required=True)
    args = parser.parse_args()
    access, access_sha = locked_record()
    ids = load_episode_splits(ROOT / access["saved_test_ids_path"])["test"]
    if len(ids) != 1519 or id_hash(ids) != access["saved_test_id_ordered_sha256"]:
        raise AssertionError("Access-record test IDs do not match saved order.")
    model = next(model for model in access["models"] if model["rate_suffix"] == args.rate)
    evaluate_rate(model, access_sha, ids)


if __name__ == "__main__":
    main()
