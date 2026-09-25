"""Phase 5 accounting anchors and capped train/validation smoke runs only."""

from __future__ import annotations

import json
import hashlib
import math
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_reward_a_baselines import evaluate_brazil_validation  # noqa: E402
from scripts.train_dqn_reward_a_full import load_yaml, make_env, train_full  # noqa: E402
from src.config.tax_profiles import resolve_economic_scenario_from_config  # noqa: E402
from src.environment.tax_aware_env import TaxAwareEnv  # noqa: E402


RATES = ("0700", "1000", "1050", "1200", "1375", "1500")
SMOKE_RATES = ("0700", "1200", "1500")
OUT = ROOT / "runs" / "brazil_phase5_smoke"


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10):
        raise AssertionError(f"{label}: environment={actual:.12g}, independent={expected:.12g}")


def accounting_anchors() -> list[dict]:
    """Use direct basis-100 arithmetic, independent of accounting helper functions."""
    results = []
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "three_rows.parquet"
        pd.DataFrame({
            "episode_id": ["anchor"] * 3,
            "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-07-01"]),
            "adj_close": [120.0] * 3,
            "simulated_purchase_price": [100.0] * 3,
            "feature": [0.0, 1.0, 2.0],
        }).to_parquet(path, index=False)
        for suffix in RATES:
            config = load_yaml(ROOT / "configs" / f"brazil_sensitivity_rate_{suffix}_v1.yaml")
            rate = float(config["economic_scenario"]["cash_account"]["annual_gross_rate"])
            for policy, actions in (("hold", [0, 0, 0]), ("immediate", [4]), ("staggered", [1, 1, 0])):
                env = TaxAwareEnv(path, ["feature"], reward_config=config["reward"],
                                  economic_scenario=resolve_economic_scenario_from_config(config))
                _, reset = env.reset("anchor")
                infos = []
                for action in actions:
                    _, _, done, _, info = env.step(action)
                    infos.append(info)
                if not done:
                    raise AssertionError(f"{suffix} {policy} did not terminate.")
                final = infos[-1]
                if policy == "hold":
                    principal = interest = interest_tax = 0.0
                    wealth = 1.17
                elif policy == "immediate":
                    principal = 1.17
                    interest = principal * ((1 + rate) ** (2 / 252) - 1)
                    interest_tax = 0.20 * interest  # 181 calendar days
                    wealth = principal + interest - interest_tax
                else:
                    principal = 0.585
                    first_interest = 0.2925 * ((1 + rate) ** (2 / 252) - 1)
                    second_interest = 0.2925 * ((1 + rate) ** (1 / 252) - 1)
                    interest = first_interest + second_interest
                    interest_tax = 0.20 * first_interest + 0.225 * second_interest
                    wealth = 0.585 + principal + interest - interest_tax
                for key, expected in (
                    ("gross_sale_proceeds_cumulative", 1.2),
                    ("equity_tax_cumulative", 0.03),
                    ("cash_principal_cumulative", principal),
                    ("cash_interest_gross_cumulative", interest),
                    ("fixed_income_tax_cumulative", interest_tax),
                    ("total_after_tax_wealth", wealth),
                ):
                    _close(float(final[key]), expected, f"{suffix} {policy} {key}")
                _close(interest - interest_tax, float(final["cash_interest_gross_cumulative"])
                       - float(final["fixed_income_tax_cumulative"]), "net interest")
                _close(sum(float(x["base_reward"]) for x in infos),
                       wealth - float(reset["total_after_tax_wealth"]), "telescoping")
                results.append({"rate": rate, "policy": policy, "final_wealth": wealth,
                                "gross_interest": interest, "settled_interest_tax": interest_tax,
                                "net_interest": interest - interest_tax, "gross_proceeds": 1.2,
                                "equity_tax": 0.03, "cash_principal": principal})
    return results


def run_smoke(suffix: str) -> dict:
    source = ROOT / "configs" / f"brazil_sensitivity_rate_{suffix}_v1.yaml"
    config = deepcopy(load_yaml(source))
    output = OUT / f"rate_{suffix}"
    config["run"].update({"name": f"brazil_phase5_smoke_rate_{suffix}", "phase": "phase5_smoke", "status": "debug_only"})
    config["logging"]["output_dir"] = output.relative_to(ROOT).as_posix()
    config["training"]["num_epochs"] = 1
    config["data"]["max_episodes_train"] = 20
    config["evaluation"]["max_eval_episodes"] = 20
    config["evaluation"]["evaluation_frequency_episodes"] = 10
    config["training"]["checkpoint_frequency_episodes"] = 10
    for key, value in config["logging"]["csv_logs"].items():
        config["logging"]["csv_logs"][key] = f"{config['logging']['output_dir']}/{Path(value).name}"
    start = time.perf_counter()
    summary = train_full(config, config_path=source)
    elapsed = time.perf_counter() - start
    env = make_env(config)
    saved = pd.read_csv(ROOT / config["splits"]["source_csv"], dtype=str)
    first_train_id = saved.loc[saved["split"] == "train", "episode_id"].iloc[0]
    observation, _ = env.reset(first_train_id)
    if observation.shape != (36,) or not np.isfinite(observation).all():
        raise AssertionError("Smoke observation is not finite 36-input state.")
    _, _, done, _, probe = env.step(4)
    expected_terminal_date = pd.Timestamp(env._current_episode_df.iloc[-1]["date"])
    if (not done or probe["terminal_valuation_date"] != expected_terminal_date
            or probe["terminal_valuation_date"] <= probe["date"]
            or probe["cash_interest_gross_cumulative"] <= 0):
        raise AssertionError("Forced full sale did not settle at original horizon.")
    if any(summary[key] for key in ("nan_reward_count", "inf_reward_count", "nan_loss_count", "inf_loss_count")):
        raise AssertionError("Smoke training produced nonfinite rewards or losses.")
    benchmark_episodes, benchmark_steps = evaluate_brazil_validation(
        config, max_episodes=20, output_dir=output / "validation_baselines")
    if set(benchmark_episodes["split"]) != {"validation"} or set(benchmark_steps["split"]) != {"validation"}:
        raise AssertionError("Smoke benchmark touched a non-validation split.")
    bytes_used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    checkpoint_bytes = sum(path.stat().st_size for path in output.rglob("*.pt") if path.is_file())
    return {"rate": suffix, "train_wall_seconds": elapsed, "output_bytes": bytes_used,
            "checkpoint_bytes": checkpoint_bytes, "train_steps": summary["total_environment_steps"],
            "optimization_steps": summary["num_optimization_steps"],
            "best_validation_metric": summary["best_validation_metric_value"],
            "best_model_metric": summary["best_model_metric"], "obs_dim": summary["obs_dim"],
            "forced_sale_date": str(probe["date"]),
            "forced_terminal_valuation_date": str(probe["terminal_valuation_date"]),
            "validation_benchmark_episodes": len(benchmark_episodes),
            "effective_config": summary["config_used_path"],
            "output_dir": output.relative_to(ROOT).as_posix()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    anchors = accounting_anchors()
    pd.DataFrame(anchors).to_csv(OUT / "independent_accounting_anchors.csv", index=False)
    jobs = [run_smoke(suffix) for suffix in SMOKE_RATES]
    split_csv = ROOT / "runs/train_reward_c_lite_v5_full/episode_splits.csv"
    saved = pd.read_csv(split_csv, dtype=str)
    split_hashes = {
        name: hashlib.sha256("\n".join(saved.loc[saved["split"] == name, "episode_id"]).encode()).hexdigest()
        for name in ("train", "validation", "test")
    }
    full_passes = 7083 * 3
    report = {"accounting_cases": len(anchors), "smoke_runs": jobs,
              "control_split_ordered_id_sha256": split_hashes,
              "estimated_full_train_seconds_per_rate": [
                  job["train_wall_seconds"] * (full_passes / 20) for job in jobs],
              "estimated_full_storage_bytes_per_rate": [
                  (job["output_bytes"] - job["checkpoint_bytes"]) * (full_passes / 20)
                  + (job["checkpoint_bytes"] / 5) * (math.ceil(full_passes / 500) + 2)
                  for job in jobs],
              "estimate_note": "Approximate linear scaling of non-checkpoint artifacts plus 500-episode checkpoints; validation and I/O overhead will differ."}
    (OUT / "phase5_smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
