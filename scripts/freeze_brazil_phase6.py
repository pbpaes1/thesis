"""Capture and verify validation-selected Brazil Phase 6 training artifacts.

No test episode outcomes are loaded. Split IDs are read only for membership and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
RATES = ("0700", "1000", "1050", "1200", "1375", "1500")
SPLITS = ROOT / "runs/train_reward_c_lite_v5_full/episode_splits.csv"
SCENARIO = ROOT / "data/episodes/drl_episodes_brazil_v1.parquet"
SCHEMA = ROOT / "data/freeze/v2/allowed_state_columns_v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def split_hashes() -> dict[str, str]:
    saved = pd.read_csv(SPLITS, dtype=str)
    return {
        name: hashlib.sha256("\n".join(saved.loc[saved["split"] == name, "episode_id"]).encode()).hexdigest()
        for name in ("train", "validation", "test")
    }


def run_dir(rate: str) -> Path:
    if rate not in RATES:
        raise ValueError(f"Unknown rate suffix {rate}.")
    return ROOT / "runs" / f"brazil_sensitivity_rate_{rate}_v1"


def capture_start(rate: str) -> dict:
    directory = run_dir(rate)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs" / f"brazil_sensitivity_rate_{rate}_v1.yaml"
    record = {
        "rate_suffix": rate,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_text("rev-parse", "HEAD"),
        "working_tree_status": git_text("status", "--short", "--untracked-files=normal"),
        "tracked_diff_sha256": hashlib.sha256(git_text("diff", "--binary").encode()).hexdigest(),
        "source_config_sha256": sha256(config_path),
        "trainer_sha256": sha256(ROOT / "scripts/train_dqn_reward_a_full.py"),
        "environment_sha256": sha256(ROOT / "src/environment/tax_aware_env.py"),
        "scenario_parquet_sha256": sha256(SCENARIO),
        "v2_state_freeze_sha256": sha256(SCHEMA),
        "ordered_split_id_sha256": split_hashes(),
    }
    (directory / "run_provenance_start.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _summary(path: Path) -> dict[str, str]:
    return dict(line.split(": ", 1) for line in path.read_text(encoding="utf-8").splitlines()[1:] if ": " in line)


def freeze_run(rate: str, wall_seconds: float) -> dict:
    directory = run_dir(rate)
    start = json.loads((directory / "run_provenance_start.json").read_text(encoding="utf-8"))
    summary = _summary(directory / "training_summary.txt")
    config_path = directory / "config_used.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["training"]["seed"] != 42 or config["training"]["num_epochs"] != 3:
        raise AssertionError("Full run seed or epoch count differs from freeze.")
    if config["training"]["discount_factor_gamma"] != 1.0:
        raise AssertionError("Full run gamma differs from 1.0.")
    if config["evaluation"]["fixed_decision_margins"] != {"first_sale": 0.07, "subsequent_sale": 0.02}:
        raise AssertionError("Full run margins differ from 0.070/0.020.")
    if config["data"]["max_episodes_train"] is not None:
        raise AssertionError("Full run unexpectedly capped training episodes.")
    if int(summary["num_train_episodes"]) != 7083 or int(summary["num_validation_episodes"]) != 1517:
        raise AssertionError("Full run split coverage differs from control.")
    if int(summary["total_training_episode_passes"]) != 7083 * 3 or int(summary["obs_dim"]) != 36:
        raise AssertionError("Full run passes or state width differ from freeze.")
    for field in ("nan_reward_count", "inf_reward_count", "nan_loss_count", "inf_loss_count"):
        if int(summary[field]) != 0:
            raise AssertionError(f"Nonfinite training outcome: {field}={summary[field]}.")
    for path in (SCENARIO, SCHEMA):
        expected = start["scenario_parquet_sha256" if path == SCENARIO else "v2_state_freeze_sha256"]
        if sha256(path) != expected:
            raise AssertionError(f"Input changed during run: {path}.")
    if split_hashes() != start["ordered_split_id_sha256"]:
        raise AssertionError("Saved split IDs changed during run.")
    if sha256(ROOT / "scripts/train_dqn_reward_a_full.py") != start["trainer_sha256"]:
        raise AssertionError("Trainer changed during run.")

    saved = pd.read_csv(directory / "episode_splits.csv", dtype=str)
    control = pd.read_csv(SPLITS, dtype=str)
    if not saved.equals(control):
        raise AssertionError("Run split file differs from saved C-lite v5 control.")

    checkpoint = directory / "best_validation_model.pt"
    if not checkpoint.is_file():
        raise AssertionError("Missing best validation checkpoint.")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if payload["obs_dim"] != 36 or payload["best_model_metric"] != "mean_final_total_after_tax_wealth":
        raise AssertionError("Selected checkpoint has the wrong state or metric.")
    score = float(summary["best_validation_metric_value"])
    if not math.isfinite(score) or not math.isclose(score, float(payload["best_metric_value"]), rel_tol=1e-10):
        raise AssertionError("Selected checkpoint score does not match training summary.")
    metrics = pd.read_csv(directory / "eval_metrics.csv")
    if len(metrics) != math.ceil(7083 * 3 / 500) or not (metrics["num_validation_episodes"] == 500).all():
        raise AssertionError("Validation checkpoint coverage is incomplete.")
    if not math.isclose(score, metrics["mean_final_total_after_tax_wealth"].max(), rel_tol=1e-10):
        raise AssertionError("Selected checkpoint was not the validation wealth maximum.")
    if not math.isclose(score, float(summary["best_validation_metric_value"]), rel_tol=1e-10):
        raise AssertionError("Selected score differs from summary.")
    if int(payload["global_train_episode_idx"]) not in metrics["global_train_episode_idx"].to_numpy():
        raise AssertionError("Selected checkpoint episode is absent from validation records.")
    for filename in ("train_episode_rollouts.csv", "validation_episode_rollouts.csv", "train_metrics.csv"):
        if not (directory / filename).is_file():
            raise AssertionError(f"Missing run record {filename}.")
    rollout_parts = sorted(directory.glob("validation_episode_rollouts*.csv"))
    recorded_validation_checks: set[int] = set()
    for part in rollout_parts:
        indices = pd.read_csv(part, usecols=["evaluation_episode_idx"])["evaluation_episode_idx"]
        recorded_validation_checks.update(indices.dropna().astype(int).unique().tolist())
    if recorded_validation_checks != set(metrics["global_train_episode_idx"].astype(int)):
        raise AssertionError("Validation rollout files do not cover every checkpoint evaluation.")

    bytes_used = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    record = {
        **start,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall_seconds,
        "storage_bytes": bytes_used,
        "effective_config_path": config_path.relative_to(ROOT).as_posix(),
        "effective_config_sha256": sha256(config_path),
        "seed": 42,
        "gamma": 1.0,
        "fixed_decision_margins": {"first_sale": 0.07, "subsequent_sale": 0.02},
        "train_episode_count": 7083,
        "validation_episode_count": 1517,
        "validation_episodes_per_checkpoint": 500,
        "training_passes": 7083 * 3,
        "validation_checkpoints": len(metrics),
        "training_steps": int(summary["total_environment_steps"]),
        "optimization_steps": int(summary["num_optimization_steps"]),
        "selected_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "selected_checkpoint_sha256": sha256(checkpoint),
        "selection_metric": "mean_final_total_after_tax_wealth",
        "selected_validation_score": score,
        "selected_global_train_episode_idx": int(payload["global_train_episode_idx"]),
        "validation_rollout_parts": [part.relative_to(ROOT).as_posix() for part in rollout_parts],
        "finite_training_checks": "passed",
        "test_outcomes_opened": False,
    }
    (directory / "reproducibility_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def write_model_freeze() -> None:
    records = [json.loads((run_dir(rate) / "reproducibility_manifest.json").read_text(encoding="utf-8")) for rate in RATES]
    lines = ["# Brazil sensitivity model freeze v1", "",
             "Six seed-42 DQNs were trained from scratch on the saved C-lite v5 episode assignments.",
             "Checkpoints maximize validation mean final normalized after-tax wealth under fixed first/subsequent-sale margins 0.070/0.020. Shaped reward and test outcomes were not selection metrics.",
             "", "| Annual cash rate | Validation wealth | Selected checkpoint | SHA-256 | Run manifest |",
             "| ---: | ---: | --- | --- | --- |"]
    for rate, record in zip(RATES, records):
        annual_rate = yaml.safe_load((run_dir(rate) / "config_used.yaml").read_text(encoding="utf-8"))["economic_scenario"]["cash_account"]["annual_gross_rate"]
        lines.append(f"| {annual_rate:.2%} | {record['selected_validation_score']:.9f} | `{record['selected_checkpoint']}` | `{record['selected_checkpoint_sha256']}` | `runs/brazil_sensitivity_rate_{rate}_v1/reproducibility_manifest.json` |")
    lines += ["", "The scenario parquet and v2 state freeze hashes, ordered split hashes, effective configs, repository commit and working-tree status are recorded in each run manifest.",
              "Each run used seed 42, gamma 1.0, the 36-input v2 state, 7,083 saved training IDs for three epochs (21,249 passes), and the same 1,517-ID saved validation assignment. The full trainer evaluated the first 500 validation IDs at each of 43 checkpoints, as specified by its frozen `evaluation.max_eval_episodes` setting; the listed scores are means over those 500 IDs. The saved test IDs were used only for split identity and hashing.",
              "The C-lite cooldown penalty shaped training rewards but did not reduce economic wealth or select checkpoints. The fixed Q-value margins were 0.070 before the first sale and 0.020 after it at every rate.",
              "The 12% initial attempt reached its final training pass but failed while appending a large validation CSV on the synced Windows drive. That attempt is archived under `runs/brazil_phase6_failed_1200_first_attempt/`. The successful 12% run was restarted from scratch after validation rollout logging was segmented into smaller CSV parts; the logging change did not alter model actions, accounting, or selection. The first three rates used the prior single-file logger, and the later three used segmented logging. Their trainer hashes and working-tree states are recorded per run.",
              "All selection used validation episodes only. Test outcomes remain unopened; Phase 7 must record the test-access checkpoint before evaluation."]
    path = ROOT / "docs/brazil_sensitivity_model_freeze_v1.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "freeze", "document"))
    parser.add_argument("--rate", choices=RATES)
    parser.add_argument("--wall-seconds", type=float)
    args = parser.parse_args()
    if args.action in {"start", "freeze"} and not args.rate:
        parser.error("--rate is required for start/freeze")
    if args.action == "start":
        print(json.dumps(capture_start(args.rate), indent=2))
    elif args.action == "freeze":
        if args.wall_seconds is None:
            parser.error("--wall-seconds is required for freeze")
        print(json.dumps(freeze_run(args.rate, args.wall_seconds), indent=2))
    else:
        write_model_freeze()


if __name__ == "__main__":
    main()
