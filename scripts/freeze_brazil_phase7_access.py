"""Verify frozen Brazil models and record test access without reading outcomes."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RATES = ("0700", "1000", "1050", "1200", "1375", "1500")
RECORD = ROOT / "runs/brazil_phase7_test_access/test_access_checkpoint.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["episode_id"] for row in csv.DictReader(handle) if row["split"] == "test"]


def id_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def verify_frozen_inputs(*, require_unopened: bool = True) -> dict:
    freeze = (ROOT / "docs/brazil_sensitivity_model_freeze_v1.md").read_text(encoding="utf-8")
    frozen = dict(re.findall(
        r"runs/brazil_sensitivity_rate_(\d{4})_v1/best_validation_model\.pt` \| `([0-9a-f]{64})`",
        freeze,
    ))
    if set(frozen) != set(RATES):
        raise AssertionError("Model freeze must list exactly six checkpoints.")
    ids = split_ids(ROOT / "runs/train_reward_c_lite_v5_full/episode_splits.csv")
    if len(ids) != 1519 or len(set(ids)) != len(ids):
        raise AssertionError("Frozen test assignment must contain 1,519 unique IDs.")
    test_hash = id_hash(ids)
    models = []
    for rate in RATES:
        run = ROOT / "runs" / f"brazil_sensitivity_rate_{rate}_v1"
        manifest_path = run / "reproducibility_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = run / "best_validation_model.pt"
        config_path = run / "config_used.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data_path = ROOT / config["environment"]["parquet_path"]
        state_path = ROOT / config["environment"]["state_schema_path"]
        checks = {
            "selected_checkpoint_sha256": (sha256(checkpoint), frozen[rate]),
            "effective_config_sha256": (sha256(config_path), manifest["effective_config_sha256"]),
            "scenario_parquet_sha256": (sha256(data_path), manifest["scenario_parquet_sha256"]),
            "v2_state_freeze_sha256": (sha256(state_path), manifest["v2_state_freeze_sha256"]),
            "test_id_sha256": (test_hash, manifest["ordered_split_id_sha256"]["test"]),
            "run_test_id_sha256": (id_hash(split_ids(run / "episode_splits.csv")), test_hash),
        }
        for name, (actual, expected) in checks.items():
            if actual != expected:
                raise AssertionError(f"{rate}: {name} mismatch: {actual} != {expected}")
        if manifest["selected_checkpoint"] != checkpoint.relative_to(ROOT).as_posix():
            raise AssertionError(f"{rate}: selected checkpoint path mismatch")
        if manifest["seed"] != config["training"]["seed"] or manifest["seed"] != 42:
            raise AssertionError(f"{rate}: seed mismatch")
        margins = {"first_sale": 0.07, "subsequent_sale": 0.02}
        if manifest["fixed_decision_margins"] != margins or config["evaluation"]["fixed_decision_margins"] != margins:
            raise AssertionError(f"{rate}: margin mismatch")
        expected_rate = int(rate) / 10000
        if config["economic_scenario"]["cash_account"]["annual_gross_rate"] != expected_rate:
            raise AssertionError(f"{rate}: scenario rate mismatch")
        if require_unopened and manifest.get("test_outcomes_opened") is not False:
            raise AssertionError(f"{rate}: manifest does not affirm unopened test outcomes")
        models.append({
            "rate_suffix": rate, "annual_cash_rate": expected_rate,
            "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": checks["selected_checkpoint_sha256"][0],
            "effective_config_path": config_path.relative_to(ROOT).as_posix(),
            "effective_config_sha256": checks["effective_config_sha256"][0],
            "scenario_parquet_path": data_path.relative_to(ROOT).as_posix(),
            "scenario_parquet_sha256": checks["scenario_parquet_sha256"][0],
            "state_freeze_path": state_path.relative_to(ROOT).as_posix(),
            "state_freeze_sha256": checks["v2_state_freeze_sha256"][0],
            "training_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "training_manifest_pre_access_sha256": sha256(manifest_path),
            "seed": 42, "fixed_decision_margins": margins,
        })
    return {
        "record_type": "brazil_phase7_test_access_checkpoint",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "model_freeze_path": "docs/brazil_sensitivity_model_freeze_v1.md",
        "model_freeze_sha256": sha256(ROOT / "docs/brazil_sensitivity_model_freeze_v1.md"),
        "saved_test_ids_path": "runs/train_reward_c_lite_v5_full/episode_splits.csv",
        "saved_test_id_count": 1519, "saved_test_id_ordered_sha256": test_hash,
        "policies": ["trained_dqn_fixed_margin", "hold_to_terminal", "sell_immediately",
                     "sell_half_then_hold", "sell_quarters_over_time", "random_policy"],
        "models": models,
    }


def main() -> None:
    if RECORD.exists():
        raise FileExistsError(f"Access record already exists: {RECORD}")
    record = verify_frozen_inputs()
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    with RECORD.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    # Each training manifest points to the immutable pre-outcome access record.
    for model in record["models"]:
        path = ROOT / model["training_manifest_path"]
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["phase7_test_access_checkpoint"] = RECORD.relative_to(ROOT).as_posix()
        manifest["phase7_test_access_checkpoint_sha256"] = sha256(RECORD)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified six frozen checkpoints; wrote {RECORD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
