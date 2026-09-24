"""Build and verify Brazil v1 episodes and the 36-input v2 state freeze.

Run from the repository root with `python src/generate_brazil_episodes_v1.py`.
The original episode parquet and v1 state artifacts are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


OLD_TIME = "days_to_tax_transition_norm"
NEW_TIME = "time_to_terminal_horizon_norm"
FORBIDDEN_STATE = {
    OLD_TIME, "episode_id", "date", "tax_transition_date", "adj_close",
    "simulated_purchase_price", "remaining_fraction", "cash_balance",
    "cash_balance_gross", "equity_tax_step", "fixed_income_tax_step",
    "total_after_tax_wealth", "terminal_price", "terminal_return",
}
RATES = ("0700", "1000", "1050", "1200", "1375", "1500")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_list_hash(ids: list[str]) -> str:
    """Hash UTF-8 IDs joined by LF, without a trailing newline."""
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def validate_source_frame(frame: pd.DataFrame) -> pd.Series:
    required = {"episode_id", "date", "adj_close", "simulated_purchase_price"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Source episodes are missing columns: {missing}.")
    if frame.empty:
        raise ValueError("Source episodes are empty.")
    if frame["episode_id"].isna().any() or frame["date"].isna().any():
        raise ValueError("Source episode IDs and dates must not be missing.")
    if frame["episode_id"].astype(str).eq("").any():
        raise ValueError("Source episode IDs must not be empty.")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Source episode dates must be valid.")
    if frame.duplicated(["episode_id", "date"]).any():
        raise ValueError("Duplicate episode/date pairs in source episodes.")
    date_gaps = dates.groupby(frame["episode_id"], sort=False).diff()
    if (date_gaps.dropna() <= pd.Timedelta(0)).any():
        raise ValueError("Episode dates must be strictly increasing in source row order.")
    for column in ("adj_close", "simulated_purchase_price"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{column} must contain finite positive values.")
    return dates


def derive_terminal_horizon(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the new clock from each episode's actual last observed date."""
    if NEW_TIME in frame.columns:
        raise ValueError(f"Source already contains {NEW_TIME}.")
    dates = validate_source_frame(frame)
    terminal_dates = dates.groupby(frame["episode_id"], sort=False).transform("max")
    calendar_days = (terminal_dates - dates).dt.days
    result = frame.copy(deep=False)
    result[NEW_TIME] = calendar_days.clip(lower=0, upper=365).astype(float) / 365.0
    return result


def validate_scenario_frame(frame: pd.DataFrame, allowed: list[str]) -> None:
    dates = validate_source_frame(frame)
    if NEW_TIME not in frame.columns:
        raise ValueError(f"Scenario episodes are missing {NEW_TIME}.")
    if len(allowed) != 36 or len(set(allowed)) != 36:
        raise ValueError("v2 state must contain exactly 36 unique inputs.")
    if any(column not in frame.columns for column in allowed):
        raise ValueError("v2 state contains a column missing from scenario episodes.")
    if set(allowed) & FORBIDDEN_STATE:
        raise ValueError(f"v2 state contains forbidden/leaking inputs: {sorted(set(allowed) & FORBIDDEN_STATE)}")
    for column in allowed:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"v2 state column {column!r} contains missing or nonfinite values.")
    horizon = pd.to_numeric(frame[NEW_TIME], errors="coerce")
    if horizon.isna().any() or not np.isfinite(horizon.to_numpy(dtype=float)).all():
        raise ValueError("Terminal-horizon values must be finite and nonmissing.")
    if not horizon.between(0.0, 1.0).all():
        raise ValueError("Terminal-horizon values must lie in [0, 1].")
    changes = horizon.groupby(frame["episode_id"], sort=False).diff()
    if (changes.dropna() > 1e-12).any():
        raise ValueError("Terminal-horizon values must be non-increasing within episodes.")
    last_rows = frame.groupby("episode_id", sort=False).tail(1)
    if not np.allclose(last_rows[NEW_TIME].to_numpy(dtype=float), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("Terminal-horizon value must be zero on every final row.")
    expected = (dates.groupby(frame["episode_id"], sort=False).transform("max") - dates).dt.days.clip(lower=0, upper=365) / 365.0
    if not np.allclose(horizon.to_numpy(dtype=float), expected.to_numpy(dtype=float), rtol=0.0, atol=1e-12):
        raise ValueError("Terminal-horizon values do not match actual episode end dates.")


def build_allowed_columns(v1_schema: dict[str, Any]) -> list[str]:
    columns = list(v1_schema["allowed_state_columns"])
    if columns.count(OLD_TIME) != 1 or NEW_TIME in columns or len(columns) != 36:
        raise ValueError("v1 freeze must have exactly 36 inputs and one old tax countdown.")
    columns[columns.index(OLD_TIME)] = NEW_TIME
    return columns


def compare_control_splits(frame: pd.DataFrame, split_path: Path) -> dict[str, Any]:
    saved = pd.read_csv(split_path, dtype=str)
    if not {"episode_id", "split"}.issubset(saved.columns):
        raise ValueError("Control split CSV needs episode_id and split columns.")
    if saved[["episode_id", "split"]].isna().any().any() or saved["episode_id"].duplicated().any():
        raise ValueError("Control split CSV has missing or duplicate episode IDs.")
    if set(saved["split"]) != {"train", "validation", "test"}:
        raise ValueError("Control split CSV must contain train, validation, and test only.")
    first_dates = pd.to_datetime(frame["date"]).groupby(frame["episode_id"], sort=False).first()
    ordered = sorted((str(episode_id) for episode_id in first_dates.index), key=lambda eid: (first_dates.loc[eid], eid))
    n_train = int(len(ordered) * 0.70)
    n_validation = int(len(ordered) * 0.15)
    expected = {
        "train": ordered[:n_train],
        "validation": ordered[n_train:n_train + n_validation],
        "test": ordered[n_train + n_validation:],
    }
    if len(saved) != len(ordered) or set(saved["episode_id"]) != set(ordered):
        raise ValueError("Control splits do not cover the scenario episode universe exactly.")
    details: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        actual = saved.loc[saved["split"] == split, "episode_id"].tolist()
        if actual != expected[split]:
            raise ValueError(f"Control {split} split differs from chronological scenario membership or ordering.")
        details[split] = {"count": len(actual), "ordered_id_sha256": id_list_hash(actual)}
    return details


def validate_config_targets(config_dir: Path) -> None:
    expected_data = "data/episodes/drl_episodes_brazil_v1.parquet"
    expected_schema = "data/freeze/v2/allowed_state_columns_v2.json"
    for suffix in RATES:
        path = config_dir / f"brazil_sensitivity_rate_{suffix}_v1.yaml"
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        env = config["environment"]
        if env["parquet_path"] != expected_data or env["state_schema_path"] != expected_schema:
            raise ValueError(f"Brazil config {path.name} references the wrong episode parquet or v2 freeze.")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_artifacts(
    source_path: Path,
    output_path: Path,
    v1_dir: Path,
    v2_dir: Path,
    split_path: Path,
    config_dir: Path,
) -> dict[str, Any]:
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Scenario output must differ from the original episode parquet.")
    source_hash_before = file_hash(source_path)
    source = pd.read_parquet(source_path)
    with (v1_dir / "allowed_state_columns_v1.json").open(encoding="utf-8") as handle:
        v1_allowed = json.load(handle)
    with (v1_dir / "excluded_columns_v1.json").open(encoding="utf-8") as handle:
        v1_excluded = json.load(handle)
    allowed = build_allowed_columns(v1_allowed)

    # The inventory decision is frozen in brazil_sensitivity_design_v1.md:
    # preserve 36 inputs and disclose partial observability in the summary.
    scenario = derive_terminal_horizon(source)
    if list(scenario.columns) != list(source.columns) + [NEW_TIME]:
        raise AssertionError("Scenario builder must preserve all original columns in order.")
    validate_scenario_frame(scenario, allowed)
    splits = compare_control_splits(scenario, split_path)
    validate_config_targets(config_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario.to_parquet(output_path, index=False)
    read_back = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(read_back[source.columns], source, check_dtype=True)
    validate_scenario_frame(read_back, allowed)
    if file_hash(source_path) != source_hash_before:
        raise AssertionError("Original episode parquet changed during scenario generation.")

    v2_dir.mkdir(parents=True, exist_ok=True)
    source_label = output_path.as_posix()
    write_json(v2_dir / "allowed_state_columns_v2.json", {
        "version": "v2", "source_file": source_label, "allowed_state_columns": allowed,
    })
    v1_reasons = {item["column_name"]: item["reason"] for item in v1_excluded["excluded_columns"]}
    excluded = [
        {"column_name": column, "reason": (
            "U.S. tax-transition countdown; excluded from Brazil v2 observations"
            if column == OLD_TIME else v1_reasons.get(column, "simulator/bookkeeping field excluded from agent state")
        )}
        for column in scenario.columns if column not in allowed
    ]
    write_json(v2_dir / "excluded_columns_v2.json", {
        "version": "v2", "source_file": source_label, "excluded_columns": excluded,
    })

    ids = sorted(scenario["episode_id"].astype(str).unique().tolist())
    summary = {
        "rows": len(scenario), "episodes": len(ids), "columns": len(scenario.columns),
        "allowed_state_columns": len(allowed), "excluded_columns": len(excluded),
        "original_parquet_sha256": source_hash_before,
        "scenario_parquet_sha256": file_hash(output_path),
        "sorted_episode_ids_sha256": id_list_hash(ids),
        "splits": splits,
    }
    lines = [
        "# Brazil sensitivity state freeze v2", "",
        "## Inventory audit and freeze decision", "",
        "The v2 state contains 36 inputs: 35 unchanged v1 inputs and the new terminal-horizon input in the old countdown's position.",
        "Remaining inventory and cash-lot history are known to the simulator but deliberately omitted from observations to preserve the 36-input comparison with U.S. C-lite v5, as chosen for this freeze.",
        "The feedforward policy is therefore partially observable: identical market/time inputs can imply different remaining sale capacity and interest-tax exposure. No static inventory column is created.",
        "", "## Data and validation", "",
        f"- Source: `{source_path.as_posix()}`", f"- Scenario: `{source_label}`",
        f"- Rows: {summary['rows']:,}; episodes: {summary['episodes']:,}; columns: {summary['columns']}; allowed: {len(allowed)}; excluded: {len(excluded)}.",
        "- Source rows and columns retained in order; original parquet hash unchanged after generation.",
        "- Episode dates strictly increase; episode/date pairs are unique; required prices, basis, and state inputs are finite and nonmissing.",
        "- Terminal horizon uses each episode's actual last observation date, is in [0, 1], non-increasing, and zero on every terminal row.",
        "- Old tax countdown remains in the scenario parquet for provenance but is excluded from observations; future prices and simulator accounting are not state inputs.",
        "- Scenario IDs and chronological train/validation/test membership and ordering match the C-lite v5 control exactly.",
        "- All six Brazil configurations reference this parquet and v2 allowed-state JSON.",
        "", "## SHA-256", "",
        "ID-list hashes use UTF-8 IDs joined by LF without a trailing newline. The universe list is sorted lexicographically; split lists retain control CSV order.",
        f"- Original parquet: `{summary['original_parquet_sha256']}`",
        f"- Scenario parquet: `{summary['scenario_parquet_sha256']}`",
        f"- Sorted episode IDs: `{summary['sorted_episode_ids_sha256']}`",
    ]
    for split in ("train", "validation", "test"):
        item = splits[split]
        lines.append(f"- {split}: {item['count']:,} IDs; ordered-list hash `{item['ordered_id_sha256']}`")
    (v2_dir / "state_freeze_v2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/episodes/drl_episodes.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/episodes/drl_episodes_brazil_v1.parquet"))
    parser.add_argument("--v1-freeze", type=Path, default=Path("data/freeze/v1"))
    parser.add_argument("--v2-freeze", type=Path, default=Path("data/freeze/v2"))
    parser.add_argument("--control-splits", type=Path, default=Path("runs/train_reward_c_lite_v5_full/episode_splits.csv"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    args = parser.parse_args()
    summary = build_artifacts(args.input, args.output, args.v1_freeze, args.v2_freeze, args.control_splits, args.configs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
