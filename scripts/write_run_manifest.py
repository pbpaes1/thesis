"""Write a reproducibility manifest for a training run.

Run from the project root:
    python scripts/write_run_manifest.py --config configs/train_reward_c_lite_v1.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError(
        "PyYAML is required to run scripts/write_run_manifest.py. "
        "Install it with: pip install pyyaml"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.tax_profiles import resolve_tax_profile_from_config  # noqa: E402


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_reward_c_lite_v1.yaml"


def _require(config: dict, path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing required config value: {path}")
        current = current[key]
    return current


def _get_nested(config: dict, path: str, default: Any = None) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping at top level: {path}")
    return payload


def hash_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in ("numpy", "pandas", "torch", "pyyaml"):
        try:
            versions[package_name if package_name != "pyyaml" else "yaml"] = (
                metadata.version(package_name)
            )
        except metadata.PackageNotFoundError:
            versions[package_name if package_name != "pyyaml" else "yaml"] = None
    return versions


def run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_metadata() -> tuple[str | None, str | None]:
    commit_hash = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    dirty_status = None if status is None else ("dirty" if status else "clean")
    return commit_hash, dirty_status


def list_created_artifacts(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    artifacts = [
        relative_project_path(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    ]
    return artifacts


def build_manifest(config: dict, config_path: Path) -> dict[str, Any]:
    output_dir = resolve_project_path(_require(config, "logging.output_dir"))
    dataset_path = resolve_project_path(_require(config, "environment.parquet_path"))
    state_schema_path = resolve_project_path(
        _require(config, "environment.state_schema_path")
    )
    resolved_tax_profile = resolve_tax_profile_from_config(
        config,
        base_dir=PROJECT_ROOT,
    )
    git_commit_hash, git_dirty_status = git_metadata()

    final_model_path = output_dir / "final_model.pt"
    best_validation_model_path = output_dir / "best_validation_model.pt"
    exploitation_policy = str(
        _get_nested(config, "training.exploitation_policy", "greedy")
    )
    thresholded_margin = _get_nested(config, "training.thresholded_greedy.margin")
    if thresholded_margin is not None:
        thresholded_margin = float(thresholded_margin)
    exploration_action_probabilities = _get_nested(
        config,
        "exploration.action_probabilities",
    )

    return {
        "run_name": _require(config, "run.name"),
        "reward_version": _require(config, "reward.version"),
        "base_reward_version": config.get("reward", {}).get("base_reward_version"),
        "config_path": relative_project_path(config_path),
        "config_hash": hash_file(config_path),
        "reward_config_hash": hash_json_payload(config.get("reward", {})),
        "dataset_path": relative_project_path(dataset_path),
        "dataset_file_size": dataset_path.stat().st_size if dataset_path.exists() else None,
        "dataset_hash": hash_file(dataset_path),
        "state_schema_path": relative_project_path(state_schema_path),
        "state_schema_hash": hash_file(state_schema_path),
        "tax_profile_name": resolved_tax_profile["profile_name"],
        "tax_profile_values": resolved_tax_profile,
        "discount_factor_gamma": float(
            _require(config, "training.discount_factor_gamma")
        ),
        "exploitation_policy": exploitation_policy,
        "thresholded_greedy_margin": thresholded_margin,
        "exploration_action_probabilities": exploration_action_probabilities,
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "git_commit_hash": git_commit_hash,
        "git_dirty_status": git_dirty_status,
        "training_command": (
            "python scripts/train_dqn_reward_a_full.py --config "
            f"{relative_project_path(config_path)}"
        ),
        "baseline_command": (
            "python scripts/evaluate_reward_a_baselines.py --config "
            f"{relative_project_path(config_path)}"
        ),
        "behavior_inspection_command": (
            "python scripts/inspect_reward_a_policy_behavior.py --config "
            f"{relative_project_path(config_path)}"
        ),
        "created_artifacts": list_created_artifacts(output_dir),
        "final_model_path": relative_project_path(final_model_path),
        "best_validation_model_path": relative_project_path(best_validation_model_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a run reproducibility manifest.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Training config path. Defaults to "
            f"{relative_project_path(DEFAULT_CONFIG_PATH)}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)
    output_dir = resolve_project_path(_require(config, "logging.output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config, config_path)
    manifest_path = output_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote run manifest: {relative_project_path(manifest_path)}")


if __name__ == "__main__":
    main()
