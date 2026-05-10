"""Inspect learned action behavior from existing baseline rollouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError(
        "PyYAML is required to run scripts/inspect_reward_a_policy_behavior.py. "
        "Install it with: pip install pyyaml"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.tax_profiles import resolve_tax_profile_from_config  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "train_reward_a_v3.yaml"
DQN_POLICY_NAME = "trained_dqn_greedy"

STEP_REQUIRED_COLUMNS = [
    "split",
    "policy_name",
    "episode_id",
    "step_in_episode",
    "date",
    "action_idx",
    "action_fraction_requested",
    "action_fraction_executed",
    "reward",
    "reward_A",
    "reward_C_lite",
    "reward_C_lite_v2",
    "transaction_penalty",
    "transaction_penalty_applied",
    "cooldown_penalty",
    "cooldown_penalty_applied",
    "days_since_last_sale",
    "sale_count",
    "last_sale_date_before_step",
    "last_sale_date_after_step",
    "after_tax_total_value",
    "sold_fraction",
    "remaining_fraction",
    "tax_regime",
    "applicable_tax_rate",
    "after_tax_liquidation_tax_regime",
    "is_automatic_terminal_liquidation",
    "terminal_liquidation_executed",
    "done",
]

EPISODE_REQUIRED_COLUMNS = [
    "split",
    "policy_name",
    "episode_id",
    "episode_total_reward",
    "episode_total_reward_A",
    "episode_total_reward_C_lite",
    "episode_total_reward_C_lite_v2",
    "episode_total_transaction_penalty",
    "episode_total_cooldown_penalty",
    "transaction_penalty_frequency",
    "mean_transaction_penalty_when_applied",
    "cooldown_penalty_frequency",
    "mean_cooldown_penalty_when_applied",
    "episode_final_after_tax_total_value",
    "episode_realized_after_tax_pnl",
    "episode_steps",
    "episode_terminal_liquidation_executed",
    "episode_final_remaining_fraction",
    "episode_final_sold_fraction",
    "episode_full_liquidation",
    "episode_cut_occurred",
    "first_cut_step",
    "first_cut_date",
    "first_cut_fraction_executed",
    "days_to_first_sale",
    "steps_to_first_sale",
    "first_sale_date",
    "first_sale_before_tax_transition",
    "pct_episode_position_sold_short_term",
    "pct_episode_position_sold_long_term",
    "total_position_sold_short_term",
    "total_position_sold_long_term",
    "mean_effective_tax_rate_on_sales",
    "total_tax_paid",
    "total_transaction_penalty",
    "total_cooldown_penalty",
    "num_discretionary_sales",
    "num_cooldown_penalized_sales",
    "sold_before_tax_transition_flag",
    "reached_tax_transition_before_first_sale_flag",
]

SUMMARY_REQUIRED_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    "mean_episode_total_reward",
    "mean_episode_total_reward_A",
    "mean_episode_total_reward_C_lite",
    "mean_episode_total_reward_C_lite_v2",
    "mean_episode_total_transaction_penalty",
    "mean_episode_total_cooldown_penalty",
    "mean_final_after_tax_total_value",
    "mean_episode_steps",
    "terminal_liquidation_frequency",
    "full_liquidation_frequency",
    "cut_frequency",
    "sell_immediately_frequency",
    "mean_first_cut_step",
    "mean_first_cut_fraction_executed",
    "average_days_to_first_sale",
    "median_days_to_first_sale",
    "average_steps_to_first_sale",
    "pct_episodes_with_first_sale_before_tax_transition",
    "pct_episodes_sold_before_tax_transition",
    "mean_pct_position_sold_short_term",
    "mean_pct_position_sold_long_term",
    "mean_effective_tax_rate",
    "mean_total_tax_paid",
    "mean_transaction_penalty",
    "mean_cooldown_penalty",
    "mean_num_discretionary_sales",
    "mean_num_cooldown_penalized_sales",
]

ACTION_DISTRIBUTION_COLUMNS = [
    "split",
    "action_idx",
    "action_fraction_requested",
    "step_count",
    "step_share",
    "mean_executed_fraction",
    "mean_reward",
    "median_reward",
    "mean_remaining_fraction_after_action",
    "terminal_step_count",
]

FIRST_CUT_SUMMARY_COLUMNS = [
    "split",
    "num_episodes",
    "cut_frequency",
    "no_cut_frequency",
    "mean_first_cut_step",
    "median_first_cut_step",
    "p25_first_cut_step",
    "p75_first_cut_step",
    "mean_first_cut_fraction_executed",
    "median_first_cut_fraction_executed",
    "sell_immediately_frequency",
    "terminal_liquidation_frequency",
    "full_liquidation_frequency",
    "mean_episode_steps",
    "median_episode_steps",
]

DQN_EPISODE_BEHAVIOR_COLUMNS = [
    "split",
    "episode_id",
    "episode_total_reward",
    "episode_final_after_tax_total_value",
    "episode_realized_after_tax_pnl",
    "episode_steps",
    "episode_terminal_liquidation_executed",
    "episode_final_remaining_fraction",
    "episode_final_sold_fraction",
    "episode_full_liquidation",
    "episode_cut_occurred",
    "first_cut_step",
    "first_cut_date",
    "first_cut_fraction_executed",
    "behavior_label",
]

POLICY_COMPARISON_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    "mean_episode_total_reward",
    "median_episode_total_reward",
    "mean_episode_total_reward_A",
    "mean_episode_total_reward_C_lite",
    "mean_episode_total_reward_C_lite_v2",
    "mean_episode_total_transaction_penalty",
    "mean_episode_total_cooldown_penalty",
    "mean_final_after_tax_total_value",
    "median_final_after_tax_total_value",
    "mean_realized_after_tax_pnl",
    "median_realized_after_tax_pnl",
    "mean_episode_steps",
    "median_episode_steps",
    "terminal_liquidation_frequency",
    "full_liquidation_frequency",
    "cut_frequency",
    "sell_immediately_frequency",
    "mean_first_cut_step",
    "median_first_cut_step",
    "mean_first_cut_fraction_executed",
    "average_days_to_first_sale",
    "pct_episodes_with_first_sale_before_tax_transition",
    "mean_pct_position_sold_short_term",
    "mean_pct_position_sold_long_term",
    "mean_total_tax_paid",
    "mean_transaction_penalty",
    "mean_cooldown_penalty",
]

COOLDOWN_PENALTY_SUMMARY_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    "mean_episode_total_cooldown_penalty",
    "median_episode_total_cooldown_penalty",
    "cooldown_penalty_episode_frequency",
    "mean_cooldown_penalty_frequency",
    "mean_episode_steps",
    "cut_frequency",
    "terminal_liquidation_frequency",
    "mean_final_after_tax_total_value",
]

TRANSACTION_PENALTY_SUMMARY_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    "mean_episode_total_transaction_penalty",
    "median_episode_total_transaction_penalty",
    "transaction_penalty_episode_frequency",
    "mean_transaction_penalty_frequency",
    "mean_episode_steps",
    "cut_frequency",
    "terminal_liquidation_frequency",
    "mean_final_after_tax_total_value",
]

EARLY_SELLING_SUMMARY_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    "average_days_to_first_sale",
    "median_days_to_first_sale",
    "average_steps_to_first_sale",
    "pct_episodes_with_first_sale_before_tax_transition",
    "mean_pct_position_sold_short_term",
    "mean_pct_position_sold_long_term",
    "mean_effective_tax_rate",
    "mean_total_tax_paid",
    "mean_final_after_tax_total_value",
    "terminal_liquidation_frequency",
    "cut_frequency",
    "sell_immediately_frequency",
]

INPUT_PATHS: dict[str, Path] = {}


def _relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _require(config: dict, path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing required config value: {path}")
        current = current[key]
    return current


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping at top level: {path}")
    return payload


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0.0)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y"])


def _compact_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    table_df = df if max_rows is None else df.head(max_rows)
    if table_df.empty:
        return "(empty)"
    return table_df.to_string(index=False)


def load_inputs(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step_path = base_dir / "baseline_step_rollouts.csv"
    episode_path = base_dir / "baseline_episode_metrics.csv"
    summary_path = base_dir / "baseline_summary_by_policy.csv"
    for path in (step_path, episode_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Required baseline input file not found: {path}")

    global INPUT_PATHS
    INPUT_PATHS = {
        "baseline_step_rollouts": step_path,
        "baseline_episode_metrics": episode_path,
        "baseline_summary_by_policy": summary_path,
    }

    step_df = pd.read_csv(
        step_path,
        usecols=lambda col: col in STEP_REQUIRED_COLUMNS,
        low_memory=False,
    )
    episode_df = pd.read_csv(
        episode_path,
        usecols=lambda col: col in EPISODE_REQUIRED_COLUMNS,
    )
    summary_df = pd.read_csv(
        summary_path,
        usecols=lambda col: col in SUMMARY_REQUIRED_COLUMNS,
    )
    return step_df, episode_df, summary_df


def assert_tax_profiles_match(
    saved_tax_profile: dict[str, Any],
    current_tax_profile: dict[str, Any],
    *,
    artifact_path: Path,
) -> None:
    fields = (
        "profile_name",
        "short_term_rate",
        "long_term_rate",
        "niit_rate",
        "apply_niit",
    )
    mismatches: list[str] = []
    for field in fields:
        saved_value = saved_tax_profile.get(field)
        current_value = current_tax_profile.get(field)
        if saved_value is None or current_value is None:
            if saved_value != current_value:
                mismatches.append(
                    f"{field}: saved={saved_value!r}, current={current_value!r}"
                )
            continue
        if isinstance(current_value, bool):
            if not isinstance(saved_value, bool) or saved_value != current_value:
                mismatches.append(
                    f"{field}: saved={saved_value!r}, current={current_value!r}"
                )
        elif isinstance(current_value, (int, float)):
            try:
                saved_number = float(saved_value)
                current_number = float(current_value)
            except (TypeError, ValueError):
                mismatches.append(
                    f"{field}: saved={saved_value!r}, current={current_value!r}"
                )
                continue
            if not np.isclose(saved_number, current_number):
                mismatches.append(
                    f"{field}: saved={saved_value!r}, current={current_value!r}"
                )
        elif saved_value != current_value:
            mismatches.append(
                f"{field}: saved={saved_value!r}, current={current_value!r}"
            )

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Baseline artifact tax profile does not match the current behavior "
            f"inspection config for {artifact_path}: {mismatch_text}. Rerun "
            "baseline evaluation before inspecting policy behavior."
        )


def validate_baseline_tax_profile(
    baseline_dir: Path,
    current_tax_profile: dict[str, Any],
) -> None:
    baseline_config_path = baseline_dir / "baseline_config_used.yaml"
    if not baseline_config_path.exists():
        raise FileNotFoundError(
            f"Required baseline config file not found: {baseline_config_path}"
        )
    baseline_config = load_yaml(baseline_config_path)
    saved_tax_profile = baseline_config.get("resolved_tax_profile")
    if saved_tax_profile is None:
        training_config = baseline_config.get("training_config")
        if not isinstance(training_config, dict):
            raise ValueError(
                "Baseline config does not include resolved_tax_profile or "
                f"training_config: {baseline_config_path}"
            )
        saved_tax_profile = resolve_tax_profile_from_config(
            training_config,
            base_dir=PROJECT_ROOT,
        )
    if not isinstance(saved_tax_profile, dict):
        raise ValueError(
            "Baseline config resolved_tax_profile must be a mapping: "
            f"{baseline_config_path}"
        )
    assert_tax_profiles_match(
        saved_tax_profile=saved_tax_profile,
        current_tax_profile=current_tax_profile,
        artifact_path=baseline_config_path,
    )


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def summarize_action_distribution(step_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(step_df, STEP_REQUIRED_COLUMNS, "baseline_step_rollouts.csv")
    dqn_steps = step_df.loc[step_df["policy_name"] == DQN_POLICY_NAME].copy()
    if dqn_steps.empty:
        raise ValueError("No trained_dqn_greedy rows found in baseline_step_rollouts.csv.")

    dqn_steps["done_bool"] = _bool_series(dqn_steps["done"])
    grouped = (
        dqn_steps.groupby(["split", "action_idx", "action_fraction_requested"], sort=True)
        .agg(
            step_count=("episode_id", "size"),
            mean_executed_fraction=("action_fraction_executed", "mean"),
            mean_reward=("reward", "mean"),
            median_reward=("reward", "median"),
            mean_remaining_fraction_after_action=("remaining_fraction", "mean"),
            terminal_step_count=("done_bool", "sum"),
        )
        .reset_index()
    )
    split_step_counts = dqn_steps.groupby("split").size()
    grouped["step_share"] = grouped["step_count"] / grouped["split"].map(
        split_step_counts
    )
    grouped["terminal_step_count"] = grouped["terminal_step_count"].astype(int)
    return grouped[ACTION_DISTRIBUTION_COLUMNS]


def summarize_first_cut(episode_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(episode_df, EPISODE_REQUIRED_COLUMNS, "baseline_episode_metrics.csv")
    dqn_episodes = episode_df.loc[episode_df["policy_name"] == DQN_POLICY_NAME].copy()
    if dqn_episodes.empty:
        raise ValueError("No trained_dqn_greedy rows found in baseline_episode_metrics.csv.")

    dqn_episodes["episode_cut_occurred_bool"] = _bool_series(
        dqn_episodes["episode_cut_occurred"]
    )
    dqn_episodes["terminal_liquidation_bool"] = _bool_series(
        dqn_episodes["episode_terminal_liquidation_executed"]
    )
    dqn_episodes["full_liquidation_bool"] = _bool_series(
        dqn_episodes["episode_full_liquidation"]
    )
    dqn_episodes["first_cut_step_numeric"] = pd.to_numeric(
        dqn_episodes["first_cut_step"],
        errors="coerce",
    )
    dqn_episodes["first_cut_fraction_numeric"] = pd.to_numeric(
        dqn_episodes["first_cut_fraction_executed"],
        errors="coerce",
    )
    dqn_episodes["sell_immediately_bool"] = dqn_episodes[
        "first_cut_step_numeric"
    ].eq(0)

    summary = (
        dqn_episodes.groupby("split", sort=True)
        .agg(
            num_episodes=("episode_id", "size"),
            cut_frequency=("episode_cut_occurred_bool", "mean"),
            mean_first_cut_step=("first_cut_step_numeric", "mean"),
            median_first_cut_step=("first_cut_step_numeric", "median"),
            p25_first_cut_step=(
                "first_cut_step_numeric",
                lambda series: series.quantile(0.25),
            ),
            p75_first_cut_step=(
                "first_cut_step_numeric",
                lambda series: series.quantile(0.75),
            ),
            mean_first_cut_fraction_executed=(
                "first_cut_fraction_numeric",
                "mean",
            ),
            median_first_cut_fraction_executed=(
                "first_cut_fraction_numeric",
                "median",
            ),
            sell_immediately_frequency=("sell_immediately_bool", "mean"),
            terminal_liquidation_frequency=("terminal_liquidation_bool", "mean"),
            full_liquidation_frequency=("full_liquidation_bool", "mean"),
            mean_episode_steps=("episode_steps", "mean"),
            median_episode_steps=("episode_steps", "median"),
        )
        .reset_index()
    )
    summary["no_cut_frequency"] = 1.0 - summary["cut_frequency"]
    return summary[FIRST_CUT_SUMMARY_COLUMNS]


def build_dqn_episode_behavior(episode_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(episode_df, EPISODE_REQUIRED_COLUMNS, "baseline_episode_metrics.csv")
    dqn_episodes = episode_df.loc[episode_df["policy_name"] == DQN_POLICY_NAME].copy()
    if dqn_episodes.empty:
        raise ValueError("No trained_dqn_greedy rows found in baseline_episode_metrics.csv.")

    dqn_episodes["episode_cut_occurred_bool"] = _bool_series(
        dqn_episodes["episode_cut_occurred"]
    )
    dqn_episodes["terminal_liquidation_bool"] = _bool_series(
        dqn_episodes["episode_terminal_liquidation_executed"]
    )
    dqn_episodes["first_cut_step_numeric"] = pd.to_numeric(
        dqn_episodes["first_cut_step"],
        errors="coerce",
    )
    dqn_episodes["first_cut_fraction_numeric"] = pd.to_numeric(
        dqn_episodes["first_cut_fraction_executed"],
        errors="coerce",
    )

    no_cut_terminal = (
        ~dqn_episodes["episode_cut_occurred_bool"]
        & dqn_episodes["terminal_liquidation_bool"]
    )
    immediate_full = (
        dqn_episodes["first_cut_step_numeric"].eq(0)
        & dqn_episodes["first_cut_fraction_numeric"].ge(0.999)
    )
    immediate_partial = (
        dqn_episodes["first_cut_step_numeric"].eq(0)
        & dqn_episodes["first_cut_fraction_numeric"].lt(0.999)
    )
    delayed_exit = dqn_episodes["first_cut_step_numeric"].gt(0)

    dqn_episodes["behavior_label"] = np.select(
        [no_cut_terminal, immediate_full, immediate_partial, delayed_exit],
        [
            "no_cut_terminal_liquidation",
            "immediate_full_exit",
            "immediate_partial_exit",
            "delayed_exit",
        ],
        default="other",
    )
    return dqn_episodes[DQN_EPISODE_BEHAVIOR_COLUMNS]


def build_policy_behavior_comparison(episode_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(episode_df, EPISODE_REQUIRED_COLUMNS, "baseline_episode_metrics.csv")
    metrics = episode_df.copy()
    metrics["episode_terminal_liquidation_bool"] = _bool_series(
        metrics["episode_terminal_liquidation_executed"]
    )
    metrics["episode_full_liquidation_bool"] = _bool_series(
        metrics["episode_full_liquidation"]
    )
    metrics["episode_cut_occurred_bool"] = _bool_series(
        metrics["episode_cut_occurred"]
    )
    metrics["first_cut_step_numeric"] = pd.to_numeric(
        metrics["first_cut_step"],
        errors="coerce",
    )
    metrics["first_cut_fraction_numeric"] = pd.to_numeric(
        metrics["first_cut_fraction_executed"],
        errors="coerce",
    )
    for column in [
        "days_to_first_sale",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "total_tax_paid",
        "total_transaction_penalty",
        "total_cooldown_penalty",
    ]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics["first_sale_before_tax_transition"] = _bool_series(
        metrics["first_sale_before_tax_transition"]
    )
    metrics["sell_immediately_bool"] = metrics["first_cut_step_numeric"].eq(0)

    comparison = (
        metrics.groupby(["split", "policy_name"], sort=True)
        .agg(
            num_episodes=("episode_id", "size"),
            mean_episode_total_reward=("episode_total_reward", "mean"),
            median_episode_total_reward=("episode_total_reward", "median"),
            mean_episode_total_reward_A=("episode_total_reward_A", "mean"),
            mean_episode_total_reward_C_lite=(
                "episode_total_reward_C_lite",
                "mean",
            ),
            mean_episode_total_reward_C_lite_v2=(
                "episode_total_reward_C_lite_v2",
                "mean",
            ),
            mean_episode_total_transaction_penalty=(
                "episode_total_transaction_penalty",
                "mean",
            ),
            mean_episode_total_cooldown_penalty=(
                "episode_total_cooldown_penalty",
                "mean",
            ),
            mean_final_after_tax_total_value=(
                "episode_final_after_tax_total_value",
                "mean",
            ),
            median_final_after_tax_total_value=(
                "episode_final_after_tax_total_value",
                "median",
            ),
            mean_realized_after_tax_pnl=("episode_realized_after_tax_pnl", "mean"),
            median_realized_after_tax_pnl=(
                "episode_realized_after_tax_pnl",
                "median",
            ),
            mean_episode_steps=("episode_steps", "mean"),
            median_episode_steps=("episode_steps", "median"),
            terminal_liquidation_frequency=(
                "episode_terminal_liquidation_bool",
                "mean",
            ),
            full_liquidation_frequency=("episode_full_liquidation_bool", "mean"),
            cut_frequency=("episode_cut_occurred_bool", "mean"),
            sell_immediately_frequency=("sell_immediately_bool", "mean"),
            mean_first_cut_step=("first_cut_step_numeric", "mean"),
            median_first_cut_step=("first_cut_step_numeric", "median"),
            mean_first_cut_fraction_executed=(
                "first_cut_fraction_numeric",
                "mean",
            ),
            average_days_to_first_sale=("days_to_first_sale", "mean"),
            pct_episodes_with_first_sale_before_tax_transition=(
                "first_sale_before_tax_transition",
                "mean",
            ),
            mean_pct_position_sold_short_term=(
                "pct_episode_position_sold_short_term",
                "mean",
            ),
            mean_pct_position_sold_long_term=(
                "pct_episode_position_sold_long_term",
                "mean",
            ),
            mean_total_tax_paid=("total_tax_paid", "mean"),
            mean_transaction_penalty=("total_transaction_penalty", "mean"),
            mean_cooldown_penalty=("total_cooldown_penalty", "mean"),
        )
        .reset_index()
    )
    return comparison[POLICY_COMPARISON_COLUMNS]


def summarize_cooldown_penalty(episode_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(episode_df, EPISODE_REQUIRED_COLUMNS, "baseline_episode_metrics.csv")
    metrics = episode_df.copy()
    metrics["episode_total_cooldown_penalty_numeric"] = pd.to_numeric(
        metrics["episode_total_cooldown_penalty"],
        errors="coerce",
    ).fillna(0.0)
    metrics["cooldown_penalty_frequency_numeric"] = pd.to_numeric(
        metrics["cooldown_penalty_frequency"],
        errors="coerce",
    ).fillna(0.0)
    metrics["episode_cut_occurred_bool"] = _bool_series(
        metrics["episode_cut_occurred"]
    )
    metrics["episode_terminal_liquidation_bool"] = _bool_series(
        metrics["episode_terminal_liquidation_executed"]
    )
    metrics["cooldown_penalty_episode_bool"] = metrics[
        "episode_total_cooldown_penalty_numeric"
    ].gt(0.0)

    summary = (
        metrics.groupby(["split", "policy_name"], sort=True)
        .agg(
            num_episodes=("episode_id", "size"),
            mean_episode_total_cooldown_penalty=(
                "episode_total_cooldown_penalty_numeric",
                "mean",
            ),
            median_episode_total_cooldown_penalty=(
                "episode_total_cooldown_penalty_numeric",
                "median",
            ),
            cooldown_penalty_episode_frequency=(
                "cooldown_penalty_episode_bool",
                "mean",
            ),
            mean_cooldown_penalty_frequency=(
                "cooldown_penalty_frequency_numeric",
                "mean",
            ),
            mean_episode_steps=("episode_steps", "mean"),
            cut_frequency=("episode_cut_occurred_bool", "mean"),
            terminal_liquidation_frequency=(
                "episode_terminal_liquidation_bool",
                "mean",
            ),
            mean_final_after_tax_total_value=(
                "episode_final_after_tax_total_value",
                "mean",
            ),
        )
        .reset_index()
    )
    return summary[COOLDOWN_PENALTY_SUMMARY_COLUMNS]


def summarize_transaction_penalty(episode_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(episode_df, EPISODE_REQUIRED_COLUMNS, "baseline_episode_metrics.csv")
    metrics = episode_df.copy()
    metrics["episode_total_transaction_penalty_numeric"] = pd.to_numeric(
        metrics["episode_total_transaction_penalty"],
        errors="coerce",
    ).fillna(0.0)
    metrics["transaction_penalty_frequency_numeric"] = pd.to_numeric(
        metrics["transaction_penalty_frequency"],
        errors="coerce",
    ).fillna(0.0)
    metrics["episode_cut_occurred_bool"] = _bool_series(
        metrics["episode_cut_occurred"]
    )
    metrics["episode_terminal_liquidation_bool"] = _bool_series(
        metrics["episode_terminal_liquidation_executed"]
    )
    metrics["transaction_penalty_episode_bool"] = metrics[
        "episode_total_transaction_penalty_numeric"
    ].gt(0.0)

    summary = (
        metrics.groupby(["split", "policy_name"], sort=True)
        .agg(
            num_episodes=("episode_id", "size"),
            mean_episode_total_transaction_penalty=(
                "episode_total_transaction_penalty_numeric",
                "mean",
            ),
            median_episode_total_transaction_penalty=(
                "episode_total_transaction_penalty_numeric",
                "median",
            ),
            transaction_penalty_episode_frequency=(
                "transaction_penalty_episode_bool",
                "mean",
            ),
            mean_transaction_penalty_frequency=(
                "transaction_penalty_frequency_numeric",
                "mean",
            ),
            mean_episode_steps=("episode_steps", "mean"),
            cut_frequency=("episode_cut_occurred_bool", "mean"),
            terminal_liquidation_frequency=(
                "episode_terminal_liquidation_bool",
                "mean",
            ),
            mean_final_after_tax_total_value=(
                "episode_final_after_tax_total_value",
                "mean",
            ),
        )
        .reset_index()
    )
    return summary[TRANSACTION_PENALTY_SUMMARY_COLUMNS]


def summarize_early_selling(episode_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(episode_df, EPISODE_REQUIRED_COLUMNS, "baseline_episode_metrics.csv")
    metrics = episode_df.copy()
    for column in [
        "days_to_first_sale",
        "steps_to_first_sale",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "mean_effective_tax_rate_on_sales",
        "total_tax_paid",
    ]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics["first_sale_before_tax_transition_bool"] = _bool_series(
        metrics["first_sale_before_tax_transition"]
    )
    metrics["episode_terminal_liquidation_bool"] = _bool_series(
        metrics["episode_terminal_liquidation_executed"]
    )
    metrics["episode_cut_occurred_bool"] = _bool_series(
        metrics["episode_cut_occurred"]
    )
    metrics["first_cut_step_numeric"] = pd.to_numeric(
        metrics["first_cut_step"],
        errors="coerce",
    )
    metrics["sell_immediately_bool"] = metrics["first_cut_step_numeric"].eq(0)

    summary = (
        metrics.groupby(["split", "policy_name"], sort=True)
        .agg(
            num_episodes=("episode_id", "size"),
            average_days_to_first_sale=("days_to_first_sale", "mean"),
            median_days_to_first_sale=("days_to_first_sale", "median"),
            average_steps_to_first_sale=("steps_to_first_sale", "mean"),
            pct_episodes_with_first_sale_before_tax_transition=(
                "first_sale_before_tax_transition_bool",
                "mean",
            ),
            mean_pct_position_sold_short_term=(
                "pct_episode_position_sold_short_term",
                "mean",
            ),
            mean_pct_position_sold_long_term=(
                "pct_episode_position_sold_long_term",
                "mean",
            ),
            mean_effective_tax_rate=(
                "mean_effective_tax_rate_on_sales",
                "mean",
            ),
            mean_total_tax_paid=("total_tax_paid", "mean"),
            mean_final_after_tax_total_value=(
                "episode_final_after_tax_total_value",
                "mean",
            ),
            terminal_liquidation_frequency=(
                "episode_terminal_liquidation_bool",
                "mean",
            ),
            cut_frequency=("episode_cut_occurred_bool", "mean"),
            sell_immediately_frequency=("sell_immediately_bool", "mean"),
        )
        .reset_index()
    )
    return summary[EARLY_SELLING_SUMMARY_COLUMNS]


def build_behavior_summary_text(
    dqn_action_distribution: pd.DataFrame,
    dqn_first_cut_summary: pd.DataFrame,
    dqn_episode_behavior: pd.DataFrame,
    policy_behavior_comparison: pd.DataFrame,
    cooldown_penalty_summary: pd.DataFrame,
    transaction_penalty_summary: pd.DataFrame,
    early_selling_summary: pd.DataFrame,
    resolved_tax_profile: dict[str, Any],
    config_path: Path,
    reward_version: str,
) -> str:
    behavior_counts = (
        dqn_episode_behavior.groupby(["split", "behavior_label"], sort=True)
        .size()
        .rename("episode_count")
        .reset_index()
    )
    split_counts = dqn_episode_behavior.groupby("split").size()
    behavior_counts["episode_share"] = behavior_counts["episode_count"] / behavior_counts[
        "split"
    ].map(split_counts)

    title = (
        "Reward C-lite Behavior Inspection"
        if "C_lite" in reward_version
        else "Reward Policy Behavior Inspection"
    )
    lines = [
        title,
        f"config_path: {_relative_project_path(config_path)}",
        f"reward_version: {reward_version}",
        f"tax_profile_name: {resolved_tax_profile['profile_name']}",
        f"resolved_tax_profile: {resolved_tax_profile}",
        "transaction_penalty_note: transaction penalty is a training reward-shaping term for discretionary sales; final after-tax value remains the economic metric.",
        "cooldown_penalty_note: cooldown penalty is a training reward-shaping term; final after-tax value remains the economic metric.",
        "comparison_note: compare C-lite learned behavior against Reward A v3 later; this file does not state final thesis conclusions.",
        "",
        "Inputs",
    ]
    for path in INPUT_PATHS.values():
        lines.append(f"- {_relative_project_path(path)}")

    lines.extend(
        [
            "",
            "DQN action distribution",
            _compact_table(dqn_action_distribution),
            "",
            "DQN first-cut behavior",
            _compact_table(dqn_first_cut_summary),
            "",
            "DQN episode behavior labels",
            _compact_table(behavior_counts),
            "",
            "Policy behavior comparison",
            _compact_table(policy_behavior_comparison),
            "",
            "Cooldown penalty summary",
            _compact_table(cooldown_penalty_summary),
            "",
            "Transaction penalty summary",
            _compact_table(transaction_penalty_summary),
            "",
            "Early-selling summary",
            _compact_table(early_selling_summary),
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect DQN behavior from baseline rollouts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=(
            "Training/evaluation config path. Defaults to "
            f"{_relative_project_path(CONFIG_PATH)}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    print("Reward Policy Behavior Inspection")
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)
    reward_version = str(_require(config, "reward.expected_info_reward_version"))
    run_dir = resolve_project_path(_require(config, "logging.output_dir"))
    baseline_dir = run_dir / "baselines"
    output_dir = run_dir / "behavior_inspection"
    resolved_tax_profile = resolve_tax_profile_from_config(
        config,
        base_dir=PROJECT_ROOT,
    )
    validate_baseline_tax_profile(
        baseline_dir=baseline_dir,
        current_tax_profile=resolved_tax_profile,
    )

    step_df, episode_df, summary_df = load_inputs(baseline_dir)
    require_columns(step_df, STEP_REQUIRED_COLUMNS, "baseline_step_rollouts.csv")
    require_columns(episode_df, EPISODE_REQUIRED_COLUMNS, "baseline_episode_metrics.csv")
    require_columns(summary_df, SUMMARY_REQUIRED_COLUMNS, "baseline_summary_by_policy.csv")
    print(f"Loaded baseline files from {_relative_project_path(baseline_dir)}")

    dqn_action_distribution = summarize_action_distribution(step_df)
    dqn_first_cut_summary = summarize_first_cut(episode_df)
    dqn_episode_behavior = build_dqn_episode_behavior(episode_df)
    policy_behavior_comparison = build_policy_behavior_comparison(episode_df)
    cooldown_penalty_summary = summarize_cooldown_penalty(episode_df)
    transaction_penalty_summary = summarize_transaction_penalty(episode_df)
    early_selling_summary = summarize_early_selling(episode_df)
    behavior_summary_text = build_behavior_summary_text(
        dqn_action_distribution=dqn_action_distribution,
        dqn_first_cut_summary=dqn_first_cut_summary,
        dqn_episode_behavior=dqn_episode_behavior,
        policy_behavior_comparison=policy_behavior_comparison,
        cooldown_penalty_summary=cooldown_penalty_summary,
        transaction_penalty_summary=transaction_penalty_summary,
        early_selling_summary=early_selling_summary,
        resolved_tax_profile=resolved_tax_profile,
        config_path=config_path,
        reward_version=reward_version,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dqn_action_distribution.to_csv(
        output_dir / "dqn_action_distribution.csv",
        index=False,
    )
    dqn_first_cut_summary.to_csv(
        output_dir / "dqn_first_cut_summary.csv",
        index=False,
    )
    dqn_episode_behavior.to_csv(
        output_dir / "dqn_episode_behavior.csv",
        index=False,
    )
    policy_behavior_comparison.to_csv(
        output_dir / "policy_behavior_comparison.csv",
        index=False,
    )
    cooldown_penalty_summary.to_csv(
        output_dir / "cooldown_penalty_summary.csv",
        index=False,
    )
    transaction_penalty_summary.to_csv(
        output_dir / "transaction_penalty_summary.csv",
        index=False,
    )
    early_selling_summary.to_csv(
        output_dir / "early_selling_summary.csv",
        index=False,
    )
    with (output_dir / "behavior_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(behavior_summary_text)

    print(f"Saved behavior inspection outputs to {_relative_project_path(output_dir)}")
    print(f"POLICY BEHAVIOR INSPECTION COMPLETE reward_version={reward_version}")


if __name__ == "__main__":
    main()
