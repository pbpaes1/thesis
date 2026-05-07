"""Inspect learned action behavior from existing Reward A baseline rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "runs" / "train_reward_a_v1"
BASELINE_DIR = RUN_DIR / "baselines"
OUTPUT_DIR = RUN_DIR / "behavior_inspection"
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
    "after_tax_total_value",
    "sold_fraction",
    "remaining_fraction",
    "tax_regime",
    "after_tax_liquidation_tax_regime",
    "terminal_liquidation_executed",
    "done",
]

EPISODE_REQUIRED_COLUMNS = [
    "split",
    "policy_name",
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
]

SUMMARY_REQUIRED_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    "mean_episode_total_reward",
    "mean_final_after_tax_total_value",
    "mean_episode_steps",
    "terminal_liquidation_frequency",
    "full_liquidation_frequency",
    "cut_frequency",
    "mean_first_cut_step",
    "mean_first_cut_fraction_executed",
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
]

INPUT_PATHS: dict[str, Path] = {}


def _relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0.0)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y"])


def _format_optional_float(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


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

    step_df = pd.read_csv(step_path, usecols=lambda col: col in STEP_REQUIRED_COLUMNS)
    episode_df = pd.read_csv(
        episode_path,
        usecols=lambda col: col in EPISODE_REQUIRED_COLUMNS,
    )
    summary_df = pd.read_csv(
        summary_path,
        usecols=lambda col: col in SUMMARY_REQUIRED_COLUMNS,
    )
    return step_df, episode_df, summary_df


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
    metrics["sell_immediately_bool"] = metrics["first_cut_step_numeric"].eq(0)

    comparison = (
        metrics.groupby(["split", "policy_name"], sort=True)
        .agg(
            num_episodes=("episode_id", "size"),
            mean_episode_total_reward=("episode_total_reward", "mean"),
            median_episode_total_reward=("episode_total_reward", "median"),
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
        )
        .reset_index()
    )
    return comparison[POLICY_COMPARISON_COLUMNS]


def build_behavior_summary_text(
    dqn_action_distribution: pd.DataFrame,
    dqn_first_cut_summary: pd.DataFrame,
    dqn_episode_behavior: pd.DataFrame,
    policy_behavior_comparison: pd.DataFrame,
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

    interpretation_lines: list[str] = []
    for split_name in sorted(dqn_first_cut_summary["split"].unique()):
        dqn_row = dqn_first_cut_summary.loc[
            dqn_first_cut_summary["split"] == split_name
        ].iloc[0]
        comparison_split = policy_behavior_comparison[
            policy_behavior_comparison["split"] == split_name
        ]
        dqn_comparison = comparison_split[
            comparison_split["policy_name"] == DQN_POLICY_NAME
        ]
        hold_comparison = comparison_split[
            comparison_split["policy_name"] == "hold_to_terminal"
        ]
        cut_frequency = float(dqn_row["cut_frequency"])
        immediate_frequency = float(dqn_row["sell_immediately_frequency"])
        terminal_frequency = float(dqn_row["terminal_liquidation_frequency"])

        interpretation_lines.append(
            f"- {split_name}: trained_dqn_greedy cuts in "
            f"{cut_frequency:.1%} of episodes, cuts immediately in "
            f"{immediate_frequency:.1%}, and reaches terminal liquidation in "
            f"{terminal_frequency:.1%}."
        )
        if not dqn_comparison.empty and not hold_comparison.empty:
            dqn_mean_steps = float(dqn_comparison.iloc[0]["mean_episode_steps"])
            hold_mean_steps = float(hold_comparison.iloc[0]["mean_episode_steps"])
            dqn_mean_reward = float(
                dqn_comparison.iloc[0]["mean_episode_total_reward"]
            )
            hold_mean_reward = float(
                hold_comparison.iloc[0]["mean_episode_total_reward"]
            )
            step_ratio = dqn_mean_steps / hold_mean_steps if hold_mean_steps else np.nan
            interpretation_lines.append(
                f"- {split_name}: mean DQN episode length is "
                f"{_format_optional_float(dqn_mean_steps)} steps versus "
                f"{_format_optional_float(hold_mean_steps)} for hold_to_terminal "
                f"(ratio {_format_optional_float(step_ratio)})."
            )
            if dqn_mean_reward < hold_mean_reward:
                interpretation_lines.append(
                    f"- {split_name}: the pilot DQN mean reward is below "
                    "hold_to_terminal in these existing baseline summaries."
                )

    dqn_underperforms_hold = False
    early_cut_heavy = False
    for _, dqn_row in dqn_first_cut_summary.iterrows():
        if float(dqn_row["sell_immediately_frequency"]) >= 0.25:
            early_cut_heavy = True
        if float(dqn_row["mean_episode_steps"]) <= 0:
            early_cut_heavy = True
    for split_name in sorted(policy_behavior_comparison["split"].unique()):
        comparison_split = policy_behavior_comparison[
            policy_behavior_comparison["split"] == split_name
        ]
        dqn_reward = comparison_split.loc[
            comparison_split["policy_name"] == DQN_POLICY_NAME,
            "mean_episode_total_reward",
        ]
        hold_reward = comparison_split.loc[
            comparison_split["policy_name"] == "hold_to_terminal",
            "mean_episode_total_reward",
        ]
        if not dqn_reward.empty and not hold_reward.empty:
            dqn_underperforms_hold = dqn_underperforms_hold or bool(
                float(dqn_reward.iloc[0]) < float(hold_reward.iloc[0])
            )

    recommendation_note = (
        "The DQN either exits early in a large share of episodes or underperforms "
        "hold_to_terminal in the existing summaries, so treat the 500-episode "
        "model as a pilot before final analysis."
        if early_cut_heavy or dqn_underperforms_hold
        else "The behavior summary does not by itself trigger a final conclusion; "
        "use it only to decide the next training pass."
    )

    lines = [
        "Reward A DQN Behavior Inspection",
        "",
        "1. Purpose",
        "This is a build-and-train behavior inspection of the trained Reward A "
        "greedy policy. It is not final quantitative analysis, not a thesis "
        "results table, and not a statistical significance exercise.",
        "",
        "2. Inputs",
    ]
    for path in INPUT_PATHS.values():
        lines.append(f"- {_relative_project_path(path)}")

    lines.extend(
        [
            "",
            "3. DQN action distribution",
            _compact_table(dqn_action_distribution),
            "",
            "4. DQN first-cut behavior",
            _compact_table(dqn_first_cut_summary),
            "",
            "5. DQN episode behavior labels",
            _compact_table(behavior_counts),
            "",
            "6. Policy behavior comparison",
            _compact_table(policy_behavior_comparison),
            "",
            "7. Build/train interpretation",
        ]
    )
    lines.extend(interpretation_lines)
    lines.append(
        "- This inspection should be read as a mechanical behavior check before "
        "deciding whether to train beyond 500 episodes, not as final evidence."
    )
    lines.extend(
        [
            "",
            "8. Recommended next build/train decision",
            f"Recommendation note: {recommendation_note}",
            "Recommended next step: decide whether to train on more than 500 episodes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print("Reward A DQN Behavior Inspection")
    step_df, episode_df, summary_df = load_inputs(BASELINE_DIR)
    require_columns(step_df, STEP_REQUIRED_COLUMNS, "baseline_step_rollouts.csv")
    require_columns(episode_df, EPISODE_REQUIRED_COLUMNS, "baseline_episode_metrics.csv")
    require_columns(summary_df, SUMMARY_REQUIRED_COLUMNS, "baseline_summary_by_policy.csv")
    print("Loaded baseline files from runs/train_reward_a_v1/baselines/")

    dqn_action_distribution = summarize_action_distribution(step_df)
    dqn_first_cut_summary = summarize_first_cut(episode_df)
    dqn_episode_behavior = build_dqn_episode_behavior(episode_df)
    policy_behavior_comparison = build_policy_behavior_comparison(episode_df)
    behavior_summary_text = build_behavior_summary_text(
        dqn_action_distribution=dqn_action_distribution,
        dqn_first_cut_summary=dqn_first_cut_summary,
        dqn_episode_behavior=dqn_episode_behavior,
        policy_behavior_comparison=policy_behavior_comparison,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dqn_action_distribution.to_csv(
        OUTPUT_DIR / "dqn_action_distribution.csv",
        index=False,
    )
    dqn_first_cut_summary.to_csv(
        OUTPUT_DIR / "dqn_first_cut_summary.csv",
        index=False,
    )
    dqn_episode_behavior.to_csv(
        OUTPUT_DIR / "dqn_episode_behavior.csv",
        index=False,
    )
    policy_behavior_comparison.to_csv(
        OUTPUT_DIR / "policy_behavior_comparison.csv",
        index=False,
    )
    with (OUTPUT_DIR / "behavior_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(behavior_summary_text)

    print("Saved behavior inspection outputs to runs/train_reward_a_v1/behavior_inspection/")
    print("REWARD A POLICY BEHAVIOR INSPECTION COMPLETE")


if __name__ == "__main__":
    main()
