"""Build quantitative analysis phase 4-11 outputs for the frozen final run.

Run from the project root:
    python scripts/build_quant_analysis_phase_4_11.py --config configs/train_reward_c_lite_v5.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError(
        "PyYAML is required to run scripts/build_quant_analysis_phase_4_11.py. "
        "Install it with: pip install pyyaml"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_reward_c_lite_v5.yaml"
FINAL_RUN_NAME = "train_reward_c_lite_v5_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / FINAL_RUN_NAME / "quant_analysis"
PREFERRED_POLICY = "trained_dqn_first_sale_margin_0p070_normal_0p020"
VALIDATION_TEST_SPLITS = ["validation", "test"]
DIAGNOSTIC_SPLITS = ["train", "validation", "test", "all"]
RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 10000
ANNUAL_RISK_FREE_RATE = 0.04
ANNUALIZATION_FACTOR = 252
DAILY_RISK_FREE_RATE = (1.0 + ANNUAL_RISK_FREE_RATE) ** (
    1.0 / float(ANNUALIZATION_FACTOR)
) - 1.0

STEP4_FIRST_SALE_MARGINS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
STEP4_FIRST_SALE_MARGIN_POLICIES = [
    "trained_dqn_first_sale_margin_0p020_normal_0p020",
    "trained_dqn_first_sale_margin_0p030_normal_0p020",
    "trained_dqn_first_sale_margin_0p040_normal_0p020",
    "trained_dqn_first_sale_margin_0p050_normal_0p020",
    "trained_dqn_first_sale_margin_0p060_normal_0p020",
    PREFERRED_POLICY,
    "trained_dqn_first_sale_margin_0p080_normal_0p020",
    "trained_dqn_first_sale_margin_0p090_normal_0p020",
]

POLICY_UNIVERSE = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "sell_quarters_over_time",
    "random_policy",
    "trained_dqn_greedy",
    "trained_dqn_thresholded_margin_0p020",
    "trained_dqn_first_sale_margin_0p060_normal_0p020",
    PREFERRED_POLICY,
    "trained_dqn_first_sale_margin_0p080_normal_0p020",
    "trained_dqn_first_sale_margin_0p090_normal_0p020",
]
DQN_DECISION_POLICIES = [
    "trained_dqn_greedy",
    "trained_dqn_thresholded_margin_0p020",
    *STEP4_FIRST_SALE_MARGIN_POLICIES,
]
FIRST_SALE_MARGIN_POLICIES = STEP4_FIRST_SALE_MARGIN_POLICIES
BENCHMARK_POLICIES = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "sell_quarters_over_time",
    "random_policy",
]
STEP4_STEP3B_POLICIES = [*BENCHMARK_POLICIES, *DQN_DECISION_POLICIES]
STEP10_COMPARISON_POLICIES = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "sell_quarters_over_time",
    "trained_dqn_greedy",
    "trained_dqn_thresholded_margin_0p020",
    "trained_dqn_first_sale_margin_0p060_normal_0p020",
    "trained_dqn_first_sale_margin_0p080_normal_0p020",
    "trained_dqn_first_sale_margin_0p090_normal_0p020",
]
ACTION_LABELS = {
    0.0: "hold",
    0.25: "sell_25",
    0.5: "sell_50",
    0.75: "sell_75",
    1.0: "sell_100",
}
ACTION_ORDER = ["hold", "sell_25", "sell_50", "sell_75", "sell_100"]

STEP4_BASE_COLUMNS = [
    "split",
    "policy_name",
    "first_sale_margin",
    "num_episodes",
    "mean_final_after_tax_total_value",
    "median_final_after_tax_total_value",
    "std_final_after_tax_total_value",
    "mean_excess_value_vs_trained_dqn_greedy",
    "mean_excess_value_vs_trained_dqn_thresholded_margin_0p020",
    "mean_total_tax_paid",
    "mean_effective_tax_rate",
    "mean_pct_position_sold_short_term",
    "mean_pct_position_sold_long_term",
    "terminal_liquidation_frequency",
    "no_cut_episode_pct",
    "discretionary_sale_episode_pct",
    "average_days_to_first_sale",
    "median_days_to_first_sale",
    "pct_episodes_with_first_sale_before_tax_transition",
    "mean_num_discretionary_sales",
]
STEP4_EAAT_SHARPE_COLUMNS = [
    "num_valid_EAAT_Sharpe_episodes",
    "mean_EAAT_Sharpe",
    "median_EAAT_Sharpe",
    "pct_positive_EAAT_Sharpe",
    "mean_EAAT_annualized_after_tax_return",
    "median_EAAT_annualized_after_tax_return",
    "mean_EAAT_annualized_volatility",
    "median_EAAT_annualized_volatility",
    "num_valid_TA_EAAT_Sharpe_episodes",
    "median_TA_EAAT_Sharpe",
    "pct_positive_TA_EAAT_Sharpe",
    "median_TA_EAAT_annualized_after_tax_return",
    "median_TA_EAAT_annualized_volatility",
]
STEP6_EAAT_SHARPE_COLUMNS = [
    "num_valid_EAAT_Sharpe_episodes",
    "median_EAAT_Sharpe",
    "pct_positive_EAAT_Sharpe",
    "mean_EAAT_annualized_after_tax_return",
    "median_EAAT_annualized_after_tax_return",
    "mean_EAAT_annualized_volatility",
    "median_EAAT_annualized_volatility",
    "num_valid_TA_EAAT_Sharpe_episodes",
    "median_TA_EAAT_Sharpe",
    "pct_positive_TA_EAAT_Sharpe",
    "median_TA_EAAT_annualized_after_tax_return",
    "median_TA_EAAT_annualized_volatility",
]
STEP6_HEADLINE_SHARPE_COLUMNS = [
    "median_EAAT_Sharpe",
    "median_TA_EAAT_Sharpe",
]
STEP4_SOURCE_REQUIRED_SHARPE_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    *STEP4_EAAT_SHARPE_COLUMNS,
    "annual_risk_free_rate",
    "daily_risk_free_rate",
    "annualization_factor",
]
STEP4_REQUIRED_SHARPE_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    *STEP4_EAAT_SHARPE_COLUMNS,
]
STEP4_LEGACY_FORBIDDEN_COLUMNS = {
    "sharpe_episode",
    "sortino_episode",
    "episode_step_sharpe",
    "pooled_step_sharpe",
    "step_return_proxy",
    "mean_TA_EAAT_Sharpe",
    "std_TA_EAAT_Sharpe",
    "mean_episode_step_sharpe",
    "median_episode_step_sharpe",
    "std_episode_step_sharpe",
    "pct_positive_episode_step_sharpe",
    "coefficient_of_variation",
    "step_return_proxy_source",
    "mean_step_return_proxy_all_steps",
    "std_step_return_proxy_all_steps",
}
STEP4_LEGACY_FORBIDDEN_COLUMN_TOKENS = [
    "reward_path",
    "reward-path",
    "step_return_proxy",
    "reward_a_proxy",
    "reward_A_proxy",
    "full_horizon_annualized_sharpe",
    "invested_period_annualized_sharpe",
]
STEP6_FORBIDDEN_LEGACY_SHARPE_COLUMNS = {
    "sharpe_episode",
    "sortino_episode",
    "episode_step_sharpe",
    "pooled_step_sharpe",
    "full_horizon_annualized_sharpe",
    "invested_period_annualized_sharpe",
    "mean_TA_EAAT_Sharpe",
    "std_TA_EAAT_Sharpe",
}
STEP6_REQUIRED_TAX_ACCOUNTING_COLUMNS = [
    "short_term_realized_fraction",
    "long_term_realized_fraction",
    "mean_total_tax_paid",
    "median_total_tax_paid",
    "tax_paid_as_pct_of_positive_taxable_pre_tax_increment",
    "mean_effective_tax_rate",
    "mean_pre_tax_realized_gain",
    "mean_after_tax_realized_gain",
    "mean_after_tax_value_loss_vs_pre_tax",
    "mean_after_tax_value_loss_pct_vs_pre_tax",
    "mean_final_after_tax_total_value",
    "mean_realized_after_tax_pnl",
    "mean_tax_paid_difference_vs_hold_to_terminal",
    "mean_tax_paid_difference_vs_sell_immediately",
]
STEP7_EAAT_SHARPE_COLUMNS = [
    "num_valid_EAAT_Sharpe_episodes",
    "median_EAAT_Sharpe",
    "pct_positive_EAAT_Sharpe",
    "median_EAAT_annualized_after_tax_return",
    "median_EAAT_annualized_volatility",
    "num_valid_TA_EAAT_Sharpe_episodes",
    "median_TA_EAAT_Sharpe",
    "pct_positive_TA_EAAT_Sharpe",
    "median_TA_EAAT_annualized_after_tax_return",
    "median_TA_EAAT_annualized_volatility",
]
STEP7_FORBIDDEN_LEGACY_SHARPE_COLUMNS = STEP6_FORBIDDEN_LEGACY_SHARPE_COLUMNS
STEP7_REQUIRED_BEHAVIOR_COLUMNS = [
    "split",
    "diagnostic_scope",
    "num_episodes",
    "action_distribution",
    "first_action_distribution",
    "first_discretionary_sale_action_distribution",
    "first_sale_timing_distribution",
    "no_cut_episode_count",
    "no_cut_episode_pct",
    "discretionary_sale_episode_count",
    "discretionary_sale_episode_pct",
    "partial_discretionary_liquidation_pct",
    "full_early_liquidation_pct",
    "terminal_liquidation_frequency",
    "mean_num_discretionary_sales",
    "average_days_to_first_sale",
    "median_days_to_first_sale",
    "pct_episodes_with_first_sale_before_tax_transition",
    "mean_pct_position_sold_short_term",
    "mean_pct_position_sold_long_term",
    "mean_total_tax_paid",
    "mean_effective_tax_rate",
    "mean_final_after_tax_total_value",
]
STEP7_ECONOMIC_PERIODS = [
    "post_crisis_early_recovery",
    "qe_bull_market",
    "late_cycle_volatility_return",
    "covid_stimulus",
    "inflation_tightening",
    "unknown_or_outside_defined_period",
]
STEP7_PAIR_BENCHMARKS = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
]
STEP8_BENCHMARK_POLICIES = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "sell_quarters_over_time",
    "random_policy",
]
STEP8_REQUIRED_EPISODE_COLUMNS = [
    "split",
    "policy_name",
    "episode_id",
    "episode_final_after_tax_total_value",
    "total_tax_paid",
    "pct_episode_position_sold_short_term",
    "pct_episode_position_sold_long_term",
    "episode_cut_occurred",
    "num_discretionary_sales",
    "days_to_first_sale",
]
STEP8_FORBIDDEN_LEGACY_SHARPE_COLUMNS = STEP6_FORBIDDEN_LEGACY_SHARPE_COLUMNS
STEP9_POLICIES = [
    PREFERRED_POLICY,
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
]
STEP9_POLICY_LABELS = {
    PREFERRED_POLICY: "preferred",
    "hold_to_terminal": "hold_to_terminal",
    "sell_immediately": "sell_immediately",
    "sell_half_then_hold": "sell_half_then_hold",
}
STEP9_REQUIRED_OUTPUT_COLUMNS = [
    "split",
    "group_name",
    "group_bucket",
    "num_episodes",
    "preferred_mean_final_value",
    "hold_to_terminal_mean_final_value",
    "sell_immediately_mean_final_value",
    "sell_half_then_hold_mean_final_value",
    "preferred_minus_hold_mean",
    "preferred_minus_sell_immediately_mean",
    "preferred_minus_sell_half_mean",
    "preferred_win_rate_vs_hold",
    "preferred_win_rate_vs_sell_immediately",
    "preferred_win_rate_vs_sell_half",
    "preferred_median_EAAT_Sharpe",
    "hold_to_terminal_median_EAAT_Sharpe",
    "sell_immediately_median_EAAT_Sharpe",
    "sell_half_then_hold_median_EAAT_Sharpe",
    "preferred_median_TA_EAAT_Sharpe",
    "hold_to_terminal_median_TA_EAAT_Sharpe",
    "sell_immediately_median_TA_EAAT_Sharpe",
    "sell_half_then_hold_median_TA_EAAT_Sharpe",
    "preferred_minus_hold_mean_EAAT_Sharpe",
    "preferred_minus_hold_median_EAAT_Sharpe",
    "preferred_EAAT_Sharpe_win_rate_vs_hold",
    "preferred_minus_hold_mean_TA_EAAT_Sharpe",
    "preferred_minus_hold_median_TA_EAAT_Sharpe",
    "preferred_TA_EAAT_Sharpe_win_rate_vs_hold",
]
STEP9_FORBIDDEN_LEGACY_SHARPE_COLUMNS = STEP6_FORBIDDEN_LEGACY_SHARPE_COLUMNS | {
    "reward_path_sharpe",
    "reward-path Sharpe",
    "step_return_proxy_sharpe",
    "full_horizon_annualized_sharpe",
    "invested_period_annualized_sharpe",
}
STEP4_EAAT_NOTE_REQUIRED_PHRASES = [
    "EAAT Sharpe uses terminal after-tax wealth",
    "TA-EAAT Sharpe uses sale-level tranches",
    "The annual risk-free rate is 4%",
    "The daily risk-free rate is computed as (1 + 0.04) ** (1 / 252) - 1",
]
STEP4_EAAT_NOTE_FORBIDDEN_PHRASES = [
    "risk-free rate = 0",
    "risk-free rate: 0",
    "no annualization",
    "step-return proxy",
    "reward-path sharpe",
    "reward_a proxy",
    "legacy diagnostic",
    "legacy risk-adjusted",
]


class OutputRegistry:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def add(self, path: Path, file_type: str, step: str, description: str) -> None:
        self.rows.append(
            {
                "path": relative_project_path(path),
                "type": file_type,
                "step": str(step),
                "description": description,
            }
        )


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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping at top level: {path}")
    return payload


def require_nested(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing required config value: {path}")
        current = current[key]
    return current


def run_existing_script(args: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        command_text = " ".join([sys.executable, *args])
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {command_text}"
        )


def require_columns(df: pd.DataFrame, columns: list[str], *, source: Path, step: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{step} requires missing column(s) in {relative_project_path(source)}: {missing}"
        )


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
        if pd.notna(value)
        else False
    )


def action_label(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    value_float = round(float(value), 2)
    for key, label in ACTION_LABELS.items():
        if math.isclose(value_float, key, abs_tol=1e-9):
            return label
    return f"sell_{value_float:g}"


def compact_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def markdown_table_text(df: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        if value is None or pd.isna(value):
            text = ""
        elif isinstance(value, float):
            text = f"{value:.10g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")

    headers = [str(column) for column in df.columns]
    rows = [[fmt(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    lines = [
        "| "
        + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
        + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(markdown_table_text(df), encoding="utf-8")


def save_table(
    df: pd.DataFrame,
    csv_path: Path,
    md_path: Path,
    registry: OutputRegistry,
    step: str,
    description: str,
) -> None:
    df.to_csv(csv_path, index=False)
    write_markdown_table(df, md_path)
    registry.add(csv_path, "csv", step, description)
    registry.add(md_path, "md", step, description)


def save_text(
    text: str,
    path: Path,
    registry: OutputRegistry,
    step: str,
    description: str,
) -> None:
    path.write_text(text, encoding="utf-8")
    registry.add(path, "txt", step, description)


def save_plot(path: Path, registry: OutputRegistry, step: str, description: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    registry.add(path, "png", step, description)


def ordered_policy_frame(df: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    split_order = {split: index for index, split in enumerate(VALIDATION_TEST_SPLITS)}
    policy_order = {policy: index for index, policy in enumerate(policies)}
    out = df.copy()
    out["_split_order"] = out["split"].map(split_order)
    out["_policy_order"] = out["policy_name"].map(policy_order)
    return out.sort_values(["_split_order", "_policy_order"]).drop(
        columns=["_split_order", "_policy_order"]
    )


def verify_frozen_run(config_path: Path, config: dict[str, Any], output_dir: Path) -> Path:
    run_path = resolve_project_path(require_nested(config, "logging.output_dir"))
    if run_path.name != FINAL_RUN_NAME:
        raise ValueError(
            f"Expected frozen run directory {FINAL_RUN_NAME!r}, got "
            f"{relative_project_path(run_path)!r}."
        )
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not run_path.exists():
        raise FileNotFoundError(f"Missing run directory: {run_path}")
    if output_dir.resolve() != (run_path / "quant_analysis").resolve():
        raise ValueError(
            "Output directory must be the frozen run quant_analysis directory: "
            f"{relative_project_path(run_path / 'quant_analysis')}"
        )
    if not (run_path / "best_validation_model.pt").exists() and not (
        run_path / "final_model.pt"
    ).exists():
        raise FileNotFoundError("Missing best_validation_model.pt and final_model.pt.")
    if not (run_path / "episode_splits.csv").exists():
        raise FileNotFoundError(f"Missing episode splits: {run_path / 'episode_splits.csv'}")
    return run_path


def ensure_baseline_outputs(config_path: Path, run_path: Path) -> None:
    required = [
        run_path / "baselines" / "baseline_episode_metrics.csv",
        run_path / "baselines" / "baseline_step_rollouts.csv",
        run_path / "baselines" / "baseline_summary_by_policy.csv",
    ]
    if any(not path.exists() for path in required):
        run_existing_script(
            [
                "scripts/evaluate_reward_a_baselines.py",
                "--config",
                relative_project_path(config_path),
            ]
        )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required baseline outputs: "
            + ", ".join(relative_project_path(path) for path in missing)
        )


def load_baseline_data(run_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episode_path = run_path / "baselines" / "baseline_episode_metrics.csv"
    step_path = run_path / "baselines" / "baseline_step_rollouts.csv"
    summary_path = run_path / "baselines" / "baseline_summary_by_policy.csv"
    episode_df = pd.read_csv(episode_path, low_memory=False)
    step_df = pd.read_csv(step_path, low_memory=False)
    summary_df = pd.read_csv(summary_path)
    missing_policies = sorted(set(POLICY_UNIVERSE) - set(episode_df["policy_name"].unique()))
    if missing_policies:
        raise ValueError(
            "Baseline episode metrics are missing final policy universe rows: "
            + ", ".join(missing_policies)
        )
    return episode_df, step_df, summary_df


def prepare_episode_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_columns = [
        "episode_final_after_tax_total_value",
        "episode_initial_after_tax_total_value",
        "episode_total_reward_A",
        "episode_realized_after_tax_pnl",
        "total_tax_paid",
        "mean_effective_tax_rate_on_sales",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "days_to_first_sale",
        "steps_to_first_sale",
        "num_discretionary_sales",
        "terminal_liquidation_fraction",
        "episode_final_sold_fraction",
        "episode_final_remaining_fraction",
        "total_positive_taxable_pre_tax_increment",
        "first_cut_fraction_executed",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "episode_terminal_liquidation_executed" in out.columns:
        out["episode_terminal_liquidation_executed"] = coerce_bool_series(
            out["episode_terminal_liquidation_executed"]
        )
    if "first_sale_before_tax_transition" in out.columns:
        out["first_sale_before_tax_transition"] = coerce_bool_series(
            out["first_sale_before_tax_transition"]
        )
    if "episode_cut_occurred" in out.columns:
        out["episode_cut_occurred"] = coerce_bool_series(out["episode_cut_occurred"])
    elif "num_discretionary_sales" in out.columns:
        out["episode_cut_occurred"] = out["num_discretionary_sales"].fillna(0).gt(0)
    return out


def prepare_step_rollouts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_columns = [
        "step_in_episode",
        "action_idx",
        "action_fraction_requested",
        "action_fraction_executed",
        "after_tax_total_value",
        "previous_after_tax_total_value",
        "sold_fraction",
        "remaining_fraction",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["terminal_liquidation_executed", "done", "is_automatic_terminal_liquidation"]:
        if column in out.columns:
            out[column] = coerce_bool_series(out[column])
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "tax_transition_date" in out.columns:
        out["tax_transition_date"] = pd.to_datetime(
            out["tax_transition_date"], errors="coerce"
        )
    return out


def aggregate_policy_metrics(
    episode_df: pd.DataFrame,
    policies: list[str],
    splits: list[str],
) -> pd.DataFrame:
    metrics = episode_df[
        episode_df["split"].isin(splits) & episode_df["policy_name"].isin(policies)
    ].copy()
    for split in splits:
        for policy in policies:
            if metrics[metrics["split"].eq(split) & metrics["policy_name"].eq(policy)].empty:
                raise ValueError(f"Missing rows for split={split}, policy={policy}")

    metrics["no_cut_episode"] = ~metrics["episode_cut_occurred"]
    summary = (
        metrics.groupby(["split", "policy_name"], sort=False)
        .agg(
            num_episodes=("episode_id", "nunique"),
            mean_final_after_tax_total_value=(
                "episode_final_after_tax_total_value",
                "mean",
            ),
            median_final_after_tax_total_value=(
                "episode_final_after_tax_total_value",
                "median",
            ),
            std_final_after_tax_total_value=(
                "episode_final_after_tax_total_value",
                "std",
            ),
            mean_total_tax_paid=("total_tax_paid", "mean"),
            median_total_tax_paid=("total_tax_paid", "median"),
            mean_effective_tax_rate=("mean_effective_tax_rate_on_sales", "mean"),
            mean_pct_position_sold_short_term=(
                "pct_episode_position_sold_short_term",
                "mean",
            ),
            mean_pct_position_sold_long_term=(
                "pct_episode_position_sold_long_term",
                "mean",
            ),
            terminal_liquidation_frequency=(
                "episode_terminal_liquidation_executed",
                "mean",
            ),
            no_cut_episode_count=("no_cut_episode", "sum"),
            no_cut_episode_pct=("no_cut_episode", "mean"),
            average_days_to_first_sale=("days_to_first_sale", "mean"),
            median_days_to_first_sale=("days_to_first_sale", "median"),
            pct_episodes_with_first_sale_before_tax_transition=(
                "first_sale_before_tax_transition",
                "mean",
            ),
            mean_num_discretionary_sales=("num_discretionary_sales", "mean"),
            mean_realized_after_tax_pnl=("episode_realized_after_tax_pnl", "mean"),
            total_tax_paid_sum=("total_tax_paid", "sum"),
            positive_taxable_pre_tax_increment_sum=(
                "total_positive_taxable_pre_tax_increment",
                "sum",
            ),
        )
        .reset_index()
    )
    summary["no_cut_episode_count"] = summary["no_cut_episode_count"].astype(int)
    summary["discretionary_sale_episode_count"] = (
        summary["num_episodes"] - summary["no_cut_episode_count"]
    )
    summary["discretionary_sale_episode_pct"] = 1.0 - summary["no_cut_episode_pct"]
    return ordered_policy_frame(summary, policies)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.mask(denominator.eq(0.0))
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def plot_bar_by_split(
    df: pd.DataFrame,
    split: str,
    y_column: str,
    path: Path,
    registry: OutputRegistry,
    step: str,
    description: str,
    *,
    policy_column: str = "policy_name",
    title: str | None = None,
    ylabel: str | None = None,
) -> None:
    split_df = df[df["split"].eq(split)].copy()
    plt.figure(figsize=(10, 4.8))
    plt.bar(split_df[policy_column], split_df[y_column], color="#3b6ea8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(ylabel or y_column)
    plt.title(title or f"{y_column} - {split}")
    plt.grid(axis="y", alpha=0.25)
    save_plot(path, registry, step, description)


def step4_first_sale_margin(policy_name: str) -> float:
    if policy_name == "trained_dqn_thresholded_margin_0p020":
        return 0.020
    if policy_name.startswith("trained_dqn_first_sale_margin_"):
        return extract_first_sale_margin(policy_name)
    return np.nan


def validate_no_legacy_step4_metrics(columns: list[str]) -> None:
    lower_tokens = [token.lower() for token in STEP4_LEGACY_FORBIDDEN_COLUMN_TOKENS]
    bad_columns = []
    for column in columns:
        lower_column = column.lower()
        if column in STEP4_LEGACY_FORBIDDEN_COLUMNS:
            bad_columns.append(column)
            continue
        if any(token in lower_column for token in lower_tokens):
            bad_columns.append(column)
    if bad_columns:
        raise ValueError(
            "Step 4 output attempted to include legacy Step 4B or path Sharpe "
            "metric column(s): " + ", ".join(sorted(set(bad_columns)))
        )


def missing_split_policy_rows(
    df: pd.DataFrame,
    *,
    policies: list[str],
    splits: list[str],
) -> list[str]:
    missing: list[str] = []
    for split in splits:
        split_policies = set(
            df.loc[df["split"].eq(split), "policy_name"].astype(str).unique()
        )
        for policy in policies:
            if policy not in split_policies:
                missing.append(f"{split}:{policy}")
    return missing


def load_phase_1_3_module() -> Any:
    module_path = PROJECT_ROOT / "scripts" / "build_quant_analysis_phase_1_3.py"
    spec = importlib.util.spec_from_file_location(
        "build_quant_analysis_phase_1_3_for_step4",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Step 3B helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step4_step3b_policy_df() -> pd.DataFrame:
    rows = []
    for order, policy in enumerate(STEP4_STEP3B_POLICIES):
        if policy in BENCHMARK_POLICIES:
            family = "benchmark"
            role = "benchmark"
        elif policy == "trained_dqn_greedy":
            family = "raw_dqn"
            role = "raw DQN"
        elif policy == "trained_dqn_thresholded_margin_0p020":
            family = "thresholded_dqn"
            role = "thresholded DQN"
        else:
            family = "first_sale_thresholded_dqn"
            role = "preferred DQN" if policy == PREFERRED_POLICY else "sensitivity DQN"
        rows.append(
            {
                "policy_order": order,
                "policy_name": policy,
                "policy_family": family,
                "role": role,
                "included_in_main_table": True,
                "notes": "Step 4 corrected EAAT/TA-EAAT Sharpe coverage.",
            }
        )
    return pd.DataFrame(rows)


def validate_step3b_eaat_notes(notes_path: Path) -> None:
    if not notes_path.exists():
        raise FileNotFoundError(
            "Missing corrected Step 3B EAAT Sharpe notes file: "
            f"{relative_project_path(notes_path)}"
        )
    text = notes_path.read_text(encoding="utf-8")
    lower_text = text.lower()
    missing_phrases = [
        phrase for phrase in STEP4_EAAT_NOTE_REQUIRED_PHRASES if phrase.lower() not in lower_text
    ]
    if missing_phrases:
        raise ValueError(
            "Step 4 requires corrected final Step 3B EAAT/TA-EAAT notes. "
            "Missing phrase(s): " + "; ".join(missing_phrases)
        )
    forbidden_phrases = [
        phrase
        for phrase in STEP4_EAAT_NOTE_FORBIDDEN_PHRASES
        if phrase.lower() in lower_text
    ]
    if forbidden_phrases:
        raise ValueError(
            "Step 4 refused a Sharpe notes file with legacy assumptions: "
            + "; ".join(forbidden_phrases)
        )


def validate_step3b_eaat_assumptions(df: pd.DataFrame, *, source: Path) -> None:
    require_columns(
        df,
        STEP4_SOURCE_REQUIRED_SHARPE_COLUMNS,
        source=source,
        step="Step 4 corrected Step 3B Sharpe merge",
    )
    annual_rf = pd.to_numeric(df["annual_risk_free_rate"], errors="coerce")
    daily_rf = pd.to_numeric(df["daily_risk_free_rate"], errors="coerce")
    annualization = pd.to_numeric(df["annualization_factor"], errors="coerce")
    if annual_rf.isna().any() or not np.allclose(annual_rf, ANNUAL_RISK_FREE_RATE):
        raise ValueError(
            "Corrected Step 3B EAAT Sharpe rows must use annual_risk_free_rate=0.04 "
            f"in {relative_project_path(source)}."
        )
    if daily_rf.isna().any() or not np.allclose(daily_rf, DAILY_RISK_FREE_RATE):
        raise ValueError(
            "Corrected Step 3B EAAT Sharpe rows must use daily_risk_free_rate="
            "(1 + 0.04) ** (1 / 252) - 1 in "
            f"{relative_project_path(source)}."
        )
    if annualization.isna().any() or not np.allclose(
        annualization,
        ANNUALIZATION_FACTOR,
    ):
        raise ValueError(
            "Corrected Step 3B EAAT Sharpe rows must use annualization_factor=252 "
            f"in {relative_project_path(source)}."
        )


def regenerate_step3b_eaat_outputs_for_step4(
    *,
    config: dict[str, Any],
    run_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    phase_1_3 = load_phase_1_3_module()
    step_rollouts_path = run_path / "baselines" / "baseline_step_rollouts.csv"
    if not step_rollouts_path.exists():
        raise FileNotFoundError(
            f"Missing baseline step rollouts: {relative_project_path(step_rollouts_path)}"
        )
    step_rollouts_df = pd.read_csv(step_rollouts_path, low_memory=False)
    policy_df = step4_step3b_policy_df()
    stock_paths_df, metadata_notes = phase_1_3.load_episode_stock_paths_for_eaat(
        config=config,
        step_rollouts_df=step_rollouts_df,
        policy_df=policy_df,
        source_path=step_rollouts_path,
    )
    eaat_summary_df, episode_metrics_df, tranche_records_df = (
        phase_1_3.build_eaat_sharpe_metrics(
            step_rollouts_df,
            stock_paths_df,
            policy_df,
            source_path=step_rollouts_path,
        )
    )
    eaat_summary_df.to_csv(
        output_dir / "step3b_policy_eaat_sharpe_metrics.csv",
        index=False,
    )
    write_markdown_table(
        eaat_summary_df,
        output_dir / "step3b_policy_eaat_sharpe_metrics.md",
    )
    episode_metrics_df.to_csv(
        output_dir / "step3b_episode_eaat_sharpe_metrics.csv",
        index=False,
    )
    tranche_records_df.to_csv(
        output_dir / "step3b_episode_tranche_records.csv",
        index=False,
    )
    phase_1_3.write_step3b_eaat_notes(
        output_dir / "step3b_eaat_sharpe_metrics_notes.txt",
        metadata_notes=metadata_notes,
    )
    phase_1_3.write_step3b_one_case_verification(
        output_dir=output_dir,
        config=config,
        step_rollouts_df=step_rollouts_df,
        episode_metrics_df=episode_metrics_df,
        tranche_records_df=tranche_records_df,
    )
    phase_1_3.write_step3b_median_episode_verification(
        output_dir=output_dir,
        step_rollouts_df=step_rollouts_df,
        episode_metrics_df=episode_metrics_df,
        tranche_records_df=tranche_records_df,
    )
    return eaat_summary_df


def load_step3b_eaat_sharpe_metrics(
    *,
    config: dict[str, Any],
    run_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    csv_path = output_dir / "step3b_policy_eaat_sharpe_metrics.csv"
    notes_path = output_dir / "step3b_eaat_sharpe_metrics_notes.txt"
    if not csv_path.exists():
        raise FileNotFoundError(
            "Missing corrected Step 3B EAAT Sharpe file: "
            f"{relative_project_path(csv_path)}"
        )
    validate_step3b_eaat_notes(notes_path)
    df = pd.read_csv(csv_path)
    validate_step3b_eaat_assumptions(df, source=csv_path)
    missing = missing_split_policy_rows(
        df,
        policies=DQN_DECISION_POLICIES,
        splits=VALIDATION_TEST_SPLITS,
    )
    if missing:
        df = regenerate_step3b_eaat_outputs_for_step4(
            config=config,
            run_path=run_path,
            output_dir=output_dir,
        )
        validate_step3b_eaat_notes(notes_path)
        validate_step3b_eaat_assumptions(df, source=csv_path)
        missing = missing_split_policy_rows(
            df,
            policies=DQN_DECISION_POLICIES,
            splits=VALIDATION_TEST_SPLITS,
        )
        if missing:
            raise ValueError(
                "Corrected Step 3B EAAT Sharpe output is missing required Step 4 "
                "DQN split/policy rows after regeneration: " + ", ".join(missing)
            )
    return df


def validate_step4_margin_sweep(df: pd.DataFrame, *, split: str) -> None:
    split_margins = sorted(
        round(float(value), 2)
        for value in df.loc[df["split"].eq(split), "first_sale_margin"].dropna().unique()
    )
    expected = [round(value, 2) for value in STEP4_FIRST_SALE_MARGINS]
    if split_margins != expected:
        raise ValueError(
            f"Step 4 {split} margin chart data must include exactly the "
            f"0.02-0.09 first-sale sweep. Got {split_margins}; expected {expected}."
        )


def plot_step4_margin_vs_after_tax_value(
    df: pd.DataFrame,
    split: str,
    path: Path,
    registry: OutputRegistry,
) -> None:
    split_df = df[df["split"].eq(split)].copy()
    margin_df = split_df[
        split_df["policy_name"].isin(STEP4_FIRST_SALE_MARGIN_POLICIES)
    ].sort_values("first_sale_margin")
    validate_step4_margin_sweep(margin_df, split=split)
    reference_values = split_df.set_index("policy_name")[
        "mean_final_after_tax_total_value"
    ]
    for policy in ["trained_dqn_greedy", "trained_dqn_thresholded_margin_0p020"]:
        if policy not in reference_values.index:
            raise ValueError(f"Missing Step 4 chart reference policy: {split}:{policy}")

    plt.figure(figsize=(8, 5))
    plt.plot(
        margin_df["first_sale_margin"],
        margin_df["mean_final_after_tax_total_value"],
        marker="o",
        color="#2f6f4e",
        label="first-sale-thresholded DQN",
    )
    plt.axhline(
        reference_values["trained_dqn_greedy"],
        color="#8f3d3d",
        linestyle="--",
        linewidth=1.5,
        label="greedy DQN",
    )
    plt.axhline(
        reference_values["trained_dqn_thresholded_margin_0p020"],
        color="#3b6ea8",
        linestyle=":",
        linewidth=1.8,
        label="normal thresholded DQN",
    )
    plt.xticks(STEP4_FIRST_SALE_MARGINS)
    plt.xlabel("first-sale margin")
    plt.ylabel("mean final after-tax total value")
    plt.title(f"DQN decision-rule progression - {split}")
    plt.grid(alpha=0.25)
    plt.legend()
    save_plot(
        path,
        registry,
        "4",
        f"Step 4 first-sale margin versus after-tax value for {split}.",
    )


def plot_step4_margin_metric(
    df: pd.DataFrame,
    *,
    split: str,
    y_column: str,
    ylabel: str,
    path: Path,
    registry: OutputRegistry,
    description: str,
) -> None:
    margin_df = df[
        df["split"].eq(split) & df["policy_name"].isin(STEP4_FIRST_SALE_MARGIN_POLICIES)
    ].sort_values("first_sale_margin")
    validate_step4_margin_sweep(margin_df, split=split)
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(
        margin_df["first_sale_margin"],
        margin_df[y_column],
        marker="o",
        color="#2f6f4e",
    )
    plt.xticks(STEP4_FIRST_SALE_MARGINS)
    plt.xlabel("first-sale margin")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by first-sale margin - {split}")
    plt.grid(alpha=0.25)
    save_plot(path, registry, "4", description)


def plot_step4_sharpe_by_policy(
    df: pd.DataFrame,
    *,
    split: str,
    y_column: str,
    title: str,
    path: Path,
    registry: OutputRegistry,
    description: str,
) -> None:
    split_df = df[df["split"].eq(split)].copy()
    if split_df.empty:
        raise ValueError(f"Missing Step 4 Sharpe chart rows for split={split}")
    plt.figure(figsize=(11, 4.8))
    plt.bar(split_df["policy_name"].map(short_policy_label), split_df[y_column], color="#3b6ea8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(y_column)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    save_plot(path, registry, "4", description)


def build_step4(
    episode_df: pd.DataFrame,
    output_dir: Path,
    plots_dir: Path,
    registry: OutputRegistry,
    *,
    config: dict[str, Any],
    run_path: Path,
) -> pd.DataFrame:
    summary = aggregate_policy_metrics(
        episode_df,
        DQN_DECISION_POLICIES,
        VALIDATION_TEST_SPLITS,
    )
    pivot = summary.pivot(
        index="split",
        columns="policy_name",
        values="mean_final_after_tax_total_value",
    )
    summary["mean_excess_value_vs_trained_dqn_greedy"] = (
        summary["mean_final_after_tax_total_value"]
        - summary["split"].map(pivot["trained_dqn_greedy"])
    )
    summary["mean_excess_value_vs_trained_dqn_thresholded_margin_0p020"] = (
        summary["mean_final_after_tax_total_value"]
        - summary["split"].map(pivot["trained_dqn_thresholded_margin_0p020"])
    )
    summary["first_sale_margin"] = summary["policy_name"].map(step4_first_sale_margin)

    sharpe_df = load_step3b_eaat_sharpe_metrics(
        config=config,
        run_path=run_path,
        output_dir=output_dir,
    )
    sharpe_counts = sharpe_df[["split", "policy_name", "num_episodes"]].copy()
    count_check = summary[["split", "policy_name", "num_episodes"]].merge(
        sharpe_counts,
        on=["split", "policy_name"],
        how="left",
        suffixes=("_step4", "_step3b"),
    )
    mismatched_counts = count_check[
        count_check["num_episodes_step3b"].isna()
        | count_check["num_episodes_step4"].ne(count_check["num_episodes_step3b"])
    ]
    if not mismatched_counts.empty:
        raise ValueError(
            "Step 4 and corrected Step 3B EAAT Sharpe episode counts differ for: "
            + ", ".join(
                f"{row.split}:{row.policy_name}"
                for row in mismatched_counts.itertuples(index=False)
            )
        )

    sharpe_columns = ["split", "policy_name", *STEP4_EAAT_SHARPE_COLUMNS]
    out = summary[STEP4_BASE_COLUMNS].merge(
        sharpe_df[sharpe_columns],
        on=["split", "policy_name"],
        how="left",
        validate="one_to_one",
    )
    missing_sharpe_rows = out[
        out["num_valid_EAAT_Sharpe_episodes"].isna()
        | out["num_valid_TA_EAAT_Sharpe_episodes"].isna()
    ]
    if not missing_sharpe_rows.empty:
        raise ValueError(
            "Step 4 table is missing corrected EAAT/TA-EAAT Sharpe values for: "
            + ", ".join(
                f"{row.split}:{row.policy_name}"
                for row in missing_sharpe_rows.itertuples(index=False)
            )
        )
    validate_no_legacy_step4_metrics(out.columns.tolist())
    save_table(
        out,
        output_dir / "step4_dqn_decision_rule_comparison.csv",
        output_dir / "step4_dqn_decision_rule_comparison.md",
        registry,
        "4",
        "DQN decision rule comparison for validation and test splits.",
    )
    for split in VALIDATION_TEST_SPLITS:
        plot_bar_by_split(
            out,
            split,
            "mean_final_after_tax_total_value",
            plots_dir / f"step4_dqn_decision_rule_value_{split}.png",
            registry,
            "4",
            f"Step 4 DQN decision-rule mean final after-tax value for {split}.",
            title=f"DQN decision rule value - {split}",
            ylabel="mean final after-tax total value",
        )
        plot_step4_margin_vs_after_tax_value(
            out,
            split,
            plots_dir / f"step4_margin_vs_after_tax_value_{split}.png",
            registry,
        )
        plot_step4_sharpe_by_policy(
            out,
            split=split,
            y_column="median_EAAT_Sharpe",
            title=f"Median EAAT Sharpe by DQN decision rule - {split}",
            path=plots_dir / f"step4_eaat_median_sharpe_by_policy_{split}.png",
            registry=registry,
            description=f"Step 4 median EAAT Sharpe by DQN decision rule for {split}.",
        )
        plot_step4_sharpe_by_policy(
            out,
            split=split,
            y_column="median_TA_EAAT_Sharpe",
            title=f"Median TA-EAAT Sharpe by DQN decision rule - {split}",
            path=plots_dir / f"step4_ta_eaat_median_sharpe_by_policy_{split}.png",
            registry=registry,
            description=f"Step 4 median TA-EAAT Sharpe by DQN decision rule for {split}.",
        )
    plot_step4_margin_metric(
        out,
        split="test",
        y_column="mean_pct_position_sold_short_term",
        ylabel="mean short-term sold fraction",
        path=plots_dir / "step4_margin_vs_short_term_fraction_test.png",
        registry=registry,
        description="Step 4 first-sale margin versus short-term sold fraction for test split.",
    )
    plot_step4_margin_metric(
        out,
        split="test",
        y_column="no_cut_episode_pct",
        ylabel="no-cut episode pct",
        path=plots_dir / "step4_margin_vs_no_cut_pct_test.png",
        registry=registry,
        description="Step 4 first-sale margin versus no-cut percentage for test split.",
    )
    plot_step4_margin_metric(
        out,
        split="test",
        y_column="mean_total_tax_paid",
        ylabel="mean total tax paid",
        path=plots_dir / "step4_margin_vs_total_tax_paid_test.png",
        registry=registry,
        description="Step 4 first-sale margin versus total tax paid for test split.",
    )
    return out


def short_policy_label(policy_name: str) -> str:
    labels = {
        "hold_to_terminal": "hold",
        "sell_immediately": "sell_now",
        "sell_half_then_hold": "half_hold",
        "sell_quarters_over_time": "quarters",
        "random_policy": "random",
        "trained_dqn_greedy": "dqn_greedy",
        "trained_dqn_thresholded_margin_0p020": "dqn_thr_020",
        "trained_dqn_first_sale_margin_0p020_normal_0p020": "fsm_020",
        "trained_dqn_first_sale_margin_0p030_normal_0p020": "fsm_030",
        "trained_dqn_first_sale_margin_0p040_normal_0p020": "fsm_040",
        "trained_dqn_first_sale_margin_0p050_normal_0p020": "fsm_050",
        "trained_dqn_first_sale_margin_0p060_normal_0p020": "fsm_060",
        PREFERRED_POLICY: "fsm_070",
        "trained_dqn_first_sale_margin_0p080_normal_0p020": "fsm_080",
        "trained_dqn_first_sale_margin_0p090_normal_0p020": "fsm_090",
    }
    return labels.get(policy_name, policy_name[:16])


def extract_first_sale_margin(policy_name: str) -> float:
    token = policy_name.split("trained_dqn_first_sale_margin_", 1)[1].split(
        "_normal_",
        1,
    )[0]
    return float(token.replace("p", "."))


def build_step5(
    episode_df: pd.DataFrame,
    output_dir: Path,
    plots_dir: Path,
    registry: OutputRegistry,
) -> pd.DataFrame:
    summary = aggregate_policy_metrics(
        episode_df,
        FIRST_SALE_MARGIN_POLICIES,
        VALIDATION_TEST_SPLITS,
    )
    summary["first_sale_margin"] = summary["policy_name"].map(extract_first_sale_margin)
    columns = [
        "split",
        "policy_name",
        "first_sale_margin",
        "num_episodes",
        "average_days_to_first_sale",
        "median_days_to_first_sale",
        "pct_episodes_with_first_sale_before_tax_transition",
        "mean_pct_position_sold_short_term",
        "mean_pct_position_sold_long_term",
        "mean_total_tax_paid",
        "mean_effective_tax_rate",
        "mean_final_after_tax_total_value",
        "median_final_after_tax_total_value",
        "no_cut_episode_count",
        "no_cut_episode_pct",
        "terminal_liquidation_frequency",
        "mean_num_discretionary_sales",
    ]
    out = summary[columns].sort_values(["split", "first_sale_margin"])
    save_table(
        out,
        output_dir / "step5_first_sale_margin_sensitivity.csv",
        output_dir / "step5_first_sale_margin_sensitivity.md",
        registry,
        "5",
        "First-sale margin sensitivity table.",
    )
    notes = "\n".join(
        [
            "Step 5 interpretation notes",
            "The full 0.020-0.090 first-sale margin sweep is available for sensitivity checks.",
            "Step 4 is the final DQN decision-rule and first-sale-margin analysis block.",
            "0.070 is the preferred balance.",
            "0.080 and 0.090 are useful sensitivity policies but increasingly close to hold-to-terminal behavior.",
            "0.090 should not be the main interpretation policy if no-cut episodes are too high.",
            "",
        ]
    )
    save_text(
        notes,
        output_dir / "step5_interpretation_notes.txt",
        registry,
        "5",
        "Compact interpretation notes for first-sale margin sensitivity.",
    )
    return out


def validate_no_legacy_step6_metrics(columns: list[str]) -> None:
    bad_columns = [
        column for column in columns if column in STEP6_FORBIDDEN_LEGACY_SHARPE_COLUMNS
    ]
    if bad_columns:
        raise ValueError(
            "Step 6 output attempted to include forbidden legacy Sharpe column(s): "
            + ", ".join(sorted(set(bad_columns)))
        )


def validate_step6_output(out: pd.DataFrame, sharpe_df: pd.DataFrame) -> None:
    missing_output_rows = missing_split_policy_rows(
        out,
        policies=POLICY_UNIVERSE,
        splits=VALIDATION_TEST_SPLITS,
    )
    if missing_output_rows:
        raise ValueError(
            "Step 6 table is missing required split-policy rows: "
            + ", ".join(missing_output_rows)
        )

    sharpe_pairs = sharpe_df[["split", "policy_name"]].drop_duplicates()
    missing_sharpe_rows = missing_split_policy_rows(
        sharpe_pairs,
        policies=POLICY_UNIVERSE,
        splits=VALIDATION_TEST_SPLITS,
    )
    if missing_sharpe_rows:
        raise ValueError(
            "Step 6 corrected EAAT/TA-EAAT source is missing split-policy rows: "
            + ", ".join(missing_sharpe_rows)
        )

    merged_missing = out[
        out["num_valid_EAAT_Sharpe_episodes"].isna()
        | out["num_valid_TA_EAAT_Sharpe_episodes"].isna()
    ]
    if not merged_missing.empty:
        raise ValueError(
            "Step 6 table is missing corrected EAAT/TA-EAAT values for: "
            + ", ".join(
                f"{row.split}:{row.policy_name}"
                for row in merged_missing.itertuples(index=False)
            )
        )

    missing_tax_columns = [
        column for column in STEP6_REQUIRED_TAX_ACCOUNTING_COLUMNS if column not in out.columns
    ]
    missing_sharpe_columns = [
        column for column in STEP6_EAAT_SHARPE_COLUMNS if column not in out.columns
    ]
    if missing_tax_columns or missing_sharpe_columns:
        raise ValueError(
            "Step 6 table is missing required metric columns. "
            f"tax={missing_tax_columns}; sharpe={missing_sharpe_columns}"
        )

    validate_no_legacy_step6_metrics(out.columns.tolist())
    numeric = out.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Step 6 output contains infinite values after safe division.")


def plot_step6_median_eaat_ta_eaat_sharpe(
    out: pd.DataFrame,
    plots_dir: Path,
    registry: OutputRegistry,
) -> None:
    split_df = out[out["split"].eq("test")].copy()
    if split_df.empty:
        raise ValueError("Missing Step 6 test rows for median EAAT/TA-EAAT chart.")
    x = np.arange(len(split_df))
    width = 0.38
    plt.figure(figsize=(11, 4.8))
    plt.bar(
        x - width / 2.0,
        split_df["median_EAAT_Sharpe"],
        width,
        label="median EAAT Sharpe",
        color="#3b6ea8",
    )
    plt.bar(
        x + width / 2.0,
        split_df["median_TA_EAAT_Sharpe"],
        width,
        label="median TA-EAAT Sharpe",
        color="#8a6f2a",
    )
    plt.xticks(x, split_df["policy_name"].map(short_policy_label), rotation=35, ha="right")
    plt.ylabel("median Sharpe")
    plt.title("Median EAAT and TA-EAAT Sharpe - test")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_plot(
        plots_dir / "step6_median_eaat_and_ta_eaat_sharpe_test.png",
        registry,
        "6",
        "Grouped median EAAT and TA-EAAT Sharpe by policy for the test split.",
    )


def write_step6_tax_efficiency_markdown(out: pd.DataFrame, path: Path) -> None:
    headline_columns = [
        "split",
        "policy_name",
        "median_EAAT_Sharpe",
        "median_TA_EAAT_Sharpe",
    ]
    headline = out[headline_columns]
    text = "\n".join(
        [
            "# Step 6 Tax-Efficiency Analysis",
            "",
            "## Headline Corrected Sharpe Diagnostics",
            "",
            markdown_table_text(headline).rstrip(),
            "",
            "## Full Tax-Accounting And Risk-Adjusted Diagnostics",
            "",
            markdown_table_text(out).rstrip(),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def build_step6(
    episode_df: pd.DataFrame,
    output_dir: Path,
    plots_dir: Path,
    registry: OutputRegistry,
    *,
    config: dict[str, Any],
    run_path: Path,
) -> pd.DataFrame:
    summary = aggregate_policy_metrics(episode_df, POLICY_UNIVERSE, VALIDATION_TEST_SPLITS)
    step6_metrics = episode_df[
        episode_df["split"].isin(VALIDATION_TEST_SPLITS)
        & episode_df["policy_name"].isin(POLICY_UNIVERSE)
    ].copy()
    step6_metrics["episode_realized_pre_tax_pnl"] = (
        step6_metrics["episode_realized_after_tax_pnl"] + step6_metrics["total_tax_paid"]
    )
    step6_metrics["episode_final_pre_tax_total_value"] = (
        step6_metrics["episode_final_after_tax_total_value"]
        + step6_metrics["total_tax_paid"]
    )
    realized_loss = (
        step6_metrics.groupby(["split", "policy_name"], sort=False)
        .agg(
            mean_pre_tax_realized_gain=("episode_realized_pre_tax_pnl", "mean"),
            mean_after_tax_realized_gain=("episode_realized_after_tax_pnl", "mean"),
            mean_final_pre_tax_total_value=(
                "episode_final_pre_tax_total_value",
                "mean",
            ),
        )
        .reset_index()
    )
    summary = summary.merge(
        realized_loss,
        on=["split", "policy_name"],
        how="left",
        validate="one_to_one",
    )
    summary["short_term_realized_fraction"] = summary[
        "mean_pct_position_sold_short_term"
    ]
    summary["long_term_realized_fraction"] = summary[
        "mean_pct_position_sold_long_term"
    ]
    summary["tax_paid_as_pct_of_positive_taxable_pre_tax_increment"] = (
        safe_divide(
            summary["total_tax_paid_sum"],
            summary["positive_taxable_pre_tax_increment_sum"],
        )
    )
    summary["mean_after_tax_value_loss_vs_pre_tax"] = (
        summary["mean_pre_tax_realized_gain"]
        - summary["mean_after_tax_realized_gain"]
    )
    summary["mean_after_tax_value_loss_pct_vs_pre_tax"] = safe_divide(
        summary["mean_after_tax_value_loss_vs_pre_tax"],
        summary["mean_pre_tax_realized_gain"],
    )
    summary["mean_final_after_tax_value_loss_vs_pre_tax"] = (
        summary["mean_final_pre_tax_total_value"]
        - summary["mean_final_after_tax_total_value"]
    )
    summary["mean_final_after_tax_value_loss_pct_vs_pre_tax"] = safe_divide(
        summary["mean_final_after_tax_value_loss_vs_pre_tax"],
        summary["mean_final_pre_tax_total_value"],
    )
    pivot = summary.pivot(
        index="split",
        columns="policy_name",
        values="mean_total_tax_paid",
    )
    summary["mean_tax_paid_difference_vs_hold_to_terminal"] = (
        summary["mean_total_tax_paid"] - summary["split"].map(pivot["hold_to_terminal"])
    )
    summary["mean_tax_paid_difference_vs_sell_immediately"] = (
        summary["mean_total_tax_paid"] - summary["split"].map(pivot["sell_immediately"])
    )
    sharpe_df = load_step3b_eaat_sharpe_metrics(
        config=config,
        run_path=run_path,
        output_dir=output_dir,
    )
    sharpe_columns = ["split", "policy_name", *STEP6_EAAT_SHARPE_COLUMNS]
    columns = [
        "split",
        "policy_name",
        "short_term_realized_fraction",
        "long_term_realized_fraction",
        "mean_total_tax_paid",
        "median_total_tax_paid",
        "tax_paid_as_pct_of_positive_taxable_pre_tax_increment",
        "mean_effective_tax_rate",
        "mean_pre_tax_realized_gain",
        "mean_after_tax_realized_gain",
        "mean_after_tax_value_loss_vs_pre_tax",
        "mean_after_tax_value_loss_pct_vs_pre_tax",
        "mean_final_after_tax_total_value",
        "mean_final_pre_tax_total_value",
        "mean_final_after_tax_value_loss_vs_pre_tax",
        "mean_final_after_tax_value_loss_pct_vs_pre_tax",
        "mean_realized_after_tax_pnl",
        "mean_tax_paid_difference_vs_hold_to_terminal",
        "mean_tax_paid_difference_vs_sell_immediately",
    ]
    out = summary[columns].merge(
        sharpe_df[sharpe_columns],
        on=["split", "policy_name"],
        how="left",
        validate="one_to_one",
    )
    remaining_sharpe_columns = [
        column
        for column in STEP6_EAAT_SHARPE_COLUMNS
        if column not in STEP6_HEADLINE_SHARPE_COLUMNS
    ]
    out = out[
        [
            "split",
            "policy_name",
            *STEP6_HEADLINE_SHARPE_COLUMNS,
            *[column for column in columns if column not in {"split", "policy_name"}],
            *remaining_sharpe_columns,
        ]
    ]
    validate_step6_output(out, sharpe_df)
    step6_csv_path = output_dir / "step6_tax_efficiency_analysis.csv"
    step6_md_path = output_dir / "step6_tax_efficiency_analysis.md"
    out.to_csv(step6_csv_path, index=False)
    write_step6_tax_efficiency_markdown(out, step6_md_path)
    registry.add(
        step6_csv_path,
        "csv",
        "6",
        "Tax-efficiency analysis for final policy universe.",
    )
    registry.add(
        step6_md_path,
        "md",
        "6",
        "Preview-friendly tax-efficiency analysis with headline Sharpe diagnostics.",
    )
    notes = "\n".join(
        [
            "Step 6 tax-efficiency notes",
            "Step 6 combines tax-accounting diagnostics with corrected EAAT / TA-EAAT Sharpe diagnostics.",
            "EAAT Sharpe measures exposure-adjusted after-tax annual risk efficiency using terminal after-tax wealth.",
            "TA-EAAT Sharpe measures tranche-annualized after-tax risk efficiency across sale tranches.",
            "Final after-tax value remains the primary thesis metric.",
            "EAAT and TA-EAAT are secondary risk-adjusted diagnostics.",
            "Tax-paid differences versus hold_to_terminal and sell_immediately are differences in mean tax paid, not causal estimates.",
            "tax_paid_as_pct_of_positive_taxable_pre_tax_increment uses total_positive_taxable_pre_tax_increment from baseline_episode_metrics.csv.",
            "mean_pre_tax_realized_gain is computed as episode_realized_after_tax_pnl + total_tax_paid, averaged by split-policy.",
            "mean_final_pre_tax_total_value is computed as episode_final_after_tax_total_value + total_tax_paid, averaged by split-policy.",
            f"annual risk-free rate = {ANNUAL_RISK_FREE_RATE:.0%}",
            f"daily risk-free rate = (1 + {ANNUAL_RISK_FREE_RATE}) ** (1 / {ANNUALIZATION_FACTOR}) - 1 = {DAILY_RISK_FREE_RATE}",
            f"annualization factor = {ANNUALIZATION_FACTOR}",
            "liquidated after-tax proceeds earn the daily risk-free rate",
            "",
        ]
    )
    save_text(
        notes,
        output_dir / "step6_tax_efficiency_notes.txt",
        registry,
        "6",
        "Tax-efficiency denominator and interpretation notes.",
    )
    for split in ["test", "validation"]:
        split_df = out[out["split"].eq(split)]
        x = np.arange(len(split_df))
        plt.figure(figsize=(10, 4.8))
        plt.bar(x, split_df["short_term_realized_fraction"], label="short-term")
        plt.bar(
            x,
            split_df["long_term_realized_fraction"],
            bottom=split_df["short_term_realized_fraction"],
            label="long-term",
        )
        plt.xticks(x, split_df["policy_name"], rotation=35, ha="right")
        plt.ylabel("sold fraction")
        plt.title(f"Short vs long-term sold fraction - {split}")
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
        save_plot(
            plots_dir / f"step6_short_vs_long_term_sold_fraction_{split}.png",
            registry,
            "6",
            f"Stacked short/long-term sold fraction for {split}.",
        )
    plot_step6_median_eaat_ta_eaat_sharpe(out, plots_dir, registry)
    plot_bar_by_split(
        out,
        "test",
        "mean_total_tax_paid",
        plots_dir / "step6_mean_tax_paid_by_policy_test.png",
        registry,
        "6",
        "Mean tax paid by policy for test split.",
        title="Mean tax paid by policy - test",
        ylabel="mean total tax paid",
    )
    return out


def import_baseline_evaluator() -> Any:
    path = PROJECT_ROOT / "scripts" / "evaluate_reward_a_baselines.py"
    spec = importlib.util.spec_from_file_location("evaluate_reward_a_baselines", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import evaluator helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_train_preferred_diagnostics(
    config_path: Path,
    config: dict[str, Any],
    run_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    train_episode_path = (
        intermediate_dir / "step7_preferred_policy_train_episode_metrics.csv"
    )
    train_step_path = intermediate_dir / "step7_preferred_policy_train_step_rollouts.csv"
    splits_df = pd.read_csv(run_path / "episode_splits.csv")
    train_ids = splits_df.loc[splits_df["split"].eq("train"), "episode_id"].astype(str).tolist()

    if train_episode_path.exists() and train_step_path.exists():
        train_episode = pd.read_csv(train_episode_path)
        if train_episode["episode_id"].nunique() == len(train_ids):
            return (
                prepare_episode_metrics(train_episode),
                prepare_step_rollouts(pd.read_csv(train_step_path, low_memory=False)),
                False,
            )

    evaluator = import_baseline_evaluator()
    evaluator._validate_reward_config(config)
    env = evaluator.make_env(config)
    expected_reward_version = str(require_nested(config, "reward.expected_info_reward_version"))
    output_model_dir = resolve_project_path(require_nested(config, "logging.output_dir"))
    model_path = (
        output_model_dir / "best_validation_model.pt"
        if (output_model_dir / "best_validation_model.pt").exists()
        else output_model_dir / "final_model.pt"
    )
    state_columns = evaluator.load_state_columns(
        resolve_project_path(require_nested(config, "environment.state_schema_path"))
    )
    device = evaluator._resolve_device(str(require_nested(config, "training.device")))
    q_net = evaluator.load_trained_q_network(
        config=config,
        model_path=model_path,
        obs_dim=len(state_columns),
        num_actions=len(env.action_fractions),
        device=device,
    )
    policy_names = evaluator.resolve_evaluation_policy_names(config)
    rng = np.random.default_rng(RANDOM_SEED)
    episode_rows, step_rows = evaluator.evaluate_policy_on_episodes(
        env=env,
        policy_name=PREFERRED_POLICY,
        episode_ids=train_ids,
        split_name="train",
        expected_reward_version=expected_reward_version,
        rng=rng,
        q_net=q_net,
        device=device,
        policy_names=policy_names,
    )
    train_episode = pd.DataFrame(episode_rows, columns=evaluator.EPISODE_METRIC_COLUMNS)
    train_step = pd.DataFrame(step_rows, columns=evaluator.STEP_ROLLOUT_COLUMNS)
    train_episode.to_csv(train_episode_path, index=False)
    train_step.to_csv(train_step_path, index=False)
    return prepare_episode_metrics(train_episode), prepare_step_rollouts(train_step), True


def diagnostic_episode_and_step_frames(
    baseline_episode: pd.DataFrame,
    baseline_step: pd.DataFrame,
    train_episode: pd.DataFrame,
    train_step: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vt_episode = baseline_episode[
        baseline_episode["policy_name"].eq(PREFERRED_POLICY)
        & baseline_episode["split"].isin(["validation", "test"])
    ].copy()
    vt_step = baseline_step[
        baseline_step["policy_name"].eq(PREFERRED_POLICY)
        & baseline_step["split"].isin(["validation", "test"])
    ].copy()
    diag_episode = pd.concat([train_episode, vt_episode], ignore_index=True)
    diag_step = pd.concat([train_step, vt_step], ignore_index=True)
    all_episode = diag_episode.copy()
    all_episode["split"] = "all"
    all_step = diag_step.copy()
    all_step["split"] = "all"
    return (
        pd.concat([diag_episode, all_episode], ignore_index=True),
        pd.concat([diag_step, all_step], ignore_index=True),
    )


def distribution_dict(series: pd.Series, categories: list[str] | None = None) -> dict[str, float]:
    counts = series.value_counts(normalize=True, dropna=False).to_dict()
    if categories is None:
        keys = sorted(str(key) for key in counts)
        return {key: float(counts.get(key, 0.0)) for key in keys}
    return {key: float(counts.get(key, 0.0)) for key in categories}


def action_distribution(step_split: pd.DataFrame) -> dict[str, float]:
    labels = step_split["action_fraction_requested"].map(action_label)
    return distribution_dict(labels, ACTION_ORDER)


def first_action_distribution(step_split: pd.DataFrame) -> dict[str, float]:
    first_steps = step_split.sort_values(["episode_id", "step_in_episode"]).groupby(
        "episode_id",
        sort=False,
    ).head(1)
    labels = first_steps["action_fraction_requested"].map(action_label)
    return distribution_dict(labels, ACTION_ORDER)


def first_sale_action_distribution(step_split: pd.DataFrame) -> dict[str, float]:
    sale_steps = step_split[
        step_split["action_fraction_executed"].fillna(0).gt(0)
        & ~step_split["terminal_liquidation_executed"].fillna(False)
    ].sort_values(["episode_id", "step_in_episode"])
    first_sales = sale_steps.groupby("episode_id", sort=False).head(1)
    labels = first_sales["action_fraction_executed"].map(action_label)
    return distribution_dict(labels, ACTION_ORDER)


def first_sale_timing_distribution(episode_split: pd.DataFrame) -> dict[str, float]:
    buckets = pd.cut(
        episode_split["days_to_first_sale"],
        bins=[-np.inf, 0, 30, 180, np.inf],
        labels=["immediate", "1_to_30_days", "31_to_180_days", "over_180_days"],
    ).astype(object)
    buckets = buckets.fillna("no_discretionary_sale")
    return distribution_dict(
        buckets,
        [
            "immediate",
            "1_to_30_days",
            "31_to_180_days",
            "over_180_days",
            "no_discretionary_sale",
        ],
    )


def diagnostic_scope_for_split(split: str) -> str:
    if split == "train":
        return "in_sample_descriptive"
    if split == "all":
        return "all_episode_descriptive"
    return "out_of_sample"


def step7_behavior_metric_row(
    split_episode: pd.DataFrame,
    split_step: pd.DataFrame,
    *,
    split: str,
    diagnostic_scope: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split_episode = split_episode.copy()
    split_episode["no_cut_episode"] = ~split_episode["episode_cut_occurred"]
    discretionary = split_episode["num_discretionary_sales"].fillna(0).gt(0)
    partial = discretionary & split_episode["terminal_liquidation_fraction"].fillna(0).gt(0)
    full_early = discretionary & split_episode["terminal_liquidation_fraction"].fillna(0).eq(0)
    row: dict[str, Any] = {
        "split": split,
        "diagnostic_scope": diagnostic_scope,
        "num_episodes": int(split_episode["episode_id"].nunique()),
        "action_distribution": compact_json(action_distribution(split_step)),
        "first_action_distribution": compact_json(first_action_distribution(split_step)),
        "first_discretionary_sale_action_distribution": compact_json(
            first_sale_action_distribution(split_step)
        ),
        "first_sale_timing_distribution": compact_json(
            first_sale_timing_distribution(split_episode)
        ),
        "no_cut_episode_count": int(split_episode["no_cut_episode"].sum()),
        "no_cut_episode_pct": float(split_episode["no_cut_episode"].mean()),
        "discretionary_sale_episode_count": int(discretionary.sum()),
        "discretionary_sale_episode_pct": float(discretionary.mean()),
        "partial_discretionary_liquidation_pct": float(partial.mean()),
        "full_early_liquidation_pct": float(full_early.mean()),
        "terminal_liquidation_frequency": float(
            split_episode["episode_terminal_liquidation_executed"].mean()
        ),
        "mean_num_discretionary_sales": float(
            split_episode["num_discretionary_sales"].mean()
        ),
        "average_days_to_first_sale": float(split_episode["days_to_first_sale"].mean()),
        "median_days_to_first_sale": float(split_episode["days_to_first_sale"].median()),
        "pct_episodes_with_first_sale_before_tax_transition": float(
            split_episode["first_sale_before_tax_transition"].mean()
        ),
        "mean_pct_position_sold_short_term": float(
            split_episode["pct_episode_position_sold_short_term"].mean()
        ),
        "mean_pct_position_sold_long_term": float(
            split_episode["pct_episode_position_sold_long_term"].mean()
        ),
        "mean_total_tax_paid": float(split_episode["total_tax_paid"].mean()),
        "mean_effective_tax_rate": float(
            split_episode["mean_effective_tax_rate_on_sales"].mean()
        ),
        "mean_final_after_tax_total_value": float(
            split_episode["episode_final_after_tax_total_value"].mean()
        ),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def behavior_summary_rows(
    episode_df: pd.DataFrame,
    step_df: pd.DataFrame,
    baseline_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_vt = baseline_summary[
        baseline_summary["policy_name"].isin(["hold_to_terminal", "sell_immediately"])
    ].pivot(index="split", columns="policy_name", values="mean_final_after_tax_total_value")
    for split in DIAGNOSTIC_SPLITS:
        split_episode = episode_df[episode_df["split"].eq(split)].copy()
        split_step = step_df[step_df["split"].eq(split)].copy()
        final_mean = split_episode["episode_final_after_tax_total_value"].mean()
        excess_hold = np.nan
        excess_sell = np.nan
        if split in baseline_vt.index:
            excess_hold = final_mean - float(baseline_vt.loc[split, "hold_to_terminal"])
            excess_sell = final_mean - float(baseline_vt.loc[split, "sell_immediately"])
        rows.append(
            step7_behavior_metric_row(
                split_episode,
                split_step,
                split=split,
                diagnostic_scope=diagnostic_scope_for_split(split),
                extra_fields={
                    "mean_excess_value_vs_hold_to_terminal": excess_hold,
                    "mean_excess_value_vs_sell_immediately": excess_sell,
                },
            )
        )
    return pd.DataFrame(rows)


def economic_period_from_year(year: Any) -> str:
    if pd.isna(year):
        return "unknown_or_outside_defined_period"
    year_int = int(year)
    if 2010 <= year_int <= 2012:
        return "post_crisis_early_recovery"
    if 2013 <= year_int <= 2016:
        return "qe_bull_market"
    if 2017 <= year_int <= 2019:
        return "late_cycle_volatility_return"
    if 2020 <= year_int <= 2021:
        return "covid_stimulus"
    if 2022 <= year_int <= 2024:
        return "inflation_tightening"
    return "unknown_or_outside_defined_period"


def add_economic_period_columns(
    episode_df: pd.DataFrame,
    step_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_out = step_df.copy()
    step_out["date"] = pd.to_datetime(step_out["date"], errors="coerce")
    first_dates = (
        step_out.sort_values(["split", "episode_id", "step_in_episode"])
        .groupby(["split", "episode_id"], as_index=False)
        .agg(episode_start_date=("date", "first"))
    )
    first_dates["calendar_year"] = first_dates["episode_start_date"].dt.year
    first_dates["economic_period"] = first_dates["calendar_year"].map(
        economic_period_from_year
    )
    episode_out = episode_df.merge(
        first_dates[
            ["split", "episode_id", "episode_start_date", "calendar_year", "economic_period"]
        ],
        on=["split", "episode_id"],
        how="left",
        validate="many_to_one",
    )
    episode_out["economic_period"] = episode_out["economic_period"].fillna(
        "unknown_or_outside_defined_period"
    )
    step_out = step_out.merge(
        first_dates[["split", "episode_id", "calendar_year", "economic_period"]],
        on=["split", "episode_id"],
        how="left",
        validate="many_to_one",
    )
    step_out["economic_period"] = step_out["economic_period"].fillna(
        "unknown_or_outside_defined_period"
    )
    return episode_out, step_out


def paired_benchmark_diagnostics(
    preferred_episode: pd.DataFrame,
    baseline_episode: pd.DataFrame,
    *,
    split: str,
) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    preferred_values = preferred_episode[
        ["episode_id", "episode_final_after_tax_total_value"]
    ].rename(columns={"episode_final_after_tax_total_value": "preferred_value"})
    for benchmark in STEP7_PAIR_BENCHMARKS:
        suffix = benchmark
        diagnostics[f"mean_excess_value_vs_{suffix}"] = np.nan
        diagnostics[f"win_rate_vs_{suffix}"] = np.nan
        benchmark_values = baseline_episode[
            baseline_episode["split"].eq(split)
            & baseline_episode["policy_name"].eq(benchmark)
            & baseline_episode["episode_id"].isin(preferred_values["episode_id"])
        ][["episode_id", "episode_final_after_tax_total_value"]].rename(
            columns={"episode_final_after_tax_total_value": "benchmark_value"}
        )
        if benchmark_values.empty:
            continue
        paired = preferred_values.merge(
            benchmark_values,
            on="episode_id",
            how="inner",
            validate="one_to_one",
        )
        if paired.empty:
            continue
        diagnostics[f"mean_excess_value_vs_{suffix}"] = (
            float(paired["preferred_value"].mean())
            - float(paired["benchmark_value"].mean())
        )
        diagnostics[f"win_rate_vs_{suffix}"] = float(
            paired["preferred_value"].gt(paired["benchmark_value"]).mean()
        )
    return diagnostics


def build_step7_economic_period_behavior(
    diag_episode: pd.DataFrame,
    diag_step: pd.DataFrame,
    baseline_episode: pd.DataFrame,
) -> pd.DataFrame:
    econ_episode, econ_step = add_economic_period_columns(diag_episode, diag_step)
    if econ_episode["economic_period"].isna().all() or econ_episode[
        "economic_period"
    ].eq("unknown_or_outside_defined_period").all():
        raise ValueError("Step 7 economic_period is missing or unknown for every row.")

    rows: list[dict[str, Any]] = []
    for split in DIAGNOSTIC_SPLITS:
        split_episode = econ_episode[econ_episode["split"].eq(split)].copy()
        split_step = econ_step[econ_step["split"].eq(split)].copy()
        for period in STEP7_ECONOMIC_PERIODS:
            period_episode = split_episode[split_episode["economic_period"].eq(period)].copy()
            if period_episode.empty:
                continue
            period_step = split_step[split_step["economic_period"].eq(period)].copy()
            paired = {
                f"{metric}_vs_{benchmark}": np.nan
                for benchmark in STEP7_PAIR_BENCHMARKS
                for metric in ["mean_excess_value", "win_rate"]
            }
            if split in VALIDATION_TEST_SPLITS:
                paired = paired_benchmark_diagnostics(
                    period_episode,
                    baseline_episode,
                    split=split,
                )
            rows.append(
                step7_behavior_metric_row(
                    period_episode,
                    period_step,
                    split=split,
                    diagnostic_scope=diagnostic_scope_for_split(split),
                    extra_fields={
                        "economic_period": period,
                        "calendar_year_min": int(period_episode["calendar_year"].min())
                        if period_episode["calendar_year"].notna().any()
                        else np.nan,
                        "calendar_year_max": int(period_episode["calendar_year"].max())
                        if period_episode["calendar_year"].notna().any()
                        else np.nan,
                        **paired,
                    },
                )
            )
    out = pd.DataFrame(rows)
    ordered_columns = [
        "split",
        "diagnostic_scope",
        "economic_period",
        "calendar_year_min",
        "calendar_year_max",
        "num_episodes",
        "action_distribution",
        "first_action_distribution",
        "first_discretionary_sale_action_distribution",
        "first_sale_timing_distribution",
        "no_cut_episode_count",
        "no_cut_episode_pct",
        "discretionary_sale_episode_count",
        "discretionary_sale_episode_pct",
        "partial_discretionary_liquidation_pct",
        "full_early_liquidation_pct",
        "terminal_liquidation_frequency",
        "mean_num_discretionary_sales",
        "average_days_to_first_sale",
        "median_days_to_first_sale",
        "pct_episodes_with_first_sale_before_tax_transition",
        "mean_pct_position_sold_short_term",
        "mean_pct_position_sold_long_term",
        "mean_total_tax_paid",
        "mean_effective_tax_rate",
        "mean_final_after_tax_total_value",
        "mean_excess_value_vs_hold_to_terminal",
        "mean_excess_value_vs_sell_immediately",
        "mean_excess_value_vs_sell_half_then_hold",
        "win_rate_vs_hold_to_terminal",
        "win_rate_vs_sell_immediately",
        "win_rate_vs_sell_half_then_hold",
    ]
    return out[ordered_columns]


def add_step7_corrected_sharpe_columns(
    out: pd.DataFrame,
    sharpe_df: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        sharpe_df,
        ["split", "policy_name", *STEP7_EAAT_SHARPE_COLUMNS],
        source=DEFAULT_OUTPUT_DIR / "step3b_policy_eaat_sharpe_metrics.csv",
        step="Step 7 corrected EAAT/TA-EAAT merge",
    )
    preferred_sharpe = sharpe_df[
        sharpe_df["policy_name"].eq(PREFERRED_POLICY)
        & sharpe_df["split"].isin(VALIDATION_TEST_SPLITS)
    ][["split", *STEP7_EAAT_SHARPE_COLUMNS]]
    missing = sorted(set(VALIDATION_TEST_SPLITS) - set(preferred_sharpe["split"]))
    if missing:
        raise ValueError(
            "Corrected Step 3B EAAT/TA-EAAT metrics are missing preferred-policy "
            "Step 7 split(s): " + ", ".join(missing)
        )
    return out.merge(preferred_sharpe, on="split", how="left", validate="one_to_one")


def validate_no_legacy_step7_metrics(columns: list[str]) -> None:
    bad_columns = [
        column for column in columns if column in STEP7_FORBIDDEN_LEGACY_SHARPE_COLUMNS
    ]
    if bad_columns:
        raise ValueError(
            "Step 7 output attempted to include forbidden legacy Sharpe column(s): "
            + ", ".join(sorted(set(bad_columns)))
        )


def validate_step7_output(
    out: pd.DataFrame,
    *,
    required_columns: list[str],
    require_sharpe_columns: bool = False,
    require_economic_period: bool = False,
) -> None:
    missing = [column for column in required_columns if column not in out.columns]
    if require_sharpe_columns:
        missing.extend(
            column for column in STEP7_EAAT_SHARPE_COLUMNS if column not in out.columns
        )
    if missing:
        raise ValueError("Step 7 output is missing required column(s): " + ", ".join(missing))
    validate_no_legacy_step7_metrics(out.columns.tolist())
    if require_economic_period:
        if "economic_period" not in out.columns:
            raise ValueError("Step 7 economic-period output is missing economic_period.")
        if out["economic_period"].isna().all() or out["economic_period"].eq(
            "unknown_or_outside_defined_period"
        ).all():
            raise ValueError("Step 7 economic_period is missing or unknown for every row.")
    numeric = out.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Step 7 output contains infinite values after safe division.")


def validate_step7_required_data(
    diag_episode: pd.DataFrame,
    diag_step: pd.DataFrame,
) -> None:
    if diag_episode.empty or diag_step.empty:
        raise ValueError("Step 7 preferred-policy diagnostic data is empty.")
    missing_splits = missing_split_policy_rows(
        diag_episode,
        policies=[PREFERRED_POLICY],
        splits=["train", "validation", "test"],
    )
    if missing_splits:
        raise ValueError(
            "Step 7 preferred policy is missing from diagnostic episode data: "
            + ", ".join(missing_splits)
        )
    required_episode_columns = [
        "split",
        "policy_name",
        "episode_id",
        "episode_cut_occurred",
        "num_discretionary_sales",
        "terminal_liquidation_fraction",
        "episode_terminal_liquidation_executed",
        "days_to_first_sale",
        "first_sale_before_tax_transition",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "total_tax_paid",
        "mean_effective_tax_rate_on_sales",
        "episode_final_after_tax_total_value",
    ]
    required_step_columns = [
        "split",
        "policy_name",
        "episode_id",
        "step_in_episode",
        "date",
        "action_fraction_requested",
        "action_fraction_executed",
        "terminal_liquidation_executed",
    ]
    require_columns(
        diag_episode,
        required_episode_columns,
        source=Path("Step 7 preferred-policy diagnostic episode frame"),
        step="Step 7",
    )
    require_columns(
        diag_step,
        required_step_columns,
        source=Path("Step 7 preferred-policy diagnostic step frame"),
        step="Step 7",
    )


def normalized_progress_frame(step_df: pd.DataFrame, split: str) -> pd.DataFrame:
    split_step = step_df[step_df["split"].eq(split)].copy()
    max_step = split_step.groupby("episode_id")["step_in_episode"].transform("max")
    split_step["normalized_progress"] = np.where(
        max_step.gt(0),
        split_step["step_in_episode"] / max_step,
        0.0,
    )
    split_step["progress_bin"] = pd.cut(
        split_step["normalized_progress"],
        bins=np.linspace(0, 1, 21),
        include_lowest=True,
        labels=np.linspace(0.025, 0.975, 20),
    ).astype(float)
    return (
        split_step.groupby("progress_bin", as_index=False)
        .agg(
            mean_remaining_fraction=("remaining_fraction", "mean"),
            mean_sold_fraction=("sold_fraction", "mean"),
        )
        .sort_values("progress_bin")
    )


def build_step7(
    config_path: Path,
    config: dict[str, Any],
    run_path: Path,
    output_dir: Path,
    plots_dir: Path,
    baseline_episode: pd.DataFrame,
    baseline_step: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    registry: OutputRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    train_episode, train_step, train_generated = ensure_train_preferred_diagnostics(
        config_path,
        config,
        run_path,
        output_dir,
    )
    registry.add(
        output_dir / "intermediate" / "step7_preferred_policy_train_episode_metrics.csv",
        "csv",
        "7",
        "Cached train split preferred-policy diagnostic episode metrics.",
    )
    registry.add(
        output_dir / "intermediate" / "step7_preferred_policy_train_step_rollouts.csv",
        "csv",
        "7",
        "Cached train split preferred-policy diagnostic step rollouts.",
    )
    diag_episode, diag_step = diagnostic_episode_and_step_frames(
        baseline_episode,
        baseline_step,
        train_episode,
        train_step,
    )
    validate_step7_required_data(diag_episode, diag_step)
    sharpe_df = load_step3b_eaat_sharpe_metrics(
        config=config,
        run_path=run_path,
        output_dir=output_dir,
    )
    behavior_by_split = behavior_summary_rows(diag_episode, diag_step, baseline_summary)
    behavior_by_split = add_step7_corrected_sharpe_columns(
        behavior_by_split,
        sharpe_df,
    )
    validate_step7_output(
        behavior_by_split,
        required_columns=STEP7_REQUIRED_BEHAVIOR_COLUMNS,
        require_sharpe_columns=True,
    )
    save_table(
        behavior_by_split,
        output_dir / "step7_preferred_policy_behavior_by_split.csv",
        output_dir / "step7_preferred_policy_behavior_by_split.md",
        registry,
        "7",
        "Preferred policy behavior diagnostics by train, validation, test, and all episodes.",
    )
    behavior_summary = behavior_by_split.copy()
    validate_step7_output(
        behavior_summary,
        required_columns=STEP7_REQUIRED_BEHAVIOR_COLUMNS,
        require_sharpe_columns=True,
    )
    save_table(
        behavior_summary,
        output_dir / "step7_preferred_policy_behavior_summary.csv",
        output_dir / "step7_preferred_policy_behavior_summary.md",
        registry,
        "7",
        "Preferred policy behavior summary including descriptive diagnostic scope labels.",
    )

    behavior_by_economic_period = build_step7_economic_period_behavior(
        diag_episode,
        diag_step,
        baseline_episode,
    )
    validate_step7_output(
        behavior_by_economic_period,
        required_columns=[
            *STEP7_REQUIRED_BEHAVIOR_COLUMNS,
            "economic_period",
            "calendar_year_min",
            "calendar_year_max",
            "mean_excess_value_vs_hold_to_terminal",
            "mean_excess_value_vs_sell_immediately",
            "mean_excess_value_vs_sell_half_then_hold",
            "win_rate_vs_hold_to_terminal",
            "win_rate_vs_sell_immediately",
            "win_rate_vs_sell_half_then_hold",
        ],
        require_economic_period=True,
    )
    save_table(
        behavior_by_economic_period,
        output_dir / "step7_preferred_policy_behavior_by_economic_period.csv",
        output_dir / "step7_preferred_policy_behavior_by_economic_period.md",
        registry,
        "7",
        "Preferred policy behavior diagnostics by split and economic period.",
    )
    notes = "\n".join(
        [
            "Step 7 behavior diagnostics notes",
            "Step 7 tests whether the preferred DQN is behaviorally different from random and from mechanical hold-to-terminal.",
            f"The preferred policy is {PREFERRED_POLICY}.",
            "Train and all-episode diagnostics are descriptive and include in-sample observations.",
            "Validation and test are the out-of-sample behavior checks.",
            "Final after-tax value remains the primary thesis metric.",
            "EAAT / TA-EAAT Sharpe metrics are secondary risk-adjusted diagnostics.",
            "Economic-period splits are heterogeneity diagnostics only and must not be used to reselect thresholds.",
            "The 0.070 policy remains preferred because it balances active behavior with reduced premature selling; 0.080 and 0.090 are more hold-like sensitivity policies.",
            "",
        ]
    )
    save_text(
        notes,
        output_dir / "step7_behavior_diagnostics_notes.txt",
        registry,
        "7",
        "Preferred policy Step 7 behavior diagnostic interpretation notes.",
    )

    test_step = diag_step[diag_step["split"].eq("test")]
    action_dist = pd.Series(action_distribution(test_step)).reindex(ACTION_ORDER)
    plt.figure(figsize=(7, 4.5))
    plt.bar(action_dist.index, action_dist.values, color="#3b6ea8")
    plt.ylabel("step share")
    plt.title("Preferred policy action distribution - test")
    plt.grid(axis="y", alpha=0.25)
    save_plot(
        plots_dir / "step7_action_distribution_test.png",
        registry,
        "7",
        "Preferred policy test action distribution.",
    )

    test_episode = diag_episode[diag_episode["split"].eq("test")]
    plt.figure(figsize=(7, 4.5))
    test_episode["days_to_first_sale"].dropna().hist(bins=30, color="#6a8f3a")
    plt.xlabel("days to first sale")
    plt.ylabel("episode count")
    plt.title("First sale timing - test")
    save_plot(
        plots_dir / "step7_first_sale_timing_histogram_test.png",
        registry,
        "7",
        "Preferred policy test first-sale timing histogram.",
    )

    progress = normalized_progress_frame(diag_step, "test")
    plt.figure(figsize=(7, 4.5))
    plt.plot(progress["progress_bin"], progress["mean_remaining_fraction"], marker="o")
    plt.xlabel("normalized episode progress")
    plt.ylabel("mean remaining fraction")
    plt.title("Remaining fraction over normalized time - test")
    plt.grid(alpha=0.25)
    save_plot(
        plots_dir / "step7_remaining_fraction_over_time_test.png",
        registry,
        "7",
        "Preferred policy test remaining fraction over normalized time.",
    )
    plt.figure(figsize=(7, 4.5))
    plt.plot(progress["progress_bin"], progress["mean_sold_fraction"], marker="o")
    plt.xlabel("normalized episode progress")
    plt.ylabel("mean cumulative sold fraction")
    plt.title("Cumulative sold fraction over normalized time - test")
    plt.grid(alpha=0.25)
    save_plot(
        plots_dir / "step7_cumulative_sold_fraction_over_time_test.png",
        registry,
        "7",
        "Preferred policy test cumulative sold fraction over normalized time.",
    )

    action_split_rows = []
    for split in DIAGNOSTIC_SPLITS:
        dist = action_distribution(diag_step[diag_step["split"].eq(split)])
        for action in ACTION_ORDER:
            action_split_rows.append({"split": split, "action": action, "share": dist[action]})
    action_split_df = pd.DataFrame(action_split_rows)
    pivot = action_split_df.pivot(index="split", columns="action", values="share").reindex(
        DIAGNOSTIC_SPLITS
    )
    pivot.plot(kind="bar", stacked=True, figsize=(8, 4.8))
    plt.ylabel("step share")
    plt.title("Preferred policy action distribution by split")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    save_plot(
        plots_dir / "step7_action_distribution_by_split.png",
        registry,
        "7",
        "Preferred policy action distribution by split.",
    )
    for column, filename, ylabel in [
        ("no_cut_episode_pct", "step7_no_cut_pct_by_split.png", "no-cut episode pct"),
        (
            "mean_pct_position_sold_short_term",
            "step7_short_term_sold_fraction_by_split.png",
            "mean short-term sold fraction",
        ),
        (
            "average_days_to_first_sale",
            "step7_days_to_first_sale_by_split.png",
            "average days to first sale",
        ),
    ]:
        plt.figure(figsize=(7, 4.5))
        plt.bar(behavior_by_split["split"], behavior_by_split[column], color="#3b6ea8")
        plt.ylabel(ylabel)
        plt.title(ylabel + " by split")
        plt.grid(axis="y", alpha=0.25)
        save_plot(plots_dir / filename, registry, "7", f"Preferred policy {ylabel} by split.")

    for plot_split in ["test", "all"]:
        split_economic = behavior_by_economic_period[
            behavior_by_economic_period["split"].eq(plot_split)
        ].copy()
        if split_economic.empty:
            raise ValueError(
                f"Step 7 economic-period {plot_split} table is empty."
            )
        for column, filename_stem, ylabel in [
            (
                "no_cut_episode_pct",
                "step7_no_cut_pct_by_economic_period",
                "no-cut episode pct",
            ),
            (
                "discretionary_sale_episode_pct",
                "step7_discretionary_sale_pct_by_economic_period",
                "discretionary sale episode pct",
            ),
            (
                "mean_pct_position_sold_short_term",
                "step7_short_term_sold_fraction_by_economic_period",
                "mean short-term sold fraction",
            ),
            (
                "average_days_to_first_sale",
                "step7_average_days_to_first_sale_by_economic_period",
                "average days to first sale",
            ),
        ]:
            plt.figure(figsize=(9, 4.8))
            plt.bar(
                split_economic["economic_period"],
                split_economic[column],
                color="#3b6ea8",
            )
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} by economic period - {plot_split}")
            plt.xticks(rotation=35, ha="right")
            plt.grid(axis="y", alpha=0.25)
            save_plot(
                plots_dir / f"{filename_stem}_{plot_split}.png",
                registry,
                "7",
                f"Preferred policy {plot_split} {ylabel} by economic period.",
            )

    return diag_episode, diag_step, train_generated


def validate_no_legacy_step8_metrics(columns: list[str]) -> None:
    bad_columns = [
        column for column in columns if column in STEP8_FORBIDDEN_LEGACY_SHARPE_COLUMNS
    ]
    if bad_columns:
        raise ValueError(
            "Step 8 output attempted to include forbidden legacy Sharpe column(s): "
            + ", ".join(sorted(set(bad_columns)))
        )


def validate_step8_table(df: pd.DataFrame, *, name: str) -> None:
    if df.empty:
        raise ValueError(f"Step 8 generated empty table: {name}")
    validate_no_legacy_step8_metrics(df.columns.tolist())
    numeric = df.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError(f"Step 8 table contains infinite values: {name}")


def validate_step8_inputs(episode_df: pd.DataFrame, step_df: pd.DataFrame) -> None:
    require_columns(
        episode_df,
        STEP8_REQUIRED_EPISODE_COLUMNS,
        source=Path("baseline episode metrics"),
        step="Step 8",
    )
    require_columns(
        step_df,
        [
            "split",
            "policy_name",
            "episode_id",
            "step_in_episode",
            "date",
            "tax_transition_date",
            "after_tax_total_value",
        ],
        source=Path("baseline step rollouts"),
        step="Step 8",
    )
    missing = missing_split_policy_rows(
        episode_df,
        policies=[PREFERRED_POLICY, "hold_to_terminal"],
        splits=VALIDATION_TEST_SPLITS,
    )
    if missing:
        raise ValueError(
            "Step 8 requires preferred policy and hold_to_terminal rows: "
            + ", ".join(missing)
        )


def paired_preferred_benchmark_frame(
    episode_df: pd.DataFrame,
    *,
    split: str,
    benchmark: str,
) -> pd.DataFrame:
    preferred = episode_df[
        episode_df["split"].eq(split) & episode_df["policy_name"].eq(PREFERRED_POLICY)
    ][
        [
            "episode_id",
            "episode_final_after_tax_total_value",
            "total_tax_paid",
            "pct_episode_position_sold_short_term",
        ]
    ].rename(
        columns={
            "episode_final_after_tax_total_value": "preferred_value",
            "total_tax_paid": "preferred_tax_paid",
            "pct_episode_position_sold_short_term": "preferred_short_term",
        }
    )
    bench = episode_df[
        episode_df["split"].eq(split) & episode_df["policy_name"].eq(benchmark)
    ][
        [
            "episode_id",
            "episode_final_after_tax_total_value",
            "total_tax_paid",
            "pct_episode_position_sold_short_term",
        ]
    ].rename(
        columns={
            "episode_final_after_tax_total_value": "benchmark_value",
            "total_tax_paid": "benchmark_tax_paid",
            "pct_episode_position_sold_short_term": "benchmark_short_term",
        }
    )
    merged = preferred.merge(bench, on="episode_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(
            f"Step 8 found no paired episodes for split={split}, benchmark={benchmark}."
        )
    return merged


def win_loss_result(mean_difference: float) -> str:
    if mean_difference > 0:
        return "Win"
    if mean_difference < 0:
        return "Loss"
    return "Tie"


def build_step8_win_loss_table(episode_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in VALIDATION_TEST_SPLITS:
        for benchmark in STEP8_BENCHMARK_POLICIES:
            merged = paired_preferred_benchmark_frame(
                episode_df,
                split=split,
                benchmark=benchmark,
            )
            diff = merged["preferred_value"] - merged["benchmark_value"]
            mean_difference = float(diff.mean())
            rows.append(
                {
                    "split": split,
                    "benchmark_policy": benchmark,
                    "num_paired_episodes": int(len(merged)),
                    "preferred_mean_final_after_tax_value": float(
                        merged["preferred_value"].mean()
                    ),
                    "benchmark_mean_final_after_tax_value": float(
                        merged["benchmark_value"].mean()
                    ),
                    "mean_difference": mean_difference,
                    "median_difference": float(diff.median()),
                    "win_rate": float(diff.gt(0).mean()),
                    "tie_rate": float(diff.eq(0).mean()),
                    "loss_rate": float(diff.lt(0).mean()),
                    "result": win_loss_result(mean_difference),
                }
            )
    out = pd.DataFrame(rows)
    validate_step8_table(out, name="step8_win_loss_vs_benchmark_table")
    return out


def build_step8_dominance_table(episode_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in VALIDATION_TEST_SPLITS:
        for benchmark in STEP8_BENCHMARK_POLICIES:
            merged = paired_preferred_benchmark_frame(
                episode_df,
                split=split,
                benchmark=benchmark,
            )
            diff = merged["preferred_value"] - merged["benchmark_value"]
            rows.append(
                {
                    "split": split,
                    "benchmark_policy": benchmark,
                    "num_paired_episodes": int(len(merged)),
                    "mean_difference": float(diff.mean()),
                    "median_difference": float(diff.median()),
                    "win_rate": float(diff.gt(0).mean()),
                    "tie_rate": float(diff.eq(0).mean()),
                    "loss_rate": float(diff.lt(0).mean()),
                    "preferred_mean_final_after_tax_value": float(
                        merged["preferred_value"].mean()
                    ),
                    "benchmark_mean_final_after_tax_value": float(
                        merged["benchmark_value"].mean()
                    ),
                    "preferred_minus_benchmark_total_mean_tax_paid": float(
                        merged["preferred_tax_paid"].mean()
                        - merged["benchmark_tax_paid"].mean()
                    ),
                    "preferred_minus_benchmark_short_term_sold_fraction": float(
                        merged["preferred_short_term"].mean()
                        - merged["benchmark_short_term"].mean()
                    ),
                }
            )
    out = pd.DataFrame(rows)
    validate_step8_table(out, name="step8_benchmark_dominance_analysis")
    return out


def step8_hold_path_stats(step_df: pd.DataFrame) -> pd.DataFrame:
    hold_steps = step_df[
        step_df["split"].isin(VALIDATION_TEST_SPLITS)
        & step_df["policy_name"].eq("hold_to_terminal")
    ].copy()
    hold_steps["date"] = pd.to_datetime(hold_steps["date"], errors="coerce")
    hold_steps["tax_transition_date"] = pd.to_datetime(
        hold_steps["tax_transition_date"],
        errors="coerce",
    )
    hold_steps = hold_steps.sort_values(["split", "episode_id", "step_in_episode"])
    hold_steps["path_return_proxy"] = hold_steps.groupby(
        ["split", "episode_id"]
    )["after_tax_total_value"].pct_change()
    hold_steps["path_return_proxy"] = hold_steps["path_return_proxy"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    stats = (
        hold_steps.groupby(["split", "episode_id"], as_index=False)
        .agg(
            hold_path_first_value=("after_tax_total_value", "first"),
            hold_path_min_value=("after_tax_total_value", "min"),
            episode_start_date=("date", "first"),
            tax_transition_date=("tax_transition_date", "first"),
            volatility_proxy=("path_return_proxy", "std"),
        )
    )
    stats["drawdown_proxy"] = (
        stats["hold_path_first_value"] - stats["hold_path_min_value"]
    )
    stats["calendar_year"] = stats["episode_start_date"].dt.year
    stats["economic_period"] = stats["calendar_year"].map(economic_period_from_year)
    stats["days_until_tax_transition_at_start"] = (
        stats["tax_transition_date"] - stats["episode_start_date"]
    ).dt.days
    return stats


def step8_preferred_vs_hold_base(
    episode_df: pd.DataFrame,
    step_df: pd.DataFrame,
) -> pd.DataFrame:
    paired_rows: list[pd.DataFrame] = []
    path_stats = step8_hold_path_stats(step_df)
    for split in VALIDATION_TEST_SPLITS:
        preferred = episode_df[
            episode_df["split"].eq(split) & episode_df["policy_name"].eq(PREFERRED_POLICY)
        ][
            [
                "split",
                "episode_id",
                "episode_final_after_tax_total_value",
                "total_tax_paid",
                "pct_episode_position_sold_short_term",
                "pct_episode_position_sold_long_term",
                "episode_cut_occurred",
                "num_discretionary_sales",
                "days_to_first_sale",
            ]
        ].rename(
            columns={
                "episode_final_after_tax_total_value": "preferred_final_after_tax_value",
                "total_tax_paid": "preferred_tax_paid",
                "pct_episode_position_sold_short_term": "preferred_short_term_sold_fraction",
                "pct_episode_position_sold_long_term": "preferred_long_term_sold_fraction",
                "episode_cut_occurred": "preferred_cut_occurred",
                "num_discretionary_sales": "preferred_num_discretionary_sales",
                "days_to_first_sale": "preferred_days_to_first_sale",
            }
        )
        hold = episode_df[
            episode_df["split"].eq(split) & episode_df["policy_name"].eq("hold_to_terminal")
        ][
            [
                "split",
                "episode_id",
                "episode_final_after_tax_total_value",
                "total_tax_paid",
                "pct_episode_position_sold_short_term",
                "pct_episode_position_sold_long_term",
            ]
        ].rename(
            columns={
                "episode_final_after_tax_total_value": "hold_final_after_tax_value",
                "total_tax_paid": "hold_tax_paid",
                "pct_episode_position_sold_short_term": "hold_short_term_sold_fraction",
                "pct_episode_position_sold_long_term": "hold_long_term_sold_fraction",
            }
        )
        paired = preferred.merge(
            hold,
            on=["split", "episode_id"],
            how="inner",
            validate="one_to_one",
        )
        if paired.empty:
            raise ValueError(f"Step 8 found no preferred/hold paired episodes for {split}.")
        paired_rows.append(paired)
    base = pd.concat(paired_rows, ignore_index=True)
    base = base.merge(
        path_stats,
        on=["split", "episode_id"],
        how="left",
        validate="one_to_one",
    )
    base["preferred_minus_hold"] = (
        base["preferred_final_after_tax_value"] - base["hold_final_after_tax_value"]
    )
    base["preferred_beats_hold"] = base["preferred_minus_hold"].gt(0)
    base["preferred_ties_hold"] = base["preferred_minus_hold"].eq(0)
    base["preferred_loses_to_hold"] = base["preferred_minus_hold"].lt(0)
    base["preferred_no_cut"] = ~base["preferred_cut_occurred"]
    base["preferred_discretionary_sale"] = (
        base["preferred_num_discretionary_sales"].fillna(0).gt(0)
    )
    return base


def add_step8_situational_buckets(
    base: pd.DataFrame,
    notes: list[str],
    episode_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    groupings: list[tuple[str, str]] = []
    base["hold_terminal_outcome_bucket"] = pd.cut(
        base["hold_final_after_tax_value"],
        bins=[-np.inf, 0.30, 0.45, 0.60, np.inf],
        labels=[
            "hold_terminal_final_value <= 0.30",
            "0.30 < hold_terminal_final_value <= 0.45",
            "0.45 < hold_terminal_final_value <= 0.60",
            "hold_terminal_final_value > 0.60",
        ],
    ).astype(object)
    groupings.append(("hold_terminal_outcome_bucket", "hold_terminal_outcome_bucket"))
    notes.append(
        "hold_terminal_outcome_bucket uses fixed hold_to_terminal final after-tax value buckets: <=0.30, 0.30-0.45, 0.45-0.60, >0.60."
    )

    base["hold_terminal_weak"] = np.where(
        base["hold_final_after_tax_value"].le(0.30),
        "hold_terminal_weak",
        "hold_terminal_not_weak",
    )
    groupings.append(("hold_terminal_weak", "hold_terminal_weak"))
    notes.append("hold_terminal_weak is True when hold_to_terminal final value <= 0.30.")

    if base["drawdown_proxy"].notna().any():
        base["drawdown_bucket"] = quantile_bucket(
            base["drawdown_proxy"],
            ["low_drawdown", "medium_drawdown", "high_drawdown"],
        ).astype(object)
        groupings.append(("drawdown_bucket", "drawdown_bucket"))
        notes.append(
            "drawdown_bucket uses tertiles of hold_to_terminal path drawdown proxy: first after-tax path value minus minimum after-tax path value."
        )
    else:
        notes.append("drawdown_bucket skipped because hold_to_terminal path values were unavailable.")

    if base["volatility_proxy"].notna().any():
        base["volatility_bucket"] = quantile_bucket(
            base["volatility_proxy"],
            ["low_volatility", "medium_volatility", "high_volatility"],
        ).astype(object)
        groupings.append(("volatility_bucket", "volatility_bucket"))
        notes.append(
            "volatility_bucket uses tertiles of hold_to_terminal daily after-tax path percent-change volatility."
        )
    else:
        notes.append("volatility_bucket skipped because hold_to_terminal path volatility was unavailable.")

    if base["days_until_tax_transition_at_start"].notna().any():
        base["tax_transition_distance_bucket"] = quantile_bucket(
            base["days_until_tax_transition_at_start"],
            ["near_transition", "medium_transition_distance", "far_from_transition"],
        ).astype(object)
        groupings.append(
            ("tax_transition_distance_bucket", "tax_transition_distance_bucket")
        )
        notes.append(
            "tax_transition_distance_bucket uses tertiles of days from the first rollout date to tax_transition_date; lower values are near_transition."
        )
    else:
        notes.append(
            "tax_transition_distance_bucket skipped because tax_transition_date was unavailable."
        )

    if base["economic_period"].notna().any():
        base["economic_period"] = base["economic_period"].fillna(
            "unknown_or_outside_defined_period"
        )
        groupings.append(("economic_period", "economic_period"))
        notes.append("economic_period reuses the Step 7 calendar-year mapping.")
    else:
        notes.append("economic_period skipped because episode start dates were unavailable.")

    base["episode_start_year_bucket"] = base["calendar_year"].fillna("unknown").astype(str)
    groupings.append(("episode_start_year_bucket", "episode_start_year_bucket"))
    notes.append("episode_start_year_bucket uses the first hold_to_terminal rollout year.")

    for optional in ["sector", "industry"]:
        if optional in episode_df.columns:
            optional_values = episode_df[
                episode_df["split"].isin(VALIDATION_TEST_SPLITS)
                & episode_df["policy_name"].eq(PREFERRED_POLICY)
            ][["split", "episode_id", optional]]
            base = base.merge(
                optional_values,
                on=["split", "episode_id"],
                how="left",
                validate="one_to_one",
            )
            if optional in base.columns and base[optional].notna().any():
                groupings.append((optional, optional))
                notes.append(f"{optional} grouping included from episode-level metadata.")
            else:
                notes.append(f"{optional} grouping skipped because metadata was unavailable.")
        else:
            notes.append(f"{optional} grouping skipped because metadata was unavailable.")
    return base, groupings


def step8_situational_row(
    group: pd.DataFrame,
    *,
    split: str,
    group_name: str,
    group_bucket: str,
) -> dict[str, Any]:
    diff = group["preferred_minus_hold"]
    return {
        "split": split,
        "group_name": group_name,
        "group_bucket": group_bucket,
        "num_episodes": int(len(group)),
        "preferred_mean_final_after_tax_value": float(
            group["preferred_final_after_tax_value"].mean()
        ),
        "hold_mean_final_after_tax_value": float(
            group["hold_final_after_tax_value"].mean()
        ),
        "mean_difference_vs_hold": float(diff.mean()),
        "median_difference_vs_hold": float(diff.median()),
        "win_rate_vs_hold": float(diff.gt(0).mean()),
        "tie_rate_vs_hold": float(diff.eq(0).mean()),
        "loss_rate_vs_hold": float(diff.lt(0).mean()),
        "preferred_mean_tax_paid": float(group["preferred_tax_paid"].mean()),
        "hold_mean_tax_paid": float(group["hold_tax_paid"].mean()),
        "preferred_minus_hold_mean_tax_paid": float(
            group["preferred_tax_paid"].mean() - group["hold_tax_paid"].mean()
        ),
        "preferred_mean_short_term_sold_fraction": float(
            group["preferred_short_term_sold_fraction"].mean()
        ),
        "hold_mean_short_term_sold_fraction": float(
            group["hold_short_term_sold_fraction"].mean()
        ),
        "preferred_no_cut_pct": float(group["preferred_no_cut"].mean()),
        "preferred_discretionary_sale_pct": float(
            group["preferred_discretionary_sale"].mean()
        ),
        "preferred_average_days_to_first_sale": float(
            group["preferred_days_to_first_sale"].mean()
        ),
        "preferred_median_days_to_first_sale": float(
            group["preferred_days_to_first_sale"].median()
        ),
    }


def build_step8_situational_analysis(
    base: pd.DataFrame,
    episode_df: pd.DataFrame,
    notes: list[str],
) -> pd.DataFrame:
    base = base.copy()
    base, groupings = add_step8_situational_buckets(base, notes, episode_df)
    rows: list[dict[str, Any]] = []
    for split in VALIDATION_TEST_SPLITS:
        split_base = base[base["split"].eq(split)]
        for group_name, group_column in groupings:
            for bucket, group in split_base.groupby(group_column, sort=False, observed=False):
                if group.empty:
                    continue
                rows.append(
                    step8_situational_row(
                        group,
                        split=split,
                        group_name=group_name,
                        group_bucket=str(bucket),
                    )
                )
    out = pd.DataFrame(rows)
    validate_step8_table(out, name="step8_preferred_vs_hold_situational_analysis")
    return out


def build_step8_win_loss_characteristics(base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = base.copy()
    work["preferred_vs_hold_bucket"] = np.where(
        work["preferred_beats_hold"],
        "preferred_beats_hold",
        "preferred_does_not_beat_hold",
    )
    for split in VALIDATION_TEST_SPLITS:
        split_base = work[work["split"].eq(split)]
        for bucket, group in split_base.groupby("preferred_vs_hold_bucket", sort=False):
            rows.append(
                {
                    "split": split,
                    "preferred_vs_hold_bucket": str(bucket),
                    "post_hoc_scope": "descriptive_only",
                    "num_episodes": int(len(group)),
                    "mean_hold_final_after_tax_value": float(
                        group["hold_final_after_tax_value"].mean()
                    ),
                    "mean_preferred_final_after_tax_value": float(
                        group["preferred_final_after_tax_value"].mean()
                    ),
                    "mean_difference_vs_hold": float(
                        group["preferred_minus_hold"].mean()
                    ),
                    "mean_hold_tax_paid": float(group["hold_tax_paid"].mean()),
                    "mean_preferred_tax_paid": float(
                        group["preferred_tax_paid"].mean()
                    ),
                    "mean_preferred_short_term_sold_fraction": float(
                        group["preferred_short_term_sold_fraction"].mean()
                    ),
                    "mean_preferred_long_term_sold_fraction": float(
                        group["preferred_long_term_sold_fraction"].mean()
                    ),
                    "preferred_no_cut_pct": float(group["preferred_no_cut"].mean()),
                    "preferred_discretionary_sale_pct": float(
                        group["preferred_discretionary_sale"].mean()
                    ),
                    "preferred_average_days_to_first_sale": float(
                        group["preferred_days_to_first_sale"].mean()
                    ),
                    "mean_days_until_tax_transition_at_start": float(
                        group["days_until_tax_transition_at_start"].mean()
                    ),
                    "mean_drawdown_proxy": float(group["drawdown_proxy"].mean()),
                    "mean_volatility_proxy": float(group["volatility_proxy"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    validate_step8_table(out, name="step8_preferred_vs_hold_win_loss_characteristics")
    return out


def plot_step8_situational_metric(
    situational: pd.DataFrame,
    *,
    split: str,
    metric: str,
    ylabel: str,
    path: Path,
    registry: OutputRegistry,
) -> None:
    split_df = situational[situational["split"].eq(split)].copy()
    if split_df.empty:
        raise ValueError(f"Step 8 situational plot has no rows for split={split}.")
    split_df["label"] = split_df["group_name"] + ": " + split_df["group_bucket"]
    plt.figure(figsize=(12, 6.2))
    colors = np.where(split_df[metric].ge(0), "#3b6ea8", "#8a4f3d")
    plt.bar(split_df["label"], split_df[metric], color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by ex-ante group - {split}")
    plt.xticks(rotation=55, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    save_plot(
        path,
        registry,
        "8",
        f"Preferred policy versus hold_to_terminal {ylabel} by group for {split}.",
    )


def step8_safe_plot_token(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def plot_step8_group_win_tie_loss_rates(
    situational: pd.DataFrame,
    *,
    split: str,
    group_name: str,
    plots_dir: Path,
    registry: OutputRegistry,
) -> None:
    group_df = situational[
        situational["split"].eq(split) & situational["group_name"].eq(group_name)
    ].copy()
    if group_df.empty:
        raise ValueError(
            f"Step 8 win/tie/loss plot has no rows for split={split}, group={group_name}."
        )
    x = np.arange(len(group_df))
    plt.figure(figsize=(8.6, 4.8))
    plt.bar(x, group_df["win_rate_vs_hold"], label="win", color="#3b6ea8")
    plt.bar(
        x,
        group_df["tie_rate_vs_hold"],
        bottom=group_df["win_rate_vs_hold"],
        label="tie",
        color="#8a8a8a",
    )
    plt.bar(
        x,
        group_df["loss_rate_vs_hold"],
        bottom=group_df["win_rate_vs_hold"] + group_df["tie_rate_vs_hold"],
        label="loss",
        color="#8a4f3d",
    )
    plt.xticks(x, group_df["group_bucket"], rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("episode share")
    plt.title(f"Preferred vs hold win/tie/loss - {group_name} - {split}")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    token = step8_safe_plot_token(group_name)
    save_plot(
        plots_dir / f"step8_win_tie_loss_vs_hold_by_{token}_{split}.png",
        registry,
        "8",
        f"Preferred policy win/tie/loss rates versus hold_to_terminal by {group_name} for {split}.",
    )


def plot_step8_group_mean_difference(
    situational: pd.DataFrame,
    *,
    split: str,
    group_name: str,
    plots_dir: Path,
    registry: OutputRegistry,
) -> None:
    group_df = situational[
        situational["split"].eq(split) & situational["group_name"].eq(group_name)
    ].copy()
    if group_df.empty:
        raise ValueError(
            f"Step 8 mean-difference plot has no rows for split={split}, group={group_name}."
        )
    colors = np.where(group_df["mean_difference_vs_hold"].ge(0), "#3b6ea8", "#8a4f3d")
    plt.figure(figsize=(8.6, 4.8))
    plt.bar(group_df["group_bucket"], group_df["mean_difference_vs_hold"], color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("mean difference vs hold_to_terminal")
    plt.title(f"Mean difference vs hold - {group_name} - {split}")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    token = step8_safe_plot_token(group_name)
    save_plot(
        plots_dir / f"step8_mean_difference_vs_hold_by_{token}_{split}.png",
        registry,
        "8",
        f"Preferred policy mean difference versus hold_to_terminal by {group_name} for {split}.",
    )


def plot_step8_situational_group_charts(
    situational: pd.DataFrame,
    plots_dir: Path,
    registry: OutputRegistry,
) -> None:
    group_names = list(dict.fromkeys(situational["group_name"].astype(str).tolist()))
    for split in VALIDATION_TEST_SPLITS:
        for group_name in group_names:
            plot_step8_group_win_tie_loss_rates(
                situational,
                split=split,
                group_name=group_name,
                plots_dir=plots_dir,
                registry=registry,
            )
            plot_step8_group_mean_difference(
                situational,
                split=split,
                group_name=group_name,
                plots_dir=plots_dir,
                registry=registry,
            )


def build_step8(
    episode_df: pd.DataFrame,
    step_df: pd.DataFrame,
    output_dir: Path,
    plots_dir: Path,
    registry: OutputRegistry,
) -> pd.DataFrame:
    validate_step8_inputs(episode_df, step_df)
    out = build_step8_dominance_table(episode_df)
    save_table(
        out,
        output_dir / "step8_benchmark_dominance_analysis.csv",
        output_dir / "step8_benchmark_dominance_analysis.md",
        registry,
        "8",
        "Paired preferred-policy dominance analysis against benchmark policies.",
    )

    win_loss = build_step8_win_loss_table(episode_df)
    save_table(
        win_loss,
        output_dir / "step8_win_loss_vs_benchmark_table.csv",
        output_dir / "step8_win_loss_vs_benchmark_table.md",
        registry,
        "8",
        "Thesis-facing preferred-policy win/loss table versus benchmark policies.",
    )

    notes: list[str] = []
    preferred_hold_base = step8_preferred_vs_hold_base(episode_df, step_df)
    situational = build_step8_situational_analysis(
        preferred_hold_base,
        episode_df,
        notes,
    )
    save_table(
        situational,
        output_dir / "step8_preferred_vs_hold_situational_analysis.csv",
        output_dir / "step8_preferred_vs_hold_situational_analysis.md",
        registry,
        "8",
        "Preferred DQN versus hold_to_terminal situational diagnostics.",
    )

    win_loss_characteristics = build_step8_win_loss_characteristics(preferred_hold_base)
    save_table(
        win_loss_characteristics,
        output_dir / "step8_preferred_vs_hold_win_loss_characteristics.csv",
        output_dir / "step8_preferred_vs_hold_win_loss_characteristics.md",
        registry,
        "8",
        "Post-hoc preferred DQN versus hold_to_terminal win/loss characteristics.",
    )

    perf = aggregate_policy_metrics(episode_df, POLICY_UNIVERSE, VALIDATION_TEST_SPLITS)
    best_validation = perf[perf["split"].eq("validation")].sort_values(
        "mean_final_after_tax_total_value",
        ascending=False,
    ).iloc[0]["policy_name"]
    best_test = perf[perf["split"].eq("test")].sort_values(
        "mean_final_after_tax_total_value",
        ascending=False,
    ).iloc[0]["policy_name"]
    test_rows = win_loss[win_loss["split"].eq("test")].set_index("benchmark_policy")
    summary_lines = [
        "Step 8 benchmark dominance summary",
        "Primary thesis-facing output: step8_win_loss_vs_benchmark_table.csv/md",
        f"best_validation_policy_by_mean_final_after_tax_value: {best_validation}",
        f"best_test_policy_by_mean_final_after_tax_value: {best_test}",
        f"preferred_beats_hold_to_terminal_on_test: {bool(test_rows.loc['hold_to_terminal', 'mean_difference'] > 0)}",
        f"preferred_beats_sell_immediately_on_test: {bool(test_rows.loc['sell_immediately', 'mean_difference'] > 0)}",
        f"preferred_beats_sell_half_then_hold_on_test: {bool(test_rows.loc['sell_half_then_hold', 'mean_difference'] > 0)}",
        f"preferred_beats_sell_quarters_over_time_on_test: {bool(test_rows.loc['sell_quarters_over_time', 'mean_difference'] > 0)}",
        "hold_to_terminal is strongest overall by mean final after-tax value in validation and test.",
        "The preferred DQN beats naive active liquidation benchmarks but does not beat hold_to_terminal overall.",
        "",
    ]
    save_text(
        "\n".join(summary_lines),
        output_dir / "step8_benchmark_dominance_summary.txt",
        registry,
        "8",
        "Benchmark dominance summary text.",
    )

    situational_notes = [
        "Step 8 preferred versus hold_to_terminal situational notes",
        "The preferred policy does not beat hold_to_terminal overall.",
        "hold_to_terminal remains strongest in mean final after-tax value.",
        "The situational analysis is diagnostic.",
        "Ex-ante groups are used where possible to avoid selecting on DQN performance.",
        "Post-hoc win/loss characteristics are descriptive only.",
        "Situational plots are split by ex-ante group and show win, tie, and loss rates together because one minus win rate includes both ties and losses.",
        "The preferred DQN tends to add value mainly when full passive deferral fails to preserve the appreciated position, especially in weaker or riskier episode paths. However, because the sample is built from positions that already appreciated substantially, and because tax deferral is highly valuable, hold-to-terminal remains the strongest overall benchmark.",
        "",
        "Grouping implementation notes",
        *notes,
        "",
    ]
    save_text(
        "\n".join(situational_notes),
        output_dir / "step8_preferred_vs_hold_situational_notes.txt",
        registry,
        "8",
        "Preferred DQN versus hold_to_terminal situational-analysis notes.",
    )

    test_plot = out[out["split"].eq("test")]
    plt.figure(figsize=(8, 4.6))
    plt.bar(test_plot["benchmark_policy"], test_plot["mean_difference"], color="#8a4f3d")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("preferred minus benchmark mean final value")
    plt.title("Preferred policy vs benchmarks - test")
    plt.grid(axis="y", alpha=0.25)
    save_plot(
        plots_dir / "step8_preferred_vs_benchmarks_test.png",
        registry,
        "8",
        "Preferred policy mean paired differences against benchmarks on test split.",
    )
    plot_step8_situational_group_charts(situational, plots_dir, registry)
    return out


def quantile_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    ranked = series.rank(method="first")
    try:
        return pd.qcut(ranked, q=len(labels), labels=labels)
    except ValueError:
        return pd.Series(["unbucketed"] * len(series), index=series.index)


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.mean()) if values.notna().any() else np.nan


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.median()) if values.notna().any() else np.nan


def safe_win_rate(diff: pd.Series) -> float:
    values = pd.to_numeric(diff, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    return float(values.gt(0).mean()) if len(values) else np.nan


def load_step3b_episode_eaat_sharpe_metrics(
    output_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    csv_path = output_dir / "step3b_episode_eaat_sharpe_metrics.csv"
    notes_path = output_dir / "step3b_eaat_sharpe_metrics_notes.txt"
    if not csv_path.exists():
        raise FileNotFoundError(
            "Step 9 requires corrected Step 3B episode-level EAAT/TA-EAAT "
            f"Sharpe metrics, but missing: {relative_project_path(csv_path)}"
        )
    validate_step3b_eaat_notes(notes_path)
    df = pd.read_csv(csv_path, low_memory=False)
    required = [
        "split",
        "policy_name",
        "episode_id",
        "EAAT_Sharpe",
        "TA_EAAT_Sharpe",
        "annual_risk_free_rate",
        "daily_risk_free_rate",
        "annualization_factor",
    ]
    require_columns(
        df,
        required,
        source=csv_path,
        step="Step 9 corrected Step 3B episode Sharpe merge",
    )
    if not np.isclose(
        pd.to_numeric(df["annual_risk_free_rate"], errors="coerce").dropna().unique(),
        ANNUAL_RISK_FREE_RATE,
    ).all():
        raise ValueError(
            "Step 9 corrected Step 3B episode Sharpe rows must use "
            f"annual_risk_free_rate={ANNUAL_RISK_FREE_RATE}."
        )
    if not np.isclose(
        pd.to_numeric(df["daily_risk_free_rate"], errors="coerce").dropna().unique(),
        DAILY_RISK_FREE_RATE,
    ).all():
        raise ValueError(
            "Step 9 corrected Step 3B episode Sharpe rows must use "
            f"daily_risk_free_rate={DAILY_RISK_FREE_RATE}."
        )
    if not np.isclose(
        pd.to_numeric(df["annualization_factor"], errors="coerce").dropna().unique(),
        ANNUALIZATION_FACTOR,
    ).all():
        raise ValueError(
            "Step 9 corrected Step 3B episode Sharpe rows must use "
            f"annualization_factor={ANNUALIZATION_FACTOR}."
        )
    duplicate_count = int(df.duplicated(["split", "policy_name", "episode_id"]).sum())
    if duplicate_count:
        raise ValueError(
            "Step 9 cannot merge corrected Sharpe metrics because Step 3B has "
            f"{duplicate_count} duplicate split/policy/episode rows."
        )
    missing = missing_split_policy_rows(
        df,
        policies=STEP9_POLICIES,
        splits=VALIDATION_TEST_SPLITS,
    )
    if missing:
        raise ValueError(
            "Step 9 corrected Sharpe merge is missing required split/policy rows: "
            + ", ".join(missing)
        )
    df = df[df["split"].isin(VALIDATION_TEST_SPLITS) & df["policy_name"].isin(STEP9_POLICIES)].copy()
    for column in ["EAAT_Sharpe", "TA_EAAT_Sharpe"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
    return df[["split", "policy_name", "episode_id", "EAAT_Sharpe", "TA_EAAT_Sharpe"]], csv_path


def step9_hold_path_stats(step_df: pd.DataFrame) -> pd.DataFrame:
    hold_steps = step_df[
        step_df["split"].isin(VALIDATION_TEST_SPLITS)
        & step_df["policy_name"].eq("hold_to_terminal")
    ].copy()
    hold_steps = hold_steps.sort_values(["split", "episode_id", "step_in_episode"])
    hold_steps["date"] = pd.to_datetime(hold_steps["date"], errors="coerce")
    hold_steps["tax_transition_date"] = pd.to_datetime(
        hold_steps["tax_transition_date"],
        errors="coerce",
    )
    hold_steps["path_return_pct"] = hold_steps.groupby(["split", "episode_id"])[
        "after_tax_total_value"
    ].pct_change()
    hold_steps["path_return_pct"] = hold_steps["path_return_pct"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    stats = (
        hold_steps.groupby(["split", "episode_id"], as_index=False)
        .agg(
            hold_path_first_value=("after_tax_total_value", "first"),
            hold_path_max_value=("after_tax_total_value", "max"),
            hold_path_min_value=("after_tax_total_value", "min"),
            episode_start_date=("date", "first"),
            tax_transition_date=("tax_transition_date", "first"),
            volatility_proxy=("path_return_pct", "std"),
        )
    )
    stats["maximum_gain_proxy"] = (
        stats["hold_path_max_value"] - stats["hold_path_first_value"]
    )
    stats["drawdown_after_start_proxy"] = (
        stats["hold_path_first_value"] - stats["hold_path_min_value"]
    )
    stats["days_until_tax_transition_at_start"] = (
        stats["tax_transition_date"] - stats["episode_start_date"]
    ).dt.days
    stats["calendar_year"] = stats["episode_start_date"].dt.year.astype("Int64")
    return stats


def load_step9_episode_start_gain_features(
    config: dict[str, Any],
    hold_path_stats: pd.DataFrame,
    notes: list[str],
) -> pd.DataFrame:
    parquet_path = resolve_project_path(require_nested(config, "environment.parquet_path"))
    out = hold_path_stats[["split", "episode_id", "episode_start_date"]].copy()
    out["gain_at_episode_start"] = np.nan
    if not parquet_path.exists():
        notes.append(
            "gain_at_episode_start_bucket skipped because the configured episode parquet "
            f"was unavailable: {relative_project_path(parquet_path)}."
        )
        return out[["split", "episode_id", "gain_at_episode_start"]]
    try:
        raw = pd.read_parquet(
            parquet_path,
            columns=["episode_id", "date", "unrealized_gains_pct"],
        )
    except Exception as exc:
        notes.append(
            "gain_at_episode_start_bucket skipped because unrealized_gains_pct could "
            f"not be loaded from {relative_project_path(parquet_path)}: {exc}"
        )
        return out[["split", "episode_id", "gain_at_episode_start"]]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.sort_values(["episode_id", "date"])
    raw_first = (
        raw.groupby("episode_id", as_index=False)
        .first()[["episode_id", "date", "unrealized_gains_pct"]]
        .rename(
            columns={
                "date": "raw_episode_start_date",
                "unrealized_gains_pct": "gain_at_episode_start",
            }
        )
    )
    out = out.merge(raw_first, on="episode_id", how="left", validate="many_to_one")
    out["gain_at_episode_start"] = pd.to_numeric(
        out["gain_at_episode_start_y"],
        errors="coerce",
    )
    start_matches = (
        pd.to_datetime(out["episode_start_date"], errors="coerce")
        .eq(pd.to_datetime(out["raw_episode_start_date"], errors="coerce"))
        .fillna(False)
    )
    unmatched = int((out["gain_at_episode_start"].notna() & ~start_matches).sum())
    if unmatched:
        notes.append(
            "gain_at_episode_start_bucket loaded unrealized_gains_pct from the raw "
            f"episode parquet; {unmatched} rows had a start-date mismatch and were left available by episode_id."
        )
    else:
        notes.append(
            "gain_at_episode_start_bucket uses tertiles of raw unrealized_gains_pct "
            "from the first hold_to_terminal rollout step, joined from the configured episode parquet."
        )
    return out[["split", "episode_id", "gain_at_episode_start"]]


def validate_step9_inputs(episode_df: pd.DataFrame, step_df: pd.DataFrame) -> None:
    require_columns(
        episode_df,
        [
            "split",
            "policy_name",
            "episode_id",
            "episode_final_after_tax_total_value",
            "total_tax_paid",
            "pct_episode_position_sold_short_term",
            "episode_cut_occurred",
            "days_to_first_sale",
        ],
        source=Path("baseline episode metrics"),
        step="Step 9",
    )
    require_columns(
        step_df,
        [
            "split",
            "policy_name",
            "episode_id",
            "step_in_episode",
            "date",
            "tax_transition_date",
            "after_tax_total_value",
        ],
        source=Path("baseline step rollouts"),
        step="Step 9",
    )
    missing = missing_split_policy_rows(
        episode_df,
        policies=STEP9_POLICIES,
        splits=VALIDATION_TEST_SPLITS,
    )
    if missing:
        raise ValueError(
            "Step 9 requires all preferred and benchmark policy rows: "
            + ", ".join(missing)
        )
    duplicate_count = int(
        episode_df[
            episode_df["split"].isin(VALIDATION_TEST_SPLITS)
            & episode_df["policy_name"].isin(STEP9_POLICIES)
        ].duplicated(["split", "policy_name", "episode_id"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            "Step 9 paired comparisons require unique split/policy/episode rows; "
            f"found {duplicate_count} duplicates in baseline episode metrics."
        )


def validate_step9_output(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Step 9 generated an empty cross-sectional table.")
    missing = [column for column in STEP9_REQUIRED_OUTPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Step 9 output is missing required columns: " + ", ".join(missing)
        )
    bad_legacy = [
        column for column in df.columns if column in STEP9_FORBIDDEN_LEGACY_SHARPE_COLUMNS
    ]
    if bad_legacy:
        raise ValueError(
            "Step 9 output attempted to include forbidden legacy Sharpe column(s): "
            + ", ".join(sorted(set(bad_legacy)))
        )
    bad_sector_groups = sorted(
        group_name
        for group_name in df["group_name"].astype(str).unique()
        if "sector" in group_name.lower() or "industry" in group_name.lower()
    )
    if bad_sector_groups:
        raise ValueError(
            "Step 9 sector/industry analysis is out of scope for this task, but "
            "group(s) were generated: " + ", ".join(bad_sector_groups)
        )
    numeric = df.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Step 9 output contains infinite numeric values.")


def add_step9_grouping(
    base: pd.DataFrame,
    rows: list[dict[str, Any]],
    *,
    split: str,
    group_name: str,
    group_column: str,
) -> None:
    rows.extend(build_cross_section_group_rows(base, group_name, group_column, split))


def build_cross_section_group_rows(
    base: pd.DataFrame,
    group_name: str,
    group_column: str,
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for bucket, group in base.groupby(group_column, sort=False, observed=False):
        pref_minus_hold = group["preferred_value"] - group["hold_to_terminal_value"]
        pref_minus_sell = group["preferred_value"] - group["sell_immediately_value"]
        pref_minus_half = group["preferred_value"] - group["sell_half_then_hold_value"]
        pref_minus_hold_eaat = group["preferred_EAAT_Sharpe"] - group[
            "hold_to_terminal_EAAT_Sharpe"
        ]
        pref_minus_sell_eaat = group["preferred_EAAT_Sharpe"] - group[
            "sell_immediately_EAAT_Sharpe"
        ]
        pref_minus_half_eaat = group["preferred_EAAT_Sharpe"] - group[
            "sell_half_then_hold_EAAT_Sharpe"
        ]
        pref_minus_hold_ta = group["preferred_TA_EAAT_Sharpe"] - group[
            "hold_to_terminal_TA_EAAT_Sharpe"
        ]
        pref_minus_sell_ta = group["preferred_TA_EAAT_Sharpe"] - group[
            "sell_immediately_TA_EAAT_Sharpe"
        ]
        pref_minus_half_ta = group["preferred_TA_EAAT_Sharpe"] - group[
            "sell_half_then_hold_TA_EAAT_Sharpe"
        ]
        rows.append(
            {
                "split": split,
                "group_name": group_name,
                "group_bucket": str(bucket),
                "num_episodes": int(len(group)),
                "preferred_mean_final_value": safe_mean(group["preferred_value"]),
                "hold_to_terminal_mean_final_value": safe_mean(
                    group["hold_to_terminal_value"]
                ),
                "sell_immediately_mean_final_value": safe_mean(
                    group["sell_immediately_value"]
                ),
                "sell_half_then_hold_mean_final_value": safe_mean(
                    group["sell_half_then_hold_value"]
                ),
                "preferred_minus_hold_mean": safe_mean(pref_minus_hold),
                "preferred_minus_sell_immediately_mean": safe_mean(pref_minus_sell),
                "preferred_minus_sell_half_mean": safe_mean(pref_minus_half),
                "preferred_win_rate_vs_hold": safe_win_rate(pref_minus_hold),
                "preferred_win_rate_vs_sell_immediately": safe_win_rate(pref_minus_sell),
                "preferred_win_rate_vs_sell_half": safe_win_rate(pref_minus_half),
                "preferred_mean_tax_paid": safe_mean(group["preferred_tax_paid"]),
                "preferred_mean_short_term_sold_fraction": safe_mean(
                    group["preferred_short_term"]
                ),
                "preferred_no_cut_pct": safe_mean(group["preferred_no_cut"].astype(float)),
                "preferred_median_EAAT_Sharpe": safe_median(
                    group["preferred_EAAT_Sharpe"]
                ),
                "hold_to_terminal_median_EAAT_Sharpe": safe_median(
                    group["hold_to_terminal_EAAT_Sharpe"]
                ),
                "sell_immediately_median_EAAT_Sharpe": safe_median(
                    group["sell_immediately_EAAT_Sharpe"]
                ),
                "sell_half_then_hold_median_EAAT_Sharpe": safe_median(
                    group["sell_half_then_hold_EAAT_Sharpe"]
                ),
                "preferred_median_TA_EAAT_Sharpe": safe_median(
                    group["preferred_TA_EAAT_Sharpe"]
                ),
                "hold_to_terminal_median_TA_EAAT_Sharpe": safe_median(
                    group["hold_to_terminal_TA_EAAT_Sharpe"]
                ),
                "sell_immediately_median_TA_EAAT_Sharpe": safe_median(
                    group["sell_immediately_TA_EAAT_Sharpe"]
                ),
                "sell_half_then_hold_median_TA_EAAT_Sharpe": safe_median(
                    group["sell_half_then_hold_TA_EAAT_Sharpe"]
                ),
                "preferred_minus_hold_mean_EAAT_Sharpe": safe_mean(
                    pref_minus_hold_eaat
                ),
                "preferred_minus_hold_median_EAAT_Sharpe": safe_median(
                    pref_minus_hold_eaat
                ),
                "preferred_EAAT_Sharpe_win_rate_vs_hold": safe_win_rate(
                    pref_minus_hold_eaat
                ),
                "preferred_minus_hold_mean_TA_EAAT_Sharpe": safe_mean(
                    pref_minus_hold_ta
                ),
                "preferred_minus_hold_median_TA_EAAT_Sharpe": safe_median(
                    pref_minus_hold_ta
                ),
                "preferred_TA_EAAT_Sharpe_win_rate_vs_hold": safe_win_rate(
                    pref_minus_hold_ta
                ),
                "preferred_minus_sell_immediately_mean_EAAT_Sharpe": safe_mean(
                    pref_minus_sell_eaat
                ),
                "preferred_minus_sell_half_mean_EAAT_Sharpe": safe_mean(
                    pref_minus_half_eaat
                ),
                "preferred_minus_sell_immediately_mean_TA_EAAT_Sharpe": safe_mean(
                    pref_minus_sell_ta
                ),
                "preferred_minus_sell_half_mean_TA_EAAT_Sharpe": safe_mean(
                    pref_minus_half_ta
                ),
            }
        )
    return rows


def plot_step9_preferred_minus_hold(
    out: pd.DataFrame,
    *,
    group_name: str,
    path: Path,
    registry: OutputRegistry,
    title: str,
) -> None:
    plot_df = out[out["split"].eq("test") & out["group_name"].eq(group_name)].copy()
    if plot_df.empty:
        raise ValueError(
            f"Step 9 plot has no rows for test split and group_name={group_name}."
        )
    colors = np.where(plot_df["preferred_minus_hold_mean"].ge(0), "#3b6ea8", "#8a4f3d")
    plt.figure(figsize=(8.4, 4.8))
    plt.bar(plot_df["group_bucket"], plot_df["preferred_minus_hold_mean"], color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("preferred minus hold mean final value")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    save_plot(
        path,
        registry,
        "9",
        f"Preferred minus hold mean final after-tax value by {group_name} for test split.",
    )


def plot_step9_preferred_sharpe_by_new_groups(
    out: pd.DataFrame,
    *,
    path: Path,
    registry: OutputRegistry,
) -> None:
    group_names = [
        "volatility_bucket",
        "gain_at_episode_start_bucket",
        "days_until_tax_transition_at_start_bucket",
    ]
    plot_df = out[
        out["split"].eq("test") & out["group_name"].isin(group_names)
    ].copy()
    if plot_df.empty:
        raise ValueError("Step 9 Sharpe plot has no rows for the new test groupings.")
    plot_df["label"] = plot_df["group_name"] + ": " + plot_df["group_bucket"]
    x = np.arange(len(plot_df))
    width = 0.38
    plt.figure(figsize=(13.5, 5.6))
    plt.bar(
        x - width / 2,
        plot_df["preferred_median_EAAT_Sharpe"],
        width,
        label="median EAAT Sharpe",
        color="#3b6ea8",
    )
    plt.bar(
        x + width / 2,
        plot_df["preferred_median_TA_EAAT_Sharpe"],
        width,
        label="median TA-EAAT Sharpe",
        color="#6c7a40",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("preferred policy median Sharpe")
    plt.title("Preferred median EAAT and TA-EAAT Sharpe by new Step 9 groups - test")
    plt.xticks(x, plot_df["label"], rotation=55, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_plot(
        path,
        registry,
        "9",
        "Preferred policy median EAAT and TA-EAAT Sharpe across new Step 9 test groupings.",
    )


def build_step9(
    episode_df: pd.DataFrame,
    step_df: pd.DataFrame,
    output_dir: Path,
    plots_dir: Path,
    registry: OutputRegistry,
    config: dict[str, Any],
) -> pd.DataFrame:
    validate_step9_inputs(episode_df, step_df)
    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    sharpe_df, sharpe_source = load_step3b_episode_eaat_sharpe_metrics(output_dir)
    hold_path_stats = step9_hold_path_stats(step_df)
    start_gain = load_step9_episode_start_gain_features(config, hold_path_stats, notes)
    notes.append(
        "volatility_bucket uses tertiles of hold_to_terminal daily/path percent "
        "changes computed from after_tax_total_value."
    )
    notes.append(
        "days_until_tax_transition_at_start_bucket uses tertiles of the first "
        "hold_to_terminal rollout date distance to tax_transition_date."
    )
    notes.append(
        "maximum_gain_bucket uses the hold_to_terminal path proxy: maximum "
        "after_tax_total_value minus first after_tax_total_value."
    )
    notes.append(
        "drawdown_after_episode_start_bucket uses the hold_to_terminal path proxy: "
        "first after_tax_total_value minus minimum after_tax_total_value."
    )
    notes.append(
        "sector and industry groupings are intentionally skipped; sector analysis "
        "is out of scope for this Step 9 update."
    )
    notes.append(
        "EAAT and TA-EAAT Sharpe diagnostics are merged from "
        f"{relative_project_path(sharpe_source)}."
    )
    notes.append(
        "Final after-tax value remains the primary thesis metric; EAAT and TA-EAAT "
        "Sharpe ratios are secondary risk-adjusted diagnostics."
    )
    for split in VALIDATION_TEST_SPLITS:
        split_metrics = episode_df[
            episode_df["split"].eq(split) & episode_df["policy_name"].isin(STEP9_POLICIES)
        ].copy()
        missing_policies = sorted(set(STEP9_POLICIES) - set(split_metrics["policy_name"].unique()))
        if missing_policies:
            raise ValueError(
                f"Step 9 split={split} is missing required policy rows: "
                + ", ".join(missing_policies)
            )
        value_wide = split_metrics.pivot(
            index=["split", "episode_id"],
            columns="policy_name",
            values="episode_final_after_tax_total_value",
        ).reset_index()
        value_wide = value_wide.rename(
            columns={
                PREFERRED_POLICY: "preferred_value",
                "hold_to_terminal": "hold_to_terminal_value",
                "sell_immediately": "sell_immediately_value",
                "sell_half_then_hold": "sell_half_then_hold_value",
            }
        )
        value_columns = [
            "preferred_value",
            "hold_to_terminal_value",
            "sell_immediately_value",
            "sell_half_then_hold_value",
        ]
        if value_wide[value_columns].isna().any().any():
            raise ValueError(
                f"Step 9 split={split} has unpaired final-value rows for at least one required policy."
            )
        sharpe_long = sharpe_df[
            sharpe_df["split"].eq(split) & sharpe_df["policy_name"].isin(STEP9_POLICIES)
        ].copy()
        sharpe_long["policy_label"] = sharpe_long["policy_name"].map(STEP9_POLICY_LABELS)
        sharpe_wide = sharpe_long.pivot(
            index=["split", "episode_id"],
            columns="policy_label",
            values=["EAAT_Sharpe", "TA_EAAT_Sharpe"],
        )
        sharpe_wide.columns = [
            f"{policy_label}_{metric}"
            for metric, policy_label in sharpe_wide.columns.to_flat_index()
        ]
        sharpe_wide = sharpe_wide.reset_index()
        pref_extra = split_metrics[split_metrics["policy_name"].eq(PREFERRED_POLICY)][
            [
                "split",
                "episode_id",
                "total_tax_paid",
                "pct_episode_position_sold_short_term",
                "episode_cut_occurred",
                "days_to_first_sale",
            ]
        ].rename(
            columns={
                "total_tax_paid": "preferred_tax_paid",
                "pct_episode_position_sold_short_term": "preferred_short_term",
                "episode_cut_occurred": "preferred_cut_occurred",
            }
        )
        base = value_wide.merge(
            sharpe_wide,
            on=["split", "episode_id"],
            how="left",
            validate="one_to_one",
        )
        required_sharpe_columns = [
            f"{label}_{metric}"
            for label in STEP9_POLICY_LABELS.values()
            for metric in ["EAAT_Sharpe", "TA_EAAT_Sharpe"]
        ]
        missing_sharpe_columns = [
            column for column in required_sharpe_columns if column not in base.columns
        ]
        if missing_sharpe_columns:
            raise ValueError(
                "Step 9 corrected Sharpe merge is missing columns: "
                + ", ".join(missing_sharpe_columns)
            )
        base = base.merge(pref_extra, on=["split", "episode_id"], how="inner", validate="one_to_one")
        base = base.merge(
            hold_path_stats,
            on=["split", "episode_id"],
            how="left",
            validate="one_to_one",
        )
        base = base.merge(
            start_gain,
            on=["split", "episode_id"],
            how="left",
            validate="one_to_one",
        )
        base["preferred_no_cut"] = ~base["preferred_cut_occurred"]
        base["final_episode_return_bucket"] = quantile_bucket(
            base["preferred_value"],
            ["low", "mid", "high"],
        )
        add_step9_grouping(
            base,
            rows,
            split=split,
            group_name="final_episode_return_bucket",
            group_column="final_episode_return_bucket",
        )

        if base["maximum_gain_proxy"].notna().any():
            base["maximum_gain_bucket"] = quantile_bucket(
                base["maximum_gain_proxy"],
                ["low", "mid", "high"],
            )
            add_step9_grouping(
                base,
                rows,
                split=split,
                group_name="maximum_gain_bucket",
                group_column="maximum_gain_bucket",
            )
        else:
            notes.append(f"{split}: maximum gain bucket skipped; no path proxy available.")
        if base["drawdown_after_start_proxy"].notna().any():
            base["drawdown_after_episode_start_bucket"] = quantile_bucket(
                base["drawdown_after_start_proxy"],
                ["low", "mid", "high"],
            )
            add_step9_grouping(
                base,
                rows,
                split=split,
                group_name="drawdown_after_episode_start_bucket",
                group_column="drawdown_after_episode_start_bucket",
            )
        else:
            notes.append(f"{split}: drawdown bucket skipped; no path proxy available.")
        if base["volatility_proxy"].notna().any():
            base["volatility_bucket"] = quantile_bucket(
                base["volatility_proxy"],
                ["low_volatility", "medium_volatility", "high_volatility"],
            ).astype(object)
            add_step9_grouping(
                base,
                rows,
                split=split,
                group_name="volatility_bucket",
                group_column="volatility_bucket",
            )
        else:
            notes.append(
                f"{split}: volatility_bucket skipped because hold_to_terminal path volatility could not be computed."
            )
        if base["gain_at_episode_start"].notna().any():
            base["gain_at_episode_start_bucket"] = quantile_bucket(
                base["gain_at_episode_start"],
                ["low_start_gain", "medium_start_gain", "high_start_gain"],
            ).astype(object)
            add_step9_grouping(
                base,
                rows,
                split=split,
                group_name="gain_at_episode_start_bucket",
                group_column="gain_at_episode_start_bucket",
            )
        else:
            notes.append(
                f"{split}: gain_at_episode_start_bucket skipped because unrealized_gains_pct was unavailable in Step 9 inputs and no valid raw parquet join was available."
            )
        if base["days_until_tax_transition_at_start"].notna().any():
            base["days_until_tax_transition_at_start_bucket"] = quantile_bucket(
                base["days_until_tax_transition_at_start"],
                ["near_transition", "medium_transition_distance", "far_from_transition"],
            ).astype(object)
            add_step9_grouping(
                base,
                rows,
                split=split,
                group_name="days_until_tax_transition_at_start_bucket",
                group_column="days_until_tax_transition_at_start_bucket",
            )
        else:
            notes.append(
                f"{split}: days_until_tax_transition_at_start_bucket skipped because dates or tax transition dates were unavailable."
            )
        base["days_to_first_sale_bucket"] = pd.cut(
            base["days_to_first_sale"],
            bins=[-np.inf, 0, 30, 180, np.inf],
            labels=["immediate", "1_to_30_days", "31_to_180_days", "over_180_days"],
        ).astype(object)
        base["days_to_first_sale_bucket"] = base["days_to_first_sale_bucket"].fillna(
            "no_discretionary_sale"
        )
        add_step9_grouping(
            base,
            rows,
            split=split,
            group_name="days_to_first_sale_bucket",
            group_column="days_to_first_sale_bucket",
        )
        if base["calendar_year"].notna().any():
            add_step9_grouping(
                base,
                rows,
                split=split,
                group_name="calendar_year",
                group_column="calendar_year",
            )
        else:
            notes.append(f"{split}: calendar year skipped; no valid date available.")

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan)
    validate_step9_output(out)
    save_table(
        out,
        output_dir / "step9_cross_sectional_episode_analysis.csv",
        output_dir / "step9_cross_sectional_episode_analysis.md",
        registry,
        "9",
        "Cross-sectional paired episode analysis by available groupings.",
    )
    save_text(
        "\n".join(["Step 9 skipped grouping notes", *notes, ""]),
        output_dir / "step9_skipped_groupings_notes.txt",
        registry,
        "9",
        "Skipped or proxied cross-sectional grouping notes.",
    )
    plot_step9_preferred_minus_hold(
        out,
        group_name="final_episode_return_bucket",
        path=plots_dir / "step9_preferred_minus_hold_by_final_value_bucket_test.png",
        registry=registry,
        title="Preferred minus hold by final value bucket - test",
    )
    plot_step9_preferred_minus_hold(
        out,
        group_name="volatility_bucket",
        path=plots_dir / "step9_preferred_minus_hold_by_volatility_bucket_test.png",
        registry=registry,
        title="Preferred minus hold by volatility bucket - test",
    )
    plot_step9_preferred_minus_hold(
        out,
        group_name="gain_at_episode_start_bucket",
        path=plots_dir / "step9_preferred_minus_hold_by_gain_at_episode_start_bucket_test.png",
        registry=registry,
        title="Preferred minus hold by gain-at-episode-start bucket - test",
    )
    plot_step9_preferred_minus_hold(
        out,
        group_name="days_until_tax_transition_at_start_bucket",
        path=plots_dir / "step9_preferred_minus_hold_by_days_until_tax_transition_at_start_bucket_test.png",
        registry=registry,
        title="Preferred minus hold by tax-transition distance - test",
    )
    plot_step9_preferred_sharpe_by_new_groups(
        out,
        path=plots_dir / "step9_median_eaat_ta_eaat_sharpe_by_group_test.png",
        registry=registry,
    )
    return out


def bootstrap_ci_mean(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(diff) == 0:
        return np.nan, np.nan
    samples = rng.choice(diff, size=(BOOTSTRAP_ITERATIONS, len(diff)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def wilcoxon_pvalue(diff: np.ndarray) -> float | None:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 1.0
    return float(wilcoxon(nonzero).pvalue)


def build_step10(
    episode_df: pd.DataFrame,
    output_dir: Path,
    plots_dir: Path,
    registry: OutputRegistry,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for split in VALIDATION_TEST_SPLITS:
        preferred = episode_df[
            episode_df["split"].eq(split) & episode_df["policy_name"].eq(PREFERRED_POLICY)
        ][["episode_id", "episode_final_after_tax_total_value"]].rename(
            columns={"episode_final_after_tax_total_value": "preferred_value"}
        )
        for comparison in STEP10_COMPARISON_POLICIES:
            comp = episode_df[
                episode_df["split"].eq(split) & episode_df["policy_name"].eq(comparison)
            ][["episode_id", "episode_final_after_tax_total_value"]].rename(
                columns={"episode_final_after_tax_total_value": "comparison_value"}
            )
            merged = preferred.merge(comp, on="episode_id", how="inner")
            diff = (
                merged["preferred_value"] - merged["comparison_value"]
            ).to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci_mean(diff, rng)
            rows.append(
                {
                    "split": split,
                    "comparison_policy": comparison,
                    "num_paired_episodes": int(len(diff)),
                    "mean_difference": float(np.mean(diff)),
                    "median_difference": float(np.median(diff)),
                    "std_difference": float(np.std(diff, ddof=1)),
                    "p05_difference": float(np.quantile(diff, 0.05)),
                    "p95_difference": float(np.quantile(diff, 0.95)),
                    "win_rate": float(np.mean(diff > 0)),
                    "loss_rate": float(np.mean(diff < 0)),
                    "tie_rate": float(np.mean(diff == 0)),
                    "bootstrap_95ci_mean_difference_low": ci_low,
                    "bootstrap_95ci_mean_difference_high": ci_high,
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "random_seed": RANDOM_SEED,
                    "wilcoxon_signed_rank_p_value": wilcoxon_pvalue(diff),
                }
            )
    out = pd.DataFrame(rows)
    split_order = {split: index for index, split in enumerate(VALIDATION_TEST_SPLITS)}
    comp_order = {policy: index for index, policy in enumerate(STEP10_COMPARISON_POLICIES)}
    out["_split_order"] = out["split"].map(split_order)
    out["_comp_order"] = out["comparison_policy"].map(comp_order)
    out = out.sort_values(["_split_order", "_comp_order"]).drop(
        columns=["_split_order", "_comp_order"]
    )
    save_table(
        out,
        output_dir / "step10_robustness_statistical_tests.csv",
        output_dir / "step10_robustness_statistical_tests.md",
        registry,
        "10",
        "Paired robustness and statistical testing calculations.",
    )
    notes = [
        "Step 10 statistical testing notes",
        "Bootstrap confidence intervals are percentile intervals for paired mean differences.",
        f"bootstrap_iterations: {BOOTSTRAP_ITERATIONS}",
        f"random_seed: {RANDOM_SEED}",
        "Wilcoxon signed-rank p-values are included only when scipy is installed.",
        "These outputs provide calculations only and do not make significance claims.",
        "",
    ]
    save_text(
        "\n".join(notes),
        output_dir / "step10_statistical_testing_notes.txt",
        registry,
        "10",
        "Statistical testing notes.",
    )
    for comparison, filename in [
        ("hold_to_terminal", "step10_difference_distribution_vs_hold_test.png"),
        (
            "sell_immediately",
            "step10_difference_distribution_vs_sell_immediately_test.png",
        ),
    ]:
        preferred = episode_df[
            episode_df["split"].eq("test") & episode_df["policy_name"].eq(PREFERRED_POLICY)
        ][["episode_id", "episode_final_after_tax_total_value"]].rename(
            columns={"episode_final_after_tax_total_value": "preferred_value"}
        )
        comp = episode_df[
            episode_df["split"].eq("test") & episode_df["policy_name"].eq(comparison)
        ][["episode_id", "episode_final_after_tax_total_value"]].rename(
            columns={"episode_final_after_tax_total_value": "comparison_value"}
        )
        diff = preferred.merge(comp, on="episode_id", how="inner")
        diff["difference"] = diff["preferred_value"] - diff["comparison_value"]
        plt.figure(figsize=(7, 4.5))
        diff["difference"].hist(bins=40, color="#6a8f3a")
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("preferred minus comparison final value")
        plt.ylabel("episode count")
        plt.title(f"Difference distribution vs {comparison} - test")
        save_plot(
            plots_dir / filename,
            registry,
            "10",
            f"Test paired difference distribution versus {comparison}.",
        )
    return out


def select_representative_cases(episode_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    test = episode_df[episode_df["split"].eq("test")]
    pref = test[test["policy_name"].eq(PREFERRED_POLICY)].copy()
    hold = test[test["policy_name"].eq("hold_to_terminal")][
        ["episode_id", "episode_final_after_tax_total_value"]
    ].rename(columns={"episode_final_after_tax_total_value": "hold_value"})
    pref = pref.merge(hold, on="episode_id", how="inner")
    pref["preferred_minus_hold"] = (
        pref["episode_final_after_tax_total_value"] - pref["hold_value"]
    )
    selections = []
    selected_episode_ids: set[str] = set()

    def choose(case_id: int, label: str, mask: pd.Series, sort_col: str, ascending: bool) -> None:
        candidates = pref[mask].sort_values(sort_col, ascending=ascending)
        distinct_candidates = candidates[
            ~candidates["episode_id"].astype(str).isin(selected_episode_ids)
        ]
        if not distinct_candidates.empty:
            candidates = distinct_candidates
        if candidates.empty:
            notes.append(f"case_{case_id}: skipped; no episode matched {label}.")
            return
        row = candidates.iloc[0]
        selected_episode_ids.add(str(row["episode_id"]))
        selections.append(
            {
                "case_id": case_id,
                "case_label": label,
                "episode_id": row["episode_id"],
                "preferred_final_value": row["episode_final_after_tax_total_value"],
                "hold_to_terminal_final_value": row["hold_value"],
                "preferred_minus_hold": row["preferred_minus_hold"],
                "num_discretionary_sales": row["num_discretionary_sales"],
                "days_to_first_sale": row["days_to_first_sale"],
                "pct_position_sold_short_term": row[
                    "pct_episode_position_sold_short_term"
                ],
                "episode_cut_occurred": row["episode_cut_occurred"],
            }
        )

    discretionary = pref["num_discretionary_sales"].fillna(0).gt(0)
    choose(
        1,
        "dqn_sells_early_beats_hold",
        discretionary & pref["preferred_minus_hold"].gt(0),
        "preferred_minus_hold",
        False,
    )
    choose(
        2,
        "dqn_sells_early_underperforms_hold",
        discretionary & pref["preferred_minus_hold"].lt(0),
        "preferred_minus_hold",
        True,
    )
    choose(
        3,
        "dqn_waits_until_long_term",
        pref["pct_episode_position_sold_short_term"].fillna(0).eq(0),
        "episode_final_after_tax_total_value",
        False,
    )
    choose(
        4,
        "dqn_hold_like",
        ~pref["episode_cut_occurred"],
        "episode_final_after_tax_total_value",
        False,
    )
    choose(
        5,
        "partial_liquidation",
        discretionary
        & (
            pref["terminal_liquidation_fraction"].fillna(0).gt(0)
            | pref["first_cut_fraction_executed"].fillna(1).lt(1)
        ),
        "episode_final_after_tax_total_value",
        False,
    )
    return pd.DataFrame(selections), notes


def plot_case_study(
    case: pd.Series,
    step_df: pd.DataFrame,
    plots_dir: Path,
    registry: OutputRegistry,
) -> None:
    episode_id = case["episode_id"]
    case_id = int(case["case_id"])
    filenames = {
        1: "step11_case_1_dqn_sells_early_beats_hold.png",
        2: "step11_case_2_dqn_sells_early_underperforms_hold.png",
        3: "step11_case_3_dqn_waits_until_long_term.png",
        4: "step11_case_4_dqn_hold_like.png",
        5: "step11_case_5_partial_liquidation.png",
    }
    policies = [PREFERRED_POLICY, "hold_to_terminal", "sell_immediately"]
    labels = {
        PREFERRED_POLICY: "preferred DQN",
        "hold_to_terminal": "hold_to_terminal",
        "sell_immediately": "sell_immediately",
    }
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for policy in policies:
        path = step_df[
            step_df["split"].eq("test")
            & step_df["policy_name"].eq(policy)
            & step_df["episode_id"].eq(episode_id)
        ].sort_values("step_in_episode")
        if path.empty:
            continue
        x = path["date"] if path["date"].notna().all() else path["step_in_episode"]
        axes[0].plot(x, path["after_tax_total_value"], label=labels[policy])
        if policy == PREFERRED_POLICY:
            sale_steps = path[
                path["action_fraction_executed"].fillna(0).gt(0)
                & ~path["terminal_liquidation_executed"].fillna(False)
            ]
            axes[0].scatter(
                sale_steps["date"] if sale_steps["date"].notna().all() else sale_steps["step_in_episode"],
                sale_steps["after_tax_total_value"],
                marker="o",
                color="black",
                s=30,
                label="preferred sale",
            )
            axes[1].plot(x, path["remaining_fraction"], label="remaining fraction")
            axes[1].plot(x, path["sold_fraction"], label="sold fraction")
            transition = path["tax_transition_date"].dropna()
            if not transition.empty:
                axes[0].axvline(transition.iloc[0], color="gray", linestyle="--", linewidth=1)
                axes[1].axvline(transition.iloc[0], color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("after-tax total value")
    axes[0].set_title(f"Case {case_id}: {case['case_label']} ({episode_id})")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("fraction")
    axes[1].set_xlabel("date")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    save_plot(
        plots_dir / filenames[case_id],
        registry,
        "11",
        f"Representative case-study plot: {case['case_label']}.",
    )


def build_step11(
    episode_df: pd.DataFrame,
    step_df: pd.DataFrame,
    output_dir: Path,
    plots_dir: Path,
    registry: OutputRegistry,
) -> pd.DataFrame:
    selections, notes = select_representative_cases(episode_df)
    save_table(
        selections,
        output_dir / "step11_representative_episode_selection.csv",
        output_dir / "step11_representative_episode_selection.md",
        registry,
        "11",
        "Automatically selected representative test episodes.",
    )
    plotted_cases = set()
    for _, case in selections.iterrows():
        plot_case_study(case, step_df, plots_dir, registry)
        plotted_cases.add(int(case["case_id"]))
    for case_id in range(1, 6):
        if case_id not in plotted_cases and not any(
            note.startswith(f"case_{case_id}:") for note in notes
        ):
            notes.append(f"case_{case_id}: skipped; no plot generated.")
    save_text(
        "\n".join(["Step 11 case-study notes", *notes, ""]),
        output_dir / "step11_case_study_notes.txt",
        registry,
        "11",
        "Representative episode case-study notes.",
    )
    return selections


def best_policy_by_split(episode_df: pd.DataFrame, split: str) -> str:
    summary = aggregate_policy_metrics(episode_df, POLICY_UNIVERSE, [split])
    return str(
        summary.sort_values("mean_final_after_tax_total_value", ascending=False).iloc[0][
            "policy_name"
        ]
    )


def preferred_beats(episode_df: pd.DataFrame, split: str, benchmark: str) -> bool:
    summary = aggregate_policy_metrics(episode_df, [PREFERRED_POLICY, benchmark], [split])
    values = summary.set_index("policy_name")["mean_final_after_tax_total_value"]
    return bool(values[PREFERRED_POLICY] > values[benchmark])


def write_final_summary_and_manifest(
    *,
    output_dir: Path,
    registry: OutputRegistry,
    episode_df: pd.DataFrame,
    train_generated: bool,
    skipped_optional: list[str],
) -> None:
    if any(row["step"] == "4B" for row in registry.rows):
        raise RuntimeError(
            "Legacy Step 4B outputs were registered during the final Step 4 workflow."
        )
    created_tables = sum(1 for row in registry.rows if row["type"] in {"csv", "md"})
    created_plots = sum(1 for row in registry.rows if row["type"] == "png")
    lines = [
        "Quantitative Analysis Phase 4-11 Summary",
        "script_run: python scripts/build_quant_analysis_phase_4_11.py --config configs/train_reward_c_lite_v5.yaml",
        "source_files_used: runs/train_reward_c_lite_v5_full/baselines/baseline_episode_metrics.csv; runs/train_reward_c_lite_v5_full/baselines/baseline_step_rollouts.csv; runs/train_reward_c_lite_v5_full/baselines/baseline_summary_by_policy.csv; runs/train_reward_c_lite_v5_full/episode_splits.csv; runs/train_reward_c_lite_v5_full/quant_analysis/step3b_policy_eaat_sharpe_metrics.csv; runs/train_reward_c_lite_v5_full/quant_analysis/step3b_episode_eaat_sharpe_metrics.csv; runs/train_reward_c_lite_v5_full/quant_analysis/step3b_eaat_sharpe_metrics_notes.txt; data/episodes/drl_episodes.parquet",
        f"output_directory: {relative_project_path(output_dir)}",
        f"number_of_tables_created: {created_tables}",
        f"number_of_plots_created: {created_plots}",
        f"preferred_policy: {PREFERRED_POLICY}",
        f"best_validation_policy_by_mean_final_after_tax_value: {best_policy_by_split(episode_df, 'validation')}",
        f"best_test_policy_by_mean_final_after_tax_value: {best_policy_by_split(episode_df, 'test')}",
        f"preferred_policy_beats_hold_to_terminal_on_test: {preferred_beats(episode_df, 'test', 'hold_to_terminal')}",
        f"preferred_policy_beats_sell_immediately_on_test: {preferred_beats(episode_df, 'test', 'sell_immediately')}",
        f"preferred_policy_beats_sell_half_then_hold_on_test: {preferred_beats(episode_df, 'test', 'sell_half_then_hold')}",
        "step4_final_decision_rule_block_completed: True",
        "step4_policy_progression: trained_dqn_greedy -> trained_dqn_thresholded_margin_0p020 -> first-sale-thresholded DQN sweep",
        "step4_first_sale_margin_sweep: 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090",
        "step4_uses_final_annualized_eaat_ta_eaat_sharpe_diagnostics: True",
        "step4_sharpe_scope: final EAAT / TA-EAAT Sharpe metrics only; no legacy Sharpe diagnostics, reward-path Sharpe, step-return proxy Sharpe, or Step 4B metrics are included.",
        f"step4_annual_risk_free_rate: {ANNUAL_RISK_FREE_RATE}",
        f"step4_daily_risk_free_rate: {DAILY_RISK_FREE_RATE}",
        f"step4_annualization_factor: {ANNUALIZATION_FACTOR}",
        "step4_final_after_tax_value_primary_metric: True",
        "step4_eaat_ta_eaat_secondary_risk_adjusted_diagnostics: True",
        "step4_ta_eaat_summary_statistic: median; mean and standard deviation are omitted because TA-EAAT is sensitive to outliers from near-zero tranche volatility denominators.",
        "step4_margin_chart_moved_into_step4: True",
        "step4_legacy_step4b_metrics_included: False",
        "step4_legacy_step4b_outputs_regenerated: False",
        f"step4_preferred_policy_remains: {PREFERRED_POLICY}",
        "step4_scope_note: hold_to_terminal may remain the best absolute final-value benchmark, but Step 4 is about DQN decision-rule improvement.",
        "step7_corrected_eaat_ta_eaat_columns_added: True",
        "step7_economic_period_behavior_completed: True",
        "step7_economic_period_all_episode_plots_completed: True",
        "step7_train_all_descriptive_only: True",
        "step7_validation_test_out_of_sample_behavior_checks: True",
        "step8_win_loss_vs_benchmark_table_completed: True",
        "step8_preferred_vs_hold_situational_analysis_completed: True",
        "step8_situational_group_charts_split_by_ex_ante_group: True",
        "step8_situational_charts_include_win_tie_loss_rates: True",
        "step8_hold_to_terminal_strongest_overall: True",
        "step8_preferred_dqn_beats_naive_active_liquidation_benchmarks_on_test: True",
        "step9_cross_sectional_heterogeneity_extended: volatility_bucket; gain_at_episode_start_bucket; days_until_tax_transition_at_start_bucket",
        "step9_corrected_eaat_ta_eaat_sharpe_diagnostics_added: True",
        "step9_sector_industry_analysis_attempted: False",
        f"train_all_behavior_diagnostics_completed: True; train_rollout_generated_this_run={train_generated}",
        "skipped_optional_analyses: " + ("; ".join(skipped_optional) if skipped_optional else "none"),
        "step_12_writing_implemented: False",
        "",
    ]
    summary_path = output_dir / "phase_4_11_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    registry.add(summary_path, "txt", "4-11", "Final combined phase 4-11 summary.")
    manifest_path = output_dir / "phase_4_11_outputs_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(registry.rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    registry.add(manifest_path, "json", "4-11", "Manifest of phase 4-11 created files.")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(registry.rows, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build quantitative analysis phase 4-11 outputs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Final config path. Defaults to {relative_project_path(DEFAULT_CONFIG_PATH)}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Defaults to {relative_project_path(DEFAULT_OUTPUT_DIR)}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    output_dir = resolve_project_path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    registry = OutputRegistry()

    config = load_yaml(config_path)
    run_path = verify_frozen_run(config_path, config, output_dir)
    ensure_baseline_outputs(config_path, run_path)
    baseline_episode, baseline_step, baseline_summary = load_baseline_data(run_path)
    baseline_episode = prepare_episode_metrics(baseline_episode)
    baseline_step = prepare_step_rollouts(baseline_step)
    baseline_summary = prepare_episode_metrics(baseline_summary)

    episode_source = run_path / "baselines" / "baseline_episode_metrics.csv"
    step_source = run_path / "baselines" / "baseline_step_rollouts.csv"
    require_columns(
        baseline_episode,
        [
            "split",
            "policy_name",
            "episode_id",
            "episode_final_after_tax_total_value",
            "episode_realized_after_tax_pnl",
            "total_tax_paid",
            "mean_effective_tax_rate_on_sales",
            "pct_episode_position_sold_short_term",
            "pct_episode_position_sold_long_term",
            "episode_terminal_liquidation_executed",
            "episode_cut_occurred",
            "days_to_first_sale",
            "first_sale_before_tax_transition",
            "num_discretionary_sales",
            "terminal_liquidation_fraction",
            "episode_final_sold_fraction",
            "episode_final_remaining_fraction",
            "total_positive_taxable_pre_tax_increment",
        ],
        source=episode_source,
        step="steps 4-11",
    )
    require_columns(
        baseline_step,
        [
            "split",
            "policy_name",
            "episode_id",
            "step_in_episode",
            "date",
            "tax_transition_date",
            "action_idx",
            "action_fraction_requested",
            "action_fraction_executed",
            "after_tax_total_value",
            "sold_fraction",
            "remaining_fraction",
            "terminal_liquidation_executed",
        ],
        source=step_source,
        step="steps 7, 9, and 11",
    )

    build_step4(
        baseline_episode,
        output_dir,
        plots_dir,
        registry,
        config=config,
        run_path=run_path,
    )
    build_step5(baseline_episode, output_dir, plots_dir, registry)
    build_step6(
        baseline_episode,
        output_dir,
        plots_dir,
        registry,
        config=config,
        run_path=run_path,
    )
    diag_episode, diag_step, train_generated = build_step7(
        config_path,
        config,
        run_path,
        output_dir,
        plots_dir,
        baseline_episode,
        baseline_step,
        baseline_summary,
        registry,
    )
    build_step8(baseline_episode, baseline_step, output_dir, plots_dir, registry)
    build_step9(baseline_episode, baseline_step, output_dir, plots_dir, registry, config)
    build_step10(baseline_episode, output_dir, plots_dir, registry)
    build_step11(baseline_episode, baseline_step, output_dir, plots_dir, registry)
    skipped_optional = [
        "sector and industry analysis intentionally skipped for Step 9; sector analysis is out of scope for this task",
        "train/all benchmark excess values left as NA except validation/test, avoiding extra benchmark reruns",
    ]
    write_final_summary_and_manifest(
        output_dir=output_dir,
        registry=registry,
        episode_df=baseline_episode,
        train_generated=train_generated,
        skipped_optional=skipped_optional,
    )
    print(f"Wrote quantitative analysis phase 4-11 outputs to {relative_project_path(output_dir)}")
    print("Step 12 writing was not implemented.")


if __name__ == "__main__":
    main()
