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
BOOTSTRAP_ITERATIONS = 2000
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
        split_episode["no_cut_episode"] = ~split_episode["episode_cut_occurred"]
        discretionary = split_episode["num_discretionary_sales"].fillna(0).gt(0)
        partial = discretionary & split_episode["terminal_liquidation_fraction"].fillna(0).gt(0)
        full_early = discretionary & split_episode["terminal_liquidation_fraction"].fillna(0).eq(0)
        final_mean = split_episode["episode_final_after_tax_total_value"].mean()
        excess_hold = np.nan
        excess_sell = np.nan
        if split in baseline_vt.index:
            excess_hold = final_mean - float(baseline_vt.loc[split, "hold_to_terminal"])
            excess_sell = final_mean - float(baseline_vt.loc[split, "sell_immediately"])
        rows.append(
            {
                "split": split,
                "diagnostic_scope": (
                    "in_sample_descriptive"
                    if split == "train"
                    else "all_episode_descriptive"
                    if split == "all"
                    else "out_of_sample"
                ),
                "num_episodes": int(split_episode["episode_id"].nunique()),
                "action_distribution": compact_json(action_distribution(split_step)),
                "first_action_distribution": compact_json(
                    first_action_distribution(split_step)
                ),
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
                "average_days_to_first_sale": float(
                    split_episode["days_to_first_sale"].mean()
                ),
                "median_days_to_first_sale": float(
                    split_episode["days_to_first_sale"].median()
                ),
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
                "mean_final_after_tax_total_value": float(final_mean),
                "mean_excess_value_vs_hold_to_terminal": excess_hold,
                "mean_excess_value_vs_sell_immediately": excess_sell,
            }
        )
    return pd.DataFrame(rows)


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
    behavior_by_split = behavior_summary_rows(diag_episode, diag_step, baseline_summary)
    save_table(
        behavior_by_split,
        output_dir / "step7_preferred_policy_behavior_by_split.csv",
        output_dir / "step7_preferred_policy_behavior_by_split.md",
        registry,
        "7",
        "Preferred policy behavior diagnostics by train, validation, test, and all episodes.",
    )
    behavior_summary = behavior_by_split.copy()
    save_table(
        behavior_summary,
        output_dir / "step7_preferred_policy_behavior_summary.csv",
        output_dir / "step7_preferred_policy_behavior_summary.md",
        registry,
        "7",
        "Preferred policy behavior summary including descriptive diagnostic scope labels.",
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

    return diag_episode, diag_step, train_generated


def build_step8(
    episode_df: pd.DataFrame,
    output_dir: Path,
    plots_dir: Path,
    registry: OutputRegistry,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    final_values = episode_df[
        episode_df["split"].isin(VALIDATION_TEST_SPLITS)
        & episode_df["policy_name"].isin([PREFERRED_POLICY, *BENCHMARK_POLICIES])
    ]
    for split in VALIDATION_TEST_SPLITS:
        preferred = final_values[
            final_values["split"].eq(split) & final_values["policy_name"].eq(PREFERRED_POLICY)
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
        for benchmark in BENCHMARK_POLICIES:
            bench = final_values[
                final_values["split"].eq(split) & final_values["policy_name"].eq(benchmark)
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
            merged = preferred.merge(bench, on="episode_id", how="inner")
            diff = merged["preferred_value"] - merged["benchmark_value"]
            rows.append(
                {
                    "split": split,
                    "benchmark_policy": benchmark,
                    "num_paired_episodes": len(merged),
                    "mean_difference": diff.mean(),
                    "median_difference": diff.median(),
                    "win_rate": diff.gt(0).mean(),
                    "tie_rate": diff.eq(0).mean(),
                    "loss_rate": diff.lt(0).mean(),
                    "preferred_mean_final_after_tax_value": merged[
                        "preferred_value"
                    ].mean(),
                    "benchmark_mean_final_after_tax_value": merged[
                        "benchmark_value"
                    ].mean(),
                    "preferred_minus_benchmark_total_mean_tax_paid": (
                        merged["preferred_tax_paid"].mean()
                        - merged["benchmark_tax_paid"].mean()
                    ),
                    "preferred_minus_benchmark_short_term_sold_fraction": (
                        merged["preferred_short_term"].mean()
                        - merged["benchmark_short_term"].mean()
                    ),
                }
            )
    out = pd.DataFrame(rows)
    save_table(
        out,
        output_dir / "step8_benchmark_dominance_analysis.csv",
        output_dir / "step8_benchmark_dominance_analysis.md",
        registry,
        "8",
        "Paired preferred-policy dominance analysis against benchmark policies.",
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
    test_rows = out[out["split"].eq("test")].set_index("benchmark_policy")
    summary_lines = [
        "Step 8 benchmark dominance summary",
        f"best_validation_policy_by_mean_final_after_tax_value: {best_validation}",
        f"best_test_policy_by_mean_final_after_tax_value: {best_test}",
        f"preferred_beats_hold_to_terminal_on_test: {bool(test_rows.loc['hold_to_terminal', 'mean_difference'] > 0)}",
        f"preferred_beats_sell_immediately_on_test: {bool(test_rows.loc['sell_immediately', 'mean_difference'] > 0)}",
        f"preferred_beats_sell_half_then_hold_on_test: {bool(test_rows.loc['sell_half_then_hold', 'mean_difference'] > 0)}",
        f"preferred_beats_sell_quarters_over_time_on_test: {bool(test_rows.loc['sell_quarters_over_time', 'mean_difference'] > 0)}",
        "If hold_to_terminal is strongest, this file states it directly.",
        "",
    ]
    save_text(
        "\n".join(summary_lines),
        output_dir / "step8_benchmark_dominance_summary.txt",
        registry,
        "8",
        "Benchmark dominance summary text.",
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
    return out


def quantile_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    ranked = series.rank(method="first")
    try:
        return pd.qcut(ranked, q=len(labels), labels=labels)
    except ValueError:
        return pd.Series(["unbucketed"] * len(series), index=series.index)


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
        rows.append(
            {
                "split": split,
                "group_name": group_name,
                "group_bucket": str(bucket),
                "num_episodes": int(len(group)),
                "preferred_mean_final_value": group["preferred_value"].mean(),
                "hold_to_terminal_mean_final_value": group[
                    "hold_to_terminal_value"
                ].mean(),
                "sell_immediately_mean_final_value": group[
                    "sell_immediately_value"
                ].mean(),
                "sell_half_then_hold_mean_final_value": group[
                    "sell_half_then_hold_value"
                ].mean(),
                "preferred_minus_hold_mean": pref_minus_hold.mean(),
                "preferred_minus_sell_immediately_mean": pref_minus_sell.mean(),
                "preferred_minus_sell_half_mean": pref_minus_half.mean(),
                "preferred_win_rate_vs_hold": pref_minus_hold.gt(0).mean(),
                "preferred_win_rate_vs_sell_immediately": pref_minus_sell.gt(0).mean(),
                "preferred_win_rate_vs_sell_half": pref_minus_half.gt(0).mean(),
                "preferred_mean_tax_paid": group["preferred_tax_paid"].mean(),
                "preferred_mean_short_term_sold_fraction": group[
                    "preferred_short_term"
                ].mean(),
                "preferred_no_cut_pct": group["preferred_no_cut"].mean(),
            }
        )
    return rows


def build_step9(
    episode_df: pd.DataFrame,
    step_df: pd.DataFrame,
    output_dir: Path,
    plots_dir: Path,
    registry: OutputRegistry,
) -> pd.DataFrame:
    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    policies = [PREFERRED_POLICY, "hold_to_terminal", "sell_immediately", "sell_half_then_hold"]
    for split in VALIDATION_TEST_SPLITS:
        split_metrics = episode_df[
            episode_df["split"].eq(split) & episode_df["policy_name"].isin(policies)
        ].copy()
        wide = split_metrics.pivot(
            index="episode_id",
            columns="policy_name",
            values="episode_final_after_tax_total_value",
        ).reset_index()
        wide = wide.rename(
            columns={
                PREFERRED_POLICY: "preferred_value",
                "hold_to_terminal": "hold_to_terminal_value",
                "sell_immediately": "sell_immediately_value",
                "sell_half_then_hold": "sell_half_then_hold_value",
            }
        )
        pref_extra = split_metrics[split_metrics["policy_name"].eq(PREFERRED_POLICY)][
            [
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
        base = wide.merge(pref_extra, on="episode_id", how="inner")
        base["preferred_no_cut"] = ~base["preferred_cut_occurred"]
        base["final_episode_return_bucket"] = quantile_bucket(
            base["preferred_value"],
            ["low", "mid", "high"],
        )
        rows.extend(
            build_cross_section_group_rows(
                base,
                "final_episode_return_bucket",
                "final_episode_return_bucket",
                split,
            )
        )

        pref_steps = step_df[
            step_df["split"].eq(split) & step_df["policy_name"].eq(PREFERRED_POLICY)
        ].copy()
        path_stats = (
            pref_steps.groupby("episode_id")
            .agg(
                first_value=("after_tax_total_value", "first"),
                max_value=("after_tax_total_value", "max"),
                min_value=("after_tax_total_value", "min"),
                first_date=("date", "first"),
            )
            .reset_index()
        )
        path_stats["max_gain_proxy"] = (
            path_stats["max_value"] - path_stats["first_value"]
        )
        path_stats["drawdown_after_start_proxy"] = (
            path_stats["first_value"] - path_stats["min_value"]
        )
        path_stats["calendar_year"] = pd.to_datetime(
            path_stats["first_date"],
            errors="coerce",
        ).dt.year.astype("Int64")
        base_path = base.merge(path_stats, on="episode_id", how="left")
        if base_path["max_gain_proxy"].notna().any():
            notes.append(
                f"{split}: maximum gain bucket computed from max after_tax_total_value minus first after_tax_total_value proxy."
            )
            base_path["maximum_gain_bucket"] = quantile_bucket(
                base_path["max_gain_proxy"],
                ["low", "mid", "high"],
            )
            rows.extend(
                build_cross_section_group_rows(
                    base_path,
                    "maximum_gain_bucket",
                    "maximum_gain_bucket",
                    split,
                )
            )
        else:
            notes.append(f"{split}: maximum gain bucket skipped; no path proxy available.")
        if base_path["drawdown_after_start_proxy"].notna().any():
            notes.append(
                f"{split}: drawdown bucket computed from first after_tax_total_value minus minimum after_tax_total_value proxy."
            )
            base_path["drawdown_after_episode_start_bucket"] = quantile_bucket(
                base_path["drawdown_after_start_proxy"],
                ["low", "mid", "high"],
            )
            rows.extend(
                build_cross_section_group_rows(
                    base_path,
                    "drawdown_after_episode_start_bucket",
                    "drawdown_after_episode_start_bucket",
                    split,
                )
            )
        else:
            notes.append(f"{split}: drawdown bucket skipped; no path proxy available.")
        base["days_to_first_sale_bucket"] = pd.cut(
            base["days_to_first_sale"],
            bins=[-np.inf, 0, 30, 180, np.inf],
            labels=["immediate", "1_to_30_days", "31_to_180_days", "over_180_days"],
        ).astype(object)
        base["days_to_first_sale_bucket"] = base["days_to_first_sale_bucket"].fillna(
            "no_discretionary_sale"
        )
        rows.extend(
            build_cross_section_group_rows(
                base,
                "days_to_first_sale_bucket",
                "days_to_first_sale_bucket",
                split,
            )
        )
        if base_path["calendar_year"].notna().any():
            rows.extend(
                build_cross_section_group_rows(
                    base_path,
                    "calendar_year",
                    "calendar_year",
                    split,
                )
            )
        else:
            notes.append(f"{split}: calendar year skipped; no valid date available.")
        notes.append(f"{split}: sector grouping skipped; no sector column exists in baseline artifacts.")

    out = pd.DataFrame(rows)
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
    plot_df = out[
        out["split"].eq("test") & out["group_name"].eq("final_episode_return_bucket")
    ]
    plt.figure(figsize=(7, 4.5))
    plt.bar(plot_df["group_bucket"], plot_df["preferred_minus_hold_mean"], color="#3b6ea8")
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("preferred minus hold mean final value")
    plt.title("Preferred minus hold by final value bucket - test")
    plt.grid(axis="y", alpha=0.25)
    save_plot(
        plots_dir / "step9_preferred_minus_hold_by_final_value_bucket_test.png",
        registry,
        "9",
        "Preferred minus hold by final value bucket for test split.",
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
        "source_files_used: runs/train_reward_c_lite_v5_full/baselines/baseline_episode_metrics.csv; runs/train_reward_c_lite_v5_full/baselines/baseline_step_rollouts.csv; runs/train_reward_c_lite_v5_full/baselines/baseline_summary_by_policy.csv; runs/train_reward_c_lite_v5_full/episode_splits.csv; runs/train_reward_c_lite_v5_full/quant_analysis/step3b_policy_eaat_sharpe_metrics.csv; runs/train_reward_c_lite_v5_full/quant_analysis/step3b_eaat_sharpe_metrics_notes.txt",
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
    build_step8(baseline_episode, output_dir, plots_dir, registry)
    build_step9(baseline_episode, baseline_step, output_dir, plots_dir, registry)
    build_step10(baseline_episode, output_dir, plots_dir, registry)
    build_step11(baseline_episode, baseline_step, output_dir, plots_dir, registry)
    skipped_optional = [
        "sector grouping skipped because baseline artifacts do not include a sector column",
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
