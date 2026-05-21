"""Build quantitative analysis phase 1-3 outputs for the frozen final run.

Run from the project root:
    python scripts/build_quant_analysis_phase_1_3.py --config configs/train_reward_c_lite_v5.yaml
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError(
        "PyYAML is required to run scripts/build_quant_analysis_phase_1_3.py. "
        "Install it with: pip install pyyaml"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_reward_c_lite_v5.yaml"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "train_reward_c_lite_v5_full" / "quant_analysis"
)
FINAL_RUN_NAME = "train_reward_c_lite_v5_full"
PREFERRED_POLICY = "trained_dqn_first_sale_margin_0p070_normal_0p020"
SPLIT_ORDER = ["validation", "test"]
ANNUALIZATION_FACTOR = 252
SHARPE_ANNUALIZATION_FACTOR = math.sqrt(float(ANNUALIZATION_FACTOR))
ANNUAL_RISK_FREE_RATE = 0.04
DAILY_RISK_FREE_RATE = (1.0 + ANNUAL_RISK_FREE_RATE) ** (
    1.0 / float(ANNUALIZATION_FACTOR)
) - 1.0
RISK_FREE_RATE_SOURCE = "constant_annual_4_percent_thesis_assumption"
CASH_REINVESTMENT_ASSUMPTION = (
    "liquidated_after_tax_proceeds_earn_daily_risk_free_rate"
)
TAX_FREE_COUNTERFACTUAL_PROFILE = {
    "profile_name": "tax_free_counterfactual",
    "short_term_rate": 0.0,
    "long_term_rate": 0.0,
    "niit_rate": 0.0,
    "apply_niit": False,
}
TAX_FREE_COUNTERFACTUAL_DIR_NAME = "tax_free_counterfactual"
SAVE_TAX_FREE_STEP_ROLLOUTS_FOR_ALL_POLICIES = False
TAX_FREE_STEP_ROLLOUT_POLICIES = {
    "hold_to_terminal",
    "sell_immediately",
    PREFERRED_POLICY,
    "trained_dqn_first_sale_margin_0p090_normal_0p020",
}
TIE_TOLERANCE = 1e-12

POLICY_UNIVERSE_ROWS = [
    {
        "policy_name": "hold_to_terminal",
        "policy_family": "benchmark",
        "role": "benchmark",
        "included_in_main_table": True,
        "notes": "Never sells before automatic terminal liquidation.",
    },
    {
        "policy_name": "sell_immediately",
        "policy_family": "benchmark",
        "role": "benchmark",
        "included_in_main_table": True,
        "notes": "Sells the full position at the first available decision step.",
    },
    {
        "policy_name": "sell_half_then_hold",
        "policy_family": "benchmark",
        "role": "benchmark",
        "included_in_main_table": True,
        "notes": "Sells half of the original position, then holds the remainder.",
    },
    {
        "policy_name": "sell_quarters_over_time",
        "policy_family": "benchmark",
        "role": "benchmark",
        "included_in_main_table": True,
        "notes": "Sells quarter-sized portions over the episode.",
    },
    {
        "policy_name": "random_policy",
        "policy_family": "benchmark",
        "role": "benchmark",
        "included_in_main_table": True,
        "notes": "Uses the evaluator random policy with deterministic seeded evaluation.",
    },
    {
        "policy_name": "trained_dqn_greedy",
        "policy_family": "raw_dqn",
        "role": "raw DQN",
        "included_in_main_table": True,
        "notes": "Greedy policy from the frozen C-lite v5 trained DQN.",
    },
    {
        "policy_name": "trained_dqn_thresholded_margin_0p020",
        "policy_family": "thresholded_dqn",
        "role": "thresholded DQN",
        "included_in_main_table": True,
        "notes": "DQN action is filtered by a 0.020 margin threshold.",
    },
    {
        "policy_name": "trained_dqn_first_sale_margin_0p060_normal_0p020",
        "policy_family": "first_sale_thresholded_dqn",
        "role": "sensitivity DQN",
        "included_in_main_table": True,
        "notes": "First sale requires a 0.060 margin; subsequent sales use 0.020.",
    },
    {
        "policy_name": PREFERRED_POLICY,
        "policy_family": "first_sale_thresholded_dqn",
        "role": "preferred DQN",
        "included_in_main_table": True,
        "notes": "Preferred final interpretation policy.",
    },
    {
        "policy_name": "trained_dqn_first_sale_margin_0p080_normal_0p020",
        "policy_family": "first_sale_thresholded_dqn",
        "role": "sensitivity DQN",
        "included_in_main_table": True,
        "notes": "First sale requires a 0.080 margin; subsequent sales use 0.020.",
    },
    {
        "policy_name": "trained_dqn_first_sale_margin_0p090_normal_0p020",
        "policy_family": "first_sale_thresholded_dqn",
        "role": "sensitivity DQN",
        "included_in_main_table": True,
        "notes": "First sale requires a 0.090 margin; subsequent sales use 0.020.",
    },
]

MAIN_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    "mean_final_after_tax_total_value",
    "median_final_after_tax_total_value",
    "std_final_after_tax_total_value",
    "p25_final_after_tax_total_value",
    "p75_final_after_tax_total_value",
    "min_final_after_tax_total_value",
    "max_final_after_tax_total_value",
    "mean_realized_after_tax_pnl",
    "mean_total_tax_paid",
    "mean_effective_tax_rate",
    "mean_pct_position_sold_short_term",
    "mean_pct_position_sold_long_term",
    "terminal_liquidation_frequency",
    "no_cut_episode_count",
    "no_cut_episode_pct",
    "discretionary_sale_episode_count",
    "discretionary_sale_episode_pct",
    "mean_num_discretionary_sales",
    "average_days_to_first_sale",
    "median_days_to_first_sale",
    "pct_episodes_with_first_sale_before_tax_transition",
    "mean_excess_value_vs_hold_to_terminal",
    "mean_excess_value_vs_sell_immediately",
]

COMPACT_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    "mean_final_after_tax_total_value",
    "median_final_after_tax_total_value",
    "std_final_after_tax_total_value",
    "mean_total_tax_paid",
    "mean_effective_tax_rate",
    "terminal_liquidation_frequency",
    "no_cut_episode_pct",
    "discretionary_sale_episode_pct",
    "mean_excess_value_vs_hold_to_terminal",
    "mean_excess_value_vs_sell_immediately",
]

STEP3B_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    "annual_risk_free_rate",
    "daily_risk_free_rate",
    "risk_free_rate_source",
    "cash_reinvestment_assumption",
    "annualization_factor",
    "num_valid_full_horizon_sharpe_episodes",
    "num_valid_invested_period_sharpe_episodes",
    "mean_full_horizon_annualized_sharpe",
    "median_full_horizon_annualized_sharpe",
    "std_full_horizon_annualized_sharpe",
    "pct_positive_full_horizon_annualized_sharpe",
    "mean_invested_period_annualized_sharpe",
    "median_invested_period_annualized_sharpe",
    "std_invested_period_annualized_sharpe",
    "pct_positive_invested_period_annualized_sharpe",
    "mean_full_horizon_daily_volatility",
    "median_full_horizon_daily_volatility",
    "mean_invested_period_daily_volatility",
    "median_invested_period_daily_volatility",
    "mean_full_horizon_annualized_volatility",
    "median_full_horizon_annualized_volatility",
    "mean_invested_period_annualized_volatility",
    "median_invested_period_annualized_volatility",
]

STEP3B_COMPACT_COLUMNS = [
    "median_full_horizon_annualized_sharpe",
    "median_invested_period_annualized_sharpe",
]

LEGACY_COMPACT_SHARPE_COLUMNS = [
    "mean_episode_return",
    "std_episode_return",
    "sharpe_episode",
    "sortino_episode",
    "mean_episode_step_sharpe",
    "median_episode_step_sharpe",
    "std_episode_step_sharpe",
    "pct_positive_episode_step_sharpe",
    "pooled_step_sharpe",
    "step_return_proxy_source",
    "mean_step_return_proxy_all_steps",
    "std_step_return_proxy_all_steps",
]

STEP3C_COLUMNS = [
    "split",
    "bucket_scheme",
    "hold_terminal_bucket",
    "policy_name",
    "num_episodes",
    "mean_hold_terminal_final_value",
    "median_hold_terminal_final_value",
    "mean_final_after_tax_total_value",
    "median_final_after_tax_total_value",
    "std_final_after_tax_total_value",
    "mean_difference_vs_hold_to_terminal",
    "median_difference_vs_hold_to_terminal",
    "win_rate_vs_hold_to_terminal",
    "tie_rate_vs_hold_to_terminal",
    "loss_rate_vs_hold_to_terminal",
    "mean_difference_vs_sell_immediately",
    "mean_difference_vs_sell_half_then_hold",
    "mean_total_tax_paid",
    "mean_effective_tax_rate",
    "mean_pct_position_sold_short_term",
    "mean_pct_position_sold_long_term",
    "terminal_liquidation_frequency",
    "no_cut_episode_pct",
    "discretionary_sale_episode_pct",
    "average_days_to_first_sale",
    "pct_episodes_with_first_sale_before_tax_transition",
]

STEP3D_COMPARISON_COLUMNS = [
    "split",
    "policy_name",
    "mean_value_taxed",
    "mean_value_tax_free",
    "difference_tax_free_minus_taxed",
    "median_value_taxed",
    "median_value_tax_free",
    "median_difference_tax_free_minus_taxed",
    "mean_tax_paid_taxed",
    "mean_tax_paid_tax_free",
    "difference_tax_paid_tax_free_minus_taxed",
    "effective_tax_rate_taxed",
    "effective_tax_rate_tax_free",
    "rank_by_mean_value_taxed",
    "rank_by_mean_value_tax_free",
    "rank_change_tax_free_minus_taxed",
    "mean_paired_difference_tax_free_minus_taxed",
    "median_paired_difference_tax_free_minus_taxed",
    "win_rate_tax_free_vs_taxed",
]


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain a mapping at top level: {path}")
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
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command_text}")


def compact_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            text = ""
        elif isinstance(value, float):
            text = "" if math.isnan(value) else f"{value:.10g}"
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
        "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_key_value_markdown(row: dict[str, Any], path: Path) -> None:
    kv_df = pd.DataFrame(
        [{"field": key, "value": compact_json(value)} for key, value in row.items()]
    )
    write_markdown_table(kv_df, path)


def ensure_manifest(config_path: Path, run_path: Path) -> Path:
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        run_existing_script(
            [
                "scripts/write_run_manifest.py",
                "--config",
                relative_project_path(config_path),
            ]
        )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest after generation attempt: {manifest_path}")
    return manifest_path


def verify_frozen_artifacts(config_path: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    run_path = resolve_project_path(require_nested(config, "logging.output_dir"))
    if run_path.name != FINAL_RUN_NAME:
        raise ValueError(
            "This phase 1-3 package is frozen to "
            f"{FINAL_RUN_NAME!r}, but config logging.output_dir is {relative_project_path(run_path)!r}."
        )

    if not config_path.exists():
        raise FileNotFoundError(f"Missing final config: {config_path}")
    if not run_path.exists() or not run_path.is_dir():
        raise FileNotFoundError(f"Missing final run directory: {run_path}")

    manifest_path = ensure_manifest(config_path, run_path)
    final_model_path = run_path / "final_model.pt"
    best_validation_model_path = run_path / "best_validation_model.pt"
    if not final_model_path.exists() and not best_validation_model_path.exists():
        raise FileNotFoundError(
            "Missing trained model artifact. Expected at least one of: "
            f"{best_validation_model_path}, {final_model_path}"
        )

    episode_splits_path = run_path / "episode_splits.csv"
    if not episode_splits_path.exists():
        raise FileNotFoundError(f"Missing episode splits: {episode_splits_path}")

    return run_path, manifest_path


def build_reproducibility_row(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    tax_profile_values = manifest.get("tax_profile_values") or {}

    return {
        "run_name": manifest.get("run_name", require_nested(config, "run.name")),
        "run_path": relative_project_path(run_path),
        "config_path": manifest.get("config_path", relative_project_path(config_path)),
        "reward_version": manifest.get("reward_version", require_nested(config, "reward.version")),
        "base_reward_version": manifest.get(
            "base_reward_version",
            config.get("reward", {}).get("base_reward_version"),
        ),
        "dataset_path": manifest.get(
            "dataset_path",
            str(require_nested(config, "environment.parquet_path")),
        ),
        "state_schema_path": manifest.get(
            "state_schema_path",
            str(require_nested(config, "environment.state_schema_path")),
        ),
        "tax_profile_name": manifest.get("tax_profile_name", tax_profile_values.get("profile_name")),
        "tax_profile_values": compact_json(tax_profile_values),
        "short_term_rate": tax_profile_values.get("short_term_rate"),
        "long_term_rate": tax_profile_values.get("long_term_rate"),
        "niit_rate": tax_profile_values.get("niit_rate"),
        "apply_niit": tax_profile_values.get("apply_niit"),
        "discount_factor_gamma": manifest.get(
            "discount_factor_gamma",
            require_nested(config, "training.discount_factor_gamma"),
        ),
        "final_model_path": manifest.get("final_model_path", relative_project_path(run_path / "final_model.pt")),
        "best_validation_model_path": manifest.get(
            "best_validation_model_path",
            relative_project_path(run_path / "best_validation_model.pt"),
        ),
        "git_commit_hash": manifest.get("git_commit_hash"),
        "git_dirty_status": manifest.get("git_dirty_status"),
        "config_hash": manifest.get("config_hash"),
        "reward_config_hash": manifest.get("reward_config_hash"),
        "dataset_hash": manifest.get("dataset_hash"),
        "state_schema_hash": manifest.get("state_schema_hash"),
        "python_version": manifest.get("python_version"),
        "package_versions": compact_json(manifest.get("package_versions")),
    }


def write_step1_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    row = build_reproducibility_row(
        config=config,
        config_path=config_path,
        run_path=run_path,
        manifest_path=manifest_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(output_dir / "step1_reproducibility_table.csv", index=False)
    write_key_value_markdown(row, output_dir / "step1_reproducibility_table.md")
    metadata = {
        **row,
        "source_manifest_path": relative_project_path(manifest_path),
        "quant_analysis_output_dir": relative_project_path(output_dir),
    }
    with (output_dir / "step1_reproducibility_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return row


def policy_universe_df() -> pd.DataFrame:
    df = pd.DataFrame(POLICY_UNIVERSE_ROWS)
    df["policy_order"] = range(len(df))
    return df[
        [
            "policy_order",
            "policy_name",
            "policy_family",
            "role",
            "included_in_main_table",
            "notes",
        ]
    ]


def write_step2_outputs(output_dir: Path) -> pd.DataFrame:
    df = policy_universe_df()
    save_df = df.drop(columns=["policy_order"])
    save_df = save_df.assign(
        included_in_main_table=save_df["included_in_main_table"].map(
            {True: "true", False: "false"}
        )
    )
    save_df.to_csv(output_dir / "step2_policy_universe.csv", index=False)
    write_markdown_table(save_df, output_dir / "step2_policy_universe.md")
    return df


def required_baseline_paths(run_path: Path) -> list[Path]:
    baselines_dir = run_path / "baselines"
    return [
        baselines_dir / "baseline_episode_metrics.csv",
        baselines_dir / "baseline_step_rollouts.csv",
        baselines_dir / "baseline_summary_by_policy.csv",
        baselines_dir / "baseline_evaluation_summary.txt",
    ]


def ensure_baselines(config_path: Path, run_path: Path, required_policies: list[str]) -> None:
    missing_outputs = [path for path in required_baseline_paths(run_path) if not path.exists()]
    if missing_outputs:
        run_existing_script(
            [
                "scripts/evaluate_reward_a_baselines.py",
                "--config",
                relative_project_path(config_path),
            ]
        )

    summary_path = run_path / "baselines" / "baseline_summary_by_policy.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing baseline summary: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    missing_policies = sorted(set(required_policies) - set(summary_df["policy_name"].unique()))
    if missing_policies:
        run_existing_script(
            [
                "scripts/evaluate_reward_a_baselines.py",
                "--config",
                relative_project_path(config_path),
            ]
        )
        summary_df = pd.read_csv(summary_path)
        missing_policies = sorted(set(required_policies) - set(summary_df["policy_name"].unique()))
        if missing_policies:
            raise ValueError(
                "Required policies are missing from baseline outputs after regeneration: "
                + ", ".join(missing_policies)
            )


def require_columns(df: pd.DataFrame, columns: list[str], *, source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in {source}: {missing}")


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
        if pd.notna(value)
        else False
    )


def build_performance_summary(
    episode_metrics_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    *,
    source_path: Path,
) -> pd.DataFrame:
    required_columns = [
        "split",
        "policy_name",
        "episode_final_after_tax_total_value",
        "episode_realized_after_tax_pnl",
        "episode_terminal_liquidation_executed",
        "days_to_first_sale",
        "first_sale_before_tax_transition",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "mean_effective_tax_rate_on_sales",
        "total_tax_paid",
        "num_discretionary_sales",
    ]
    require_columns(episode_metrics_df, required_columns, source=source_path)

    policies = policy_df["policy_name"].tolist()
    metrics = episode_metrics_df[
        episode_metrics_df["split"].isin(SPLIT_ORDER)
        & episode_metrics_df["policy_name"].isin(policies)
    ].copy()
    missing_pairs = []
    for split in SPLIT_ORDER:
        split_policies = set(metrics.loc[metrics["split"].eq(split), "policy_name"].unique())
        for policy in policies:
            if policy not in split_policies:
                missing_pairs.append(f"{split}:{policy}")
    if missing_pairs:
        raise ValueError(
            "Missing required split/policy rows in baseline episode metrics: "
            + ", ".join(missing_pairs)
        )

    numeric_columns = [
        "episode_final_after_tax_total_value",
        "episode_realized_after_tax_pnl",
        "days_to_first_sale",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "mean_effective_tax_rate_on_sales",
        "total_tax_paid",
        "num_discretionary_sales",
    ]
    for column in numeric_columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

    if "episode_cut_occurred" in metrics.columns:
        cut_occurred = coerce_bool_series(metrics["episode_cut_occurred"])
        metrics["no_cut_episode"] = ~cut_occurred
    else:
        metrics["no_cut_episode"] = metrics["num_discretionary_sales"].fillna(0).eq(0)

    metrics["episode_terminal_liquidation_executed"] = coerce_bool_series(
        metrics["episode_terminal_liquidation_executed"]
    )
    metrics["first_sale_before_tax_transition"] = coerce_bool_series(
        metrics["first_sale_before_tax_transition"]
    )

    grouped = metrics.groupby(["split", "policy_name"], sort=False)
    summary = grouped.agg(
        num_episodes=("episode_final_after_tax_total_value", "size"),
        mean_final_after_tax_total_value=("episode_final_after_tax_total_value", "mean"),
        median_final_after_tax_total_value=("episode_final_after_tax_total_value", "median"),
        std_final_after_tax_total_value=("episode_final_after_tax_total_value", "std"),
        p25_final_after_tax_total_value=(
            "episode_final_after_tax_total_value",
            lambda series: series.quantile(0.25),
        ),
        p75_final_after_tax_total_value=(
            "episode_final_after_tax_total_value",
            lambda series: series.quantile(0.75),
        ),
        min_final_after_tax_total_value=("episode_final_after_tax_total_value", "min"),
        max_final_after_tax_total_value=("episode_final_after_tax_total_value", "max"),
        mean_realized_after_tax_pnl=("episode_realized_after_tax_pnl", "mean"),
        mean_total_tax_paid=("total_tax_paid", "mean"),
        mean_effective_tax_rate=("mean_effective_tax_rate_on_sales", "mean"),
        mean_pct_position_sold_short_term=("pct_episode_position_sold_short_term", "mean"),
        mean_pct_position_sold_long_term=("pct_episode_position_sold_long_term", "mean"),
        terminal_liquidation_frequency=("episode_terminal_liquidation_executed", "mean"),
        no_cut_episode_count=("no_cut_episode", "sum"),
        no_cut_episode_pct=("no_cut_episode", "mean"),
        mean_num_discretionary_sales=("num_discretionary_sales", "mean"),
        average_days_to_first_sale=("days_to_first_sale", "mean"),
        median_days_to_first_sale=("days_to_first_sale", "median"),
        pct_episodes_with_first_sale_before_tax_transition=(
            "first_sale_before_tax_transition",
            "mean",
        ),
    ).reset_index()

    summary["discretionary_sale_episode_count"] = (
        summary["num_episodes"] - summary["no_cut_episode_count"]
    )
    summary["discretionary_sale_episode_pct"] = 1.0 - summary["no_cut_episode_pct"]

    baseline_values = summary.pivot(
        index="split",
        columns="policy_name",
        values="mean_final_after_tax_total_value",
    )
    if "hold_to_terminal" not in baseline_values.columns:
        raise ValueError("Cannot compute excess value: hold_to_terminal is missing.")
    if "sell_immediately" not in baseline_values.columns:
        raise ValueError("Cannot compute excess value: sell_immediately is missing.")
    summary["mean_excess_value_vs_hold_to_terminal"] = (
        summary["mean_final_after_tax_total_value"]
        - summary["split"].map(baseline_values["hold_to_terminal"])
    )
    summary["mean_excess_value_vs_sell_immediately"] = (
        summary["mean_final_after_tax_total_value"]
        - summary["split"].map(baseline_values["sell_immediately"])
    )

    summary = summary.merge(
        policy_df[["policy_name", "policy_order"]],
        on="policy_name",
        how="left",
    )
    split_order_map = {split: index for index, split in enumerate(SPLIT_ORDER)}
    summary["split_order"] = summary["split"].map(split_order_map)
    summary = summary.sort_values(["split_order", "policy_order"]).drop(
        columns=["split_order", "policy_order"]
    )
    summary["no_cut_episode_count"] = summary["no_cut_episode_count"].astype(int)
    summary["discretionary_sale_episode_count"] = summary[
        "discretionary_sale_episode_count"
    ].astype(int)
    return summary[MAIN_COLUMNS]


def write_step3_outputs(run_path: Path, output_dir: Path, policy_df: pd.DataFrame) -> pd.DataFrame:
    episode_metrics_path = run_path / "baselines" / "baseline_episode_metrics.csv"
    if not episode_metrics_path.exists():
        raise FileNotFoundError(f"Missing baseline episode metrics: {episode_metrics_path}")
    episode_metrics_df = pd.read_csv(episode_metrics_path)
    summary_df = build_performance_summary(
        episode_metrics_df,
        policy_df,
        source_path=episode_metrics_path,
    )
    summary_df.to_csv(output_dir / "step3_policy_performance_summary.csv", index=False)
    write_markdown_table(summary_df, output_dir / "step3_policy_performance_summary.md")

    compact_df = summary_df[COMPACT_COLUMNS]
    compact_df.to_csv(
        output_dir / "step3_policy_performance_summary_compact.csv",
        index=False,
    )
    write_markdown_table(
        compact_df,
        output_dir / "step3_policy_performance_summary_compact.md",
    )
    return summary_df


def assign_hold_terminal_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["two_bucket_group"] = np.where(
        out["hold_terminal_final_value"].le(0.30),
        "hold_terminal_le_30pct",
        "hold_terminal_gt_30pct",
    )
    out["three_bucket_group"] = np.select(
        [
            out["hold_terminal_final_value"].le(0.30),
            out["hold_terminal_final_value"].gt(0.30)
            & out["hold_terminal_final_value"].le(0.60),
            out["hold_terminal_final_value"].gt(0.60),
        ],
        [
            "hold_terminal_le_30pct",
            "hold_terminal_30_to_60pct",
            "hold_terminal_gt_60pct",
        ],
        default="unassigned",
    )
    if out["three_bucket_group"].eq("unassigned").any():
        raise ValueError("Step 3C could not assign at least one episode to a bucket.")
    return out


def aggregate_step3c_bucket_metrics(
    metrics: pd.DataFrame,
    *,
    bucket_column: str,
    bucket_scheme: str,
) -> pd.DataFrame:
    grouped = metrics.groupby(["split", bucket_column, "policy_name"], sort=False)
    summary = grouped.agg(
        num_episodes=("episode_id", "nunique"),
        mean_hold_terminal_final_value=("hold_terminal_final_value", "mean"),
        median_hold_terminal_final_value=("hold_terminal_final_value", "median"),
        mean_final_after_tax_total_value=("episode_final_after_tax_total_value", "mean"),
        median_final_after_tax_total_value=(
            "episode_final_after_tax_total_value",
            "median",
        ),
        std_final_after_tax_total_value=("episode_final_after_tax_total_value", "std"),
        mean_difference_vs_hold_to_terminal=("difference_vs_hold_to_terminal", "mean"),
        median_difference_vs_hold_to_terminal=(
            "difference_vs_hold_to_terminal",
            "median",
        ),
        win_rate_vs_hold_to_terminal=("win_vs_hold_to_terminal", "mean"),
        tie_rate_vs_hold_to_terminal=("tie_vs_hold_to_terminal", "mean"),
        loss_rate_vs_hold_to_terminal=("loss_vs_hold_to_terminal", "mean"),
        mean_difference_vs_sell_immediately=(
            "difference_vs_sell_immediately",
            "mean",
        ),
        mean_difference_vs_sell_half_then_hold=(
            "difference_vs_sell_half_then_hold",
            "mean",
        ),
        mean_total_tax_paid=("total_tax_paid", "mean"),
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
        no_cut_episode_pct=("no_cut_episode", "mean"),
        discretionary_sale_episode_pct=("discretionary_sale_episode", "mean"),
        average_days_to_first_sale=("days_to_first_sale", "mean"),
        pct_episodes_with_first_sale_before_tax_transition=(
            "first_sale_before_tax_transition",
            "mean",
        ),
    ).reset_index()
    summary = summary.rename(columns={bucket_column: "hold_terminal_bucket"})
    summary.insert(1, "bucket_scheme", bucket_scheme)
    return summary


def validate_step3c_bucket_counts(
    metrics: pd.DataFrame,
    *,
    bucket_column: str,
    bucket_scheme: str,
) -> None:
    episode_buckets = metrics[["split", "episode_id", bucket_column]].drop_duplicates()
    duplicate_episode_assignments = episode_buckets.duplicated(
        ["split", "episode_id"],
        keep=False,
    )
    if duplicate_episode_assignments.any():
        raise ValueError(
            f"Step 3C {bucket_scheme} assigns at least one episode to multiple buckets."
        )

    split_episode_counts = metrics[["split", "episode_id"]].drop_duplicates()
    expected_counts = split_episode_counts.groupby("split").size()
    bucket_counts = episode_buckets.groupby("split").size()
    for split, expected_count in expected_counts.items():
        actual_count = int(bucket_counts.get(split, 0))
        if int(expected_count) != actual_count:
            raise ValueError(
                "Step 3C bucket counts do not sum to the full split episode count "
                f"for split={split}, scheme={bucket_scheme}: "
                f"expected={int(expected_count)}, actual={actual_count}."
            )


def write_step3c_summary(
    step3c_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    test_focus = step3c_df[
        step3c_df["split"].eq("test")
        & step3c_df["bucket_scheme"].eq("three_bucket")
        & step3c_df["hold_terminal_bucket"].eq("hold_terminal_le_30pct")
        & step3c_df["policy_name"].eq(PREFERRED_POLICY)
    ]
    focus_lines = []
    if not test_focus.empty:
        row = test_focus.iloc[0]
        focus_lines = [
            "test_preferred_dqn_le_30pct_num_episodes: "
            f"{int(row['num_episodes'])}",
            "test_preferred_dqn_le_30pct_mean_difference_vs_hold_to_terminal: "
            f"{row['mean_difference_vs_hold_to_terminal']}",
            "test_preferred_dqn_le_30pct_win_rate_vs_hold_to_terminal: "
            f"{row['win_rate_vs_hold_to_terminal']}",
            "test_preferred_dqn_le_30pct_mean_difference_vs_sell_immediately: "
            f"{row['mean_difference_vs_sell_immediately']}",
            "test_preferred_dqn_le_30pct_mean_difference_vs_sell_half_then_hold: "
            f"{row['mean_difference_vs_sell_half_then_hold']}",
        ]

    lines = [
        "Step 3C conditional hold-weak performance analysis",
        "This analysis is conditional on hold_to_terminal terminal outcome.",
        "The <=30% group is not selected based on DQN performance.",
        "It tests whether active liquidation helps when passive holding fails to preserve the initial +30% rally.",
        "It should be interpreted as a subgroup analysis, not as the primary result.",
        "The >30% group is included to avoid cherry-picking.",
        f"preferred_policy: {PREFERRED_POLICY}",
        "bucket_schemes: two_bucket, three_bucket",
        *focus_lines,
        "",
    ]
    (output_dir / "step3c_conditional_hold_weak_analysis_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_step3c_plots(step3c_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []
    bucket_order = [
        "hold_terminal_le_30pct",
        "hold_terminal_30_to_60pct",
        "hold_terminal_gt_60pct",
    ]
    label_map = {
        "hold_terminal_le_30pct": "<=30%",
        "hold_terminal_30_to_60pct": "30-60%",
        "hold_terminal_gt_60pct": ">60%",
    }
    test_three = step3c_df[
        step3c_df["split"].eq("test")
        & step3c_df["bucket_scheme"].eq("three_bucket")
    ].copy()
    if test_three.empty:
        return created_paths

    preferred_plot = test_three[
        test_three["policy_name"].isin(["hold_to_terminal", PREFERRED_POLICY])
    ]
    if not preferred_plot.empty:
        pivot = preferred_plot.pivot(
            index="hold_terminal_bucket",
            columns="policy_name",
            values="mean_final_after_tax_total_value",
        ).reindex(bucket_order)
        pivot = pivot.dropna(how="all")
        if not pivot.empty:
            ax = pivot.rename(index=label_map).plot(kind="bar", figsize=(8.5, 5.0))
            ax.set_xlabel("hold-to-terminal terminal value bucket")
            ax.set_ylabel("mean final after-tax total value")
            ax.set_title("Preferred DQN vs hold by hold-terminal bucket - test")
            ax.legend(loc="best", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            path = plots_dir / "step3c_test_preferred_vs_hold_by_hold_terminal_bucket.png"
            plt.savefig(path, dpi=160)
            plt.close()
            created_paths.append(path)

    selected_policies = [
        "hold_to_terminal",
        "sell_immediately",
        "sell_half_then_hold",
        PREFERRED_POLICY,
    ]
    selected_plot = test_three[test_three["policy_name"].isin(selected_policies)]
    if not selected_plot.empty:
        pivot = selected_plot.pivot(
            index="hold_terminal_bucket",
            columns="policy_name",
            values="mean_final_after_tax_total_value",
        ).reindex(bucket_order)
        pivot = pivot.dropna(how="all")
        if not pivot.empty:
            ax = pivot.rename(index=label_map).plot(kind="bar", figsize=(9.5, 5.2))
            ax.set_xlabel("hold-to-terminal terminal value bucket")
            ax.set_ylabel("mean final after-tax total value")
            ax.set_title("Policy values by hold-terminal bucket - test")
            ax.legend(loc="best", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            path = plots_dir / "step3c_test_policy_values_by_hold_terminal_bucket.png"
            plt.savefig(path, dpi=160)
            plt.close()
            created_paths.append(path)

    return created_paths


def build_step3c_conditional_hold_weak_analysis(
    run_path: Path,
    output_dir: Path,
    policy_df: pd.DataFrame,
) -> pd.DataFrame:
    episode_metrics_path = run_path / "baselines" / "baseline_episode_metrics.csv"
    if not episode_metrics_path.exists():
        raise FileNotFoundError(f"Missing baseline episode metrics: {episode_metrics_path}")

    episode_metrics_df = pd.read_csv(episode_metrics_path)
    required_columns = [
        "split",
        "policy_name",
        "episode_id",
        "episode_final_after_tax_total_value",
        "total_tax_paid",
        "mean_effective_tax_rate_on_sales",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "episode_terminal_liquidation_executed",
        "episode_cut_occurred",
        "num_discretionary_sales",
        "days_to_first_sale",
        "first_sale_before_tax_transition",
    ]
    require_columns(episode_metrics_df, required_columns, source=episode_metrics_path)

    policies = policy_df["policy_name"].tolist()
    metrics = episode_metrics_df[
        episode_metrics_df["split"].isin(SPLIT_ORDER)
        & episode_metrics_df["policy_name"].isin(policies)
    ].copy()
    validate_required_split_policy_pairs(metrics, policy_df, source_path=episode_metrics_path)

    numeric_columns = [
        "episode_final_after_tax_total_value",
        "total_tax_paid",
        "mean_effective_tax_rate_on_sales",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "num_discretionary_sales",
        "days_to_first_sale",
    ]
    for column in numeric_columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics["episode_terminal_liquidation_executed"] = coerce_bool_series(
        metrics["episode_terminal_liquidation_executed"]
    )
    metrics["episode_cut_occurred"] = coerce_bool_series(metrics["episode_cut_occurred"])
    metrics["first_sale_before_tax_transition"] = coerce_bool_series(
        metrics["first_sale_before_tax_transition"]
    )
    metrics["no_cut_episode"] = ~metrics["episode_cut_occurred"]
    metrics["discretionary_sale_episode"] = metrics["num_discretionary_sales"].fillna(0).gt(0)

    hold_reference = metrics[metrics["policy_name"].eq("hold_to_terminal")][
        ["split", "episode_id", "episode_final_after_tax_total_value"]
    ].copy()
    duplicate_hold_rows = hold_reference.duplicated(["split", "episode_id"], keep=False)
    if duplicate_hold_rows.any():
        raise ValueError(
            "Step 3C requires exactly one hold_to_terminal row per split/episode."
        )
    hold_reference = hold_reference.rename(
        columns={"episode_final_after_tax_total_value": "hold_terminal_final_value"}
    )

    split_episode_counts = metrics[["split", "episode_id"]].drop_duplicates()
    missing_hold = split_episode_counts.merge(
        hold_reference[["split", "episode_id"]],
        on=["split", "episode_id"],
        how="left",
        indicator=True,
    )
    missing_hold = missing_hold[missing_hold["_merge"].eq("left_only")]
    if not missing_hold.empty:
        raise ValueError(
            "Step 3C missing hold_to_terminal final value for at least one "
            "validation/test episode."
        )

    benchmark_values = metrics.pivot_table(
        index=["split", "episode_id"],
        columns="policy_name",
        values="episode_final_after_tax_total_value",
        aggfunc="first",
    )
    for benchmark in [
        "hold_to_terminal",
        "sell_immediately",
        "sell_half_then_hold",
    ]:
        if benchmark not in benchmark_values.columns:
            raise ValueError(f"Step 3C benchmark is missing: {benchmark}")
    benchmark_values = benchmark_values[
        ["hold_to_terminal", "sell_immediately", "sell_half_then_hold"]
    ].rename(
        columns={
            "hold_to_terminal": "paired_hold_to_terminal_value",
            "sell_immediately": "paired_sell_immediately_value",
            "sell_half_then_hold": "paired_sell_half_then_hold_value",
        }
    ).reset_index()

    metrics = metrics.merge(hold_reference, on=["split", "episode_id"], how="left")
    metrics = metrics.merge(benchmark_values, on=["split", "episode_id"], how="left")
    metrics = assign_hold_terminal_buckets(metrics)
    metrics["difference_vs_hold_to_terminal"] = (
        metrics["episode_final_after_tax_total_value"]
        - metrics["paired_hold_to_terminal_value"]
    )
    metrics["difference_vs_sell_immediately"] = (
        metrics["episode_final_after_tax_total_value"]
        - metrics["paired_sell_immediately_value"]
    )
    metrics["difference_vs_sell_half_then_hold"] = (
        metrics["episode_final_after_tax_total_value"]
        - metrics["paired_sell_half_then_hold_value"]
    )
    metrics["win_vs_hold_to_terminal"] = metrics[
        "difference_vs_hold_to_terminal"
    ].gt(TIE_TOLERANCE)
    metrics["tie_vs_hold_to_terminal"] = metrics[
        "difference_vs_hold_to_terminal"
    ].abs().le(TIE_TOLERANCE)
    metrics["loss_vs_hold_to_terminal"] = metrics[
        "difference_vs_hold_to_terminal"
    ].lt(-TIE_TOLERANCE)

    validate_step3c_bucket_counts(
        metrics,
        bucket_column="two_bucket_group",
        bucket_scheme="two_bucket",
    )
    validate_step3c_bucket_counts(
        metrics,
        bucket_column="three_bucket_group",
        bucket_scheme="three_bucket",
    )

    two_bucket = aggregate_step3c_bucket_metrics(
        metrics,
        bucket_column="two_bucket_group",
        bucket_scheme="two_bucket",
    )
    three_bucket = aggregate_step3c_bucket_metrics(
        metrics,
        bucket_column="three_bucket_group",
        bucket_scheme="three_bucket",
    )
    out = pd.concat([two_bucket, three_bucket], ignore_index=True)
    split_order_map = {split: index for index, split in enumerate(SPLIT_ORDER)}
    scheme_order_map = {"two_bucket": 0, "three_bucket": 1}
    bucket_order_map = {
        "hold_terminal_le_30pct": 0,
        "hold_terminal_30_to_60pct": 1,
        "hold_terminal_gt_30pct": 2,
        "hold_terminal_gt_60pct": 3,
    }
    out = out.merge(
        policy_df[["policy_name", "policy_order"]],
        on="policy_name",
        how="left",
    )
    out["_split_order"] = out["split"].map(split_order_map)
    out["_scheme_order"] = out["bucket_scheme"].map(scheme_order_map)
    out["_bucket_order"] = out["hold_terminal_bucket"].map(bucket_order_map)
    out = out.sort_values(
        ["_split_order", "_scheme_order", "_bucket_order", "policy_order"]
    ).drop(columns=["_split_order", "_scheme_order", "_bucket_order", "policy_order"])
    out["num_episodes"] = out["num_episodes"].astype(int)
    out = out[STEP3C_COLUMNS]

    out.to_csv(output_dir / "step3c_conditional_hold_weak_analysis.csv", index=False)
    write_markdown_table(out, output_dir / "step3c_conditional_hold_weak_analysis.md")
    write_step3c_summary(out, output_dir)
    write_step3c_plots(out, output_dir)
    return out


def build_conditional_hold_weak_table_from_episode_metrics(
    episode_metrics_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    *,
    source_path: Path,
) -> pd.DataFrame:
    required_columns = [
        "split",
        "policy_name",
        "episode_id",
        "episode_final_after_tax_total_value",
        "total_tax_paid",
        "mean_effective_tax_rate_on_sales",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "episode_terminal_liquidation_executed",
        "episode_cut_occurred",
        "num_discretionary_sales",
        "days_to_first_sale",
        "first_sale_before_tax_transition",
    ]
    require_columns(episode_metrics_df, required_columns, source=source_path)

    policies = policy_df["policy_name"].tolist()
    metrics = episode_metrics_df[
        episode_metrics_df["split"].isin(SPLIT_ORDER)
        & episode_metrics_df["policy_name"].isin(policies)
    ].copy()
    validate_required_split_policy_pairs(metrics, policy_df, source_path=source_path)

    numeric_columns = [
        "episode_final_after_tax_total_value",
        "total_tax_paid",
        "mean_effective_tax_rate_on_sales",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "num_discretionary_sales",
        "days_to_first_sale",
    ]
    for column in numeric_columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics["episode_terminal_liquidation_executed"] = coerce_bool_series(
        metrics["episode_terminal_liquidation_executed"]
    )
    metrics["episode_cut_occurred"] = coerce_bool_series(metrics["episode_cut_occurred"])
    metrics["first_sale_before_tax_transition"] = coerce_bool_series(
        metrics["first_sale_before_tax_transition"]
    )
    metrics["no_cut_episode"] = ~metrics["episode_cut_occurred"]
    metrics["discretionary_sale_episode"] = metrics["num_discretionary_sales"].fillna(0).gt(0)

    hold_reference = metrics[metrics["policy_name"].eq("hold_to_terminal")][
        ["split", "episode_id", "episode_final_after_tax_total_value"]
    ].copy()
    duplicate_hold_rows = hold_reference.duplicated(["split", "episode_id"], keep=False)
    if duplicate_hold_rows.any():
        raise ValueError(
            "Conditional hold-weak analysis requires exactly one hold_to_terminal "
            "row per split/episode."
        )
    hold_reference = hold_reference.rename(
        columns={"episode_final_after_tax_total_value": "hold_terminal_final_value"}
    )

    split_episode_counts = metrics[["split", "episode_id"]].drop_duplicates()
    missing_hold = split_episode_counts.merge(
        hold_reference[["split", "episode_id"]],
        on=["split", "episode_id"],
        how="left",
        indicator=True,
    )
    missing_hold = missing_hold[missing_hold["_merge"].eq("left_only")]
    if not missing_hold.empty:
        raise ValueError(
            "Conditional hold-weak analysis missing hold_to_terminal final value "
            "for at least one validation/test episode."
        )

    benchmark_values = metrics.pivot_table(
        index=["split", "episode_id"],
        columns="policy_name",
        values="episode_final_after_tax_total_value",
        aggfunc="first",
    )
    for benchmark in [
        "hold_to_terminal",
        "sell_immediately",
        "sell_half_then_hold",
    ]:
        if benchmark not in benchmark_values.columns:
            raise ValueError(f"Conditional hold-weak benchmark is missing: {benchmark}")
    benchmark_values = benchmark_values[
        ["hold_to_terminal", "sell_immediately", "sell_half_then_hold"]
    ].rename(
        columns={
            "hold_to_terminal": "paired_hold_to_terminal_value",
            "sell_immediately": "paired_sell_immediately_value",
            "sell_half_then_hold": "paired_sell_half_then_hold_value",
        }
    ).reset_index()

    metrics = metrics.merge(hold_reference, on=["split", "episode_id"], how="left")
    metrics = metrics.merge(benchmark_values, on=["split", "episode_id"], how="left")
    metrics = assign_hold_terminal_buckets(metrics)
    metrics["difference_vs_hold_to_terminal"] = (
        metrics["episode_final_after_tax_total_value"]
        - metrics["paired_hold_to_terminal_value"]
    )
    metrics["difference_vs_sell_immediately"] = (
        metrics["episode_final_after_tax_total_value"]
        - metrics["paired_sell_immediately_value"]
    )
    metrics["difference_vs_sell_half_then_hold"] = (
        metrics["episode_final_after_tax_total_value"]
        - metrics["paired_sell_half_then_hold_value"]
    )
    metrics["win_vs_hold_to_terminal"] = metrics[
        "difference_vs_hold_to_terminal"
    ].gt(TIE_TOLERANCE)
    metrics["tie_vs_hold_to_terminal"] = metrics[
        "difference_vs_hold_to_terminal"
    ].abs().le(TIE_TOLERANCE)
    metrics["loss_vs_hold_to_terminal"] = metrics[
        "difference_vs_hold_to_terminal"
    ].lt(-TIE_TOLERANCE)

    validate_step3c_bucket_counts(
        metrics,
        bucket_column="two_bucket_group",
        bucket_scheme="two_bucket",
    )
    validate_step3c_bucket_counts(
        metrics,
        bucket_column="three_bucket_group",
        bucket_scheme="three_bucket",
    )
    out = pd.concat(
        [
            aggregate_step3c_bucket_metrics(
                metrics,
                bucket_column="two_bucket_group",
                bucket_scheme="two_bucket",
            ),
            aggregate_step3c_bucket_metrics(
                metrics,
                bucket_column="three_bucket_group",
                bucket_scheme="three_bucket",
            ),
        ],
        ignore_index=True,
    )
    split_order_map = {split: index for index, split in enumerate(SPLIT_ORDER)}
    scheme_order_map = {"two_bucket": 0, "three_bucket": 1}
    bucket_order_map = {
        "hold_terminal_le_30pct": 0,
        "hold_terminal_30_to_60pct": 1,
        "hold_terminal_gt_30pct": 2,
        "hold_terminal_gt_60pct": 3,
    }
    out = out.merge(
        policy_df[["policy_name", "policy_order"]],
        on="policy_name",
        how="left",
    )
    out["_split_order"] = out["split"].map(split_order_map)
    out["_scheme_order"] = out["bucket_scheme"].map(scheme_order_map)
    out["_bucket_order"] = out["hold_terminal_bucket"].map(bucket_order_map)
    out = out.sort_values(
        ["_split_order", "_scheme_order", "_bucket_order", "policy_order"]
    ).drop(columns=["_split_order", "_scheme_order", "_bucket_order", "policy_order"])
    out["num_episodes"] = out["num_episodes"].astype(int)
    return out[STEP3C_COLUMNS]


def write_step3d_conditional_summary(
    step3d_conditional_df: pd.DataFrame,
    tax_free_output_dir: Path,
) -> None:
    test_focus = step3d_conditional_df[
        step3d_conditional_df["split"].eq("test")
        & step3d_conditional_df["bucket_scheme"].eq("three_bucket")
        & step3d_conditional_df["hold_terminal_bucket"].eq("hold_terminal_le_30pct")
        & step3d_conditional_df["policy_name"].eq(PREFERRED_POLICY)
    ]
    focus_lines = []
    if not test_focus.empty:
        row = test_focus.iloc[0]
        focus_lines = [
            "test_preferred_dqn_le_30pct_num_episodes: "
            f"{int(row['num_episodes'])}",
            "test_preferred_dqn_le_30pct_mean_difference_vs_hold_to_terminal: "
            f"{row['mean_difference_vs_hold_to_terminal']}",
            "test_preferred_dqn_le_30pct_win_rate_vs_hold_to_terminal: "
            f"{row['win_rate_vs_hold_to_terminal']}",
        ]

    lines = [
        "Step 3D tax-free conditional hold-weak analysis",
        "This is the Step 3C downside/hold-weak subgroup analysis repeated under the tax-free counterfactual.",
        "Buckets are based only on tax-free hold_to_terminal terminal outcomes, not DQN outcomes.",
        "The <=30% group tests active policy behavior when passive holding fails to preserve the initial +30% rally in the tax-free environment.",
        "This is a subgroup diagnostic, not the primary thesis result.",
        f"preferred_policy: {PREFERRED_POLICY}",
        *focus_lines,
        "",
    ]
    (
        tax_free_output_dir
        / "step3d_tax_free_conditional_hold_weak_analysis_summary.txt"
    ).write_text("\n".join(lines), encoding="utf-8")


def finite_positive_denominator(series: pd.Series) -> pd.Series:
    return series.notna() & np.isfinite(series) & series.gt(0.0)


def add_annualized_sharpe_columns(
    per_episode: pd.DataFrame,
    *,
    prefix: str,
    mean_column: str,
    std_column: str,
) -> None:
    count_column = f"{prefix}_num_valid_returns"
    sharpe_column = f"{prefix}_annualized_sharpe"
    volatility_column = f"{prefix}_annualized_volatility"

    valid_sharpe = (
        per_episode[count_column].fillna(0).ge(2)
        & per_episode[mean_column].notna()
        & finite_positive_denominator(per_episode[std_column])
    )
    per_episode[sharpe_column] = np.nan
    per_episode.loc[valid_sharpe, sharpe_column] = (
        SHARPE_ANNUALIZATION_FACTOR
        * per_episode.loc[valid_sharpe, mean_column]
        / per_episode.loc[valid_sharpe, std_column]
    )
    per_episode[volatility_column] = (
        per_episode[std_column] * SHARPE_ANNUALIZATION_FACTOR
    )


def pct_positive_nonmissing(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    return float(values.gt(0.0).mean())


def nonmissing_count(series: pd.Series) -> int:
    return int(series.notna().sum())


def validate_required_split_policy_pairs(
    df: pd.DataFrame,
    policy_df: pd.DataFrame,
    *,
    source_path: Path,
    expected_splits: list[str] | None = None,
) -> None:
    policies = policy_df["policy_name"].tolist()
    splits = SPLIT_ORDER if expected_splits is None else expected_splits
    missing_pairs = []
    for split in splits:
        split_policies = set(df.loc[df["split"].eq(split), "policy_name"].unique())
        for policy in policies:
            if policy not in split_policies:
                missing_pairs.append(f"{split}:{policy}")
    if missing_pairs:
        raise ValueError(
            "Missing required split/policy rows in "
            f"{source_path}: " + ", ".join(missing_pairs)
        )


def build_risk_adjusted_path_metrics(
    step_rollouts_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    *,
    source_path: Path,
    expected_splits: list[str] | None = None,
    terminal_steps_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required_columns = [
        "split",
        "policy_name",
        "episode_id",
        "step_in_episode",
        "after_tax_total_value",
        "remaining_fraction",
        "action_fraction_executed",
        "terminal_liquidation_executed",
    ]
    require_columns(step_rollouts_df, required_columns, source=source_path)

    splits = SPLIT_ORDER if expected_splits is None else expected_splits
    policies = policy_df["policy_name"].tolist()
    steps = step_rollouts_df[
        step_rollouts_df["split"].isin(splits)
        & step_rollouts_df["policy_name"].isin(policies)
    ].copy()
    validate_required_split_policy_pairs(
        steps,
        policy_df,
        source_path=source_path,
        expected_splits=splits,
    )

    numeric_columns = [
        "step_in_episode",
        "after_tax_total_value",
        "remaining_fraction",
    ]
    if "previous_after_tax_total_value" in steps.columns:
        numeric_columns.append("previous_after_tax_total_value")
    for column in numeric_columns:
        steps[column] = pd.to_numeric(steps[column], errors="coerce")

    if steps["step_in_episode"].isna().any():
        raise ValueError(
            "Cannot build Step 3B path metrics because step_in_episode contains "
            f"non-numeric or missing values in {source_path}."
        )

    group_keys = ["split", "policy_name", "episode_id"]
    steps = steps.sort_values(
        [*group_keys, "step_in_episode"],
        kind="mergesort",
    )

    steps["wealth"] = 1.0 + steps["after_tax_total_value"]
    if "previous_after_tax_total_value" in steps.columns:
        steps["previous_wealth"] = 1.0 + steps["previous_after_tax_total_value"]
    else:
        steps["previous_wealth"] = steps.groupby(group_keys, sort=False)["wealth"].shift(1)

    steps["previous_remaining_fraction"] = steps.groupby(group_keys, sort=False)[
        "remaining_fraction"
    ].shift(1)
    exposure_start = steps["previous_remaining_fraction"].where(
        steps["previous_remaining_fraction"].notna(),
        steps["remaining_fraction"],
    )

    valid_return = (
        steps["wealth"].notna()
        & steps["previous_wealth"].notna()
        & steps["wealth"].gt(0.0)
        & steps["previous_wealth"].gt(0.0)
    )
    steps["step_return"] = np.nan
    steps.loc[valid_return, "step_return"] = (
        steps.loc[valid_return, "wealth"]
        / steps.loc[valid_return, "previous_wealth"]
        - 1.0
    )
    steps.loc[~np.isfinite(steps["step_return"]), "step_return"] = np.nan
    steps["was_cash_at_interval_start"] = exposure_start.le(0.0)
    steps["adjusted_return"] = steps["step_return"]
    cash_return = valid_return & steps["was_cash_at_interval_start"]
    steps.loc[cash_return, "adjusted_return"] = DAILY_RISK_FREE_RATE
    steps["full_horizon_excess_return"] = (
        steps["adjusted_return"] - DAILY_RISK_FREE_RATE
    )
    steps["invested_period_return"] = valid_return & exposure_start.gt(0.0)
    steps["invested_period_excess_return"] = (
        steps["step_return"] - DAILY_RISK_FREE_RATE
    )

    if terminal_steps_df is None:
        terminal_steps = (
            steps.groupby(["split", "episode_id"], sort=False)["step_in_episode"]
            .max()
            .reset_index(name="terminal_step_in_episode")
        )
    else:
        require_columns(
            terminal_steps_df,
            ["split", "episode_id", "terminal_step_in_episode"],
            source=source_path,
        )
        terminal_steps = terminal_steps_df[
            terminal_steps_df["split"].isin(splits)
        ].copy()
        terminal_steps["terminal_step_in_episode"] = pd.to_numeric(
            terminal_steps["terminal_step_in_episode"],
            errors="coerce",
        )
    episode_last_rows = steps.groupby(group_keys, sort=False).tail(1)[
        [*group_keys, "step_in_episode", "remaining_fraction"]
    ]
    episode_last_rows = episode_last_rows.rename(
        columns={
            "step_in_episode": "last_observed_step_in_episode",
            "remaining_fraction": "final_observed_remaining_fraction",
        }
    )
    episode_end = episode_last_rows.merge(
        terminal_steps,
        on=["split", "episode_id"],
        how="left",
    )
    if episode_end["terminal_step_in_episode"].isna().any():
        raise ValueError(
            "Cannot build risk-adjusted path metrics because terminal step "
            f"metadata is missing for at least one episode in {source_path}."
        )
    episode_end["num_synthetic_cash_returns"] = (
        episode_end["terminal_step_in_episode"]
        - episode_end["last_observed_step_in_episode"]
    ).clip(lower=0)
    synthetic_cash_base = episode_end[
        episode_end["final_observed_remaining_fraction"].le(0.0)
        & episode_end["num_synthetic_cash_returns"].gt(0)
    ].copy()
    synthetic_cash_base["num_synthetic_cash_returns"] = synthetic_cash_base[
        "num_synthetic_cash_returns"
    ].astype(int)

    episode_index = (
        steps.groupby(group_keys, sort=False)
        .size()
        .reset_index(name="num_step_rows")
    )

    full_horizon_returns = steps[
        [*group_keys, "adjusted_return", "full_horizon_excess_return"]
    ].copy()
    if not synthetic_cash_base.empty:
        synthetic_cash_returns = synthetic_cash_base.loc[
            synthetic_cash_base.index.repeat(
                synthetic_cash_base["num_synthetic_cash_returns"]
            ),
            group_keys,
        ].reset_index(drop=True)
        synthetic_cash_returns["adjusted_return"] = DAILY_RISK_FREE_RATE
        synthetic_cash_returns["full_horizon_excess_return"] = 0.0
        full_horizon_returns = pd.concat(
            [full_horizon_returns, synthetic_cash_returns],
            ignore_index=True,
        )

    full_stats = (
        full_horizon_returns.groupby(group_keys, sort=False)
        .agg(
            full_horizon_num_valid_returns=("adjusted_return", "count"),
            full_horizon_daily_excess_mean=(
                "full_horizon_excess_return",
                "mean",
            ),
            full_horizon_daily_std=("adjusted_return", "std"),
        )
        .reset_index()
    )
    invested_stats = (
        steps[steps["invested_period_return"]]
        .groupby(group_keys, sort=False)
        .agg(
            invested_period_num_valid_returns=("step_return", "count"),
            invested_period_daily_excess_mean=(
                "invested_period_excess_return",
                "mean",
            ),
            invested_period_daily_std=("step_return", "std"),
        )
        .reset_index()
    )

    per_episode = episode_index.merge(full_stats, on=group_keys, how="left")
    per_episode = per_episode.merge(invested_stats, on=group_keys, how="left")
    for column in [
        "full_horizon_num_valid_returns",
        "invested_period_num_valid_returns",
    ]:
        per_episode[column] = per_episode[column].fillna(0).astype(int)

    add_annualized_sharpe_columns(
        per_episode,
        prefix="full_horizon",
        mean_column="full_horizon_daily_excess_mean",
        std_column="full_horizon_daily_std",
    )
    add_annualized_sharpe_columns(
        per_episode,
        prefix="invested_period",
        mean_column="invested_period_daily_excess_mean",
        std_column="invested_period_daily_std",
    )

    summary = (
        per_episode.groupby(["split", "policy_name"], sort=False)
        .agg(
            num_episodes=("episode_id", "nunique"),
            num_valid_full_horizon_sharpe_episodes=(
                "full_horizon_annualized_sharpe",
                nonmissing_count,
            ),
            num_valid_invested_period_sharpe_episodes=(
                "invested_period_annualized_sharpe",
                nonmissing_count,
            ),
            mean_full_horizon_annualized_sharpe=(
                "full_horizon_annualized_sharpe",
                "mean",
            ),
            median_full_horizon_annualized_sharpe=(
                "full_horizon_annualized_sharpe",
                "median",
            ),
            std_full_horizon_annualized_sharpe=(
                "full_horizon_annualized_sharpe",
                "std",
            ),
            pct_positive_full_horizon_annualized_sharpe=(
                "full_horizon_annualized_sharpe",
                pct_positive_nonmissing,
            ),
            mean_invested_period_annualized_sharpe=(
                "invested_period_annualized_sharpe",
                "mean",
            ),
            median_invested_period_annualized_sharpe=(
                "invested_period_annualized_sharpe",
                "median",
            ),
            std_invested_period_annualized_sharpe=(
                "invested_period_annualized_sharpe",
                "std",
            ),
            pct_positive_invested_period_annualized_sharpe=(
                "invested_period_annualized_sharpe",
                pct_positive_nonmissing,
            ),
            mean_full_horizon_daily_volatility=("full_horizon_daily_std", "mean"),
            median_full_horizon_daily_volatility=("full_horizon_daily_std", "median"),
            mean_invested_period_daily_volatility=("invested_period_daily_std", "mean"),
            median_invested_period_daily_volatility=(
                "invested_period_daily_std",
                "median",
            ),
            mean_full_horizon_annualized_volatility=(
                "full_horizon_annualized_volatility",
                "mean",
            ),
            median_full_horizon_annualized_volatility=(
                "full_horizon_annualized_volatility",
                "median",
            ),
            mean_invested_period_annualized_volatility=(
                "invested_period_annualized_volatility",
                "mean",
            ),
            median_invested_period_annualized_volatility=(
                "invested_period_annualized_volatility",
                "median",
            ),
        )
        .reset_index()
    )
    summary["annual_risk_free_rate"] = ANNUAL_RISK_FREE_RATE
    summary["daily_risk_free_rate"] = DAILY_RISK_FREE_RATE
    summary["risk_free_rate_source"] = RISK_FREE_RATE_SOURCE
    summary["cash_reinvestment_assumption"] = CASH_REINVESTMENT_ASSUMPTION
    summary["annualization_factor"] = ANNUALIZATION_FACTOR

    summary = summary.merge(
        policy_df[["policy_name", "policy_order"]],
        on="policy_name",
        how="left",
    )
    split_order_map = {split: index for index, split in enumerate(splits)}
    summary["split_order"] = summary["split"].map(split_order_map)
    summary = summary.sort_values(["split_order", "policy_order"]).drop(
        columns=["split_order", "policy_order"]
    )
    for column in [
        "num_episodes",
        "num_valid_full_horizon_sharpe_episodes",
        "num_valid_invested_period_sharpe_episodes",
    ]:
        summary[column] = summary[column].astype(int)
    return summary[STEP3B_COLUMNS]


def write_step3b_notes(path: Path) -> None:
    notes = [
        "Step 3B risk-adjusted path metrics notes",
        "A constant annual risk-free rate of 4% is assumed.",
        "The daily risk-free rate is computed as (1 + 0.04) ** (1 / 252) - 1.",
        "Annualization uses sqrt(252).",
        "After liquidation, realized after-tax proceeds are assumed to earn the daily risk-free rate until terminal date.",
        "If a rollout path ends at full liquidation before the episode terminal step, full-horizon diagnostics add daily cash intervals through the terminal step inferred from the episode rollout universe.",
        "Full-horizon Sharpe includes post-liquidation cash periods, but those periods earn the risk-free rate and therefore contribute approximately zero excess return.",
        "Invested-period Sharpe excludes post-liquidation flat periods and measures return efficiency only while the policy remains exposed.",
        "These are episode-level Sharpe-style diagnostics based on after-tax value paths, not classical live-portfolio Sharpe ratios from a fully investable daily portfolio return series.",
        "Final after-tax total value remains the primary thesis metric.",
        "",
    ]
    path.write_text("\n".join(notes), encoding="utf-8")


def build_step3b_outputs(run_path: Path, output_dir: Path, policy_df: pd.DataFrame) -> pd.DataFrame:
    step_rollouts_path = run_path / "baselines" / "baseline_step_rollouts.csv"
    if not step_rollouts_path.exists():
        raise FileNotFoundError(f"Missing baseline step rollouts: {step_rollouts_path}")
    step_rollouts_df = pd.read_csv(step_rollouts_path, low_memory=False)
    metrics_df = build_risk_adjusted_path_metrics(
        step_rollouts_df,
        policy_df,
        source_path=step_rollouts_path,
    )
    metrics_df.to_csv(
        output_dir / "step3b_policy_risk_adjusted_path_metrics.csv",
        index=False,
    )
    write_markdown_table(
        metrics_df,
        output_dir / "step3b_policy_risk_adjusted_path_metrics.md",
    )
    write_step3b_notes(output_dir / "step3b_risk_adjusted_path_metrics_notes.txt")
    return metrics_df


def update_step3_compact_with_step3b(output_dir: Path, step3b_df: pd.DataFrame) -> None:
    compact_csv = output_dir / "step3_policy_performance_summary_compact.csv"
    compact_md = output_dir / "step3_policy_performance_summary_compact.md"
    if not compact_csv.exists():
        raise FileNotFoundError(f"Missing Step 3 compact performance summary: {compact_csv}")

    compact_df = pd.read_csv(compact_csv)
    drop_columns = [
        column
        for column in [*LEGACY_COMPACT_SHARPE_COLUMNS, *STEP3B_COMPACT_COLUMNS]
        if column in compact_df.columns
    ]
    if drop_columns:
        compact_df = compact_df.drop(columns=drop_columns)

    compact_step3b = step3b_df[["split", "policy_name", *STEP3B_COMPACT_COLUMNS]]
    compact_df = compact_df.merge(compact_step3b, on=["split", "policy_name"], how="left")
    compact_df.to_csv(compact_csv, index=False)
    write_markdown_table(compact_df, compact_md)


def load_baseline_evaluator_module() -> Any:
    module_path = PROJECT_ROOT / "scripts" / "evaluate_reward_a_baselines.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_reward_a_baselines_for_step3d",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline evaluator module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tax_free_counterfactual_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    out = copy.deepcopy(config)
    out["tax_profile"] = dict(TAX_FREE_COUNTERFACTUAL_PROFILE)
    out.setdefault("logging", {})
    out["logging"]["output_dir"] = relative_project_path(output_dir)
    out.setdefault("notes", [])
    if isinstance(out["notes"], list):
        out["notes"] = [
            *out["notes"],
            "Tax-free counterfactual evaluation config generated by Step 3D.",
            "Uses frozen taxed-trained DQN; does not retrain or create a new model.",
        ]
    return out


def save_yaml_file(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def tax_free_intermediate_paths(
    intermediate_dir: Path,
    split_name: str,
    policy_name: str,
) -> tuple[Path, Path, Path]:
    safe_policy_name = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in policy_name
    )
    return (
        intermediate_dir / f"episode_metrics_{split_name}_{safe_policy_name}.csv",
        intermediate_dir / f"step_rollouts_{split_name}_{safe_policy_name}.csv",
        intermediate_dir / f"risk_adjusted_path_metrics_{split_name}_{safe_policy_name}.csv",
    )


def should_save_tax_free_step_rollouts(
    policy_name: str,
    *,
    episode_metrics_only: bool,
) -> bool:
    if episode_metrics_only:
        return False
    return (
        SAVE_TAX_FREE_STEP_ROLLOUTS_FOR_ALL_POLICIES
        or policy_name in TAX_FREE_STEP_ROLLOUT_POLICIES
    )


def load_reference_terminal_steps(run_path: Path) -> pd.DataFrame:
    step_rollouts_path = run_path / "baselines" / "baseline_step_rollouts.csv"
    if not step_rollouts_path.exists():
        raise FileNotFoundError(
            f"Missing baseline step rollouts for terminal step reference: {step_rollouts_path}"
        )
    columns = ["split", "policy_name", "episode_id", "step_in_episode"]
    steps = pd.read_csv(step_rollouts_path, usecols=columns, low_memory=False)
    steps = steps[
        steps["split"].isin(SPLIT_ORDER)
        & steps["policy_name"].eq("hold_to_terminal")
    ].copy()
    steps["step_in_episode"] = pd.to_numeric(
        steps["step_in_episode"],
        errors="coerce",
    )
    if steps["step_in_episode"].isna().any():
        raise ValueError(f"Invalid step_in_episode in {step_rollouts_path}")
    return (
        steps.groupby(["split", "episode_id"], sort=False)["step_in_episode"]
        .max()
        .reset_index(name="terminal_step_in_episode")
    )


def load_frozen_q_network_for_counterfactual(
    evaluator: Any,
    *,
    config: dict[str, Any],
    model_path: Path,
    obs_dim: int,
    num_actions: int,
    device: Any,
) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model artifact not found: {model_path}")

    hidden_layers = list(require_nested(config, "algorithm.policy_network.hidden_layers"))
    activation = str(require_nested(config, "algorithm.policy_network.activation"))
    expected_reward_version = str(require_nested(config, "reward.expected_info_reward_version"))
    q_net = evaluator.QNetwork(obs_dim, num_actions, hidden_layers, activation).to(device)

    try:
        checkpoint = evaluator.torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:  # pragma: no cover - for older torch versions.
        checkpoint = evaluator.torch.load(model_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected model artifact to contain a dict: {model_path}")
    required_keys = {"model_state_dict", "obs_dim", "num_actions", "reward_version"}
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(
            "Model artifact is missing required keys for Step 3D: "
            f"{sorted(missing_keys)}"
        )
    if checkpoint["reward_version"] != expected_reward_version:
        raise ValueError(
            f"Saved reward_version={checkpoint['reward_version']!r}; "
            f"expected {expected_reward_version!r}."
        )
    if int(checkpoint["obs_dim"]) != int(obs_dim):
        raise ValueError(
            f"Saved obs_dim={checkpoint['obs_dim']}; current obs_dim={obs_dim}."
        )
    if int(checkpoint["num_actions"]) != int(num_actions):
        raise ValueError(
            f"Saved num_actions={checkpoint['num_actions']}; "
            f"current num_actions={num_actions}."
        )

    q_net.load_state_dict(checkpoint["model_state_dict"])
    q_net.eval()
    with evaluator.torch.no_grad():
        dummy_obs = evaluator.torch.zeros(
            (1, obs_dim),
            dtype=evaluator.torch.float32,
            device=device,
        )
        dummy_q_values = q_net(dummy_obs)
        if tuple(dummy_q_values.shape) != (1, num_actions):
            raise ValueError(
                f"Expected dummy Q-value shape (1, {num_actions}), got "
                f"{tuple(dummy_q_values.shape)}."
            )
        if not evaluator.torch.isfinite(dummy_q_values).all():
            raise ValueError("Loaded Q-network produced NaN or infinite values.")
    return q_net


def evaluate_tax_free_policy_universe(
    *,
    evaluator: Any,
    config: dict[str, Any],
    run_path: Path,
    tax_free_output_dir: Path,
    model_path: Path,
    policy_names: list[str],
    policy_df: pd.DataFrame,
    episode_metrics_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluator._validate_reward_config(config)
    env = evaluator.make_env(config)
    state_columns = evaluator.load_state_columns(
        resolve_project_path(require_nested(config, "environment.state_schema_path"))
    )
    obs_dim = len(state_columns)
    num_actions = len(env.action_fractions)
    for action_fraction in require_nested(config, "action_space.action_fractions"):
        evaluator.action_index_for_fraction(env, float(action_fraction))

    device = evaluator._resolve_device(str(require_nested(config, "training.device")))
    q_net = load_frozen_q_network_for_counterfactual(
        evaluator,
        config=config,
        model_path=model_path,
        obs_dim=obs_dim,
        num_actions=num_actions,
        device=device,
    )
    splits = evaluator.load_episode_splits(run_path / "episode_splits.csv")
    validation_ids = evaluator._apply_episode_cap(
        splits["validation"],
        evaluator.MAX_VALIDATION_EPISODES,
        "validation",
    )
    test_ids = evaluator._apply_episode_cap(
        splits["test"],
        evaluator.MAX_TEST_EPISODES,
        "test",
    )
    if not validation_ids:
        raise ValueError("No validation episodes selected for Step 3D.")
    if not test_ids:
        raise ValueError("No test episodes selected for Step 3D.")

    intermediate_dir = tax_free_output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    terminal_steps_df = load_reference_terminal_steps(run_path)
    episode_frames: list[pd.DataFrame] = []
    step_frames: list[pd.DataFrame] = []
    risk_frames: list[pd.DataFrame] = []
    expected_reward_version = str(require_nested(config, "reward.expected_info_reward_version"))
    for policy_name in policy_names:
        for split_name, episode_ids in (
            ("validation", validation_ids),
            ("test", test_ids),
        ):
            episode_cache_path, step_cache_path, risk_cache_path = tax_free_intermediate_paths(
                intermediate_dir,
                split_name,
                policy_name,
            )
            save_step_rollouts = should_save_tax_free_step_rollouts(
                policy_name,
                episode_metrics_only=episode_metrics_only,
            )
            if (
                episode_cache_path.exists()
                and risk_cache_path.exists()
                and (not save_step_rollouts or step_cache_path.exists())
            ):
                try:
                    episode_frame = pd.read_csv(episode_cache_path)
                    risk_frame = pd.read_csv(risk_cache_path)
                    cache_valid = (
                        len(episode_frame) == len(episode_ids)
                        and set(episode_frame["split"].dropna().unique()) == {split_name}
                        and set(episode_frame["policy_name"].dropna().unique())
                        == {policy_name}
                        and len(risk_frame) == 1
                        and set(risk_frame["split"].dropna().unique()) == {split_name}
                        and set(risk_frame["policy_name"].dropna().unique())
                        == {policy_name}
                    )
                    if cache_valid:
                        print(
                            "[Step 3D] reusing cached "
                            f"split={split_name} policy={policy_name}",
                            flush=True,
                        )
                        episode_frames.append(episode_frame)
                        risk_frames.append(risk_frame)
                        if save_step_rollouts:
                            step_frames.append(
                                pd.read_csv(step_cache_path, low_memory=False)
                            )
                        continue
                except Exception as exc:
                    print(
                        "[Step 3D] ignoring invalid cache "
                        f"split={split_name} policy={policy_name}: {exc}",
                        flush=True,
                    )

            start_time = time.perf_counter()
            print(
                f"[Step 3D] evaluating split={split_name} policy={policy_name}",
                flush=True,
            )
            rng_seed = 42 + (1000 * SPLIT_ORDER.index(split_name)) + policy_names.index(
                policy_name
            )
            rng = np.random.default_rng(rng_seed)
            policy_episode_rows, policy_step_rows = evaluator.evaluate_policy_on_episodes(
                env=env,
                policy_name=policy_name,
                episode_ids=episode_ids,
                split_name=split_name,
                expected_reward_version=expected_reward_version,
                rng=rng,
                q_net=q_net,
                device=device,
                policy_names=policy_names,
            )
            episode_frame = pd.DataFrame(
                policy_episode_rows,
                columns=evaluator.EPISODE_METRIC_COLUMNS,
            )
            episode_frame.to_csv(episode_cache_path, index=False)
            episode_frames.append(episode_frame)

            step_frame = pd.DataFrame(
                policy_step_rows,
                columns=evaluator.STEP_ROLLOUT_COLUMNS,
            )
            single_policy_df = policy_df[
                policy_df["policy_name"].eq(policy_name)
            ].copy()
            risk_frame = build_risk_adjusted_path_metrics(
                step_frame,
                single_policy_df,
                source_path=risk_cache_path,
                expected_splits=[split_name],
                terminal_steps_df=terminal_steps_df,
            )
            risk_frame.to_csv(risk_cache_path, index=False)
            risk_frames.append(risk_frame)

            if save_step_rollouts:
                step_frame.to_csv(step_cache_path, index=False)
                step_frames.append(step_frame)

            elapsed_seconds = time.perf_counter() - start_time
            print(
                "[Step 3D] completed "
                f"split={split_name} policy={policy_name} "
                f"episodes={len(episode_frame)} "
                f"step_rollouts_saved={save_step_rollouts} "
                f"seconds={elapsed_seconds:.1f}",
                flush=True,
            )

    episode_metrics_df = (
        pd.concat(episode_frames, ignore_index=True)
        if episode_frames
        else pd.DataFrame(columns=evaluator.EPISODE_METRIC_COLUMNS)
    )
    step_rollouts_df = (
        pd.concat(step_frames, ignore_index=True)
        if step_frames
        else pd.DataFrame(columns=evaluator.STEP_ROLLOUT_COLUMNS)
    )
    risk_metrics_df = (
        pd.concat(risk_frames, ignore_index=True)
        if risk_frames
        else pd.DataFrame(columns=STEP3B_COLUMNS)
    )
    return episode_metrics_df, step_rollouts_df, risk_metrics_df


def build_step3d_comparison(
    *,
    taxed_summary_df: pd.DataFrame,
    tax_free_summary_df: pd.DataFrame,
    taxed_episode_df: pd.DataFrame,
    tax_free_episode_df: pd.DataFrame,
    policy_df: pd.DataFrame,
) -> pd.DataFrame:
    taxed = taxed_summary_df[
        [
            "split",
            "policy_name",
            "mean_final_after_tax_total_value",
            "median_final_after_tax_total_value",
            "mean_total_tax_paid",
            "mean_effective_tax_rate",
        ]
    ].rename(
        columns={
            "mean_final_after_tax_total_value": "mean_value_taxed",
            "median_final_after_tax_total_value": "median_value_taxed",
            "mean_total_tax_paid": "mean_tax_paid_taxed",
            "mean_effective_tax_rate": "effective_tax_rate_taxed",
        }
    )
    tax_free = tax_free_summary_df[
        [
            "split",
            "policy_name",
            "mean_final_after_tax_total_value",
            "median_final_after_tax_total_value",
            "mean_total_tax_paid",
            "mean_effective_tax_rate",
        ]
    ].rename(
        columns={
            "mean_final_after_tax_total_value": "mean_value_tax_free",
            "median_final_after_tax_total_value": "median_value_tax_free",
            "mean_total_tax_paid": "mean_tax_paid_tax_free",
            "mean_effective_tax_rate": "effective_tax_rate_tax_free",
        }
    )
    comparison = taxed.merge(tax_free, on=["split", "policy_name"], how="inner")
    comparison["difference_tax_free_minus_taxed"] = (
        comparison["mean_value_tax_free"] - comparison["mean_value_taxed"]
    )
    comparison["median_difference_tax_free_minus_taxed"] = (
        comparison["median_value_tax_free"] - comparison["median_value_taxed"]
    )
    comparison["difference_tax_paid_tax_free_minus_taxed"] = (
        comparison["mean_tax_paid_tax_free"] - comparison["mean_tax_paid_taxed"]
    )
    comparison["rank_by_mean_value_taxed"] = comparison.groupby("split")[
        "mean_value_taxed"
    ].rank(ascending=False, method="min")
    comparison["rank_by_mean_value_tax_free"] = comparison.groupby("split")[
        "mean_value_tax_free"
    ].rank(ascending=False, method="min")
    comparison["rank_change_tax_free_minus_taxed"] = (
        comparison["rank_by_mean_value_tax_free"]
        - comparison["rank_by_mean_value_taxed"]
    )

    taxed_episode = taxed_episode_df[
        taxed_episode_df["split"].isin(SPLIT_ORDER)
        & taxed_episode_df["policy_name"].isin(policy_df["policy_name"])
    ][["split", "policy_name", "episode_id", "episode_final_after_tax_total_value"]]
    tax_free_episode = tax_free_episode_df[
        tax_free_episode_df["split"].isin(SPLIT_ORDER)
        & tax_free_episode_df["policy_name"].isin(policy_df["policy_name"])
    ][["split", "policy_name", "episode_id", "episode_final_after_tax_total_value"]]
    paired = taxed_episode.merge(
        tax_free_episode,
        on=["split", "policy_name", "episode_id"],
        suffixes=("_taxed", "_tax_free"),
        how="inner",
    )
    paired["paired_difference_tax_free_minus_taxed"] = (
        paired["episode_final_after_tax_total_value_tax_free"]
        - paired["episode_final_after_tax_total_value_taxed"]
    )
    paired_summary = (
        paired.groupby(["split", "policy_name"], sort=False)[
            "paired_difference_tax_free_minus_taxed"
        ]
        .agg(
            mean_paired_difference_tax_free_minus_taxed="mean",
            median_paired_difference_tax_free_minus_taxed="median",
            win_rate_tax_free_vs_taxed=lambda series: float(series.gt(TIE_TOLERANCE).mean()),
        )
        .reset_index()
    )
    comparison = comparison.merge(
        paired_summary,
        on=["split", "policy_name"],
        how="left",
    )

    comparison = comparison.merge(
        policy_df[["policy_name", "policy_order"]],
        on="policy_name",
        how="left",
    )
    split_order_map = {split: index for index, split in enumerate(SPLIT_ORDER)}
    comparison["_split_order"] = comparison["split"].map(split_order_map)
    comparison = comparison.sort_values(["_split_order", "policy_order"]).drop(
        columns=["_split_order", "policy_order"]
    )
    for column in [
        "rank_by_mean_value_taxed",
        "rank_by_mean_value_tax_free",
        "rank_change_tax_free_minus_taxed",
    ]:
        comparison[column] = comparison[column].astype(int)
    return comparison[STEP3D_COMPARISON_COLUMNS]


def write_step3d_notes(
    *,
    path: Path,
    run_path: Path,
    model_path: Path,
    episode_metrics_only: bool,
) -> None:
    if episode_metrics_only:
        rollout_scope = "No tax-free step rollouts are saved because --step3d-episode-metrics-only was used."
    elif SAVE_TAX_FREE_STEP_ROLLOUTS_FOR_ALL_POLICIES:
        rollout_scope = "Tax-free step rollouts are saved for every policy."
    else:
        rollout_scope = (
            "Tax-free step rollouts are saved only for: "
            + ", ".join(sorted(TAX_FREE_STEP_ROLLOUT_POLICIES))
            + ". Episode-level metrics are saved for every policy."
        )
    notes = [
        "Step 3D tax-free counterfactual notes",
        "This is a counterfactual evaluation using a zero-tax profile.",
        "The DQN is not retrained.",
        "The DQN remains the frozen model trained under the original taxed C-lite v5 setup.",
        "Tax-free results test whether policy rankings and behavior depend on tax burden.",
        "Since the model was trained under taxed rewards, this does not answer what a newly trained tax-free DQN would learn.",
        "It only answers how the existing frozen policy behaves when evaluated under a no-tax environment.",
        "DQN label: frozen taxed-trained DQN evaluated under tax-free environment.",
        f"frozen_run_used: {relative_project_path(run_path)}",
        f"model_used: {relative_project_path(model_path)}",
        f"step_rollout_scope: {rollout_scope}",
        "All Step 3D files are written under quant_analysis/tax_free_counterfactual; the original taxed baselines directory is not written by Step 3D.",
        "Original taxed results remain the primary thesis results.",
        "",
    ]
    path.write_text("\n".join(notes), encoding="utf-8")


def write_step3d_risk_adjusted_notes(
    *,
    path: Path,
    episode_metrics_only: bool,
) -> None:
    notes = [
        "Step 3D tax-free risk-adjusted path metrics notes",
        "This is the Step 3B episode-level risk-adjusted path diagnostic repeated under the tax-free counterfactual.",
        "A constant annual risk-free rate of 4% is assumed.",
        "The daily risk-free rate is computed as (1 + 0.04) ** (1 / 252) - 1.",
        "Annualization uses sqrt(252).",
        "After liquidation, realized after-tax proceeds are assumed to earn the daily risk-free rate until terminal date.",
        "Full-horizon Sharpe includes post-liquidation cash periods, but those periods earn the risk-free rate and therefore contribute approximately zero excess return.",
        "Invested-period Sharpe excludes post-liquidation cash periods and measures return efficiency only while the policy remains exposed.",
        "These are episode-level Sharpe-style diagnostics based on tax-free counterfactual after-tax value paths, not classical live-portfolio Sharpe ratios.",
        "The DQN is the frozen taxed-trained model; it is not retrained for this tax-free environment.",
        "risk_metric_source: per split/policy in-memory step rollouts cached as compact intermediate risk tables.",
        f"full_step_rollouts_saved: {not episode_metrics_only}",
        "Original taxed results remain the primary thesis results.",
        "",
    ]
    path.write_text("\n".join(notes), encoding="utf-8")


def write_step3d_tax_free_risk_outputs(
    risk_metrics_df: pd.DataFrame,
    tax_free_output_dir: Path,
    *,
    episode_metrics_only: bool,
) -> pd.DataFrame:
    out = risk_metrics_df.copy()
    if out.empty:
        out = pd.DataFrame(columns=STEP3B_COLUMNS)
    else:
        out = out[STEP3B_COLUMNS]
        policy_order = {
            policy: index
            for index, policy in enumerate(
                policy_universe_df()["policy_name"].tolist()
            )
        }
        split_order_map = {split: index for index, split in enumerate(SPLIT_ORDER)}
        out["_split_order"] = out["split"].map(split_order_map)
        out["_policy_order"] = out["policy_name"].map(policy_order)
        out = out.sort_values(["_split_order", "_policy_order"]).drop(
            columns=["_split_order", "_policy_order"]
        )

    out.to_csv(
        tax_free_output_dir / "step3d_tax_free_risk_adjusted_path_metrics.csv",
        index=False,
    )
    write_markdown_table(
        out,
        tax_free_output_dir / "step3d_tax_free_risk_adjusted_path_metrics.md",
    )
    write_step3d_risk_adjusted_notes(
        path=tax_free_output_dir / "step3d_tax_free_risk_adjusted_path_metrics_notes.txt",
        episode_metrics_only=episode_metrics_only,
    )
    return out


def write_step3d_tax_free_conditional_outputs(
    episode_metrics_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    tax_free_output_dir: Path,
    *,
    source_path: Path,
) -> pd.DataFrame:
    out = build_conditional_hold_weak_table_from_episode_metrics(
        episode_metrics_df,
        policy_df,
        source_path=source_path,
    )
    out.to_csv(
        tax_free_output_dir / "step3d_tax_free_conditional_hold_weak_analysis.csv",
        index=False,
    )
    write_markdown_table(
        out,
        tax_free_output_dir / "step3d_tax_free_conditional_hold_weak_analysis.md",
    )
    write_step3d_conditional_summary(out, tax_free_output_dir)
    return out


def build_step3d_tax_free_counterfactual(
    *,
    config: dict[str, Any],
    run_path: Path,
    output_dir: Path,
    policy_df: pd.DataFrame,
    taxed_summary_df: pd.DataFrame,
    episode_metrics_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, pd.DataFrame, pd.DataFrame]:
    tax_free_output_dir = output_dir / TAX_FREE_COUNTERFACTUAL_DIR_NAME
    tax_free_output_dir.mkdir(parents=True, exist_ok=True)
    tax_free_config = tax_free_counterfactual_config(config, tax_free_output_dir)
    tax_free_config_path = tax_free_output_dir / "tax_free_eval_config.yaml"
    save_yaml_file(tax_free_config, tax_free_config_path)

    evaluator = load_baseline_evaluator_module()
    best_validation_model_path = run_path / "best_validation_model.pt"
    final_model_path = run_path / "final_model.pt"
    model_path = (
        best_validation_model_path
        if best_validation_model_path.exists()
        else final_model_path
    )
    policy_names = policy_df["policy_name"].tolist()
    episode_metrics_path = (
        tax_free_output_dir / "tax_free_counterfactual_episode_metrics.csv"
    )
    step_rollouts_path = (
        tax_free_output_dir / "tax_free_counterfactual_step_rollouts.csv"
    )

    episode_metrics_df, step_rollouts_df, risk_metrics_df = evaluate_tax_free_policy_universe(
        evaluator=evaluator,
        config=tax_free_config,
        run_path=run_path,
        tax_free_output_dir=tax_free_output_dir,
        model_path=model_path,
        policy_names=policy_names,
        policy_df=policy_df,
        episode_metrics_only=episode_metrics_only,
    )
    validate_required_split_policy_pairs(
        episode_metrics_df,
        policy_df,
        source_path=episode_metrics_path,
    )
    episode_metrics_df.to_csv(episode_metrics_path, index=False)
    step_rollouts_df.to_csv(step_rollouts_path, index=False)
    tax_free_risk_df = write_step3d_tax_free_risk_outputs(
        risk_metrics_df,
        tax_free_output_dir,
        episode_metrics_only=episode_metrics_only,
    )
    tax_free_conditional_df = write_step3d_tax_free_conditional_outputs(
        episode_metrics_df,
        policy_df,
        tax_free_output_dir,
        source_path=episode_metrics_path,
    )

    tax_free_summary_df = build_performance_summary(
        episode_metrics_df,
        policy_df,
        source_path=episode_metrics_path,
    )
    tax_free_summary_df.to_csv(
        tax_free_output_dir / "step3d_tax_free_policy_performance_summary.csv",
        index=False,
    )
    write_markdown_table(
        tax_free_summary_df,
        tax_free_output_dir / "step3d_tax_free_policy_performance_summary.md",
    )

    taxed_episode_path = run_path / "baselines" / "baseline_episode_metrics.csv"
    taxed_episode_df = pd.read_csv(taxed_episode_path)
    comparison_df = build_step3d_comparison(
        taxed_summary_df=taxed_summary_df,
        tax_free_summary_df=tax_free_summary_df,
        taxed_episode_df=taxed_episode_df,
        tax_free_episode_df=episode_metrics_df,
        policy_df=policy_df,
    )
    comparison_df.to_csv(
        tax_free_output_dir / "step3d_tax_free_vs_taxed_comparison.csv",
        index=False,
    )
    write_markdown_table(
        comparison_df,
        tax_free_output_dir / "step3d_tax_free_vs_taxed_comparison.md",
    )
    write_step3d_notes(
        path=tax_free_output_dir / "step3d_tax_free_counterfactual_notes.txt",
        run_path=run_path,
        model_path=model_path,
        episode_metrics_only=episode_metrics_only,
    )

    max_abs_tax_paid = float(tax_free_summary_df["mean_total_tax_paid"].abs().max())
    if max_abs_tax_paid > 1e-10:
        raise ValueError(
            "Step 3D tax-free counterfactual has non-zero mean_total_tax_paid: "
            f"max_abs={max_abs_tax_paid}"
        )
    return (
        tax_free_summary_df,
        comparison_df,
        tax_free_output_dir,
        tax_free_risk_df,
        tax_free_conditional_df,
    )


def value_for(summary_df: pd.DataFrame, split: str, policy: str, column: str) -> float:
    values = summary_df.loc[
        summary_df["split"].eq(split) & summary_df["policy_name"].eq(policy),
        column,
    ]
    if values.empty:
        raise ValueError(f"Missing value for split={split}, policy={policy}, column={column}")
    return float(values.iloc[0])


def best_policy(summary_df: pd.DataFrame, split: str) -> str:
    split_df = summary_df[summary_df["split"].eq(split)]
    if split_df.empty:
        raise ValueError(f"No rows available for split={split}")
    best_idx = split_df["mean_final_after_tax_total_value"].idxmax()
    return str(split_df.loc[best_idx, "policy_name"])


def write_phase_summary(
    *,
    output_dir: Path,
    run_path: Path,
    config_path: Path,
    policy_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    step3b_df: pd.DataFrame,
    step3c_df: pd.DataFrame,
    step3d_tax_free_output_dir: Path | None,
    step3d_tax_free_summary_df: pd.DataFrame | None,
    step3d_comparison_df: pd.DataFrame | None,
    step3d_risk_df: pd.DataFrame | None,
    step3d_conditional_df: pd.DataFrame | None,
    step3d_skipped: bool,
    step3d_episode_metrics_only: bool,
) -> None:
    best_validation = best_policy(performance_df, "validation")
    best_test = best_policy(performance_df, "test")
    hold_validation_best = best_validation == "hold_to_terminal"
    hold_test_best = best_test == "hold_to_terminal"
    preferred_test_value = value_for(
        performance_df,
        "test",
        PREFERRED_POLICY,
        "mean_final_after_tax_total_value",
    )
    sell_immediate_test_value = value_for(
        performance_df,
        "test",
        "sell_immediately",
        "mean_final_after_tax_total_value",
    )
    sell_half_test_value = value_for(
        performance_df,
        "test",
        "sell_half_then_hold",
        "mean_final_after_tax_total_value",
    )

    step3d_lines = []
    if step3d_skipped:
        step3d_lines = [
            "step3d_tax_free_counterfactual_completed: False",
            "step3d_tax_free_counterfactual_skipped: True",
            "tax_free_counterfactual_output_dir: "
            f"{relative_project_path(output_dir / TAX_FREE_COUNTERFACTUAL_DIR_NAME)}",
            "step3d_warning: tax-free evaluation uses the frozen taxed-trained DQN and is not retrained when enabled.",
        ]
    else:
        if (
            step3d_tax_free_output_dir is None
            or step3d_tax_free_summary_df is None
            or step3d_comparison_df is None
            or step3d_risk_df is None
            or step3d_conditional_df is None
        ):
            raise ValueError("Step 3D summary inputs are missing while Step 3D is enabled.")
        step3d_lines = [
            "step3d_tax_free_counterfactual_completed: True",
            f"step3d_tax_free_rows_written: {len(step3d_tax_free_summary_df)}",
            f"step3d_comparison_rows_written: {len(step3d_comparison_df)}",
            "step3d_tax_free_risk_adjusted_path_metrics_completed: True",
            f"step3d_tax_free_risk_adjusted_rows_written: {len(step3d_risk_df)}",
            "step3d_tax_free_conditional_hold_weak_analysis_completed: True",
            f"step3d_tax_free_conditional_rows_written: {len(step3d_conditional_df)}",
            "tax_free_counterfactual_output_dir: "
            f"{relative_project_path(step3d_tax_free_output_dir)}",
            "step3d_episode_metrics_only: "
            f"{step3d_episode_metrics_only}",
            "step3d_warning: tax-free evaluation uses the frozen taxed-trained DQN and is not retrained.",
        ]

    lines = [
        "Quantitative Analysis Phase 1-3 Summary",
        f"final_run_used: {relative_project_path(run_path)}",
        f"config_used: {relative_project_path(config_path)}",
        f"num_policies_in_final_policy_universe: {len(policy_df)}",
        f"best_validation_policy_by_mean_final_after_tax_total_value: {best_validation}",
        f"best_test_policy_by_mean_final_after_tax_total_value: {best_test}",
        f"preferred_interpretation_policy: {PREFERRED_POLICY}",
        "hold_to_terminal_remains_strongest: "
        f"validation={hold_validation_best}, test={hold_test_best}",
        "preferred_dqn_beats_sell_immediately_on_test: "
        f"{preferred_test_value > sell_immediate_test_value}",
        "preferred_dqn_beats_sell_half_then_hold_on_test: "
        f"{preferred_test_value > sell_half_test_value}",
        "step3b_risk_adjusted_path_metrics_completed: True",
        f"step3b_rows_written: {len(step3b_df)}",
        f"step3b_annual_risk_free_rate: {ANNUAL_RISK_FREE_RATE}",
        f"step3b_daily_risk_free_rate: {DAILY_RISK_FREE_RATE}",
        f"step3b_cash_reinvestment_assumption: {CASH_REINVESTMENT_ASSUMPTION}",
        "step3b_outputs: step3b_policy_risk_adjusted_path_metrics.csv; "
        "step3b_policy_risk_adjusted_path_metrics.md; "
        "step3b_risk_adjusted_path_metrics_notes.txt",
        "step3_compact_includes_step3b_median_path_sharpe: True",
        "step3c_conditional_hold_weak_analysis_completed: True",
        f"step3c_rows_written: {len(step3c_df)}",
        "step3c_outputs: step3c_conditional_hold_weak_analysis.csv; "
        "step3c_conditional_hold_weak_analysis.md; "
        "step3c_conditional_hold_weak_analysis_summary.txt",
        *step3d_lines,
        "scope_note: This package implements quantitative analysis steps 1, 2, 3, Step 3B secondary path diagnostics, Step 3C subgroup diagnostics, and Step 3D tax-free counterfactual evaluation; steps 4 onward are not implemented here.",
        "",
    ]
    (output_dir / "phase_1_3_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build quantitative analysis phase 1-3 outputs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Final training config path. Defaults to "
            f"{relative_project_path(DEFAULT_CONFIG_PATH)}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Quantitative analysis output directory. Defaults to "
            f"{relative_project_path(DEFAULT_OUTPUT_DIR)}."
        ),
    )
    parser.add_argument(
        "--skip-step3d",
        action="store_true",
        help="Build Steps 1-3C and skip the slower Step 3D tax-free counterfactual.",
    )
    parser.add_argument(
        "--step3d-episode-metrics-only",
        action="store_true",
        help=(
            "For Step 3D, write episode metrics and summary outputs but no "
            "tax-free step rollouts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    output_dir = resolve_project_path(args.output_dir)
    config = load_yaml(config_path)

    run_path, manifest_path = verify_frozen_artifacts(config_path, config)
    if output_dir.resolve() != (run_path / "quant_analysis").resolve():
        raise ValueError(
            "Output directory must remain inside the frozen final run at "
            f"{relative_project_path(run_path / 'quant_analysis')}; got "
            f"{relative_project_path(output_dir)}."
        )

    repro_row = write_step1_outputs(
        config=config,
        config_path=config_path,
        run_path=run_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
    )
    policy_df = write_step2_outputs(output_dir)
    required_policies = policy_df["policy_name"].tolist()
    ensure_baselines(config_path, run_path, required_policies)
    performance_df = write_step3_outputs(run_path, output_dir, policy_df)
    step3b_df = build_step3b_outputs(run_path, output_dir, policy_df)
    update_step3_compact_with_step3b(output_dir, step3b_df)
    step3c_df = build_step3c_conditional_hold_weak_analysis(
        run_path,
        output_dir,
        policy_df,
    )
    step3d_tax_free_summary_df: pd.DataFrame | None = None
    step3d_comparison_df: pd.DataFrame | None = None
    step3d_risk_df: pd.DataFrame | None = None
    step3d_conditional_df: pd.DataFrame | None = None
    step3d_tax_free_output_dir: Path | None = None
    if args.skip_step3d:
        print("[Step 3D] skipped by --skip-step3d", flush=True)
    else:
        (
            step3d_tax_free_summary_df,
            step3d_comparison_df,
            step3d_tax_free_output_dir,
            step3d_risk_df,
            step3d_conditional_df,
        ) = build_step3d_tax_free_counterfactual(
            config=config,
            run_path=run_path,
            output_dir=output_dir,
            policy_df=policy_df,
            taxed_summary_df=performance_df,
            episode_metrics_only=args.step3d_episode_metrics_only,
        )
    write_phase_summary(
        output_dir=output_dir,
        run_path=run_path,
        config_path=config_path,
        policy_df=policy_df,
        performance_df=performance_df,
        step3b_df=step3b_df,
        step3c_df=step3c_df,
        step3d_tax_free_output_dir=step3d_tax_free_output_dir,
        step3d_tax_free_summary_df=step3d_tax_free_summary_df,
        step3d_comparison_df=step3d_comparison_df,
        step3d_risk_df=step3d_risk_df,
        step3d_conditional_df=step3d_conditional_df,
        step3d_skipped=args.skip_step3d,
        step3d_episode_metrics_only=args.step3d_episode_metrics_only,
    )

    print(f"Wrote quantitative analysis phase 1-3 outputs to {relative_project_path(output_dir)}")
    print(f"git_dirty_status_from_manifest: {repro_row.get('git_dirty_status')}")


if __name__ == "__main__":
    main()
