"""Build Step 10 robustness/statistical significance outputs for the frozen run.

Run from the project root:
    python scripts/build_quant_analysis_step10.py --config configs/train_reward_c_lite_v5.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError(
        "PyYAML is required to run scripts/build_quant_analysis_step10.py. "
        "Install it with: pip install pyyaml"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_reward_c_lite_v5.yaml"
FINAL_RUN_NAME = "train_reward_c_lite_v5_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / FINAL_RUN_NAME / "quant_analysis"
SOURCE_RELATIVE_PATH = "baselines/baseline_episode_metrics.csv"

PREFERRED_POLICY = "trained_dqn_first_sale_margin_0p070_normal_0p020"
SENSITIVITY_POLICY = "trained_dqn_first_sale_margin_0p090_normal_0p020"
PRIMARY_SPLITS = ["validation", "test"]
MAIN_BENCHMARK_POLICIES = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "sell_quarters_over_time",
    "random_policy",
]
OPTIONAL_COMPARISONS = [
    (SENSITIVITY_POLICY, "hold_to_terminal", "sensitivity_0p090_vs_hold_to_terminal"),
    (PREFERRED_POLICY, SENSITIVITY_POLICY, "preferred_0p070_vs_sensitivity_0p090"),
]

BOOTSTRAP_REPLICATIONS = 10_000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95
CI_LOW_Q = (1.0 - CONFIDENCE_LEVEL) / 2.0
CI_HIGH_Q = 1.0 - CI_LOW_Q
WILCOXON_MIN_NONZERO = 10
RATE_TOLERANCE = 1e-10

OUTPUT_BASENAME = "step10_robustness_statistical_significance"
OUTPUT_COLUMNS = [
    "split",
    "policy_A",
    "policy_B",
    "comparison_label",
    "n_paired_episodes",
    "mean_difference",
    "mean_difference_ci_low",
    "mean_difference_ci_high",
    "median_difference",
    "median_difference_ci_low",
    "median_difference_ci_high",
    "win_rate",
    "win_rate_ci_low",
    "win_rate_ci_high",
    "tie_rate",
    "loss_rate",
    "wilcoxon_p_value",
    "statistical_interpretation",
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


def require_nested(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing required config value: {path}")
        current = current[key]
    return current


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


def verify_frozen_run(config_path: Path, config: dict[str, Any], output_dir: Path) -> Path:
    run_path = resolve_project_path(require_nested(config, "logging.output_dir"))
    if run_path.name != FINAL_RUN_NAME:
        raise ValueError(
            f"Expected frozen run directory {FINAL_RUN_NAME!r}, got "
            f"{relative_project_path(run_path)!r}."
        )
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {relative_project_path(config_path)}")
    if not run_path.exists():
        raise FileNotFoundError(f"Missing run directory: {relative_project_path(run_path)}")
    if output_dir.resolve() != (run_path / "quant_analysis").resolve():
        raise ValueError(
            "Output directory must be the frozen run quant_analysis directory: "
            f"{relative_project_path(run_path / 'quant_analysis')}"
        )
    return run_path


def normalize_episode_results(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    column_aliases = {
        "policy": "policy_name",
        "episode_final_after_tax_total_value": "final_after_tax_value",
        "final_after_tax_total_value": "final_after_tax_value",
    }
    out = df.rename(columns={k: v for k, v in column_aliases.items() if k in df.columns})
    required = ["split", "episode_id", "policy_name", "final_after_tax_value"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(
            "Step 10 requires missing column(s) in "
            f"{relative_project_path(source)}: {missing}"
        )
    out = out[required].copy()
    out["split"] = out["split"].astype(str)
    out["policy_name"] = out["policy_name"].astype(str)
    out["episode_id"] = out["episode_id"].astype(str)
    out["final_after_tax_value"] = pd.to_numeric(
        out["final_after_tax_value"], errors="coerce"
    )
    if out["final_after_tax_value"].isna().any():
        bad_count = int(out["final_after_tax_value"].isna().sum())
        raise ValueError(
            f"Step 10 found {bad_count} non-numeric final after-tax value row(s) in "
            f"{relative_project_path(source)}."
        )
    duplicate_count = int(out.duplicated(["split", "policy_name", "episode_id"]).sum())
    if duplicate_count:
        raise ValueError(
            "Step 10 paired comparisons require unique split/policy/episode rows; "
            f"found {duplicate_count} duplicates in {relative_project_path(source)}."
        )
    return out


def validate_required_inputs(episode_df: pd.DataFrame) -> None:
    missing_splits = sorted(set(PRIMARY_SPLITS) - set(episode_df["split"].unique()))
    if missing_splits:
        raise ValueError(
            "Step 10 is missing required primary split(s): " + ", ".join(missing_splits)
        )

    required_policies = {PREFERRED_POLICY, *MAIN_BENCHMARK_POLICIES}
    missing_policies = sorted(required_policies - set(episode_df["policy_name"].unique()))
    if missing_policies:
        raise ValueError(
            "Step 10 is missing required main policy row(s): "
            + ", ".join(missing_policies)
        )

    for split in PRIMARY_SPLITS:
        split_policies = set(
            episode_df.loc[episode_df["split"].eq(split), "policy_name"].unique()
        )
        missing = sorted(required_policies - split_policies)
        if missing:
            raise ValueError(
                f"Step 10 split={split} is missing required policy row(s): "
                + ", ".join(missing)
            )


def build_comparison_plan(episode_df: pd.DataFrame, notes: list[str]) -> list[tuple[str, str, str]]:
    available_policies = set(episode_df["policy_name"].unique())
    comparisons = [
        (PREFERRED_POLICY, policy_b, f"{PREFERRED_POLICY} vs {policy_b}")
        for policy_b in MAIN_BENCHMARK_POLICIES
    ]
    for policy_a, policy_b, label in OPTIONAL_COMPARISONS:
        if {policy_a, policy_b}.issubset(available_policies):
            comparisons.append((policy_a, policy_b, label))
        else:
            notes.append(
                "Optional comparison skipped because existing episode-level data are missing: "
                f"{policy_a} vs {policy_b}."
            )
    return comparisons


def paired_differences(
    episode_df: pd.DataFrame,
    *,
    split: str,
    policy_a: str,
    policy_b: str,
    notes: list[str],
) -> np.ndarray:
    a = episode_df[
        episode_df["split"].eq(split) & episode_df["policy_name"].eq(policy_a)
    ][["episode_id", "final_after_tax_value"]].rename(
        columns={"final_after_tax_value": "value_a"}
    )
    b = episode_df[
        episode_df["split"].eq(split) & episode_df["policy_name"].eq(policy_b)
    ][["episode_id", "final_after_tax_value"]].rename(
        columns={"final_after_tax_value": "value_b"}
    )
    merged = a.merge(b, on="episode_id", how="inner", validate="one_to_one")
    if len(merged) == 0:
        raise ValueError(
            f"Step 10 comparison has no paired episodes: split={split}, "
            f"policy_A={policy_a}, policy_B={policy_b}."
        )
    if len(merged) != len(a) or len(merged) != len(b):
        notes.append(
            "WARNING unmatched episodes: "
            f"split={split}; comparison={policy_a} vs {policy_b}; "
            f"episodes_policy_A={len(a)}; episodes_policy_B={len(b)}; "
            f"paired_episodes_used={len(merged)}."
        )
    return (merged["value_a"] - merged["value_b"]).to_numpy(dtype=float)


def bootstrap_difference_intervals(
    diff: np.ndarray,
    rng: np.random.Generator,
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    batch_size: int = 1_000,
) -> dict[str, tuple[float, float]]:
    if len(diff) == 0:
        return {
            "mean": (np.nan, np.nan),
            "median": (np.nan, np.nan),
            "win_rate": (np.nan, np.nan),
        }
    means = np.empty(replications, dtype=float)
    medians = np.empty(replications, dtype=float)
    win_rates = np.empty(replications, dtype=float)
    written = 0
    while written < replications:
        current = min(batch_size, replications - written)
        indices = rng.integers(0, len(diff), size=(current, len(diff)))
        samples = diff[indices]
        end = written + current
        means[written:end] = samples.mean(axis=1)
        medians[written:end] = np.median(samples, axis=1)
        win_rates[written:end] = (samples > 0).mean(axis=1)
        written = end

    return {
        "mean": (
            float(np.quantile(means, CI_LOW_Q)),
            float(np.quantile(means, CI_HIGH_Q)),
        ),
        "median": (
            float(np.quantile(medians, CI_LOW_Q)),
            float(np.quantile(medians, CI_HIGH_Q)),
        ),
        "win_rate": (
            float(np.quantile(win_rates, CI_LOW_Q)),
            float(np.quantile(win_rates, CI_HIGH_Q)),
        ),
    }


def wilcoxon_p_value(diff: np.ndarray, notes: list[str], context: str) -> float:
    nonzero = diff[diff != 0]
    if len(nonzero) < WILCOXON_MIN_NONZERO:
        notes.append(
            "Wilcoxon skipped: "
            f"{context}; nonzero paired differences={len(nonzero)} < {WILCOXON_MIN_NONZERO}."
        )
        return float("nan")
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        notes.append(f"Wilcoxon skipped: {context}; scipy is not installed.")
        return float("nan")
    try:
        return float(wilcoxon(nonzero).pvalue)
    except Exception as exc:  # pragma: no cover - depends on scipy edge cases.
        notes.append(f"Wilcoxon failed: {context}; {type(exc).__name__}: {exc}.")
        return float("nan")


def statistical_interpretation(mean_diff: float, ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "Significantly positive"
    if ci_high < 0:
        return "Significantly negative"
    if mean_diff > 0:
        return "Positive but not significant"
    if mean_diff < 0:
        return "Negative but not significant"
    return "Approximately zero / tied"


def validate_output(out: pd.DataFrame) -> None:
    if out.empty:
        raise ValueError("Step 10 generated an empty robustness table.")
    if list(out.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "Step 10 output columns do not match the required schema: "
            + ", ".join(out.columns)
        )
    if out["n_paired_episodes"].le(0).any():
        raise ValueError("Step 10 found a comparison with zero paired episodes.")
    finite_columns = [
        "mean_difference",
        "mean_difference_ci_low",
        "mean_difference_ci_high",
        "median_difference",
        "median_difference_ci_low",
        "median_difference_ci_high",
        "win_rate",
        "win_rate_ci_low",
        "win_rate_ci_high",
        "tie_rate",
        "loss_rate",
    ]
    for column in finite_columns:
        values = out[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Step 10 output contains non-finite values in {column}.")
    rate_sums = out[["win_rate", "tie_rate", "loss_rate"]].sum(axis=1)
    if not np.allclose(rate_sums, 1.0, atol=RATE_TOLERANCE):
        raise ValueError("Step 10 win/tie/loss rates do not sum to approximately 1.")


def build_step10_table(episode_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    comparisons = build_comparison_plan(episode_df, notes)
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for split in PRIMARY_SPLITS:
        for policy_a, policy_b, comparison_label in comparisons:
            diff = paired_differences(
                episode_df,
                split=split,
                policy_a=policy_a,
                policy_b=policy_b,
                notes=notes,
            )
            intervals = bootstrap_difference_intervals(diff, rng)
            mean_diff = float(np.mean(diff))
            median_diff = float(np.median(diff))
            win_rate = float(np.mean(diff > 0))
            tie_rate = float(np.mean(diff == 0))
            loss_rate = float(np.mean(diff < 0))
            mean_ci_low, mean_ci_high = intervals["mean"]
            median_ci_low, median_ci_high = intervals["median"]
            win_ci_low, win_ci_high = intervals["win_rate"]
            rows.append(
                {
                    "split": split,
                    "policy_A": policy_a,
                    "policy_B": policy_b,
                    "comparison_label": comparison_label,
                    "n_paired_episodes": int(len(diff)),
                    "mean_difference": mean_diff,
                    "mean_difference_ci_low": mean_ci_low,
                    "mean_difference_ci_high": mean_ci_high,
                    "median_difference": median_diff,
                    "median_difference_ci_low": median_ci_low,
                    "median_difference_ci_high": median_ci_high,
                    "win_rate": win_rate,
                    "win_rate_ci_low": win_ci_low,
                    "win_rate_ci_high": win_ci_high,
                    "tie_rate": tie_rate,
                    "loss_rate": loss_rate,
                    "wilcoxon_p_value": wilcoxon_p_value(
                        diff,
                        notes,
                        f"split={split}; comparison={policy_a} vs {policy_b}",
                    ),
                    "statistical_interpretation": statistical_interpretation(
                        mean_diff, mean_ci_low, mean_ci_high
                    ),
                }
            )
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    validate_output(out)
    return out, notes


def write_notes(
    path: Path,
    *,
    source_path: Path,
    row_count: int,
    splits: list[str],
    comparisons: list[str],
    notes: list[str],
) -> None:
    lines = [
        "Step 10 robustness / statistical significance notes",
        "Step 10 uses paired episode-level comparisons by episode_id.",
        "The primary metric is final after-tax value.",
        f"The preferred DQN policy is {PREFERRED_POLICY}.",
        "Bootstrap confidence intervals are computed from paired differences, not independent policy means.",
        f"Bootstrap settings: replications={BOOTSTRAP_REPLICATIONS}, seed={RANDOM_SEED}, confidence_level={CONFIDENCE_LEVEL:.0%}.",
        "Wilcoxon signed-rank tests are secondary diagnostics.",
        "Exact zero differences are removed before Wilcoxon tests.",
        "Test split is the primary thesis evidence.",
        "Validation split is supportive.",
        "Any all split rows are diagnostic only and not primary evidence.",
        "No all split rows were included because no existing episode-level all split rows were used.",
        "Training split rows are not used as primary statistical evidence and were not included in this Step 10 table.",
        "These tests are robustness diagnostics, not a new model-selection procedure.",
        "No retraining, reward redesign, or threshold reselection was performed.",
        "Win, tie, and loss rates are reported as proportions and sum to approximately 1.",
        f"source_episode_level_file: {relative_project_path(source_path)}",
        f"rows_written: {row_count}",
        "splits_included: " + ", ".join(splits),
        "comparisons_included: " + "; ".join(comparisons),
    ]
    if notes:
        lines.extend(["", "Warnings and diagnostics", *notes])
    else:
        lines.extend(["", "Warnings and diagnostics", "No unmatched episode warnings."])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def update_quant_manifest(output_dir: Path, output_paths: list[Path]) -> Path | None:
    manifest_path = output_dir / "phase_4_11_outputs_manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(
            f"Quant-analysis manifest must contain a list: {relative_project_path(manifest_path)}"
        )
    output_set = {relative_project_path(path) for path in output_paths}
    retained = [
        row
        for row in payload
        if not (isinstance(row, dict) and row.get("path") in output_set)
    ]
    descriptions = {
        "csv": "Step 10 paired robustness/statistical significance table.",
        "md": "Step 10 paired robustness/statistical significance table.",
        "txt": "Step 10 robustness/statistical significance notes.",
    }
    for path in output_paths:
        file_type = path.suffix.lstrip(".")
        retained.append(
            {
                "path": relative_project_path(path),
                "type": file_type,
                "step": "10",
                "description": descriptions.get(file_type, "Step 10 output."),
            }
        )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(retained, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Step 10 robustness/statistical significance outputs."
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
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(config_path)
    run_path = verify_frozen_run(config_path, config, output_dir)
    source_path = run_path / SOURCE_RELATIVE_PATH
    if not source_path.exists():
        raise FileNotFoundError(
            "Missing existing episode-level policy evaluation file: "
            f"{relative_project_path(source_path)}"
        )

    raw_episode = pd.read_csv(source_path, low_memory=False)
    episode_df = normalize_episode_results(raw_episode, source_path)
    validate_required_inputs(episode_df)
    table, notes = build_step10_table(episode_df)

    csv_path = output_dir / f"{OUTPUT_BASENAME}.csv"
    md_path = output_dir / f"{OUTPUT_BASENAME}.md"
    notes_path = output_dir / f"{OUTPUT_BASENAME}_notes.txt"
    table.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table_text(table), encoding="utf-8")
    write_notes(
        notes_path,
        source_path=source_path,
        row_count=len(table),
        splits=sorted(table["split"].unique(), key=PRIMARY_SPLITS.index),
        comparisons=table["comparison_label"].drop_duplicates().tolist(),
        notes=notes,
    )
    manifest_path = update_quant_manifest(output_dir, [csv_path, md_path, notes_path])

    print(f"Wrote {relative_project_path(csv_path)}")
    print(f"Wrote {relative_project_path(md_path)}")
    print(f"Wrote {relative_project_path(notes_path)}")
    if manifest_path is not None:
        print(f"Updated {relative_project_path(manifest_path)}")
    print(f"rows: {len(table)}")


if __name__ == "__main__":
    main()
