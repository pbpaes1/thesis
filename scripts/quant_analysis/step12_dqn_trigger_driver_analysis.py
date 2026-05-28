"""Build Step 12 DQN trigger-driver / state sensitivity diagnostics.

Run from the project root:
    python scripts/quant_analysis/step12_dqn_trigger_driver_analysis.py \
        --run-dir runs/train_reward_c_lite_v5_full \
        --preferred-policy trained_dqn_first_sale_margin_0p070_normal_0p020
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError("PyTorch is required for Step 12 Q-value analysis.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FINAL_RUN_NAME = "train_reward_c_lite_v5_full"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / FINAL_RUN_NAME
DEFAULT_PREFERRED_POLICY = "trained_dqn_first_sale_margin_0p070_normal_0p020"
PRIMARY_SPLITS = ["validation", "test"]
FIRST_SALE_MARGIN = 0.070
NORMAL_MARGIN = 0.020
RANDOM_SEED = 42
BATCH_SIZE = 8192
TOP_N = 15
ACTION_LABELS = {
    0.0: "hold",
    0.25: "sell_25",
    0.5: "sell_50",
    0.75: "sell_75",
    1.0: "sell_100",
}
OUTPUTS = {
    "feature_summary_csv": "step12_dqn_trigger_driver_feature_summary.csv",
    "feature_summary_md": "step12_dqn_trigger_driver_feature_summary.md",
    "event_summary_csv": "step12_dqn_trigger_event_summary.csv",
    "event_summary_md": "step12_dqn_trigger_event_summary.md",
    "correlations_csv": "step12_dqn_q_margin_feature_correlations.csv",
    "correlations_md": "step12_dqn_q_margin_feature_correlations.md",
    "permutation_csv": "step12_dqn_permutation_sensitivity.csv",
    "permutation_md": "step12_dqn_permutation_sensitivity.md",
    "group_csv": "step12_dqn_trigger_driver_group_summary.csv",
    "group_md": "step12_dqn_trigger_driver_group_summary.md",
    "notes": "step12_dqn_trigger_driver_notes.txt",
}
PLOT_FILES = [
    "step12_top_q_margin_correlations_test.png",
    "step12_top_trigger_vs_hold_feature_differences_test.png",
    "step12_top_permutation_sensitivity_test.png",
    "step12_q_margin_distribution_by_action_test.png",
    "step12_q_margin_by_days_until_tax_transition_test.png",
]


def relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_evaluator_module() -> Any:
    module_path = PROJECT_ROOT / "scripts" / "evaluate_reward_a_baselines.py"
    spec = importlib.util.spec_from_file_location("evaluate_reward_a_baselines", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import evaluator module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def save_table(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table_text(df), encoding="utf-8")


def load_state_schema(schema_path: Path) -> list[str]:
    with schema_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    columns = payload.get("allowed_state_columns") if isinstance(payload, dict) else payload
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        raise ValueError(f"Invalid state schema: {relative_project_path(schema_path)}")
    if not columns:
        raise ValueError(f"State schema has no columns: {relative_project_path(schema_path)}")
    return columns


def action_label(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    rounded = round(float(value), 2)
    for fraction, label in ACTION_LABELS.items():
        if math.isclose(rounded, fraction, abs_tol=1e-9):
            return label
    return f"sell_{rounded:g}"


def validate_run(run_dir: Path, preferred_policy: str) -> None:
    if run_dir.name != FINAL_RUN_NAME:
        raise ValueError(f"Expected frozen run directory {FINAL_RUN_NAME!r}, got {relative_project_path(run_dir)!r}.")
    required = [
        run_dir,
        run_dir / "config_used.yaml",
        run_dir / "best_validation_model.pt",
        run_dir / "episode_splits.csv",
        run_dir / "baselines" / "baseline_step_rollouts.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required Step 12 input: {relative_project_path(path)}")
    step_columns = pd.read_csv(
        run_dir / "baselines" / "baseline_step_rollouts.csv",
        nrows=0,
    ).columns
    for column in ["split", "policy_name", "episode_id", "date", "action_idx", "action_fraction_requested", "action_fraction_executed"]:
        if column not in step_columns:
            raise ValueError(f"Preferred rollout file is missing required column: {column}")
    policies = pd.read_csv(
        run_dir / "baselines" / "baseline_step_rollouts.csv",
        usecols=["policy_name"],
        low_memory=False,
    )["policy_name"].unique()
    if preferred_policy not in set(policies):
        raise ValueError(f"Preferred policy is missing from baseline_step_rollouts.csv: {preferred_policy}")
    splits = pd.read_csv(run_dir / "episode_splits.csv", usecols=["split"])["split"].unique()
    missing_splits = sorted(set(PRIMARY_SPLITS) - set(splits))
    if missing_splits:
        raise ValueError("Validation/test splits are unavailable: " + ", ".join(missing_splits))


def load_preferred_rollout(run_dir: Path, preferred_policy: str) -> pd.DataFrame:
    path = run_dir / "baselines" / "baseline_step_rollouts.csv"
    columns = [
        "split",
        "policy_name",
        "episode_id",
        "step_in_episode",
        "date",
        "tax_transition_date",
        "action_idx",
        "action_fraction_requested",
        "action_fraction_executed",
        "remaining_fraction",
        "sold_fraction",
        "tax_regime",
        "terminal_liquidation_executed",
        "is_automatic_terminal_liquidation",
    ]
    df = pd.read_csv(path, usecols=columns, low_memory=False)
    df = df[
        df["split"].isin(PRIMARY_SPLITS) & df["policy_name"].eq(preferred_policy)
    ].copy()
    if df.empty:
        raise ValueError("No preferred-policy validation/test rollout rows found.")
    df["episode_id"] = df["episode_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["tax_transition_date"] = pd.to_datetime(df["tax_transition_date"], errors="coerce")
    numeric_columns = [
        "step_in_episode",
        "action_idx",
        "action_fraction_requested",
        "action_fraction_executed",
        "remaining_fraction",
        "sold_fraction",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ["terminal_liquidation_executed", "is_automatic_terminal_liquidation"]:
        df[column] = df[column].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
            if pd.notna(value)
            else False
        )
    return df.sort_values(["split", "episode_id", "step_in_episode"]).reset_index(drop=True)


def load_state_rows(config: dict[str, Any], state_columns: list[str], rollout: pd.DataFrame) -> pd.DataFrame:
    parquet_path = resolve_project_path(config["environment"]["parquet_path"])
    meta_columns = [
        "episode_id",
        "date",
        "ticker",
        "days_until_tax_transition",
        "unrealized_gains_pct",
    ]
    columns = list(dict.fromkeys([*meta_columns, *state_columns]))
    raw = pd.read_parquet(parquet_path, columns=columns)
    raw["episode_id"] = raw["episode_id"].astype(str)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw[raw["episode_id"].isin(set(rollout["episode_id"]))].copy()
    merged = rollout.merge(
        raw,
        on=["episode_id", "date"],
        how="left",
        validate="one_to_one",
    )
    missing_state = merged[state_columns].isna().any(axis=1)
    if missing_state.any():
        raise ValueError(
            f"Could not join state features for {int(missing_state.sum())} preferred-policy rollout rows."
        )
    for column in state_columns + ["days_until_tax_transition", "unrealized_gains_pct"]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


def q_values_for_matrix(
    q_net: torch.nn.Module,
    matrix: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    q_net.eval()
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            batch = torch.as_tensor(matrix[start : start + batch_size], dtype=torch.float32, device=device)
            q_values = q_net(batch)
            if not torch.isfinite(q_values).all():
                raise ValueError("Q-values contain NaN or infinite values.")
            outputs.append(q_values.detach().cpu().numpy())
    return np.vstack(outputs)


def add_q_margin_columns(records: pd.DataFrame, q_values: np.ndarray) -> pd.DataFrame:
    if q_values.shape[1] != 5:
        raise ValueError(f"Expected 5 DQN actions, got q_values shape {q_values.shape}.")
    out = records.copy()
    out["Q_hold"] = q_values[:, 0]
    out["Q_sell_25"] = q_values[:, 1]
    out["Q_sell_50"] = q_values[:, 2]
    out["Q_sell_75"] = q_values[:, 3]
    out["Q_sell_100"] = q_values[:, 4]
    sell_q = q_values[:, 1:]
    best_sell_offsets = np.argmax(sell_q, axis=1)
    sell_action_fractions = np.array([0.25, 0.50, 0.75, 1.00])
    out["max_sell_Q"] = sell_q[np.arange(len(sell_q)), best_sell_offsets]
    out["best_sell_action"] = [action_label(value) for value in sell_action_fractions[best_sell_offsets]]
    out["sell_q_margin"] = out["max_sell_Q"] - out["Q_hold"]
    if not np.isfinite(out["sell_q_margin"].to_numpy(dtype=float)).all():
        raise ValueError("sell_q_margin has NaN or infinite values.")
    return out


def identify_trigger_events(records: pd.DataFrame) -> pd.DataFrame:
    out = records.copy()
    out["chosen_action"] = out["action_fraction_requested"].map(action_label)
    terminal = out["terminal_liquidation_executed"].fillna(False) | out[
        "is_automatic_terminal_liquidation"
    ].fillna(False)
    out["is_sell_step"] = (
        out["action_fraction_requested"].fillna(0).gt(0)
        & out["action_fraction_executed"].fillna(0).gt(0)
        & ~terminal
    )
    out["is_hold_step"] = out["action_fraction_requested"].fillna(0).eq(0)
    out["is_positive_margin"] = out["sell_q_margin"].gt(0)
    out["crosses_normal_margin"] = out["sell_q_margin"].gt(NORMAL_MARGIN)
    sorted_index = out.sort_values(["split", "episode_id", "step_in_episode"]).index
    grouped_sells = out.loc[sorted_index].groupby(["split", "episode_id"], sort=False)[
        "is_sell_step"
    ]
    prior_sells = grouped_sells.cumsum().groupby(
        [out.loc[sorted_index, "split"], out.loc[sorted_index, "episode_id"]],
        sort=False,
    ).shift(fill_value=0)
    out["is_before_first_discretionary_sale"] = False
    out.loc[sorted_index, "is_before_first_discretionary_sale"] = (
        prior_sells.eq(0).to_numpy()
    )
    out["is_first_discretionary_sale"] = out["is_sell_step"] & out["is_before_first_discretionary_sale"]
    out["crosses_first_sale_margin"] = (
        out["is_before_first_discretionary_sale"] & out["sell_q_margin"].gt(FIRST_SALE_MARGIN)
    )
    return out


def pooled_standardized_difference(sell: pd.Series, hold: pd.Series) -> float:
    sell = pd.to_numeric(sell, errors="coerce").dropna()
    hold = pd.to_numeric(hold, errors="coerce").dropna()
    if len(sell) == 0 or len(hold) == 0:
        return np.nan
    sell_std = float(sell.std(ddof=1)) if len(sell) > 1 else 0.0
    hold_std = float(hold.std(ddof=1)) if len(hold) > 1 else 0.0
    pooled = math.sqrt((sell_std**2 + hold_std**2) / 2.0)
    if pooled <= 0 or not np.isfinite(pooled):
        return 0.0
    return float((sell.mean() - hold.mean()) / pooled)


def compute_trigger_vs_hold_feature_summary(records: pd.DataFrame, state_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [
        ("sell_steps_vs_hold_steps", "is_sell_step", "is_hold_step"),
        ("first_sale_vs_pre_first_hold", "is_first_discretionary_sale", "is_pre_first_hold"),
    ]
    data = records.copy()
    data["is_pre_first_hold"] = data["is_hold_step"] & data["is_before_first_discretionary_sale"]
    for split in PRIMARY_SPLITS:
        split_df = data[data["split"].eq(split)]
        for comparison_scope, sell_flag, hold_flag in scopes:
            sell_rows = split_df[split_df[sell_flag]]
            hold_rows = split_df[split_df[hold_flag]]
            for feature in state_columns:
                sell_values = sell_rows[feature]
                hold_values = hold_rows[feature]
                row = {
                    "split": split,
                    "comparison_scope": comparison_scope,
                    "feature": feature,
                    "feature_group": assign_feature_group(feature),
                    "n_sell_or_trigger_steps": int(sell_values.notna().sum()),
                    "n_hold_steps": int(hold_values.notna().sum()),
                    "mean_on_sell_or_trigger_steps": sell_values.mean(),
                    "mean_on_hold_steps": hold_values.mean(),
                    "median_on_sell_or_trigger_steps": sell_values.median(),
                    "median_on_hold_steps": hold_values.median(),
                    "difference_in_means": sell_values.mean() - hold_values.mean(),
                    "standardized_difference": pooled_standardized_difference(sell_values, hold_values),
                }
                rows.append(row)
    out = pd.DataFrame(rows)
    out["_split_order"] = out["split"].map({"test": 0, "validation": 1})
    test_order = (
        out[out["split"].eq("test") & out["comparison_scope"].eq("sell_steps_vs_hold_steps")]
        .set_index("feature")["standardized_difference"]
        .abs()
        .sort_values(ascending=False)
    )
    out["_feature_order"] = out["feature"].map({feature: idx for idx, feature in enumerate(test_order.index)})
    out["_scope_order"] = out["comparison_scope"].map({"sell_steps_vs_hold_steps": 0, "first_sale_vs_pre_first_hold": 1})
    return out.sort_values(["_scope_order", "_split_order", "_feature_order"]).drop(columns=["_split_order", "_feature_order", "_scope_order"])


def compute_trigger_event_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in PRIMARY_SPLITS:
        data = records[records["split"].eq(split)]
        n_obs = len(data)
        n_episodes = data["episode_id"].nunique()
        action_counts = data["chosen_action"].value_counts().to_dict()
        rows.append(
            {
                "split": split,
                "n_observations": int(n_obs),
                "n_episodes": int(n_episodes),
                "sell_step_count": int(data["is_sell_step"].sum()),
                "sell_step_rate": float(data["is_sell_step"].mean()),
                "first_discretionary_sale_count": int(data["is_first_discretionary_sale"].sum()),
                "hold_step_count": int(data["is_hold_step"].sum()),
                "positive_margin_count": int(data["is_positive_margin"].sum()),
                "positive_margin_rate": float(data["is_positive_margin"].mean()),
                "crosses_normal_margin_count": int(data["crosses_normal_margin"].sum()),
                "crosses_normal_margin_rate": float(data["crosses_normal_margin"].mean()),
                "crosses_first_sale_margin_count": int(data["crosses_first_sale_margin"].sum()),
                "crosses_first_sale_margin_rate": float(data["crosses_first_sale_margin"].mean()),
                "mean_sell_q_margin": float(data["sell_q_margin"].mean()),
                "median_sell_q_margin": float(data["sell_q_margin"].median()),
                "mean_sell_q_margin_on_sell_steps": float(data.loc[data["is_sell_step"], "sell_q_margin"].mean()),
                "mean_sell_q_margin_on_hold_steps": float(data.loc[data["is_hold_step"], "sell_q_margin"].mean()),
                "hold_action_count": int(action_counts.get("hold", 0)),
                "sell_25_action_count": int(action_counts.get("sell_25", 0)),
                "sell_50_action_count": int(action_counts.get("sell_50", 0)),
                "sell_75_action_count": int(action_counts.get("sell_75", 0)),
                "sell_100_action_count": int(action_counts.get("sell_100", 0)),
            }
        )
    return pd.DataFrame(rows)


def compute_q_margin_correlations(records: pd.DataFrame, state_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in PRIMARY_SPLITS:
        data = records[records["split"].eq(split)]
        y = data["sell_q_margin"]
        for feature in state_columns:
            x = data[feature]
            valid = x.notna() & y.notna()
            n_obs = int(valid.sum())
            if n_obs < 3 or x[valid].nunique(dropna=True) < 2:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = float(x[valid].corr(y[valid], method="pearson"))
                spearman = float(x[valid].corr(y[valid], method="spearman"))
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "feature_group": assign_feature_group(feature),
                    "n_obs": n_obs,
                    "pearson_corr": pearson,
                    "spearman_corr": spearman,
                    "mean_feature_value": float(x.mean()),
                    "std_feature_value": float(x.std(ddof=1)),
                }
            )
    out = pd.DataFrame(rows)
    test_order = (
        out[out["split"].eq("test")]
        .set_index("feature")["spearman_corr"]
        .abs()
        .sort_values(ascending=False)
    )
    out["_feature_order"] = out["feature"].map({feature: idx for idx, feature in enumerate(test_order.index)})
    out["_split_order"] = out["split"].map({"test": 0, "validation": 1})
    return out.sort_values(["_split_order", "_feature_order"]).drop(columns=["_split_order", "_feature_order"])


def compute_permutation_sensitivity(
    records: pd.DataFrame,
    state_columns: list[str],
    q_net: torch.nn.Module,
    *,
    device: torch.device,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_seed)
    for split in PRIMARY_SPLITS:
        data = records[records["split"].eq(split)].copy()
        matrix = data[state_columns].to_numpy(dtype=np.float32)
        baseline_margin = data["sell_q_margin"].to_numpy(dtype=float)
        for feature_idx, feature in enumerate(state_columns):
            permuted = matrix.copy()
            permuted[:, feature_idx] = rng.permutation(permuted[:, feature_idx])
            q_values = q_values_for_matrix(q_net, permuted, device=device)
            perm_margin = q_values[:, 1:].max(axis=1) - q_values[:, 0]
            delta = perm_margin - baseline_margin
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "feature_group": assign_feature_group(feature),
                    "n_obs": int(len(data)),
                    "mean_abs_change_in_q_margin": float(np.mean(np.abs(delta))),
                    "median_abs_change_in_q_margin": float(np.median(np.abs(delta))),
                    "mean_signed_change_in_q_margin": float(np.mean(delta)),
                    "pct_sign_flip": float(np.mean(np.sign(perm_margin) != np.sign(baseline_margin))),
                    "pct_threshold_flip_0p020": float(
                        np.mean((perm_margin > NORMAL_MARGIN) != (baseline_margin > NORMAL_MARGIN))
                    ),
                    "pct_threshold_flip_0p070": float(
                        np.mean((perm_margin > FIRST_SALE_MARGIN) != (baseline_margin > FIRST_SALE_MARGIN))
                    ),
                    "random_seed": random_seed,
                }
            )
    out = pd.DataFrame(rows)
    test_order = (
        out[out["split"].eq("test")]
        .set_index("feature")["mean_abs_change_in_q_margin"]
        .sort_values(ascending=False)
    )
    out["_feature_order"] = out["feature"].map({feature: idx for idx, feature in enumerate(test_order.index)})
    out["_split_order"] = out["split"].map({"test": 0, "validation": 1})
    return out.sort_values(["_split_order", "_feature_order"]).drop(columns=["_split_order", "_feature_order"])


def assign_feature_group(feature: str) -> str:
    lower = feature.lower()
    if "days" in lower and "tax" in lower or "tax_transition" in lower:
        return "tax_timing"
    if "unrealized" in lower or "gain" in lower or "pnl" in lower:
        return "pnl_gain_path"
    if "drawdown" in lower or "volatility" in lower or "vol" in lower:
        return "risk_drawdown_volatility"
    if "remaining_fraction" in lower or "sold_fraction" in lower or "position" in lower:
        return "position_state"
    if any(token in lower for token in ["sma", "ema", "rsi", "macd", "bb_"]):
        return "technical_indicator"
    if any(token in lower for token in ["gold", "wti", "semiconductor", "eur_", "usd_", "wheat", "spy", "vix"]):
        return "macro_market"
    if lower.startswith("pc") or lower.startswith("pca"):
        return "pca_feature"
    if "market_cap" in lower or "cap_bucket" in lower:
        return "market_cap"
    return "other"


def compute_group_summary(
    correlations: pd.DataFrame,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    merged = correlations.merge(
        permutation[
            [
                "split",
                "feature",
                "mean_abs_change_in_q_margin",
            ]
        ],
        on=["split", "feature"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (split, group), data in merged.groupby(["split", "feature_group"], sort=False):
        abs_spearman = data["spearman_corr"].abs()
        sensitivity = data["mean_abs_change_in_q_margin"]
        top_idx = sensitivity.fillna(-np.inf).idxmax()
        rows.append(
            {
                "split": split,
                "feature_group": group,
                "num_features": int(data["feature"].nunique()),
                "mean_abs_spearman_corr": float(abs_spearman.mean()),
                "max_abs_spearman_corr": float(abs_spearman.max()),
                "mean_permutation_sensitivity": float(sensitivity.mean()),
                "max_permutation_sensitivity": float(sensitivity.max()),
                "top_feature_in_group": data.loc[top_idx, "feature"] if len(data) else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "max_permutation_sensitivity"], ascending=[True, False])


def plot_top_bar(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    xlabel: str,
    path: Path,
    top_n: int = TOP_N,
) -> None:
    plot_df = df[df["split"].eq("test")].copy()
    plot_df["_abs"] = plot_df[metric].abs()
    plot_df = plot_df.sort_values("_abs", ascending=False).head(top_n).sort_values(metric)
    colors = np.where(plot_df[metric].ge(0), "#3b6ea8", "#8a4f3d")
    plt.figure(figsize=(9.2, 6.0))
    plt.barh(plot_df["feature"], plot_df[metric], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_step12_plots(
    records: pd.DataFrame,
    feature_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    permutation: pd.DataFrame,
    plots_dir: Path,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = [plots_dir / name for name in PLOT_FILES]
    plot_top_bar(
        correlations,
        metric="spearman_corr",
        title="Top state-feature correlations with sell Q-margin - test",
        xlabel="Spearman correlation with sell Q-margin",
        path=paths[0],
    )
    diff_df = feature_summary[
        feature_summary["comparison_scope"].eq("sell_steps_vs_hold_steps")
    ]
    plot_top_bar(
        diff_df,
        metric="standardized_difference",
        title="Top sell-step vs hold-step state differences - test",
        xlabel="standardized difference",
        path=paths[1],
    )
    plot_top_bar(
        permutation,
        metric="mean_abs_change_in_q_margin",
        title="Top permutation sensitivity of sell Q-margin - test",
        xlabel="mean absolute change in sell Q-margin",
        path=paths[2],
    )

    test = records[records["split"].eq("test")].copy()
    action_order = ["hold", "sell_25", "sell_50", "sell_75", "sell_100"]
    grouped = [test.loc[test["chosen_action"].eq(action), "sell_q_margin"].dropna() for action in action_order]
    plt.figure(figsize=(8.8, 5.2))
    plt.boxplot(grouped, tick_labels=action_order, showfliers=False)
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline(NORMAL_MARGIN, color="#666666", linestyle="--", linewidth=1, label="0.020 normal margin")
    plt.axhline(FIRST_SALE_MARGIN, color="#999999", linestyle=":", linewidth=1, label="0.070 first-sale margin")
    plt.ylabel("sell Q-margin")
    plt.title("Sell Q-margin distribution by chosen action - test")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(paths[3], dpi=160)
    plt.close()

    if "days_until_tax_transition" in test.columns and test["days_until_tax_transition"].notna().any():
        plot_df = test[["days_until_tax_transition", "sell_q_margin"]].dropna().copy()
        plot_df["days_bucket"] = pd.qcut(
            plot_df["days_until_tax_transition"].rank(method="first"),
            q=20,
            duplicates="drop",
        )
        binned = (
            plot_df.groupby("days_bucket", observed=False)
            .agg(
                mean_days_until_tax_transition=("days_until_tax_transition", "mean"),
                mean_sell_q_margin=("sell_q_margin", "mean"),
                median_sell_q_margin=("sell_q_margin", "median"),
            )
            .sort_values("mean_days_until_tax_transition")
        )
        plt.figure(figsize=(8.8, 5.2))
        plt.plot(
            binned["mean_days_until_tax_transition"],
            binned["mean_sell_q_margin"],
            marker="o",
            label="mean sell Q-margin",
            color="#3b6ea8",
        )
        plt.plot(
            binned["mean_days_until_tax_transition"],
            binned["median_sell_q_margin"],
            marker="s",
            label="median sell Q-margin",
            color="#6c7a40",
        )
        plt.axhline(0, color="black", linewidth=1)
        plt.gca().invert_xaxis()
        plt.xlabel("days until tax transition")
        plt.ylabel("sell Q-margin")
        plt.title("Sell Q-margin as tax transition approaches - test")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(paths[4], dpi=160)
        plt.close()
    else:
        plt.figure(figsize=(8.8, 5.2))
        plt.text(0.5, 0.5, "days_until_tax_transition unavailable", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(paths[4], dpi=160)
        plt.close()
    return paths


def top_feature_list(df: pd.DataFrame, metric: str, *, split: str = "test", n: int = 10) -> list[str]:
    data = df[df["split"].eq(split)].copy()
    data["_sort"] = data[metric].abs()
    return [
        f"{row.feature}: {getattr(row, metric):.6g}"
        for row in data.sort_values("_sort", ascending=False).head(n).itertuples(index=False)
    ]


def write_step12_notes(
    path: Path,
    *,
    run_dir: Path,
    preferred_policy: str,
    records: pd.DataFrame,
    correlations: pd.DataFrame,
    permutation: pd.DataFrame,
    feature_summary: pd.DataFrame,
    warnings: list[str],
) -> None:
    perm_top = top_feature_list(permutation, "mean_abs_change_in_q_margin")
    corr_top = top_feature_list(correlations, "spearman_corr")
    diff_top = top_feature_list(
        feature_summary[feature_summary["comparison_scope"].eq("sell_steps_vs_hold_steps")],
        "standardized_difference",
    )
    test_events = records[records["split"].eq("test")]
    top_perm_feature = perm_top[0].split(":", 1)[0] if perm_top else "none"
    top_corr_feature = corr_top[0].split(":", 1)[0] if corr_top else "none"
    lines = [
        "Step 12 DQN trigger-driver / state sensitivity notes",
        "Purpose: identify state variables associated with the preferred DQN moving from hold toward a sell action.",
        f"model_run_path: {relative_project_path(run_dir)}",
        f"preferred_policy: {preferred_policy}",
        "splits_analyzed: validation, test",
        "sell_q_margin_t = max_a_sell Q(s_t, a) - Q(s_t, hold), where sell actions are 25%, 50%, 75%, and 100%.",
        "Trigger events exclude automatic terminal liquidation.",
        "is_sell_step = chosen_action != hold and executed_fraction > 0.",
        "is_first_discretionary_sale = first non-terminal discretionary sale in an episode.",
        "crosses_normal_margin uses sell_q_margin > 0.020.",
        "crosses_first_sale_margin uses sell_q_margin > 0.070 before first discretionary sale.",
        "This analysis does not interpret raw neural-network weights. Instead, it studies the DQN's sell-vs-hold Q-value margin and observed trigger events.",
        "The results should be interpreted as model-sensitivity and association evidence, not causal evidence.",
        "No retraining, reward redesign, environment change, policy-threshold change, or preferred-policy reselection was performed.",
        f"validation_observations: {len(records[records['split'].eq('validation')])}",
        f"test_observations: {len(test_events)}",
        f"test_sell_steps: {int(test_events['is_sell_step'].sum())}",
        f"test_first_discretionary_sales: {int(test_events['is_first_discretionary_sale'].sum())}",
        "market_cap_trigger_analysis: skipped because no market-cap feature was found in the model state schema."
        if not any(assign_feature_group(feature) == "market_cap" for feature in correlations["feature"].unique())
        else "market_cap_trigger_analysis: included for available market-cap state features.",
        "",
        "Top 10 test features by permutation sensitivity:",
        *perm_top,
        "",
        "Top 10 test features by Spearman correlation with sell Q-margin:",
        *corr_top,
        "",
        "Top 10 test features by sell-vs-hold standardized difference:",
        *diff_top,
        "",
        "Short interpretation:",
        f"On the test split, `{top_perm_feature}` produces the largest average absolute change in sell Q-margin under permutation, while `{top_corr_feature}` has the strongest rank association with the sell Q-margin. These diagnostics indicate which frozen state inputs the nonlinear DQN margin is most sensitive to; they do not prove that changing those variables would causally change realized performance.",
    ]
    if warnings:
        lines.extend(["", "Warnings:", *warnings])
    else:
        lines.extend(["", "Warnings:", "No Step 12 warnings."])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def update_manifest(output_dir: Path, output_paths: list[Path]) -> None:
    manifest_path = output_dir / "phase_4_11_outputs_manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a list: {relative_project_path(manifest_path)}")
    existing = {
        row.get("path")
        for row in payload
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    descriptions = {
        "step12_dqn_trigger_driver_feature_summary": "Step 12 sell-trigger versus hold state-feature summary.",
        "step12_dqn_trigger_event_summary": "Step 12 DQN trigger event summary.",
        "step12_dqn_q_margin_feature_correlations": "Step 12 Q-margin state-feature correlations.",
        "step12_dqn_permutation_sensitivity": "Step 12 Q-margin permutation sensitivity.",
        "step12_dqn_trigger_driver_group_summary": "Step 12 feature-group trigger-driver summary.",
        "step12_dqn_trigger_driver_notes": "Step 12 trigger-driver notes.",
    }
    for output_path in output_paths:
        rel = relative_project_path(output_path)
        if rel in existing:
            continue
        payload.append(
            {
                "path": rel,
                "type": output_path.suffix.lstrip("."),
                "step": "12",
                "description": descriptions.get(output_path.stem, "Step 12 trigger-driver plot."),
            }
        )
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_outputs(output_paths: list[Path]) -> None:
    for path in output_paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected Step 12 output was not written: {relative_project_path(path)}")
        if path.suffix == ".png" and path.stat().st_size <= 0:
            raise ValueError(f"Step 12 plot is empty: {relative_project_path(path)}")
    for csv_path in [path for path in output_paths if path.suffix == ".csv"]:
        md_path = csv_path.with_suffix(".md")
        if md_path.exists():
            csv_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
            md_text = md_path.read_text(encoding="utf-8")
            for column in csv_columns:
                if column not in md_text:
                    raise ValueError(f"Markdown output does not appear to match CSV columns: {relative_project_path(md_path)} missing {column}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 12 DQN trigger-driver analysis.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preferred-policy", default=DEFAULT_PREFERRED_POLICY)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_project_path(args.run_dir)
    output_dir = run_dir / "quant_analysis"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    validate_run(run_dir, args.preferred_policy)
    evaluator = load_evaluator_module()
    config = evaluator.load_yaml(run_dir / "config_used.yaml")
    schema_path = resolve_project_path(config["environment"]["state_schema_path"])
    state_columns = load_state_schema(schema_path)
    if not any(assign_feature_group(feature) == "market_cap" for feature in state_columns):
        warnings.append(
            "Market-cap trigger analysis was skipped because no market-cap feature was found in the model state schema."
        )
    action_fractions = [float(value) for value in config["action_space"]["action_fractions"]]
    if action_fractions != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError(f"Unexpected action space for Step 12: {action_fractions}")
    device = evaluator._resolve_device(args.device)
    q_net = evaluator.load_trained_q_network(
        config,
        run_dir / "best_validation_model.pt",
        obs_dim=len(state_columns),
        num_actions=len(action_fractions),
        device=device,
    )

    rollout = load_preferred_rollout(run_dir, args.preferred_policy)
    records = load_state_rows(config, state_columns, rollout)
    state_matrix = records[state_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(state_matrix).all():
        raise ValueError("State matrix contains NaN or infinite values.")
    q_values = q_values_for_matrix(q_net, state_matrix, device=device)
    records = add_q_margin_columns(records, q_values)
    records = identify_trigger_events(records)

    feature_summary = compute_trigger_vs_hold_feature_summary(records, state_columns)
    event_summary = compute_trigger_event_summary(records)
    correlations = compute_q_margin_correlations(records, state_columns)
    permutation = compute_permutation_sensitivity(
        records,
        state_columns,
        q_net,
        device=device,
        random_seed=int(args.random_seed),
    )
    group_summary = compute_group_summary(correlations, permutation)

    output_paths = [
        output_dir / OUTPUTS["feature_summary_csv"],
        output_dir / OUTPUTS["feature_summary_md"],
        output_dir / OUTPUTS["event_summary_csv"],
        output_dir / OUTPUTS["event_summary_md"],
        output_dir / OUTPUTS["correlations_csv"],
        output_dir / OUTPUTS["correlations_md"],
        output_dir / OUTPUTS["permutation_csv"],
        output_dir / OUTPUTS["permutation_md"],
        output_dir / OUTPUTS["group_csv"],
        output_dir / OUTPUTS["group_md"],
        output_dir / OUTPUTS["notes"],
    ]
    save_table(feature_summary, output_paths[0], output_paths[1])
    save_table(event_summary, output_paths[2], output_paths[3])
    save_table(correlations, output_paths[4], output_paths[5])
    save_table(permutation, output_paths[6], output_paths[7])
    save_table(group_summary, output_paths[8], output_paths[9])
    plot_paths = make_step12_plots(
        records,
        feature_summary,
        correlations,
        permutation,
        plots_dir,
    )
    write_step12_notes(
        output_paths[10],
        run_dir=run_dir,
        preferred_policy=args.preferred_policy,
        records=records,
        correlations=correlations,
        permutation=permutation,
        feature_summary=feature_summary,
        warnings=warnings,
    )
    all_outputs = [*output_paths, *plot_paths]
    validate_outputs(all_outputs)
    update_manifest(output_dir, all_outputs)

    print(f"Wrote Step 12 outputs to {relative_project_path(output_dir)}")
    print(f"records: {len(records)}")
    print("Top 10 test features by permutation sensitivity:")
    for line in top_feature_list(permutation, "mean_abs_change_in_q_margin"):
        print(f"  {line}")
    print("Top 10 test features by Spearman correlation with sell Q-margin:")
    for line in top_feature_list(correlations, "spearman_corr"):
        print(f"  {line}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("Warnings: none")


if __name__ == "__main__":
    main()
