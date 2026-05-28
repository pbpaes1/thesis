"""Build Step 11 case-study episode diagnostics for the frozen final run.

Run from the project root:
    python scripts/quant_analysis/step11_case_study_diagnostics.py \
        --run-dir runs/train_reward_c_lite_v5_full \
        --preferred-policy trained_dqn_first_sale_margin_0p070_normal_0p020
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FINAL_RUN_NAME = "train_reward_c_lite_v5_full"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / FINAL_RUN_NAME
DEFAULT_PREFERRED_POLICY = "trained_dqn_first_sale_margin_0p070_normal_0p020"
OPTIONAL_DQN_POLICY = "trained_dqn_first_sale_margin_0p090_normal_0p020"
PRIMARY_SPLITS = ["validation", "test"]
BENCHMARK_POLICIES = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "sell_quarters_over_time",
    "random_policy",
]
PLOT_BENCHMARK_POLICIES = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "sell_quarters_over_time",
]
SCENARIOS = [
    "dqn_sells_early_avoids_later_loss",
    "dqn_sells_too_early_underperforms_hold",
    "dqn_waits_correctly_until_long_term",
    "dqn_behaves_like_hold_to_terminal",
    "partial_liquidation_useful",
    "dqn_loses_to_all_major_benchmarks",
]
PREFERRED_SELECTION_SCENARIOS = [
    "dqn_sells_early_avoids_later_loss",
    "dqn_sells_too_early_underperforms_hold",
    "dqn_waits_correctly_until_long_term",
    "dqn_behaves_like_hold_to_terminal",
    "partial_liquidation_useful",
]
OUTPUT_FILES = {
    "classification_csv": "step11_case_study_scenario_classification.csv",
    "classification_md": "step11_case_study_scenario_classification.md",
    "summary_csv": "step11_case_study_scenario_summary.csv",
    "summary_md": "step11_case_study_scenario_summary.md",
    "ranking_csv": "step11_case_study_candidate_ranking.csv",
    "ranking_md": "step11_case_study_candidate_ranking.md",
    "selected_csv": "step11_selected_case_studies.csv",
    "selected_md": "step11_selected_case_studies.md",
    "notes": "step11_case_study_notes.txt",
}
HOLD_LIKE_ABS_TOLERANCE = 1e-6
POST_SALE_DRAWDOWN_THRESHOLD = 0.10
POST_SALE_UPSIDE_THRESHOLD = 0.10
EPS = 1e-12


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


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
        if pd.notna(value)
        else False
    )


def prepare_episode_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_columns = [
        "episode_final_after_tax_total_value",
        "episode_realized_after_tax_pnl",
        "total_tax_paid",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "terminal_liquidation_fraction",
        "episode_final_sold_fraction",
        "episode_final_remaining_fraction",
        "num_discretionary_sales",
        "days_to_first_sale",
        "steps_to_first_sale",
        "first_cut_step",
        "first_cut_fraction_executed",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in [
        "episode_terminal_liquidation_executed",
        "episode_cut_occurred",
        "first_sale_before_tax_transition",
    ]:
        if column in out.columns:
            out[column] = coerce_bool(out[column])
    for column in ["first_cut_date", "first_sale_date"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def prepare_step_rollouts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_columns = [
        "step_in_episode",
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
    for column in [
        "terminal_liquidation_executed",
        "is_automatic_terminal_liquidation",
        "done",
    ]:
        if column in out.columns:
            out[column] = coerce_bool(out[column])
    for column in ["date", "tax_transition_date"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def parse_ticker(episode_id: str) -> str:
    text = str(episode_id)
    if "_" not in text:
        return ""
    return text.split("_", 1)[0]


def validate_inputs(
    episode_df: pd.DataFrame,
    step_df: pd.DataFrame,
    *,
    preferred_policy: str,
) -> None:
    episode_required = [
        "split",
        "episode_id",
        "policy_name",
        "episode_final_after_tax_total_value",
        "episode_realized_after_tax_pnl",
        "total_tax_paid",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "episode_terminal_liquidation_executed",
        "episode_cut_occurred",
        "days_to_first_sale",
        "steps_to_first_sale",
        "first_sale_date",
        "first_cut_fraction_executed",
        "terminal_liquidation_fraction",
    ]
    step_required = [
        "split",
        "episode_id",
        "policy_name",
        "step_in_episode",
        "date",
        "tax_transition_date",
        "action_fraction_executed",
        "after_tax_total_value",
        "sold_fraction",
        "remaining_fraction",
        "terminal_liquidation_executed",
    ]
    missing_episode = [column for column in episode_required if column not in episode_df.columns]
    if missing_episode:
        raise ValueError(
            "Step 11 requires missing episode-level column(s): "
            + ", ".join(missing_episode)
        )
    missing_step = [column for column in step_required if column not in step_df.columns]
    if missing_step:
        raise ValueError(
            "Step 11 requires missing path-level column(s): "
            + ", ".join(missing_step)
        )
    missing_splits = sorted(set(PRIMARY_SPLITS) - set(episode_df["split"].unique()))
    if missing_splits:
        raise ValueError(
            "Step 11 is missing required validation/test split(s): "
            + ", ".join(missing_splits)
        )
    required_policies = {preferred_policy, *BENCHMARK_POLICIES}
    missing_policies = sorted(required_policies - set(episode_df["policy_name"].unique()))
    if missing_policies:
        raise ValueError(
            "Step 11 is missing required episode-level policy row(s): "
            + ", ".join(missing_policies)
        )
    for split in PRIMARY_SPLITS:
        split_policies = set(
            episode_df.loc[episode_df["split"].eq(split), "policy_name"].unique()
        )
        missing = sorted(required_policies - split_policies)
        if missing:
            raise ValueError(
                f"Step 11 split={split} is missing required policy row(s): "
                + ", ".join(missing)
            )


def load_raw_episode_metadata(run_dir: Path, warnings: list[str]) -> pd.DataFrame:
    config_path = run_dir / "config_used.yaml"
    parquet_path = PROJECT_ROOT / "data" / "episodes" / "drl_episodes.parquet"
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        match = re.search(r"parquet_path:\s*(.+)", text)
        if match:
            parquet_path = resolve_project_path(match.group(1).strip())
    columns = [
        "episode_id",
        "date",
        "ticker",
        "tax_transition_date",
        "unrealized_gains_pct",
    ]
    if not parquet_path.exists():
        warnings.append(
            "WARNING ticker/sector metadata unavailable: raw episode parquet was not found; "
            "ticker will be parsed from episode_id and gain path will use hold-path proxy."
        )
        return pd.DataFrame()
    try:
        raw = pd.read_parquet(parquet_path, columns=columns)
    except Exception as exc:
        warnings.append(
            "WARNING raw episode metadata unavailable: "
            f"{relative_project_path(parquet_path)} could not be read "
            f"({type(exc).__name__}: {exc}); gain path will use hold-path proxy."
        )
        return pd.DataFrame()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["tax_transition_date"] = pd.to_datetime(raw["tax_transition_date"], errors="coerce")
    raw["unrealized_gains_pct"] = pd.to_numeric(
        raw["unrealized_gains_pct"], errors="coerce"
    )
    return raw


def summarize_raw_metadata(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    summary = (
        raw_df.sort_values(["episode_id", "date"])
        .groupby("episode_id", as_index=False)
        .agg(
            ticker=("ticker", "first"),
            episode_start_date=("date", "min"),
            terminal_date=("date", "max"),
            tax_transition_date=("tax_transition_date", "first"),
            terminal_unrealized_gain=("unrealized_gains_pct", "last"),
        )
    )
    return summary


def sale_mask(path: pd.DataFrame) -> pd.Series:
    terminal = (
        path["terminal_liquidation_executed"].fillna(False)
        | path.get("is_automatic_terminal_liquidation", False)
    )
    return path["action_fraction_executed"].fillna(0).gt(EPS) & ~terminal


def build_path_features(
    step_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    preferred_policy: str,
    warnings: list[str],
) -> pd.DataFrame:
    preferred_steps = step_df[
        step_df["split"].isin(PRIMARY_SPLITS)
        & step_df["policy_name"].eq(preferred_policy)
    ].copy()
    raw_by_episode = {
        str(episode_id): group.sort_values("date")
        for episode_id, group in raw_df.groupby("episode_id", sort=False)
    } if not raw_df.empty else {}
    hold_steps = step_df[
        step_df["split"].isin(PRIMARY_SPLITS)
        & step_df["policy_name"].eq("hold_to_terminal")
    ]
    hold_by_key = {
        (str(split), str(episode_id)): group.sort_values("step_in_episode")
        for (split, episode_id), group in hold_steps.groupby(["split", "episode_id"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    for (split, episode_id), group in preferred_steps.groupby(["split", "episode_id"], sort=False):
        group = group.sort_values("step_in_episode")
        sales = group[sale_mask(group)]
        first_sale = sales.iloc[0] if not sales.empty else None
        transition_values = group["tax_transition_date"].dropna()
        transition = transition_values.iloc[0] if not transition_values.empty else pd.NaT
        first_sale_date = first_sale["date"] if first_sale is not None else pd.NaT
        first_sale_step = (
            float(first_sale["step_in_episode"]) if first_sale is not None else np.nan
        )
        first_sale_fraction = (
            float(first_sale["action_fraction_executed"]) if first_sale is not None else np.nan
        )
        first_sale_remaining = (
            float(first_sale["remaining_fraction"]) if first_sale is not None else np.nan
        )
        discretionary_sold_fraction = (
            float(sales["sold_fraction"].max()) if not sales.empty else 0.0
        )
        short_term_sales = sales[
            sales["tax_transition_date"].notna()
            & sales["date"].notna()
            & sales["date"].lt(sales["tax_transition_date"])
        ]
        short_term_discretionary_fraction = (
            float(short_term_sales["action_fraction_executed"].sum())
            if not short_term_sales.empty
            else 0.0
        )
        after_transition_sales = sales[
            sales["tax_transition_date"].notna()
            & sales["date"].notna()
            & sales["date"].ge(sales["tax_transition_date"])
        ]
        action_history = "; ".join(
            f"step {int(row.step_in_episode)} {row.date.date() if pd.notna(row.date) else 'no_date'} "
            f"sell {float(row.action_fraction_executed):.2f}"
            for row in sales.itertuples(index=False)
        )
        post_sale_drawdown = np.nan
        post_sale_upside = np.nan
        first_sale_unrealized_gain = np.nan
        gain_path_source = "raw_unrealized_gains_pct"
        raw_path = raw_by_episode.get(str(episode_id), pd.DataFrame())
        if first_sale is not None and not raw_path.empty and pd.notna(first_sale_date):
            post = raw_path[raw_path["date"].ge(first_sale_date)]
            if not post.empty and post["unrealized_gains_pct"].notna().any():
                post_gain = post["unrealized_gains_pct"].astype(float)
                first_sale_unrealized_gain = float(post_gain.iloc[0])
                terminal_gain = float(post_gain.iloc[-1])
                post_sale_drawdown = float(post_gain.max() - terminal_gain)
                post_sale_upside = float(post_gain.max() - first_sale_unrealized_gain)
            else:
                gain_path_source = "hold_to_terminal_after_tax_value_proxy"
        elif first_sale is not None:
            gain_path_source = "hold_to_terminal_after_tax_value_proxy"

        if first_sale is not None and not np.isfinite(post_sale_drawdown):
            hold_path = hold_by_key.get((str(split), str(episode_id)), pd.DataFrame())
            if not hold_path.empty:
                post_hold = hold_path[hold_path["step_in_episode"].ge(first_sale_step)]
                if not post_hold.empty:
                    values = post_hold["after_tax_total_value"].astype(float)
                    first_sale_unrealized_gain = float(values.iloc[0])
                    post_sale_drawdown = float(values.max() - values.iloc[-1])
                    post_sale_upside = float(values.max() - values.iloc[0])
                    gain_path_source = "hold_to_terminal_after_tax_value_proxy"

        rows.append(
            {
                "split": split,
                "episode_id": str(episode_id),
                "preferred_path_num_points": int(len(group)),
                "preferred_path_unique_dates": int(group["date"].nunique()),
                "preferred_after_tax_value_path_range": float(
                    group["after_tax_total_value"].max()
                    - group["after_tax_total_value"].min()
                ),
                "episode_start_date_from_path": group["date"].min(),
                "terminal_date_from_path": group["date"].max(),
                "tax_transition_date_from_path": transition,
                "first_sale_step_from_path": first_sale_step,
                "first_sale_date_from_path": first_sale_date,
                "first_sale_executed_fraction_from_path": first_sale_fraction,
                "first_sale_remaining_fraction": first_sale_remaining,
                "discretionary_sold_fraction_before_terminal": discretionary_sold_fraction,
                "short_term_discretionary_sold_fraction": short_term_discretionary_fraction,
                "has_discretionary_sale_after_transition": bool(not after_transition_sales.empty),
                "action_history": action_history,
                "post_sale_drawdown": post_sale_drawdown,
                "post_sale_upside": post_sale_upside,
                "first_sale_unrealized_gain": first_sale_unrealized_gain,
                "gain_path_source": gain_path_source,
            }
        )
    if raw_df.empty:
        warnings.append(
            "WARNING unrealized gain path unavailable from raw parquet; "
            "post-sale drawdown/upside uses hold-to-terminal after-tax path proxy where needed."
        )
    return pd.DataFrame(rows)


def build_episode_case_dataset(
    episode_df: pd.DataFrame,
    step_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    preferred_policy: str,
    warnings: list[str],
) -> pd.DataFrame:
    policies = [preferred_policy, *BENCHMARK_POLICIES]
    if OPTIONAL_DQN_POLICY in set(episode_df["policy_name"]):
        policies.append(OPTIONAL_DQN_POLICY)
    metrics = episode_df[
        episode_df["split"].isin(PRIMARY_SPLITS) & episode_df["policy_name"].isin(policies)
    ].copy()
    value_wide = metrics.pivot(
        index=["split", "episode_id"],
        columns="policy_name",
        values="episode_final_after_tax_total_value",
    ).reset_index()
    rename_values = {
        preferred_policy: "preferred_final_after_tax_value",
        "hold_to_terminal": "hold_to_terminal_final_after_tax_value",
        "sell_immediately": "sell_immediately_final_after_tax_value",
        "sell_half_then_hold": "sell_half_then_hold_final_after_tax_value",
        "sell_quarters_over_time": "sell_quarters_over_time_final_after_tax_value",
        "random_policy": "random_policy_final_after_tax_value",
        OPTIONAL_DQN_POLICY: "optional_0p090_dqn_final_after_tax_value",
    }
    value_wide = value_wide.rename(columns=rename_values)
    required_value_columns = [
        "preferred_final_after_tax_value",
        "hold_to_terminal_final_after_tax_value",
        "sell_immediately_final_after_tax_value",
        "sell_half_then_hold_final_after_tax_value",
        "sell_quarters_over_time_final_after_tax_value",
    ]
    if value_wide[required_value_columns].isna().any().any():
        raise ValueError("Step 11 has unpaired final-value rows for required policies.")
    preferred = metrics[metrics["policy_name"].eq(preferred_policy)].copy()
    preferred = preferred.rename(
        columns={
            "episode_final_after_tax_total_value": "preferred_final_after_tax_value_source",
            "episode_realized_after_tax_pnl": "preferred_realized_after_tax_pnl",
            "total_tax_paid": "preferred_tax_paid",
            "pct_episode_position_sold_short_term": "short_term_sold_fraction",
            "pct_episode_position_sold_long_term": "long_term_sold_fraction",
            "episode_terminal_liquidation_executed": "terminal_liquidation_flag",
            "episode_cut_occurred": "discretionary_sale_flag",
            "days_to_first_sale": "first_sale_day",
            "steps_to_first_sale": "first_sale_step",
            "first_sale_date": "first_sale_date",
            "first_cut_fraction_executed": "first_sale_executed_fraction",
            "terminal_liquidation_fraction": "terminal_liquidation_fraction",
            "episode_final_sold_fraction": "cumulative_sold_fraction",
            "episode_final_remaining_fraction": "remaining_fraction",
        }
    )
    preferred["no_cut_flag"] = ~preferred["discretionary_sale_flag"].fillna(False)
    keep_columns = [
        "split",
        "episode_id",
        "preferred_realized_after_tax_pnl",
        "preferred_tax_paid",
        "short_term_sold_fraction",
        "long_term_sold_fraction",
        "terminal_liquidation_flag",
        "no_cut_flag",
        "discretionary_sale_flag",
        "first_sale_day",
        "first_sale_step",
        "first_sale_date",
        "first_sale_executed_fraction",
        "terminal_liquidation_fraction",
        "cumulative_sold_fraction",
        "remaining_fraction",
        "num_discretionary_sales",
    ]
    base = value_wide.merge(
        preferred[keep_columns], on=["split", "episode_id"], how="inner", validate="one_to_one"
    )
    raw_summary = summarize_raw_metadata(raw_df)
    if not raw_summary.empty:
        base = base.merge(raw_summary, on="episode_id", how="left", validate="many_to_one")
    else:
        base["ticker"] = base["episode_id"].map(parse_ticker)
        base["episode_start_date"] = pd.NaT
        base["terminal_date"] = pd.NaT
        base["tax_transition_date"] = pd.NaT
        base["terminal_unrealized_gain"] = np.nan
    path_features = build_path_features(
        step_df, raw_df, preferred_policy=preferred_policy, warnings=warnings
    )
    base = base.merge(path_features, on=["split", "episode_id"], how="left", validate="one_to_one")
    for column, fallback in [
        ("ticker", base["episode_id"].map(parse_ticker)),
        ("episode_start_date", base.get("episode_start_date_from_path")),
        ("terminal_date", base.get("terminal_date_from_path")),
        ("tax_transition_date", base.get("tax_transition_date_from_path")),
        ("first_sale_day", base.get("first_sale_step_from_path")),
        ("first_sale_step", base.get("first_sale_step_from_path")),
        ("first_sale_date", base.get("first_sale_date_from_path")),
        ("first_sale_executed_fraction", base.get("first_sale_executed_fraction_from_path")),
    ]:
        if column in base.columns and fallback is not None:
            base[column] = base[column].where(base[column].notna(), fallback)
    if base["ticker"].isna().any() or base["ticker"].astype(str).eq("").any():
        warnings.append(
            "WARNING ticker metadata is incomplete; missing tickers are parsed from episode_id where possible."
        )
        base["ticker"] = base["ticker"].fillna(base["episode_id"].map(parse_ticker))
    base["dqn_minus_hold"] = (
        base["preferred_final_after_tax_value"]
        - base["hold_to_terminal_final_after_tax_value"]
    )
    base["dqn_minus_sell_immediately"] = (
        base["preferred_final_after_tax_value"]
        - base["sell_immediately_final_after_tax_value"]
    )
    base["dqn_minus_sell_half_then_hold"] = (
        base["preferred_final_after_tax_value"]
        - base["sell_half_then_hold_final_after_tax_value"]
    )
    base["dqn_minus_sell_quarters_over_time"] = (
        base["preferred_final_after_tax_value"]
        - base["sell_quarters_over_time_final_after_tax_value"]
    )
    base["discretionary_sale_flag"] = base["discretionary_sale_flag"].fillna(False).astype(bool)
    base["no_cut_flag"] = base["no_cut_flag"].fillna(~base["discretionary_sale_flag"]).astype(bool)
    base["terminal_liquidation_flag"] = (
        base["terminal_liquidation_flag"].fillna(False).astype(bool)
    )
    return base


def classify_scenarios(
    base: pd.DataFrame,
    *,
    post_sale_drawdown_threshold: float,
    post_sale_upside_threshold: float,
    hold_like_abs_tolerance: float,
) -> pd.DataFrame:
    has_sale = base["discretionary_sale_flag"].fillna(False)
    no_short_term_discretionary = base["short_term_discretionary_sold_fraction"].fillna(0).le(EPS)
    active_wins = (
        base["dqn_minus_sell_immediately"].gt(0)
        | base["dqn_minus_sell_half_then_hold"].gt(0)
        | base["dqn_minus_sell_quarters_over_time"].gt(0)
    )
    masks = {
        "dqn_sells_early_avoids_later_loss": (
            has_sale
            & base["dqn_minus_hold"].gt(0)
            & base["post_sale_drawdown"].ge(post_sale_drawdown_threshold)
        ),
        "dqn_sells_too_early_underperforms_hold": (
            has_sale
            & base["dqn_minus_hold"].lt(0)
            & base["post_sale_upside"].ge(post_sale_upside_threshold)
        ),
        "dqn_waits_correctly_until_long_term": (
            no_short_term_discretionary
            & (
                base["has_discretionary_sale_after_transition"].fillna(False)
                | base["terminal_liquidation_flag"].fillna(False)
            )
            & base["dqn_minus_sell_immediately"].gt(0)
            & (
                base["dqn_minus_sell_half_then_hold"].gt(0)
                | base["dqn_minus_sell_quarters_over_time"].gt(0)
            )
        ),
        "dqn_behaves_like_hold_to_terminal": (
            ~has_sale
            & base["terminal_liquidation_flag"].fillna(False)
            & base["dqn_minus_hold"].abs().le(hold_like_abs_tolerance)
        ),
        "partial_liquidation_useful": (
            base["discretionary_sold_fraction_before_terminal"].fillna(0).gt(EPS)
            & base["discretionary_sold_fraction_before_terminal"].fillna(0).lt(1.0 - EPS)
            & base["first_sale_remaining_fraction"].fillna(0).gt(EPS)
            & active_wins
        ),
        "dqn_loses_to_all_major_benchmarks": (
            base["dqn_minus_hold"].lt(0)
            & base["dqn_minus_sell_immediately"].lt(0)
            & base["dqn_minus_sell_half_then_hold"].lt(0)
            & base["dqn_minus_sell_quarters_over_time"].lt(0)
        ),
    }
    ranking_metrics = {
        "dqn_sells_early_avoids_later_loss": base["dqn_minus_hold"] + base["post_sale_drawdown"],
        "dqn_sells_too_early_underperforms_hold": (
            -base["dqn_minus_hold"] + base["post_sale_upside"]
        ),
        "dqn_waits_correctly_until_long_term": base[
            [
                "dqn_minus_sell_immediately",
                "dqn_minus_sell_half_then_hold",
                "dqn_minus_sell_quarters_over_time",
            ]
        ].max(axis=1),
        "dqn_behaves_like_hold_to_terminal": -base["dqn_minus_hold"].abs(),
        "partial_liquidation_useful": base[
            [
                "dqn_minus_sell_immediately",
                "dqn_minus_sell_half_then_hold",
                "dqn_minus_sell_quarters_over_time",
            ]
        ].max(axis=1),
        "dqn_loses_to_all_major_benchmarks": -base[
            [
                "dqn_minus_hold",
                "dqn_minus_sell_immediately",
                "dqn_minus_sell_half_then_hold",
                "dqn_minus_sell_quarters_over_time",
            ]
        ].max(axis=1),
    }
    rows: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        frame = base[["split", "episode_id"]].copy()
        frame["scenario"] = scenario
        frame["is_member"] = masks[scenario].fillna(False).astype(bool)
        frame["ranking_metric"] = ranking_metrics[scenario].where(frame["is_member"], np.nan)
        frame["notes"] = np.where(
            frame["is_member"],
            "Scenario definition satisfied by preferred DQN behavior and paired benchmark outcomes.",
            "Scenario definition not satisfied.",
        )
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["split", "scenario", "episode_id"]).reset_index(drop=True)


def build_scenario_summary(base: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_episode_counts = base.groupby("split")["episode_id"].nunique().to_dict()
    for split in PRIMARY_SPLITS:
        split_count = int(split_episode_counts.get(split, 0))
        for scenario in SCENARIOS:
            members = classification[
                classification["split"].eq(split)
                & classification["scenario"].eq(scenario)
                & classification["is_member"]
            ][["split", "episode_id"]]
            data = members.merge(base, on=["split", "episode_id"], how="left")
            rows.append(
                {
                    "split": split,
                    "scenario": scenario,
                    "number_of_candidate_episodes": int(len(data)),
                    "percentage_of_split": float(len(data) / split_count) if split_count else np.nan,
                    "mean_preferred_dqn_final_after_tax_value": data[
                        "preferred_final_after_tax_value"
                    ].mean(),
                    "mean_hold_to_terminal_final_after_tax_value": data[
                        "hold_to_terminal_final_after_tax_value"
                    ].mean(),
                    "mean_dqn_minus_hold_difference": data["dqn_minus_hold"].mean(),
                    "median_dqn_minus_hold_difference": data["dqn_minus_hold"].median(),
                    "dqn_win_rate_versus_hold": data["dqn_minus_hold"].gt(0).mean()
                    if len(data)
                    else np.nan,
                    "dqn_win_rate_versus_sell_immediately": data[
                        "dqn_minus_sell_immediately"
                    ].gt(0).mean()
                    if len(data)
                    else np.nan,
                    "dqn_win_rate_versus_sell_half_then_hold": data[
                        "dqn_minus_sell_half_then_hold"
                    ].gt(0).mean()
                    if len(data)
                    else np.nan,
                    "dqn_win_rate_versus_sell_quarters_over_time": data[
                        "dqn_minus_sell_quarters_over_time"
                    ].gt(0).mean()
                    if len(data)
                    else np.nan,
                    "mean_first_sale_day": data["first_sale_day"].mean(),
                    "median_first_sale_day": data["first_sale_day"].median(),
                    "mean_short_term_sold_fraction": data["short_term_sold_fraction"].mean(),
                    "mean_long_term_sold_fraction": data["long_term_sold_fraction"].mean(),
                    "mean_terminal_liquidation_frequency": data[
                        "terminal_liquidation_flag"
                    ].astype(float).mean()
                    if len(data)
                    else np.nan,
                    "mean_no_cut_frequency": data["no_cut_flag"].astype(float).mean()
                    if len(data)
                    else np.nan,
                    "mean_discretionary_sale_frequency": data[
                        "discretionary_sale_flag"
                    ].astype(float).mean()
                    if len(data)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def normalized_abs_diff(series: pd.Series, median_value: float, scale: float = 1.0) -> pd.Series:
    if not np.isfinite(median_value):
        return pd.Series(0.0, index=series.index)
    filled = series.fillna(median_value)
    return (filled - median_value).abs() / scale


def build_candidate_ranking(base: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split in PRIMARY_SPLITS:
        for scenario in SCENARIOS:
            members = classification[
                classification["split"].eq(split)
                & classification["scenario"].eq(scenario)
                & classification["is_member"]
            ][["split", "episode_id", "scenario"]]
            if members.empty:
                continue
            data = members.merge(base, on=["split", "episode_id"], how="left")
            median_diff = float(data["dqn_minus_hold"].median())
            median_sale_day = float(data["first_sale_day"].median()) if data["first_sale_day"].notna().any() else np.nan
            median_short_term = float(data["short_term_sold_fraction"].median())
            data["representativeness_score"] = (
                normalized_abs_diff(data["dqn_minus_hold"], median_diff)
                + normalized_abs_diff(data["first_sale_day"], median_sale_day, scale=252.0)
                + normalized_abs_diff(data["short_term_sold_fraction"], median_short_term)
            )
            data = data.sort_values(
                ["representativeness_score", "episode_id"], ascending=[True, True]
            ).head(10)
            data["rank"] = np.arange(1, len(data) + 1)
            rows.append(data)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    columns = [
        "split",
        "scenario",
        "rank",
        "episode_id",
        "ticker",
        "episode_start_date",
        "tax_transition_date",
        "terminal_date",
        "preferred_final_after_tax_value",
        "hold_to_terminal_final_after_tax_value",
        "sell_immediately_final_after_tax_value",
        "sell_half_then_hold_final_after_tax_value",
        "sell_quarters_over_time_final_after_tax_value",
        "random_policy_final_after_tax_value",
        "optional_0p090_dqn_final_after_tax_value",
        "dqn_minus_hold",
        "dqn_minus_sell_immediately",
        "dqn_minus_sell_half_then_hold",
        "dqn_minus_sell_quarters_over_time",
        "first_sale_day",
        "first_sale_executed_fraction",
        "short_term_sold_fraction",
        "long_term_sold_fraction",
        "terminal_liquidation_flag",
        "no_cut_flag",
        "discretionary_sale_flag",
        "post_sale_drawdown",
        "post_sale_upside",
        "representativeness_score",
        "preferred_path_num_points",
        "preferred_path_unique_dates",
        "preferred_after_tax_value_path_range",
    ]
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out[columns]


def policy_value_columns() -> dict[str, str]:
    return {
        "preferred DQN": "preferred_final_after_tax_value",
        "hold_to_terminal": "hold_to_terminal_final_after_tax_value",
        "sell_immediately": "sell_immediately_final_after_tax_value",
        "sell_half_then_hold": "sell_half_then_hold_final_after_tax_value",
        "sell_quarters_over_time": "sell_quarters_over_time_final_after_tax_value",
        "random_policy": "random_policy_final_after_tax_value",
        "0p090 DQN": "optional_0p090_dqn_final_after_tax_value",
    }


def benchmark_winner_and_rank(row: pd.Series) -> tuple[str, int]:
    values = {
        label: row[column]
        for label, column in policy_value_columns().items()
        if column in row.index and pd.notna(row[column])
    }
    sorted_values = sorted(values.items(), key=lambda item: item[1], reverse=True)
    benchmark_labels = {
        "hold_to_terminal",
        "sell_immediately",
        "sell_half_then_hold",
        "sell_quarters_over_time",
        "random_policy",
    }
    benchmark_values = {
        label: value for label, value in values.items() if label in benchmark_labels
    }
    winner = max(benchmark_values.items(), key=lambda item: item[1])[0]
    rank = 1 + sum(value > values["preferred DQN"] for _, value in sorted_values)
    return winner, int(rank)


def first_sale_action_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"Sell {float(value) * 100:.0f}%"


def select_case_studies(
    ranking: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    warnings: list[str],
) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    selected_episode_ids: set[str] = set()

    def pick_for_scenario(scenario: str, allow_validation: bool = True) -> pd.Series | None:
        for split in (["test", "validation"] if allow_validation else ["test"]):
            candidates = ranking[
                ranking["scenario"].eq(scenario)
                & ranking["split"].eq(split)
                & ~ranking["episode_id"].astype(str).isin(selected_episode_ids)
            ].sort_values(["representativeness_score", "rank", "episode_id"])
            readable = candidates[
                candidates["preferred_path_unique_dates"].fillna(0).ge(5)
                & candidates["preferred_after_tax_value_path_range"].fillna(0).gt(0.0025)
            ]
            if not readable.empty:
                return readable.iloc[0]
            if not candidates.empty:
                warnings.append(
                    "WARNING selected case-study candidate has a short or nearly flat path: "
                    f"split={split}, scenario={scenario}, episode_id={candidates.iloc[0]['episode_id']}."
                )
                return candidates.iloc[0]
        return None

    for scenario in PREFERRED_SELECTION_SCENARIOS:
        row = pick_for_scenario(scenario)
        if row is None:
            warnings.append(f"WARNING scenario has no selectable case-study candidate: {scenario}.")
            continue
        selected_episode_ids.add(str(row["episode_id"]))
        selected_rows.append(row)

    if len(selected_rows) < 4:
        fallback_order = ["dqn_loses_to_all_major_benchmarks", "partial_liquidation_useful"]
        for scenario in fallback_order:
            row = pick_for_scenario(scenario)
            if row is not None:
                selected_episode_ids.add(str(row["episode_id"]))
                selected_rows.append(row)
            if len(selected_rows) >= 4:
                break

    has_failure = any(
        row["scenario"]
        in {"dqn_sells_too_early_underperforms_hold", "dqn_loses_to_all_major_benchmarks"}
        for row in selected_rows
    )
    if not has_failure:
        row = pick_for_scenario("dqn_loses_to_all_major_benchmarks")
        if row is not None:
            selected_episode_ids.add(str(row["episode_id"]))
            selected_rows.append(row)
        else:
            warnings.append("WARNING no clear DQN failure case was available for final selection.")

    selected_rows = selected_rows[:5]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        member = classification[
            classification["split"].eq(row["split"])
            & classification["episode_id"].eq(row["episode_id"])
            & classification["scenario"].eq(row["scenario"])
            & classification["is_member"]
        ]
        if member.empty:
            raise ValueError(
                "Selected Step 11 episode does not belong to its assigned scenario: "
                f"{row['split']} {row['episode_id']} {row['scenario']}"
            )
        winner, rank = benchmark_winner_and_rank(row)
        rows.append(
            {
                "case_study_number": index,
                "scenario": row["scenario"],
                "split": row["split"],
                "episode_id": row["episode_id"],
                "ticker": row["ticker"],
                "reason_selected": (
                    "Lowest representativeness score among available readable-path "
                    f"{row['split']} candidates for {row['scenario']}."
                ),
                "preferred_dqn_final_after_tax_value": row[
                    "preferred_final_after_tax_value"
                ],
                "benchmark_winner": winner,
                "dqn_rank_among_policies": rank,
                "dqn_minus_hold": row["dqn_minus_hold"],
                "first_sale_day": row["first_sale_day"],
                "first_sale_action": first_sale_action_label(
                    row["first_sale_executed_fraction"]
                ),
                "short_term_sold_fraction": row["short_term_sold_fraction"],
                "long_term_sold_fraction": row["long_term_sold_fraction"],
                "terminal_liquidation_flag": row["terminal_liquidation_flag"],
                "plot_file": "",
            }
        )
    return pd.DataFrame(rows)


def safe_filename_part(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def load_gain_path(raw_df: pd.DataFrame, episode_id: str) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    path = raw_df[raw_df["episode_id"].astype(str).eq(str(episode_id))].sort_values("date")
    return path[["date", "unrealized_gains_pct"]].dropna()


def plot_case_study(
    selected_row: pd.Series,
    step_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    preferred_policy: str,
    plot_dir: Path,
) -> Path:
    split = str(selected_row["split"])
    episode_id = str(selected_row["episode_id"])
    scenario = str(selected_row["scenario"])
    case_number = int(selected_row["case_study_number"])
    policies = [preferred_policy, *PLOT_BENCHMARK_POLICIES]
    label_map = {
        preferred_policy: "Preferred DQN",
        "hold_to_terminal": "Hold to terminal",
        "sell_immediately": "Sell immediately",
        "sell_half_then_hold": "Sell half then hold",
        "sell_quarters_over_time": "Sell quarters",
    }
    colors = {
        preferred_policy: "#1b4e8a",
        "hold_to_terminal": "#444444",
        "sell_immediately": "#9a4c2e",
        "sell_half_then_hold": "#4f7d3a",
        "sell_quarters_over_time": "#7a5aa6",
    }
    linestyles = {
        preferred_policy: "-",
        "hold_to_terminal": "-",
        "sell_immediately": "-.",
        "sell_half_then_hold": "-",
        "sell_quarters_over_time": (0, (3, 1, 1, 1)),
    }
    markers = {
        preferred_policy: None,
        "hold_to_terminal": None,
        "sell_immediately": "s",
        "sell_half_then_hold": None,
        "sell_quarters_over_time": "D",
    }
    paths: dict[str, pd.DataFrame] = {}
    for policy in policies:
        path = step_df[
            step_df["split"].eq(split)
            & step_df["episode_id"].astype(str).eq(episode_id)
            & step_df["policy_name"].eq(policy)
        ].sort_values("step_in_episode")
        if path.empty:
            raise ValueError(
                "Path-level data cannot be reconstructed for selected episode: "
                f"split={split}, episode_id={episode_id}, policy={policy}"
            )
        paths[policy] = path
    preferred_path = paths[preferred_policy]
    x_date_available = preferred_path["date"].notna().all()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0]},
    )
    ax = axes[0]
    for policy in policies:
        path = paths[policy]
        x = path["date"] if x_date_available and path["date"].notna().all() else path["step_in_episode"]
        linewidth = 2.4 if policy == preferred_policy else 1.6
        ax.plot(
            x,
            path["after_tax_total_value"],
            label=label_map[policy],
            color=colors[policy],
            linewidth=linewidth,
            linestyle=linestyles[policy],
            marker=markers[policy],
            markersize=5.0 if len(path) <= 8 else 0.0,
            markerfacecolor="white" if markers[policy] else None,
            markeredgewidth=1.2 if markers[policy] else None,
            markevery=1 if len(path) <= 8 else None,
            alpha=0.95,
            zorder=4 if policy == "sell_quarters_over_time" else 3,
        )
        if policy == "sell_quarters_over_time" and not path.empty:
            last = path.iloc[-1]
            last_x = last["date"] if x_date_available and pd.notna(last["date"]) else last["step_in_episode"]
            ax.annotate(
                "quarters",
                xy=(last_x, last["after_tax_total_value"]),
                xytext=(6, -12),
                textcoords="offset points",
                fontsize=7,
                color=colors[policy],
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": colors[policy],
                    "lw": 0.5,
                },
            )
    gain_path = load_gain_path(raw_df, episode_id)
    twin = None
    if not gain_path.empty and x_date_available:
        twin = ax.twinx()
        twin.plot(
            gain_path["date"],
            gain_path["unrealized_gains_pct"],
            color="#777777",
            linestyle="--",
            linewidth=1.2,
            alpha=0.65,
            label="Unrealized gain",
        )
        twin.set_ylabel("unrealized gain")
        twin.tick_params(axis="y", labelsize=8)
    sales = preferred_path[sale_mask(preferred_path)]
    if not sales.empty:
        sale_x = sales["date"] if x_date_available else sales["step_in_episode"]
        ax.scatter(
            sale_x,
            sales["after_tax_total_value"],
            marker="o",
            color="black",
            s=38,
            zorder=5,
            label="Preferred sale",
        )
        y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for row in sales.itertuples(index=False):
            x_value = row.date if x_date_available else row.step_in_episode
            label = f"Sell {float(row.action_fraction_executed) * 100:.0f}%"
            ax.annotate(
                label,
                xy=(x_value, row.after_tax_total_value),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=7,
                color="black",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "lw": 0.4},
            )
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.05 * y_span)
    transition = preferred_path["tax_transition_date"].dropna()
    if not transition.empty:
        transition_x = transition.iloc[0] if x_date_available else None
        if transition_x is not None:
            for plot_ax in axes:
                plot_ax.axvline(
                    transition_x,
                    color="#666666",
                    linestyle=":",
                    linewidth=1.2,
                    label="Tax transition" if plot_ax is ax else None,
                )
    lines, labels = ax.get_legend_handles_labels()
    if twin is not None:
        twin_lines, twin_labels = twin.get_legend_handles_labels()
        lines.extend(twin_lines)
        labels.extend(twin_labels)
    deduped: dict[str, Any] = {}
    for line, label in zip(lines, labels):
        if label and not label.startswith("_"):
            deduped.setdefault(label, line)
    ax.legend(deduped.values(), deduped.keys(), fontsize=8, loc="best")
    title_ticker = selected_row["ticker"] if pd.notna(selected_row["ticker"]) else ""
    ax.set_title(
        f"Case {case_number}: {scenario} | {split} | {episode_id} | {title_ticker}",
        fontsize=11,
    )
    ax.set_ylabel("after-tax value")
    ax.grid(alpha=0.25)
    lower = axes[1]
    lower_x = preferred_path["date"] if x_date_available else preferred_path["step_in_episode"]
    lower.plot(
        lower_x,
        preferred_path["remaining_fraction"],
        label="Preferred remaining fraction",
        color="#1b4e8a",
        linewidth=2.0,
    )
    lower.plot(
        lower_x,
        preferred_path["sold_fraction"],
        label="Preferred sold fraction",
        color="#8a8a8a",
        linewidth=1.4,
        linestyle="--",
    )
    lower.set_ylabel("position fraction")
    lower.set_ylim(-0.03, 1.03)
    lower.grid(alpha=0.25)
    lower.legend(fontsize=8, loc="best")
    lower.set_xlabel("date" if x_date_available else "step in episode")
    if x_date_available:
        lower.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate(rotation=30, ha="right")
    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"step11_case_{case_number:02d}_{safe_filename_part(scenario)}_"
        f"{safe_filename_part(split)}_{safe_filename_part(episode_id)}.png"
    )
    path = plot_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    if not path.exists():
        raise FileNotFoundError(f"Step 11 plot was not written: {relative_project_path(path)}")
    return path


def write_notes(
    path: Path,
    *,
    warnings: list[str],
    selected: pd.DataFrame,
    raw_df: pd.DataFrame,
    preferred_policy: str,
    post_sale_drawdown_threshold: float,
    post_sale_upside_threshold: float,
    hold_like_abs_tolerance: float,
) -> None:
    lines = [
        "Step 11 case-study episode diagnostics notes",
        "Step 11 is interpretive and does not change the model.",
        f"The preferred DQN remains {preferred_policy}.",
        "Validation/test only are used for case selection.",
        "The scenario table is used to avoid cherry-picking.",
        "The candidate ranking table ranks representative episodes within each scenario before plots are selected.",
        "The case studies show both success and failure modes when available.",
        "The plots should be interpreted alongside Steps 7-10, not as standalone performance evidence.",
        "Final after-tax value remains the primary metric.",
        "The plots include benchmark paths to show whether DQN behavior is actually economically useful.",
        "No retraining, reward redesign, environment change, threshold reselection, or policy behavior change was performed.",
        f"post_sale_drawdown_threshold: {post_sale_drawdown_threshold}",
        f"post_sale_upside_threshold: {post_sale_upside_threshold}",
        f"hold_like_abs_tolerance: {hold_like_abs_tolerance}",
        "unrealized_gain_path_source: "
        + ("data/episodes/drl_episodes.parquet unrealized_gains_pct" if not raw_df.empty else "hold-to-terminal after-tax path proxy"),
        "selected_case_count: " + str(len(selected)),
        "selected_cases: "
        + "; ".join(
            f"{row.case_study_number}:{row.scenario}:{row.split}:{row.episode_id}"
            for row in selected.itertuples(index=False)
        ),
    ]
    if warnings:
        lines.extend(["", "Warnings and diagnostics", *warnings])
    else:
        lines.extend(["", "Warnings and diagnostics", "No Step 11 warnings."])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def update_manifest(output_dir: Path, output_paths: list[Path]) -> None:
    manifest_path = output_dir / "phase_4_11_outputs_manifest.json"
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(
            f"Quant-analysis manifest must contain a list: {relative_project_path(manifest_path)}"
        )
    current_step11_plot_paths = {
        relative_project_path(path)
        for path in output_paths
        if "/plots/step11_case_studies/" in relative_project_path(path)
    }
    payload = [
        row
        for row in payload
        if not (
            isinstance(row, dict)
            and isinstance(row.get("path"), str)
            and "/plots/step11_case_studies/" in row["path"]
            and row["path"] not in current_step11_plot_paths
        )
    ]
    existing_paths = {
        row.get("path")
        for row in payload
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    descriptions = {
        "step11_case_study_scenario_classification": "Step 11 long-format scenario membership classification.",
        "step11_case_study_scenario_summary": "Step 11 scenario summary table.",
        "step11_case_study_candidate_ranking": "Step 11 representative candidate ranking table.",
        "step11_selected_case_studies": "Step 11 selected case-study episodes.",
        "step11_case_study_notes": "Step 11 case-study notes.",
    }
    for path in output_paths:
        rel = relative_project_path(path)
        if rel in existing_paths:
            continue
        stem = path.stem
        description = descriptions.get(stem, "Step 11 case-study plot.")
        payload.append(
            {
                "path": rel,
                "type": path.suffix.lstrip("."),
                "step": "11",
                "description": description,
            }
        )
        existing_paths.add(rel)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_selected_paths(selected: pd.DataFrame, step_df: pd.DataFrame, preferred_policy: str) -> None:
    for row in selected.itertuples(index=False):
        for policy in [preferred_policy, *PLOT_BENCHMARK_POLICIES]:
            path = step_df[
                step_df["split"].eq(row.split)
                & step_df["episode_id"].astype(str).eq(str(row.episode_id))
                & step_df["policy_name"].eq(policy)
            ]
            if path.empty:
                raise ValueError(
                    "Path-level data cannot be reconstructed for selected episode: "
                    f"split={row.split}, episode_id={row.episode_id}, policy={policy}"
                )


def clear_existing_case_study_plots(plot_dir: Path) -> None:
    if not plot_dir.exists():
        return
    for path in plot_dir.glob("step11_case_*.png"):
        path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Step 11 case-study episode diagnostics."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Frozen run directory. Defaults to {relative_project_path(DEFAULT_RUN_DIR)}.",
    )
    parser.add_argument(
        "--preferred-policy",
        default=DEFAULT_PREFERRED_POLICY,
        help="Preferred DQN policy name.",
    )
    parser.add_argument(
        "--post-sale-drawdown-threshold",
        type=float,
        default=POST_SALE_DRAWDOWN_THRESHOLD,
    )
    parser.add_argument(
        "--post-sale-upside-threshold",
        type=float,
        default=POST_SALE_UPSIDE_THRESHOLD,
    )
    parser.add_argument(
        "--hold-like-abs-tolerance",
        type=float,
        default=HOLD_LIKE_ABS_TOLERANCE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_project_path(args.run_dir)
    if run_dir.name != FINAL_RUN_NAME:
        raise ValueError(
            f"Expected frozen run directory {FINAL_RUN_NAME!r}, got {relative_project_path(run_dir)!r}."
        )
    output_dir = run_dir / "quant_analysis"
    plot_dir = output_dir / "plots" / "step11_case_studies"
    episode_path = run_dir / "baselines" / "baseline_episode_metrics.csv"
    step_path = run_dir / "baselines" / "baseline_step_rollouts.csv"
    if not episode_path.exists():
        raise FileNotFoundError(
            f"Missing episode-level policy evaluation file: {relative_project_path(episode_path)}"
        )
    if not step_path.exists():
        raise FileNotFoundError(
            f"Missing path-level policy evaluation file: {relative_project_path(step_path)}"
        )

    warnings: list[str] = []
    episode_df = prepare_episode_metrics(pd.read_csv(episode_path, low_memory=False))
    step_df = prepare_step_rollouts(pd.read_csv(step_path, low_memory=False))
    validate_inputs(episode_df, step_df, preferred_policy=args.preferred_policy)
    raw_df = load_raw_episode_metadata(run_dir, warnings)
    if "random_policy" not in set(step_df["policy_name"]):
        warnings.append("WARNING random policy path is unavailable.")
    if OPTIONAL_DQN_POLICY not in set(step_df["policy_name"]):
        warnings.append("WARNING optional 0p090 DQN path is unavailable.")

    base = build_episode_case_dataset(
        episode_df,
        step_df,
        raw_df,
        preferred_policy=args.preferred_policy,
        warnings=warnings,
    )
    classification = classify_scenarios(
        base,
        post_sale_drawdown_threshold=args.post_sale_drawdown_threshold,
        post_sale_upside_threshold=args.post_sale_upside_threshold,
        hold_like_abs_tolerance=args.hold_like_abs_tolerance,
    )
    for split in PRIMARY_SPLITS:
        for scenario in SCENARIOS:
            count = int(
                classification[
                    classification["split"].eq(split)
                    & classification["scenario"].eq(scenario)
                    & classification["is_member"]
                ].shape[0]
            )
            if count == 0:
                warnings.append(f"WARNING scenario has zero candidates: split={split}, scenario={scenario}.")
    summary = build_scenario_summary(base, classification)
    ranking = build_candidate_ranking(base, classification)
    if ranking.empty:
        raise ValueError("Step 11 candidate ranking is empty; no scenario candidates found.")
    selected = select_case_studies(ranking, classification, warnings=warnings)
    if len(selected) < 4:
        raise ValueError("Step 11 selected fewer than four case-study episodes.")
    validate_selected_paths(selected, step_df, args.preferred_policy)
    clear_existing_case_study_plots(plot_dir)
    plot_paths: list[Path] = []
    for index, row in selected.iterrows():
        plot_path = plot_case_study(
            row,
            step_df,
            raw_df,
            preferred_policy=args.preferred_policy,
            plot_dir=plot_dir,
        )
        selected.loc[index, "plot_file"] = relative_project_path(plot_path)
        plot_paths.append(plot_path)

    output_paths = [
        output_dir / OUTPUT_FILES["classification_csv"],
        output_dir / OUTPUT_FILES["classification_md"],
        output_dir / OUTPUT_FILES["summary_csv"],
        output_dir / OUTPUT_FILES["summary_md"],
        output_dir / OUTPUT_FILES["ranking_csv"],
        output_dir / OUTPUT_FILES["ranking_md"],
        output_dir / OUTPUT_FILES["selected_csv"],
        output_dir / OUTPUT_FILES["selected_md"],
        output_dir / OUTPUT_FILES["notes"],
        *plot_paths,
    ]
    save_table(
        classification,
        output_dir / OUTPUT_FILES["classification_csv"],
        output_dir / OUTPUT_FILES["classification_md"],
    )
    save_table(
        summary,
        output_dir / OUTPUT_FILES["summary_csv"],
        output_dir / OUTPUT_FILES["summary_md"],
    )
    save_table(
        ranking,
        output_dir / OUTPUT_FILES["ranking_csv"],
        output_dir / OUTPUT_FILES["ranking_md"],
    )
    save_table(
        selected,
        output_dir / OUTPUT_FILES["selected_csv"],
        output_dir / OUTPUT_FILES["selected_md"],
    )
    write_notes(
        output_dir / OUTPUT_FILES["notes"],
        warnings=warnings,
        selected=selected,
        raw_df=raw_df,
        preferred_policy=args.preferred_policy,
        post_sale_drawdown_threshold=args.post_sale_drawdown_threshold,
        post_sale_upside_threshold=args.post_sale_upside_threshold,
        hold_like_abs_tolerance=args.hold_like_abs_tolerance,
    )
    for path in output_paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected Step 11 output was not written: {relative_project_path(path)}")
    update_manifest(output_dir, output_paths)

    print(f"Wrote Step 11 outputs to {relative_project_path(output_dir)}")
    print(f"Scenario classification rows: {len(classification)}")
    print(f"Candidate ranking rows: {len(ranking)}")
    print(f"Selected case studies: {len(selected)}")
    print("Selected plots:")
    for path in plot_paths:
        print(f"  {relative_project_path(path)}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  {warning}")


if __name__ == "__main__":
    main()
