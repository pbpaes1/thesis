"""Build RQ2/H2 tax-transition timing diagnostics for Chapter 4.

The script uses the final out-of-sample baseline rollout artifacts. These files
contain both the preferred DQN policy and the benchmark policies, which keeps
the timing comparison on the same episode universe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path("runs/train_reward_c_lite_v5_full")
PREFERRED_POLICY = "trained_dqn_first_sale_margin_0p070_normal_0p020"
POLICY_LABELS = {
    PREFERRED_POLICY: "Preferred DQN",
    "hold_to_terminal": "Passive hold-to-terminal",
    "sell_immediately": "Naive active: sell immediately",
    "sell_half_then_hold": "Naive active: sell half then hold",
    "sell_quarters_over_time": "Naive active: sell quarters",
    "random_policy": "Random benchmark",
}
POLICIES = list(POLICY_LABELS)
SPLIT_ORDER = ["validation", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tax-transition timing diagnostics for RQ2/H2."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def pct(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return 100.0 * float(value)


def safe_mean(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def safe_median(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    return float(values.median())


def read_inputs(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_dir = run_dir / "baselines"
    episode_path = baseline_dir / "baseline_episode_metrics.csv"
    step_path = baseline_dir / "baseline_step_rollouts.csv"
    if not episode_path.exists():
        raise FileNotFoundError(f"Missing episode metrics: {episode_path}")
    if not step_path.exists():
        raise FileNotFoundError(f"Missing step rollouts: {step_path}")

    episode_cols = [
        "split",
        "policy_name",
        "episode_id",
        "episode_final_after_tax_total_value",
        "episode_cut_occurred",
        "first_sale_before_tax_transition",
        "total_position_sold_short_term",
        "total_position_sold_long_term",
        "terminal_liquidation_fraction",
        "num_discretionary_sales",
    ]
    step_cols = [
        "split",
        "policy_name",
        "episode_id",
        "step_in_episode",
        "date",
        "tax_transition_date",
        "action_fraction_executed",
        "sold_fraction",
        "remaining_fraction",
        "tax_regime",
        "after_tax_liquidation_tax_regime",
        "after_tax_total_value",
        "terminal_liquidation_executed",
        "done",
    ]

    episode = pd.read_csv(episode_path, usecols=episode_cols)
    step = pd.read_csv(step_path, usecols=step_cols, low_memory=False)

    episode = episode[episode["policy_name"].isin(POLICIES)].copy()
    step = step[step["policy_name"].isin(POLICIES)].copy()
    episode["policy_label"] = episode["policy_name"].map(POLICY_LABELS)
    step["policy_label"] = step["policy_name"].map(POLICY_LABELS)
    step["date"] = pd.to_datetime(step["date"])
    step["tax_transition_date"] = pd.to_datetime(step["tax_transition_date"])
    step["terminal_liquidation_executed"] = step[
        "terminal_liquidation_executed"
    ].fillna(False).astype(bool)
    return episode, step


def add_sale_increments(step: pd.DataFrame) -> pd.DataFrame:
    ordered = step.sort_values(
        ["split", "policy_name", "episode_id", "step_in_episode"]
    ).copy()
    prev_sold = ordered.groupby(["split", "policy_name", "episode_id"])[
        "sold_fraction"
    ].shift(1)
    ordered["sale_fraction_of_original"] = (
        ordered["sold_fraction"].fillna(0.0) - prev_sold.fillna(0.0)
    ).clip(lower=0.0)
    ordered["is_discretionary_sale"] = (
        ordered["action_fraction_executed"].fillna(0.0) > 1e-10
    ) & ~ordered["terminal_liquidation_executed"]
    ordered["days_remaining_to_tax_transition"] = (
        ordered["tax_transition_date"] - ordered["date"]
    ).dt.days

    regime = ordered["tax_regime"].astype("string").fillna("")
    terminal_regime = (
        ordered["after_tax_liquidation_tax_regime"].astype("string").fillna("")
    )
    ordered["realization_tax_regime"] = np.where(
        ordered["terminal_liquidation_executed"] & terminal_regime.ne(""),
        terminal_regime,
        regime,
    )
    ordered["adjusted_days_remaining_to_tax_transition"] = np.where(
        ordered["realization_tax_regime"].eq("long_term"),
        np.minimum(ordered["days_remaining_to_tax_transition"], 0),
        ordered["days_remaining_to_tax_transition"],
    )
    return ordered


def timing_bucket(days_remaining: float) -> str:
    if pd.isna(days_remaining):
        return "no_sale"
    if days_remaining <= 0:
        return "at_or_after_long_term"
    if days_remaining <= 7:
        return "1_to_7_days_before"
    if days_remaining <= 30:
        return "8_to_30_days_before"
    if days_remaining <= 90:
        return "31_to_90_days_before"
    if days_remaining <= 180:
        return "91_to_180_days_before"
    return "over_180_days_before"


def timing_bucket_order(bucket: str) -> int:
    order = {
        "over_180_days_before": 0,
        "91_to_180_days_before": 1,
        "31_to_90_days_before": 2,
        "8_to_30_days_before": 3,
        "1_to_7_days_before": 4,
        "at_or_after_long_term": 5,
        "no_sale": 6,
    }
    return order.get(bucket, 99)


def build_sale_events(step: pd.DataFrame) -> pd.DataFrame:
    sale_events = step[step["sale_fraction_of_original"] > 1e-10].copy()
    sale_events["sale_timing_bucket"] = sale_events[
        "adjusted_days_remaining_to_tax_transition"
    ].map(timing_bucket)
    sale_events["sale_timing_bucket_order"] = sale_events["sale_timing_bucket"].map(
        timing_bucket_order
    )
    return sale_events


def first_sale_table(
    sale_events: pd.DataFrame, discretionary_only: bool = False
) -> pd.DataFrame:
    events = sale_events
    if discretionary_only:
        events = events[events["is_discretionary_sale"]]
    if events.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "policy_name",
                "episode_id",
                "first_sale_date",
                "first_sale_days_remaining",
                "first_sale_timing_bucket",
            ]
        )
    first = (
        events.sort_values(["split", "policy_name", "episode_id", "date"])
        .groupby(["split", "policy_name", "episode_id"], as_index=False)
        .first()
    )
    first = first[
        [
            "split",
            "policy_name",
            "episode_id",
            "date",
            "adjusted_days_remaining_to_tax_transition",
            "sale_timing_bucket",
        ]
    ].rename(
        columns={
            "date": "first_sale_date",
            "adjusted_days_remaining_to_tax_transition": "first_sale_days_remaining",
            "sale_timing_bucket": "first_sale_timing_bucket",
        }
    )
    return first


def build_summary(
    episode: pd.DataFrame, sale_events: pd.DataFrame
) -> pd.DataFrame:
    first_realization = first_sale_table(sale_events, discretionary_only=False)
    first_disc = first_sale_table(sale_events, discretionary_only=True).rename(
        columns={
            "first_sale_date": "first_discretionary_sale_date",
            "first_sale_days_remaining": "first_discretionary_sale_days_remaining",
            "first_sale_timing_bucket": "first_discretionary_sale_timing_bucket",
        }
    )

    enriched = episode.merge(
        first_realization,
        on=["split", "policy_name", "episode_id"],
        how="left",
    ).merge(first_disc, on=["split", "policy_name", "episode_id"], how="left")

    hold_values = (
        enriched[enriched["policy_name"].eq("hold_to_terminal")][
            ["split", "episode_id", "episode_final_after_tax_total_value"]
        ]
        .rename(
            columns={
                "episode_final_after_tax_total_value": "hold_final_after_tax_total_value"
            }
        )
        .copy()
    )
    enriched = enriched.merge(hold_values, on=["split", "episode_id"], how="left")
    enriched["excess_final_after_tax_value_vs_hold"] = (
        enriched["episode_final_after_tax_total_value"]
        - enriched["hold_final_after_tax_total_value"]
    )

    rows = []
    for (split, policy), group in enriched.groupby(["split", "policy_name"], sort=False):
        rows.append(
            {
                "split": split,
                "policy_name": policy,
                "policy_label": POLICY_LABELS[policy],
                "num_episodes": len(group),
                "mean_pct_original_sold_before_long_term": pct(
                    group["total_position_sold_short_term"].mean()
                ),
                "mean_pct_original_sold_at_or_after_long_term": pct(
                    group["total_position_sold_long_term"].mean()
                ),
                "pct_episodes_with_discretionary_sale_before_long_term": pct(
                    group["first_sale_before_tax_transition"].fillna(False).mean()
                ),
                "pct_episodes_with_no_discretionary_sale": pct(
                    (group["num_discretionary_sales"].fillna(0) == 0).mean()
                ),
                "median_days_remaining_at_first_realization": group[
                    "first_sale_days_remaining"
                ].pipe(safe_median),
                "mean_days_remaining_at_first_realization": group[
                    "first_sale_days_remaining"
                ].pipe(safe_mean),
                "median_days_remaining_at_first_discretionary_sale": group[
                    "first_discretionary_sale_days_remaining"
                ].pipe(safe_median),
                "mean_days_remaining_at_first_discretionary_sale": group[
                    "first_discretionary_sale_days_remaining"
                ].pipe(safe_mean),
                "mean_final_after_tax_total_value": group[
                    "episode_final_after_tax_total_value"
                ].mean(),
                "mean_excess_final_after_tax_value_vs_hold": group[
                    "excess_final_after_tax_value_vs_hold"
                ].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary["split"] = pd.Categorical(summary["split"], SPLIT_ORDER, ordered=True)
    summary["policy_name"] = pd.Categorical(summary["policy_name"], POLICIES, ordered=True)
    return summary.sort_values(["split", "policy_name"]).reset_index(drop=True)


def build_sale_distribution(sale_events: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        sale_events.groupby(
            [
                "split",
                "policy_name",
                "policy_label",
                "sale_timing_bucket",
                "sale_timing_bucket_order",
            ],
            as_index=False,
        )
        .agg(
            sale_event_count=("sale_fraction_of_original", "size"),
            sold_fraction_of_original=("sale_fraction_of_original", "sum"),
        )
        .copy()
    )
    totals = distribution.groupby(["split", "policy_name"])[
        "sold_fraction_of_original"
    ].transform("sum")
    distribution["share_of_policy_sales_pct"] = 100.0 * (
        distribution["sold_fraction_of_original"] / totals
    )
    distribution["split"] = pd.Categorical(
        distribution["split"], SPLIT_ORDER, ordered=True
    )
    distribution["policy_name"] = pd.Categorical(
        distribution["policy_name"], POLICIES, ordered=True
    )
    return distribution.sort_values(
        ["split", "policy_name", "sale_timing_bucket_order"]
    ).reset_index(drop=True)


def build_expost_dqn_diagnostic(
    episode: pd.DataFrame, step: pd.DataFrame, sale_events: pd.DataFrame
) -> pd.DataFrame:
    dqn_policy = PREFERRED_POLICY
    dqn_episode = episode[episode["policy_name"].eq(dqn_policy)].copy()
    dqn_first_disc = first_sale_table(
        sale_events[sale_events["policy_name"].eq(dqn_policy)],
        discretionary_only=True,
    ).rename(
        columns={
            "first_sale_date": "first_discretionary_sale_date",
            "first_sale_days_remaining": "first_discretionary_sale_days_remaining",
            "first_sale_timing_bucket": "first_discretionary_sale_timing_bucket",
        }
    )
    dqn_episode = dqn_episode.merge(
        dqn_first_disc, on=["split", "policy_name", "episode_id"], how="left"
    )
    dqn_episode["first_discretionary_sale_timing_bucket"] = dqn_episode[
        "first_discretionary_sale_timing_bucket"
    ].fillna("no_discretionary_sale")
    dqn_episode["first_discretionary_sale_bucket_order"] = dqn_episode[
        "first_discretionary_sale_timing_bucket"
    ].map(timing_bucket_order)

    hold_episode = (
        episode[episode["policy_name"].eq("hold_to_terminal")][
            ["split", "episode_id", "episode_final_after_tax_total_value"]
        ]
        .rename(
            columns={
                "episode_final_after_tax_total_value": "hold_final_after_tax_total_value"
            }
        )
        .copy()
    )
    dqn_episode = dqn_episode.merge(hold_episode, on=["split", "episode_id"], how="left")
    dqn_episode["dqn_minus_hold_final_after_tax_value"] = (
        dqn_episode["episode_final_after_tax_total_value"]
        - dqn_episode["hold_final_after_tax_total_value"]
    )

    hold_step = step[step["policy_name"].eq("hold_to_terminal")].copy()
    hold_step = hold_step.sort_values(["split", "episode_id", "date"])
    path_rows = []
    sale_rows = dqn_episode.dropna(subset=["first_discretionary_sale_date"])
    for row in sale_rows.itertuples(index=False):
        path = hold_step[
            (hold_step["split"].eq(row.split))
            & (hold_step["episode_id"].eq(row.episode_id))
            & (hold_step["date"].ge(row.first_discretionary_sale_date))
        ]
        if path.empty:
            continue
        value_at_sale = path.iloc[0]["after_tax_total_value"]
        if pd.isna(value_at_sale) or value_at_sale <= 1e-12:
            continue
        final_value = row.hold_final_after_tax_total_value
        min_value_after_sale = path["after_tax_total_value"].min()
        path_rows.append(
            {
                "split": row.split,
                "episode_id": row.episode_id,
                "hold_return_after_first_dqn_sale": final_value / value_at_sale - 1.0,
                "hold_max_drawdown_after_first_dqn_sale": min_value_after_sale
                / value_at_sale
                - 1.0,
            }
        )
    path_diag = pd.DataFrame(path_rows)
    dqn_episode = dqn_episode.merge(path_diag, on=["split", "episode_id"], how="left")

    rows = []
    for (split, bucket), group in dqn_episode.groupby(
        ["split", "first_discretionary_sale_timing_bucket"], sort=False
    ):
        rows.append(
            {
                "split": split,
                "first_discretionary_sale_timing_bucket": bucket,
                "bucket_order": timing_bucket_order(bucket),
                "num_episodes": len(group),
                "mean_dqn_minus_hold_final_after_tax_value": group[
                    "dqn_minus_hold_final_after_tax_value"
                ].mean(),
                "mean_hold_terminal_after_tax_value": group[
                    "hold_final_after_tax_total_value"
                ].mean(),
                "mean_hold_return_after_first_dqn_sale_expost": group[
                    "hold_return_after_first_dqn_sale"
                ].mean(),
                "mean_hold_max_drawdown_after_first_dqn_sale_expost": group[
                    "hold_max_drawdown_after_first_dqn_sale"
                ].mean(),
                "num_valid_expost_path_episodes": group[
                    "hold_return_after_first_dqn_sale"
                ].notna().sum(),
            }
        )
    diagnostic = pd.DataFrame(rows)
    diagnostic["split"] = pd.Categorical(
        diagnostic["split"], SPLIT_ORDER, ordered=True
    )
    return diagnostic.sort_values(["split", "bucket_order"]).reset_index(drop=True)


def start_distance_bucket(days: float) -> str:
    if pd.isna(days):
        return "missing"
    if days <= 30:
        return "0_to_30_days"
    if days <= 60:
        return "31_to_60_days"
    if days <= 90:
        return "61_to_90_days"
    if days <= 120:
        return "91_to_120_days"
    if days <= 180:
        return "121_to_180_days"
    return "over_180_days"


def start_distance_bucket_order(bucket: str) -> int:
    order = {
        "0_to_30_days": 0,
        "31_to_60_days": 1,
        "61_to_90_days": 2,
        "91_to_120_days": 3,
        "121_to_180_days": 4,
        "over_180_days": 5,
        "missing": 99,
    }
    return order.get(bucket, 99)


def build_start_distance_analysis(episode: pd.DataFrame, step: pd.DataFrame) -> pd.DataFrame:
    starts = (
        step[step["step_in_episode"].eq(0)][
            ["split", "policy_name", "episode_id", "date", "tax_transition_date"]
        ]
        .copy()
        .drop_duplicates(["split", "policy_name", "episode_id"])
    )
    starts["start_days_to_tax_transition"] = (
        starts["tax_transition_date"] - starts["date"]
    ).dt.days
    starts["start_tax_transition_bucket"] = starts[
        "start_days_to_tax_transition"
    ].map(start_distance_bucket)
    starts["start_tax_transition_bucket_order"] = starts[
        "start_tax_transition_bucket"
    ].map(start_distance_bucket_order)

    enriched = episode.merge(
        starts,
        on=["split", "policy_name", "episode_id"],
        how="left",
    )
    hold_values = (
        enriched[enriched["policy_name"].eq("hold_to_terminal")][
            ["split", "episode_id", "episode_final_after_tax_total_value"]
        ]
        .rename(
            columns={
                "episode_final_after_tax_total_value": "hold_final_after_tax_total_value"
            }
        )
        .copy()
    )
    enriched = enriched.merge(hold_values, on=["split", "episode_id"], how="left")
    enriched["excess_final_after_tax_value_vs_hold"] = (
        enriched["episode_final_after_tax_total_value"]
        - enriched["hold_final_after_tax_total_value"]
    )

    grouped = (
        enriched.groupby(
            [
                "split",
                "policy_name",
                "policy_label",
                "start_tax_transition_bucket",
                "start_tax_transition_bucket_order",
            ],
            as_index=False,
        )
        .agg(
            num_episodes=("episode_id", "size"),
            mean_start_days_to_tax_transition=("start_days_to_tax_transition", "mean"),
            median_start_days_to_tax_transition=(
                "start_days_to_tax_transition",
                "median",
            ),
            mean_pct_original_sold_before_long_term=(
                "total_position_sold_short_term",
                lambda x: pct(x.mean()),
            ),
            mean_pct_original_sold_at_or_after_long_term=(
                "total_position_sold_long_term",
                lambda x: pct(x.mean()),
            ),
            pct_episodes_with_no_discretionary_sale=(
                "num_discretionary_sales",
                lambda x: pct((x.fillna(0) == 0).mean()),
            ),
            mean_final_after_tax_total_value=(
                "episode_final_after_tax_total_value",
                "mean",
            ),
            mean_excess_final_after_tax_value_vs_hold=(
                "excess_final_after_tax_value_vs_hold",
                "mean",
            ),
        )
        .copy()
    )
    grouped["split"] = pd.Categorical(grouped["split"], SPLIT_ORDER, ordered=True)
    grouped["policy_name"] = pd.Categorical(
        grouped["policy_name"], POLICIES, ordered=True
    )
    return grouped.sort_values(
        ["split", "policy_name", "start_tax_transition_bucket_order"]
    ).reset_index(drop=True)


def write_markdown_table(df: pd.DataFrame, path: Path, title: str) -> None:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(
                lambda value: "" if pd.isna(value) else f"{value:.6g}"
            )
        else:
            display[col] = display[col].map(
                lambda value: "" if pd.isna(value) else str(value)
            )

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        columns = list(display.columns)
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in display.itertuples(index=False):
            handle.write("| " + " | ".join(row) + " |\n")
        handle.write("\n")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = run_dir / "quant_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    episode, step = read_inputs(run_dir)
    step = add_sale_increments(step)
    sale_events = build_sale_events(step)

    summary = build_summary(episode, sale_events)
    distribution = build_sale_distribution(sale_events)
    expost = build_expost_dqn_diagnostic(episode, step, sale_events)
    start_distance = build_start_distance_analysis(episode, step)

    summary_path = output_dir / "step13_tax_transition_timing_summary.csv"
    distribution_path = output_dir / "step13_tax_transition_sale_distribution.csv"
    expost_path = output_dir / "step13_dqn_early_sale_expost_diagnostic.csv"
    start_distance_path = output_dir / "step13_long_term_sold_by_start_distance.csv"

    summary.to_csv(summary_path, index=False)
    distribution.to_csv(distribution_path, index=False)
    expost.to_csv(expost_path, index=False)
    start_distance.to_csv(start_distance_path, index=False)

    table_cols = [
        "split",
        "policy_label",
        "num_episodes",
        "mean_pct_original_sold_before_long_term",
        "mean_pct_original_sold_at_or_after_long_term",
        "pct_episodes_with_discretionary_sale_before_long_term",
        "pct_episodes_with_no_discretionary_sale",
        "median_days_remaining_at_first_realization",
        "median_days_remaining_at_first_discretionary_sale",
        "mean_final_after_tax_total_value",
        "mean_excess_final_after_tax_value_vs_hold",
    ]
    write_markdown_table(
        summary[table_cols],
        output_dir / "step13_tax_transition_timing_summary.md",
        "Step 13 Tax-Transition Timing Summary",
    )
    write_markdown_table(
        expost,
        output_dir / "step13_dqn_early_sale_expost_diagnostic.md",
        "Step 13 DQN Early-Sale Ex-Post Diagnostic",
    )
    write_markdown_table(
        start_distance[
            [
                "split",
                "policy_label",
                "start_tax_transition_bucket",
                "num_episodes",
                "mean_start_days_to_tax_transition",
                "mean_pct_original_sold_before_long_term",
                "mean_pct_original_sold_at_or_after_long_term",
                "pct_episodes_with_no_discretionary_sale",
                "mean_final_after_tax_total_value",
                "mean_excess_final_after_tax_value_vs_hold",
            ]
        ],
        output_dir / "step13_long_term_sold_by_start_distance.md",
        "Step 13 Long-Term Sold Percentage By Starting Tax-Transition Distance",
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {distribution_path}")
    print(f"Wrote {expost_path}")
    print(f"Wrote {start_distance_path}")


if __name__ == "__main__":
    main()
