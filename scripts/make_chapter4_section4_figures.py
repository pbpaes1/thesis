"""Create thesis Section 4.4 robustness and heterogeneity figures.

Run from the project root:
    python scripts/make_chapter4_section4_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUANT_ANALYSIS_DIR = (
    PROJECT_ROOT / "runs" / "train_reward_c_lite_v5_full" / "quant_analysis"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"

FIGURE_4_2_SOURCE = QUANT_ANALYSIS_DIR / "step10_robustness_statistical_significance.csv"
FIGURE_4_3_SOURCE = QUANT_ANALYSIS_DIR / "step7_preferred_policy_behavior_by_economic_period.csv"
FIGURE_4_4_SOURCE = (
    QUANT_ANALYSIS_DIR / "sector_analysis" / "sector_policy_performance.csv"
)

FIGURE_OUTPUTS = {
    "figure_4_2": (
        "figure_4_2_paired_robustness_vs_benchmarks.svg",
        "figure_4_2_paired_robustness_vs_benchmarks.png",
    ),
    "figure_4_3": (
        "figure_4_3_period_liquidation_behaviour.svg",
        "figure_4_3_period_liquidation_behaviour.png",
    ),
    "figure_4_4": (
        "figure_4_4_sector_preferred_minus_hold.svg",
        "figure_4_4_sector_preferred_minus_hold.png",
    ),
}
NOTES_OUTPUT = "chapter4_section4_figure_notes.md"

PREFERRED_POLICY = "trained_dqn_first_sale_margin_0p070_normal_0p020"
COMPARATOR_ORDER = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "random_policy",
]
COMPARATOR_LABELS = {
    "hold_to_terminal": "Hold to terminal",
    "sell_immediately": "Sell immediately",
    "sell_half_then_hold": "Sell half, then hold",
    "random_policy": "Random policy",
}
PERIOD_ORDER = [
    "post_crisis_early_recovery",
    "qe_bull_market",
    "late_cycle_volatility_return",
    "covid_stimulus",
    "inflation_tightening",
    "unknown_or_outside_defined_period",
]
PERIOD_LABELS = {
    "post_crisis_early_recovery": "Post-crisis early recovery",
    "qe_bull_market": "QE bull market",
    "late_cycle_volatility_return": "Late-cycle volatility return",
    "covid_stimulus": "COVID stimulus",
    "inflation_tightening": "Inflation tightening",
}

FIGURE_4_2_COLUMNS = [
    "split",
    "policy_A",
    "policy_B",
    "mean_difference",
    "mean_difference_ci_low",
    "mean_difference_ci_high",
]
FIGURE_4_3_COLUMNS = [
    "split",
    "diagnostic_scope",
    "economic_period",
    "calendar_year_min",
    "calendar_year_max",
    "num_episodes",
    "no_cut_episode_pct",
    "discretionary_sale_episode_pct",
    "mean_pct_position_sold_short_term",
    "mean_pct_position_sold_long_term",
    "mean_excess_value_vs_hold_to_terminal",
    "mean_excess_value_vs_sell_immediately",
]
FIGURE_4_4_COLUMNS = [
    "split",
    "sector",
    "num_episodes",
    "preferred_minus_hold_mean",
    "preferred_win_rate_vs_hold",
    "preferred_minus_sell_immediately_mean",
]


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(colors="#333333")


def read_csv_checked(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing source file. Expected path: {relative_path(path)}")
    df = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required column(s) in {relative_path(path)}: {missing_columns}"
        )
    return df


def require_rows(data: pd.DataFrame, figure_label: str) -> None:
    if data.empty:
        raise ValueError(f"No rows remain after applying filters for {figure_label}.")


def load_figure_4_2_data(source: Path) -> pd.DataFrame:
    df = read_csv_checked(source, FIGURE_4_2_COLUMNS)
    data = df.loc[
        (df["split"] == "test")
        & (df["policy_A"] == PREFERRED_POLICY)
        & (df["policy_B"].isin(COMPARATOR_ORDER)),
        FIGURE_4_2_COLUMNS,
    ].copy()
    require_rows(data, "Figure 4.2")

    duplicate_rows = data[data["policy_B"].duplicated(keep=False)]
    if not duplicate_rows.empty:
        duplicates = sorted(duplicate_rows["policy_B"].unique())
        raise ValueError(f"Duplicate Figure 4.2 comparator rows: {duplicates}")

    missing_comparators = [
        comparator for comparator in COMPARATOR_ORDER if comparator not in set(data["policy_B"])
    ]
    if missing_comparators:
        raise ValueError(f"Missing Figure 4.2 comparator rows: {missing_comparators}")

    for column in ["mean_difference", "mean_difference_ci_low", "mean_difference_ci_high"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data["_order"] = data["policy_B"].map(
        {comparator: index for index, comparator in enumerate(COMPARATOR_ORDER)}
    )
    return data.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def load_figure_4_3_data(source: Path) -> pd.DataFrame:
    df = read_csv_checked(source, FIGURE_4_3_COLUMNS)
    data = df.loc[
        (df["split"] == "all")
        & (df["diagnostic_scope"] == "all_episode_descriptive"),
        FIGURE_4_3_COLUMNS,
    ].copy()
    require_rows(data, "Figure 4.3")

    for column in [
        "num_episodes",
        "calendar_year_min",
        "calendar_year_max",
        "no_cut_episode_pct",
        "discretionary_sale_episode_pct",
        "mean_pct_position_sold_short_term",
        "mean_pct_position_sold_long_term",
        "mean_excess_value_vs_hold_to_terminal",
        "mean_excess_value_vs_sell_immediately",
    ]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    period_order = {period: index for index, period in enumerate(PERIOD_ORDER)}
    data["_source_order"] = range(len(data))
    data["_period_order"] = data["economic_period"].map(period_order).fillna(
        len(PERIOD_ORDER) + data["_source_order"]
    )
    return (
        data.sort_values(["_period_order", "_source_order"])
        .drop(columns=["_period_order", "_source_order"])
        .reset_index(drop=True)
    )


def load_figure_4_4_data(source: Path) -> pd.DataFrame:
    df = read_csv_checked(source, FIGURE_4_4_COLUMNS)
    data = df.loc[
        (df["split"] == "test") & (df["sector"] != "Unknown"),
        FIGURE_4_4_COLUMNS,
    ].copy()
    require_rows(data, "Figure 4.4")

    for column in [
        "num_episodes",
        "preferred_minus_hold_mean",
        "preferred_win_rate_vs_hold",
        "preferred_minus_sell_immediately_mean",
    ]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    return data.sort_values("preferred_minus_hold_mean", ascending=True).reset_index(drop=True)


def format_year_range(row: pd.Series) -> str:
    year_min = int(row["calendar_year_min"])
    year_max = int(row["calendar_year_max"])
    if year_min == year_max:
        return str(year_min)
    return f"{year_min}-{year_max}"


def format_period_label(row: pd.Series) -> str:
    year_range = format_year_range(row)
    period = str(row["economic_period"])
    if period == "unknown_or_outside_defined_period":
        clean_period = year_range
    else:
        clean_period = PERIOD_LABELS.get(period, period.replace("_", " ").title())
        clean_period = f"{clean_period}\n{year_range}"
    return f"{clean_period}\n(n={int(row['num_episodes']):,})"


def set_percentage_ylim(ax: plt.Axes, max_value: float) -> None:
    upper = max(100.0, max_value + 10.0)
    ax.set_ylim(0.0, min(upper, 115.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def add_bar_value_labels(ax: plt.Axes, bars, *, suffix: str = "%") -> None:
    y_max = ax.get_ylim()[1]
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_max * 0.018,
            f"{height:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#333333",
        )


def make_figure_4_2(data: pd.DataFrame, output_dir: Path) -> list[Path]:
    svg_path = output_dir / FIGURE_OUTPUTS["figure_4_2"][0]
    png_path = output_dir / FIGURE_OUTPUTS["figure_4_2"][1]

    labels = [COMPARATOR_LABELS[comparator] for comparator in data["policy_B"]]
    y_positions = list(range(len(data)))
    means = data["mean_difference"]
    xerr_low = means - data["mean_difference_ci_low"]
    xerr_high = data["mean_difference_ci_high"] - means

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.axvline(0.0, color="#4F4F4F", linewidth=1.0, zorder=1)
    ax.errorbar(
        means,
        y_positions,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        markersize=5.5,
        color="#2F5F7F",
        ecolor="#7E93A4",
        elinewidth=1.6,
        capsize=4.0,
        capthick=1.4,
        zorder=3,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired difference in final after-tax value")
    style_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.08, y=0.18)

    fig.tight_layout(pad=0.5)
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return [svg_path, png_path]


def make_figure_4_3(data: pd.DataFrame, output_dir: Path) -> list[Path]:
    svg_path = output_dir / FIGURE_OUTPUTS["figure_4_3"][0]
    png_path = output_dir / FIGURE_OUTPUTS["figure_4_3"][1]

    labels = [format_period_label(row) for _, row in data.iterrows()]
    x_positions = list(range(len(data)))
    bar_width = 0.34
    left_positions = [x - bar_width / 2.0 for x in x_positions]
    right_positions = [x + bar_width / 2.0 for x in x_positions]

    panel_a_no_cut = data["no_cut_episode_pct"] * 100.0
    panel_a_discretionary = data["discretionary_sale_episode_pct"] * 100.0
    panel_b_short = data["mean_pct_position_sold_short_term"] * 100.0
    panel_b_long = data["mean_pct_position_sold_long_term"] * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)
    colors = {
        "no_cut": "#6F8EA3",
        "discretionary": "#C6925E",
        "short": "#8D6D9F",
        "long": "#7C9B6E",
    }

    bars_no_cut = axes[0].bar(
        left_positions,
        panel_a_no_cut,
        width=bar_width,
        color=colors["no_cut"],
        edgecolor="white",
        linewidth=0.8,
        label="No cut",
    )
    bars_discretionary = axes[0].bar(
        right_positions,
        panel_a_discretionary,
        width=bar_width,
        color=colors["discretionary"],
        edgecolor="white",
        linewidth=0.8,
        label="Discretionary sale",
    )
    axes[0].text(
        -0.07,
        1.02,
        "A",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#333333",
    )
    axes[0].set_ylabel("Episodes (%)")
    set_percentage_ylim(axes[0], max(panel_a_no_cut.max(), panel_a_discretionary.max()))
    style_axis(axes[0], grid_axis="y")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)
    add_bar_value_labels(axes[0], bars_no_cut)
    add_bar_value_labels(axes[0], bars_discretionary)

    bars_short = axes[1].bar(
        left_positions,
        panel_b_short,
        width=bar_width,
        color=colors["short"],
        edgecolor="white",
        linewidth=0.8,
        label="Short-term sold",
    )
    bars_long = axes[1].bar(
        right_positions,
        panel_b_long,
        width=bar_width,
        color=colors["long"],
        edgecolor="white",
        linewidth=0.8,
        label="Long-term sold",
    )
    axes[1].text(
        -0.07,
        1.02,
        "B",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#333333",
    )
    axes[1].set_ylabel("Mean position sold (%)")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(labels, fontsize=8.2, linespacing=1.05)
    set_percentage_ylim(axes[1], max(panel_b_short.max(), panel_b_long.max()))
    style_axis(axes[1], grid_axis="y")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)
    add_bar_value_labels(axes[1], bars_short)
    add_bar_value_labels(axes[1], bars_long)

    for ax in axes:
        ax.tick_params(axis="x", length=0, pad=7)

    fig.subplots_adjust(hspace=0.42)
    fig.tight_layout(pad=0.6)
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return [svg_path, png_path]


def make_figure_4_4(data: pd.DataFrame, output_dir: Path) -> list[Path]:
    svg_path = output_dir / FIGURE_OUTPUTS["figure_4_4"][0]
    png_path = output_dir / FIGURE_OUTPUTS["figure_4_4"][1]

    labels = data["sector"].astype(str).tolist()
    y_positions = list(range(len(data)))
    values = data["preferred_minus_hold_mean"]
    colors = ["#6F8EA3" for _ in labels]

    fig_height = max(4.2, 0.36 * len(data) + 1.1)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    ax.barh(
        y_positions,
        values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axvline(0.0, color="#4F4F4F", linewidth=1.0, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Preferred DQN minus hold-to-terminal mean final after-tax value")
    style_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    min_value = float(values.min())
    max_label_space = 0.018
    ax.set_xlim(min_value - 0.015, max_label_space)
    for y_pos, episodes in zip(y_positions, data["num_episodes"]):
        ax.text(
            0.002,
            y_pos,
            f"n={int(episodes):,}",
            ha="left",
            va="center",
            fontsize=8.8,
            color="#333333",
        )

    fig.tight_layout(pad=0.5)
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return [svg_path, png_path]


def figure_4_2_interpretation(data: pd.DataFrame) -> str:
    hold_row = data.loc[data["policy_B"] == "hold_to_terminal"].iloc[0]
    positive_comparators = data.loc[data["mean_difference"] > 0, "policy_B"].tolist()
    positive_labels = [COMPARATOR_LABELS[comparator] for comparator in positive_comparators]
    if len(positive_labels) > 1:
        positive_text = f"{', '.join(positive_labels[:-1])}, and {positive_labels[-1]}"
    else:
        positive_text = positive_labels[0]
    return (
        "On the test split, the preferred DQN is significantly below hold-to-terminal "
        f"(mean difference {hold_row['mean_difference']:.3f}) but has positive mean paired "
        f"differences against {positive_text}."
    )


def figure_4_3_interpretation(data: pd.DataFrame) -> str:
    highest_sale = data.loc[data["discretionary_sale_episode_pct"].idxmax()]
    lowest_sale = data.loc[data["discretionary_sale_episode_pct"].idxmin()]
    high_label = (
        format_year_range(highest_sale)
        if highest_sale["economic_period"] == "unknown_or_outside_defined_period"
        else PERIOD_LABELS.get(
            highest_sale["economic_period"],
            str(highest_sale["economic_period"]).replace("_", " ").title(),
        )
    )
    low_label = (
        format_year_range(lowest_sale)
        if lowest_sale["economic_period"] == "unknown_or_outside_defined_period"
        else PERIOD_LABELS.get(
            lowest_sale["economic_period"],
            str(lowest_sale["economic_period"]).replace("_", " ").title(),
        )
    )
    return (
        f"Across the full period summary, behaviour is most liquidation-heavy during {high_label} "
        f"({highest_sale['discretionary_sale_episode_pct'] * 100.0:.1f}% discretionary-sale episodes) "
        f"and more hold-like in {low_label} "
        f"({lowest_sale['discretionary_sale_episode_pct'] * 100.0:.1f}%)."
    )


def figure_4_4_interpretation(data: pd.DataFrame) -> str:
    worst = data.iloc[0]
    best = data.iloc[-1]
    return (
        "Across classified test sectors, the preferred DQN underperforms hold-to-terminal on mean "
        f"final after-tax value in every sector, with the largest shortfall in {worst['sector']} "
        f"({worst['preferred_minus_hold_mean']:.3f}) and the least negative gap in {best['sector']} "
        f"({best['preferred_minus_hold_mean']:.3f})."
    )


def write_notes(
    output_dir: Path,
    figure_4_2_data: pd.DataFrame,
    figure_4_3_data: pd.DataFrame,
    figure_4_4_data: pd.DataFrame,
) -> Path:
    notes_path = output_dir / NOTES_OUTPUT
    lines = [
        "# Chapter 4 Section 4 Figure Notes",
        "",
        "## Figure 4.2: Paired robustness against benchmarks",
        f"- Source file used: `{relative_path(FIGURE_4_2_SOURCE)}`",
        (
            "- Filters applied: `split == \"test\"`; "
            f"`policy_A == \"{PREFERRED_POLICY}\"`; "
            "`policy_B` in hold-to-terminal, sell-immediately, sell-half-then-hold, "
            "and random-policy benchmarks."
        ),
        f"- Columns used: `{', '.join(FIGURE_4_2_COLUMNS)}`",
        f"- Interpretation: {figure_4_2_interpretation(figure_4_2_data)}",
        "",
        "## Figure 4.3: Preferred DQN behaviour by economic period",
        f"- Source file used: `{relative_path(FIGURE_4_3_SOURCE)}`",
        "- Filters applied: `split == \"all\"`; `diagnostic_scope == \"all_episode_descriptive\"`.",
        f"- Columns used: `{', '.join(FIGURE_4_3_COLUMNS)}`",
        f"- Interpretation: {figure_4_3_interpretation(figure_4_3_data)}",
        "",
        "## Figure 4.4: Sector heterogeneity versus hold-to-terminal",
        f"- Source file used: `{relative_path(FIGURE_4_4_SOURCE)}`",
        "- Filters applied: `split == \"test\"`; `sector != \"Unknown\"`.",
        f"- Columns used: `{', '.join(FIGURE_4_4_COLUMNS)}`",
        f"- Interpretation: {figure_4_4_interpretation(figure_4_4_data)}",
        "",
    ]
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    return notes_path


def verify_outputs(paths: list[Path]) -> None:
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(relative_path(path) for path in missing_paths)
        raise FileNotFoundError(f"Expected output file(s) were not created: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chapter 4 Section 4 thesis figures and figure notes."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {relative_path(DEFAULT_OUTPUT_DIR)}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    figure_4_2_data = load_figure_4_2_data(FIGURE_4_2_SOURCE)
    figure_4_3_data = load_figure_4_3_data(FIGURE_4_3_SOURCE)
    figure_4_4_data = load_figure_4_4_data(FIGURE_4_4_SOURCE)

    output_paths: list[Path] = []
    output_paths.extend(make_figure_4_2(figure_4_2_data, output_dir))
    output_paths.extend(make_figure_4_3(figure_4_3_data, output_dir))
    output_paths.extend(make_figure_4_4(figure_4_4_data, output_dir))
    notes_path = write_notes(output_dir, figure_4_2_data, figure_4_3_data, figure_4_4_data)

    verify_outputs(output_paths)
    print("Verified figure outputs:")
    for path in output_paths:
        print(relative_path(path))
    print(f"Wrote {relative_path(notes_path)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
