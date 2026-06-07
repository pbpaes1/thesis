"""Create thesis Figure 4.1 liquidation fraction chart.

Run from the project root:
    python scripts/make_figure_4_1_liquidation_fractions.py
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "runs"
    / "train_reward_c_lite_v5_full"
    / "quant_analysis"
    / "step3_policy_performance_summary.md"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"
SVG_OUTPUT = "figure_4_1_liquidation_fractions.svg"
PNG_OUTPUT = "figure_4_1_liquidation_fractions.png"

POLICY_ORDER = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "trained_dqn_greedy",
    "trained_dqn_thresholded_margin_0p020",
    "trained_dqn_first_sale_margin_0p070_normal_0p020",
]
POLICY_LABELS = {
    "hold_to_terminal": "Hold to terminal",
    "sell_immediately": "Sell immediately",
    "sell_half_then_hold": "Sell half, then hold",
    "trained_dqn_greedy": "Raw greedy DQN",
    "trained_dqn_thresholded_margin_0p020": "Thresholded DQN",
    "trained_dqn_first_sale_margin_0p070_normal_0p020": "Preferred DQN",
}
REQUIRED_COLUMNS = [
    "split",
    "policy_name",
    "mean_pct_position_sold_short_term",
    "mean_pct_position_sold_long_term",
]


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_markdown_table(path: Path) -> pd.DataFrame:
    """Read the first pipe-delimited markdown table in ``path``."""
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    ]
    if len(lines) < 3:
        raise ValueError(f"No markdown table found in {relative_path(path)}.")

    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)
    df = df.dropna(axis=1, how="all")
    df.columns = [column.strip() for column in df.columns]
    df = df.loc[:, [column for column in df.columns if column]]
    df = df.apply(lambda column: column.str.strip() if column.dtype == object else column)

    separator_mask = df.apply(
        lambda row: all(str(value).strip().replace("-", "") == "" for value in row),
        axis=1,
    )
    return df.loc[~separator_mask].reset_index(drop=True)


def load_figure_data(source: Path) -> pd.DataFrame:
    df = read_markdown_table(source)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required column(s) in {relative_path(source)}: {missing_columns}"
        )

    selected = df.loc[df["split"] == "test", REQUIRED_COLUMNS].copy()
    selected = selected[selected["policy_name"].isin(POLICY_ORDER)]
    duplicate_rows = selected[selected["policy_name"].duplicated(keep=False)]
    if not duplicate_rows.empty:
        duplicates = sorted(duplicate_rows["policy_name"].unique())
        raise ValueError(f"Duplicate test rows for selected policies: {duplicates}")

    missing_policies = [policy for policy in POLICY_ORDER if policy not in set(selected["policy_name"])]
    if missing_policies:
        raise ValueError(f"Missing test rows for selected policies: {missing_policies}")

    selected["_policy_order"] = selected["policy_name"].map(
        {policy: index for index, policy in enumerate(POLICY_ORDER)}
    )
    selected = selected.sort_values("_policy_order").drop(columns="_policy_order")

    for column in [
        "mean_pct_position_sold_short_term",
        "mean_pct_position_sold_long_term",
    ]:
        selected[column] = pd.to_numeric(selected[column], errors="raise") * 100.0

    totals = (
        selected["mean_pct_position_sold_short_term"]
        + selected["mean_pct_position_sold_long_term"]
    )
    if not np.allclose(totals, 100.0, atol=0.05):
        bad = selected.loc[
            ~np.isclose(totals, 100.0, atol=0.05),
            ["policy_name", "mean_pct_position_sold_short_term", "mean_pct_position_sold_long_term"],
        ]
        raise ValueError(
            "Selected short-term and long-term liquidation fractions do not sum to "
            f"100%:\n{bad.to_string(index=False)}"
        )

    return selected


def add_segment_labels(
    ax: plt.Axes,
    y_positions: np.ndarray,
    left_values: pd.Series | np.ndarray,
    widths: pd.Series | np.ndarray,
    *,
    min_width: float = 7.0,
    text_color: str = "white",
) -> None:
    for y_pos, left, width in zip(y_positions, left_values, widths):
        if width < min_width:
            continue
        ax.text(
            float(left) + float(width) / 2.0,
            y_pos,
            f"{float(width):.1f}%",
            ha="center",
            va="center",
            color=text_color,
            fontsize=9.0,
            fontweight="semibold",
        )


def make_plot(data: pd.DataFrame, svg_path: Path, png_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [POLICY_LABELS[policy] for policy in data["policy_name"]]
    short_term = data["mean_pct_position_sold_short_term"]
    long_term = data["mean_pct_position_sold_long_term"]
    y_positions = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    short_color = "#5E7F9A"
    long_color = "#C6925E"

    ax.barh(
        y_positions,
        short_term,
        color=short_color,
        edgecolor="white",
        linewidth=0.9,
        label="Short-term liquidation",
    )
    ax.barh(
        y_positions,
        long_term,
        left=short_term,
        color=long_color,
        edgecolor="white",
        linewidth=0.9,
        label="Long-term liquidation",
    )

    add_segment_labels(ax, y_positions, np.zeros(len(data)), short_term, text_color="white")
    add_segment_labels(ax, y_positions, short_term, long_term, text_color="#2F2F2F")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Position liquidated (%)")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xticks(np.arange(0, 101, 20))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#6F6F6F")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="#333333")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.8,
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 4.1 liquidation fractions by policy."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Markdown source table. Default: {relative_path(DEFAULT_SOURCE)}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {relative_path(DEFAULT_OUTPUT_DIR)}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source if args.source.is_absolute() else PROJECT_ROOT / args.source
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    svg_path = output_dir / SVG_OUTPUT
    png_path = output_dir / PNG_OUTPUT

    if not source.exists():
        raise FileNotFoundError(f"Missing source file: {relative_path(source)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_figure_data(source)
    make_plot(data, svg_path, png_path)

    print(f"Wrote {relative_path(svg_path)}")
    print(f"Wrote {relative_path(png_path)}")


if __name__ == "__main__":
    main()
