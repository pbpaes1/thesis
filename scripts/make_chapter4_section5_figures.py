"""Create thesis Figure 4.5 representative preferred-DQN episode paths.

Run from the project root:
    python scripts/make_chapter4_section5_figures.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_RUN_DIR = PROJECT_ROOT / "runs" / "train_reward_c_lite_v5_full"
QUANT_ANALYSIS_DIR = FINAL_RUN_DIR / "quant_analysis"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"

PNG_OUTPUT = "figure_4_5_representative_dqn_episode_paths.png"
SVG_OUTPUT = "figure_4_5_representative_dqn_episode_paths.svg"
NOTES_OUTPUT = "chapter4_section5_figure_notes.md"

PREFERRED_POLICY = "trained_dqn_first_sale_margin_0p070_normal_0p020"
PLOT_BENCHMARK_POLICIES = [
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
]
STEP11_SCRIPT = PROJECT_ROOT / "scripts" / "quant_analysis" / "step11_case_study_diagnostics.py"
SELECTED_CASES_SOURCE = QUANT_ANALYSIS_DIR / "step11_selected_case_studies.csv"
SCENARIO_CLASSIFICATION_SOURCE = (
    QUANT_ANALYSIS_DIR / "step11_case_study_scenario_classification.csv"
)
BASELINE_STEP_ROLLOUTS = FINAL_RUN_DIR / "baselines" / "baseline_step_rollouts.csv"


@dataclass(frozen=True)
class CasePanel:
    panel_label: str
    episode_id: str
    illustration: str
    split: str = "test"


CASE_PANELS = [
    CasePanel(
        panel_label="A. Early sale avoids later loss: ABNB_2023-05-25",
        episode_id="ABNB_2023-05-25",
        illustration=(
            "The preferred DQN sells early enough to reduce exposure before the later "
            "drawdown, finishing above hold-to-terminal in this episode."
        ),
    ),
    CasePanel(
        panel_label="B. Early sale underperforms hold: AEP_2023-10-05",
        episode_id="AEP_2023-10-05",
        illustration=(
            "The preferred DQN liquidates before a subsequent recovery, so the early "
            "sale underperforms the hold-to-terminal benchmark."
        ),
    ),
    CasePanel(
        panel_label="C. No discretionary sale: HMC_2024-12-19",
        episode_id="HMC_2024-12-19",
        illustration=(
            "The preferred DQN makes no discretionary sale and remains aligned with "
            "hold-to-terminal through the tax-transition date."
        ),
    ),
]

NOTES_INTERPRETATION = (
    "The three representative paths show that the preferred DQN's liquidation rule can "
    "protect against later losses, can also sell too early and miss upside, and can "
    "choose a hold-like path when no discretionary sale is favored."
)
THESIS_CAPTION = (
    "Figure 4.5: Representative preferred-DQN liquidation paths across selected test episodes"
)


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
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 7.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_step11_module() -> ModuleType:
    if not STEP11_SCRIPT.exists():
        raise FileNotFoundError(
            f"Missing Step 11 fallback script: {relative_path(STEP11_SCRIPT)}"
        )
    module_name = "step11_case_study_diagnostics_for_figure_4_5"
    spec = importlib.util.spec_from_file_location(module_name, STEP11_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import fallback script: {relative_path(STEP11_SCRIPT)}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_selected_case_rows() -> pd.DataFrame:
    selected = pd.DataFrame()
    if SELECTED_CASES_SOURCE.exists():
        selected = pd.read_csv(SELECTED_CASES_SOURCE)
        required_columns = {"split", "episode_id"}
        missing_columns = required_columns - set(selected.columns)
        if missing_columns:
            raise ValueError(
                "Selected case-study table is missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )

    rows: list[dict[str, object]] = []
    for panel in CASE_PANELS:
        matching_rows = (
            selected[selected["episode_id"].astype(str).eq(panel.episode_id)]
            if not selected.empty
            else pd.DataFrame()
        )
        if matching_rows.empty:
            rows.append({"split": panel.split, "episode_id": panel.episode_id})
        else:
            row = matching_rows.iloc[0].to_dict()
            row["split"] = str(row.get("split", panel.split))
            row["episode_id"] = panel.episode_id
            rows.append(row)

    return pd.DataFrame(rows)


def read_filtered_step_rollouts(
    episode_ids: set[str],
    policies: set[str],
    *,
    chunksize: int = 750_000,
) -> pd.DataFrame:
    if not BASELINE_STEP_ROLLOUTS.exists():
        raise FileNotFoundError(
            f"Missing baseline step rollout file: {relative_path(BASELINE_STEP_ROLLOUTS)}"
        )

    usecols = [
        "split",
        "episode_id",
        "policy_name",
        "step_in_episode",
        "date",
        "tax_transition_date",
        "action_fraction_executed",
        "after_tax_total_value",
        "previous_after_tax_total_value",
        "sold_fraction",
        "remaining_fraction",
        "terminal_liquidation_executed",
        "is_automatic_terminal_liquidation",
        "done",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        BASELINE_STEP_ROLLOUTS,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        mask = (
            chunk["split"].eq("test")
            & chunk["episode_id"].astype(str).isin(episode_ids)
            & chunk["policy_name"].isin(policies)
        )
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        raise ValueError(
            "No path-level rows found for missing selected episode plot(s): "
            + ", ".join(sorted(episode_ids))
        )
    return pd.concat(chunks, ignore_index=True)


def load_plot_data() -> tuple[ModuleType, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    module = load_step11_module()
    selected_rows = read_selected_case_rows()
    policies = {PREFERRED_POLICY, *PLOT_BENCHMARK_POLICIES}
    episode_ids = set(selected_rows["episode_id"].astype(str))

    step_df = read_filtered_step_rollouts(episode_ids, policies)
    step_df = module.prepare_step_rollouts(step_df)
    warnings: list[str] = []
    raw_df = module.load_raw_episode_metadata(FINAL_RUN_DIR, warnings)

    for row in selected_rows.itertuples(index=False):
        for policy in policies:
            path = step_df[
                step_df["split"].eq(row.split)
                & step_df["episode_id"].astype(str).eq(str(row.episode_id))
                & step_df["policy_name"].eq(policy)
            ]
            if path.empty:
                raise ValueError(
                    "Missing path-level data for Figure 4.5: "
                    f"split={row.split}, episode_id={row.episode_id}, policy={policy}"
                )
    return module, selected_rows, step_df, raw_df


def style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(colors="#333333")


def plot_price_panel(
    ax: plt.Axes,
    *,
    panel: CasePanel,
    selected_row: pd.Series,
    step_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    step11_module: ModuleType,
) -> tuple[list[object], list[str]]:
    split = str(selected_row["split"])
    episode_id = str(selected_row["episode_id"])
    policies = [PREFERRED_POLICY, *PLOT_BENCHMARK_POLICIES]
    label_map = {
        PREFERRED_POLICY: "Preferred DQN",
        "hold_to_terminal": "Hold to terminal",
        "sell_immediately": "Sell immediately",
        "sell_half_then_hold": "Sell half then hold",
    }
    colors = {
        PREFERRED_POLICY: "#1B4E8A",
        "hold_to_terminal": "#444444",
        "sell_immediately": "#9A4C2E",
        "sell_half_then_hold": "#4F7D3A",
    }
    linestyles = {
        PREFERRED_POLICY: "-",
        "hold_to_terminal": "-",
        "sell_immediately": "-.",
        "sell_half_then_hold": "-",
    }
    markers = {
        PREFERRED_POLICY: None,
        "hold_to_terminal": None,
        "sell_immediately": "s",
        "sell_half_then_hold": None,
    }

    paths: dict[str, pd.DataFrame] = {}
    for policy in policies:
        path = step_df[
            step_df["split"].eq(split)
            & step_df["episode_id"].astype(str).eq(episode_id)
            & step_df["policy_name"].eq(policy)
        ].sort_values("step_in_episode")
        paths[policy] = path

    preferred_path = paths[PREFERRED_POLICY]
    x_date_available = preferred_path["date"].notna().all()
    for policy in policies:
        path = paths[policy]
        x_values = path["date"] if x_date_available and path["date"].notna().all() else path[
            "step_in_episode"
        ]
        marker = markers[policy]
        ax.plot(
            x_values,
            path["after_tax_total_value"],
            label=label_map[policy],
            color=colors[policy],
            linewidth=2.1 if policy == PREFERRED_POLICY else 1.4,
            linestyle=linestyles[policy],
            marker=marker,
            markersize=4.0 if marker and len(path) <= 8 else 0.0,
            markerfacecolor="white" if marker else None,
            markeredgewidth=1.0 if marker else None,
            markevery=1 if marker and len(path) <= 8 else None,
            alpha=0.96,
            zorder=4 if policy == PREFERRED_POLICY else 3,
        )

    twin = None
    gain_path = step11_module.load_gain_path(raw_df, episode_id)
    if not gain_path.empty and x_date_available:
        twin = ax.twinx()
        twin.plot(
            gain_path["date"],
            gain_path["unrealized_gains_pct"],
            color="#777777",
            linestyle="--",
            linewidth=1.1,
            alpha=0.65,
            label="Unrealized gain",
        )
        twin.set_ylabel("unrealized gain", fontsize=9.0)
        twin.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        twin.tick_params(axis="y", labelsize=8.5, colors="#333333")
        twin.spines["top"].set_visible(False)
        twin.spines["right"].set_color("#777777")

    sales = preferred_path[step11_module.sale_mask(preferred_path)]
    if not sales.empty:
        sale_x = sales["date"] if x_date_available else sales["step_in_episode"]
        ax.scatter(
            sale_x,
            sales["after_tax_total_value"],
            marker="o",
            color="black",
            s=30,
            zorder=5,
            label="Preferred sale",
        )
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min
        for row in sales.itertuples(index=False):
            x_value = row.date if x_date_available else row.step_in_episode
            ax.annotate(
                f"Sell {float(row.action_fraction_executed) * 100:.0f}%",
                xy=(x_value, row.after_tax_total_value),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=7.0,
                color="black",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "lw": 0.4},
            )
        ax.set_ylim(y_min, y_max + 0.04 * y_span)

    transition = preferred_path["tax_transition_date"].dropna()
    if not transition.empty and x_date_available:
        ax.axvline(
            transition.iloc[0],
            color="#666666",
            linestyle=":",
            linewidth=1.1,
            label="Tax transition",
        )

    ax.text(
        0.0,
        1.075,
        panel.panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.2,
        fontweight="semibold",
        color="#222222",
    )
    ax.set_ylabel("after-tax value")
    ax.set_xlabel("date" if x_date_available else "step in episode")
    ax.set_ylim(0.0, ax.get_ylim()[1])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if twin is not None:
        twin.set_ylim(0.0, twin.get_ylim()[1])
    style_axis(ax)

    if x_date_available:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_ha("right")

    handles, labels = ax.get_legend_handles_labels()
    if twin is not None:
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        handles.extend(twin_handles)
        labels.extend(twin_labels)
    deduped: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_"):
            deduped.setdefault(label, handle)
    return list(deduped.values()), list(deduped.keys())


def make_composite_figure(
    output_dir: Path,
    *,
    step11_module: ModuleType,
    selected_rows: pd.DataFrame,
    step_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> list[Path]:
    png_path = output_dir / PNG_OUTPUT
    svg_path = output_dir / SVG_OUTPUT

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.9), facecolor="white")
    legend_entries: dict[str, object] = {}
    for ax, panel, (_, selected_row) in zip(axes, CASE_PANELS, selected_rows.iterrows()):
        handles, labels = plot_price_panel(
            ax,
            panel=panel,
            selected_row=selected_row,
            step_df=step_df,
            raw_df=raw_df,
            step11_module=step11_module,
        )
        for handle, label in zip(handles, labels):
            legend_entries.setdefault(label, handle)

    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        frameon=False,
        fontsize=7.8,
        handlelength=2.0,
        columnspacing=1.4,
        labelspacing=0.45,
    )
    fig.subplots_adjust(left=0.095, right=0.905, top=0.965, bottom=0.145, hspace=0.58)
    fig.savefig(png_path, dpi=320, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return [png_path, svg_path]


def write_notes(output_dir: Path) -> Path:
    notes_path = output_dir / NOTES_OUTPUT
    lines = [
        "# Chapter 4 Section 5 Figure Notes",
        "",
        "## Figure 4.5: Representative preferred-DQN liquidation paths across selected test episodes",
        f"- Thesis caption: {THESIS_CAPTION}",
        "- Source files used:",
        f"  - `{relative_path(BASELINE_STEP_ROLLOUTS)}`",
        "  - `data/episodes/drl_episodes.parquet`",
        f"- Original Step 11 selected-case table: `{relative_path(SELECTED_CASES_SOURCE)}`",
        f"- Replacement panel-C classification source: `{relative_path(SCENARIO_CLASSIFICATION_SOURCE)}`",
        "- Selected episodes:",
        *[f"  - `{panel.episode_id}`" for panel in CASE_PANELS],
        "- Figure construction: the panels are regenerated from the rollout data using only the value-path axis from the Step 11 diagnostics; the position-fraction subplot is omitted.",
        "- Panel-C replacement note: `HMC_2024-12-19` is a test episode classified as both `dqn_waits_correctly_until_long_term` and `dqn_behaves_like_hold_to_terminal`; it reaches the tax-transition date on 2025-12-19 and has no preferred-DQN discretionary sale.",
        "- Panel illustrations:",
        *[f"  - {panel.panel_label}: {panel.illustration}" for panel in CASE_PANELS],
        f"- One-sentence interpretation: {NOTES_INTERPRETATION}",
        "",
    ]
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    return notes_path


def verify_outputs(paths: list[Path]) -> None:
    missing_paths = [path for path in paths if not path.exists()]
    empty_paths = [path for path in paths if path.exists() and path.stat().st_size == 0]
    if missing_paths or empty_paths:
        problems = []
        if missing_paths:
            problems.append(
                "missing: " + ", ".join(relative_path(path) for path in missing_paths)
            )
        if empty_paths:
            problems.append("empty: " + ", ".join(relative_path(path) for path in empty_paths))
        raise FileNotFoundError("Expected output file(s) failed verification: " + "; ".join(problems))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chapter 4 Section 5 thesis Figure 4.5 and figure notes."
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
    step11_module, selected_rows, step_df, raw_df = load_plot_data()
    output_paths = make_composite_figure(
        output_dir,
        step11_module=step11_module,
        selected_rows=selected_rows,
        step_df=step_df,
        raw_df=raw_df,
    )
    notes_path = write_notes(output_dir)

    verify_outputs(output_paths)
    print("Verified Figure 4.5 outputs:")
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
