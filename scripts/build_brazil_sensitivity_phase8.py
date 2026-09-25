"""Rebuild Phase 8 from locked Phase 7 artifacts, without model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/brazil_sensitivity_v1"
RATES = ("0700", "1000", "1050", "1200", "1375", "1500")
POLICIES = ("trained_dqn_fixed_margin", "hold_to_terminal", "sell_immediately",
            "sell_half_then_hold", "sell_quarters_over_time", "random_policy")
DQN = POLICIES[0]
HOLD = POLICIES[1]
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_REPS = 4000
TIE_TOL = 1e-10
CHECK_TOL = 1e-8
FOCUS_URL = "https://www.bcb.gov.br/content/focus/focus/R20260911.pdf"
HISTORICAL_URL = "https://www.bcb.gov.br/controleinflacao/comunicadoscopom/20733"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def paired_difference(a: pd.Series, b: pd.Series, ids: list[str]) -> np.ndarray:
    if not a.index.is_unique or not b.index.is_unique:
        raise AssertionError("Paired inputs contain duplicate episode IDs.")
    if set(a.index) != set(ids) or set(b.index) != set(ids):
        raise AssertionError("Paired inputs have missing or extra episode IDs.")
    return a.loc[ids].to_numpy(dtype=float) - b.loc[ids].to_numpy(dtype=float)


def paired_counts(diff: np.ndarray) -> tuple[int, int, int]:
    return (int(np.sum(diff > TIE_TOL)), int(np.sum(np.abs(diff) <= TIE_TOL)),
            int(np.sum(diff < -TIE_TOL)))


def bootstrap_ci(diff: np.ndarray, indices: np.ndarray, *, median: bool = False) -> tuple[float, float]:
    values = np.median(diff[indices], axis=1) if median else np.mean(diff[indices], axis=1)
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def assert_close(a: np.ndarray, b: np.ndarray | float, label: str) -> float:
    residual = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if not np.isfinite(residual).all():
        raise AssertionError(f"Nonfinite {label}.")
    maximum = float(np.max(np.abs(residual)))
    if maximum > CHECK_TOL:
        raise AssertionError(f"{label} residual {maximum} exceeds {CHECK_TOL}.")
    return maximum


def verified_inputs() -> tuple[dict, list[str], dict[str, pd.DataFrame], dict[str, dict]]:
    access_path = ROOT / "runs/brazil_phase7_test_access/test_access_checkpoint.json"
    access = json.loads(access_path.read_text(encoding="utf-8"))
    if access["saved_test_id_count"] != 1519 or tuple(access["policies"]) != POLICIES:
        raise AssertionError("Test access record has wrong coverage or policy lock.")
    access_hash = sha256(access_path)
    split_file = ROOT / access["saved_test_ids_path"]
    saved = pd.read_csv(split_file, dtype=str)
    ids = saved.loc[saved["split"] == "test", "episode_id"].tolist()
    if len(ids) != 1519 or len(set(ids)) != 1519 or ordered_id_hash(ids) != access["saved_test_id_ordered_sha256"]:
        raise AssertionError("Saved ordered test IDs differ from access record.")
    if tuple(model["rate_suffix"] for model in access["models"]) != RATES:
        raise AssertionError("Access record does not list the six ordered rates.")
    frames: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict] = {}
    hash_cache: dict[str, str] = {}
    def verified_hash(relative_path: str, expected: str) -> None:
        path = ROOT / relative_path
        if relative_path not in hash_cache:
            hash_cache[relative_path] = sha256(path)
        if hash_cache[relative_path] != expected:
            raise AssertionError(f"Input hash mismatch: {relative_path}")
    verified_hash(access["model_freeze_path"], access["model_freeze_sha256"])
    for model in access["models"]:
        rate = model["rate_suffix"]
        manifest_path = f"runs/brazil_sensitivity_rate_{rate}_v1/phase7_test/evaluation_manifest.json"
        manifest = json.loads((ROOT / manifest_path).read_text(encoding="utf-8"))
        manifests[rate] = manifest
        if manifest["phase"] != 7 or manifest["rate_suffix"] != rate:
            raise AssertionError(f"{rate}: wrong Phase 7 manifest.")
        if manifest["test_access_checkpoint_sha256"] != access_hash or manifest["test_access_checkpoint"] != access_path.relative_to(ROOT).as_posix():
            raise AssertionError(f"{rate}: test-access linkage mismatch.")
        if manifest["test_id_ordered_sha256"] != ordered_id_hash(ids) or manifest["test_id_count"] != 1519:
            raise AssertionError(f"{rate}: test ID hash/count mismatch.")
        if tuple(manifest["policies"]) != POLICIES or manifest["seed"] != 42:
            raise AssertionError(f"{rate}: policy/seed mismatch.")
        if manifest["fixed_decision_margins"] != {"first_sale": 0.07, "subsequent_sale": 0.02}:
            raise AssertionError(f"{rate}: margin mismatch.")
        for manifest_key, model_key, hash_key in (
            ("frozen_checkpoint", "checkpoint_path", "frozen_checkpoint_sha256"),
            ("effective_config", "effective_config_path", "effective_config_sha256"),
            ("scenario_parquet", "scenario_parquet_path", "scenario_parquet_sha256"),
            ("state_freeze", "state_freeze_path", "state_freeze_sha256"),
        ):
            if manifest[manifest_key] != model[model_key] or manifest[hash_key] != model[hash_key if hash_key != "frozen_checkpoint_sha256" else "checkpoint_sha256"]:
                raise AssertionError(f"{rate}: frozen input mismatch for {manifest_key}.")
            verified_hash(manifest[manifest_key], manifest[hash_key])
        episode_file = manifest["episode_metrics"]
        verified_hash(episode_file["path"], episode_file["sha256"])
        frame = pd.read_csv(ROOT / episode_file["path"], dtype={"episode_id": str})
        if len(frame) != 1519 * len(POLICIES) or len(frame) != episode_file["rows"]:
            raise AssertionError(f"{rate}: episode row count mismatch.")
        if frame.duplicated(["policy_name", "episode_id"]).any():
            raise AssertionError(f"{rate}: duplicate policy/episode key.")
        for policy in POLICIES:
            if frame.loc[frame.policy_name == policy, "episode_id"].tolist() != ids:
                raise AssertionError(f"{rate}/{policy}: missing, extra, or out-of-order IDs.")
        if not frame.done.all() or not (frame.split == "test").all():
            raise AssertionError(f"{rate}: incomplete or non-test episodes.")
        if not np.isclose(frame.annual_cash_rate.to_numpy(), model["annual_cash_rate"]).all():
            raise AssertionError(f"{rate}: scenario rate mismatch.")
        if not (frame.checkpoint_sha256 == model["checkpoint_sha256"]).all():
            raise AssertionError(f"{rate}: checkpoint identifier mismatch.")
        if not (frame.seed == 42).all() or not np.isclose(frame.first_sale_margin, 0.07).all() or not np.isclose(frame.subsequent_sale_margin, 0.02).all():
            raise AssertionError(f"{rate}: seed or margin identifier mismatch.")
        if {entry["policy_name"] for entry in manifest["step_rollouts"]} != set(POLICIES):
            raise AssertionError(f"{rate}: missing step output.")
        for entry in manifest["step_rollouts"]:
            verified_hash(entry["path"], entry["sha256"])
            if pq.ParquetFile(ROOT / entry["path"]).metadata.num_rows != entry["rows"]:
                raise AssertionError(f"{rate}: step row count mismatch.")
        frames[rate] = frame
    return access, ids, frames, manifests


def reconciliation(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for rate in RATES:
        frame = frames[rate]
        for policy in POLICIES:
            x = frame.loc[frame.policy_name == policy]
            cash = assert_close(x.gross_interest, x.settled_interest_tax + x.net_interest, "net interest")
            wealth = assert_close(x.final_total_after_tax_wealth,
                                  x.cash_principal + x.net_interest + x.mandatory_after_tax_proceeds,
                                  "final wealth")
            proceeds = assert_close(x.gross_sale_proceeds_cumulative,
                                    x.cash_principal + x.equity_tax + x.mandatory_after_tax_proceeds,
                                    "sale proceeds")
            reward = assert_close(x.base_reward_sum,
                                  x.final_total_after_tax_wealth - x.initial_total_after_tax_wealth,
                                  "base reward")
            fractions = assert_close(x.discretionary_sale_fraction + x.mandatory_sale_fraction, 1,
                                     "sale fractions")
            assert_close(x.remaining_fraction_after_terminal, 0, "remaining inventory")
            hold = frame.loc[frame.policy_name == HOLD].set_index("episode_id").final_total_after_tax_wealth
            excess = assert_close(x.excess_wealth_vs_same_rate_hold,
                                  x.final_total_after_tax_wealth - x.episode_id.map(hold),
                                  "same-rate hold excess")
            rows.append({"rate_suffix": rate, "annual_cash_rate": float(x.annual_cash_rate.iloc[0]),
                         "policy_name": policy, "episodes": len(x),
                         "max_net_interest_residual": cash, "max_final_wealth_residual": wealth,
                         "max_sale_proceeds_residual": proceeds, "max_base_reward_residual": reward,
                         "max_sale_fraction_residual": fractions, "max_hold_excess_residual": excess})
    return pd.DataFrame(rows)


def comparisons(frames: dict[str, pd.DataFrame], ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(ids), size=(BOOTSTRAP_REPS, len(ids)))
    summary_rows = []
    cross_rows = []
    for rate in RATES:
        frame = frames[rate]
        hold = frame.loc[frame.policy_name == HOLD].set_index("episode_id").final_total_after_tax_wealth
        for policy in POLICIES:
            x = frame.loc[frame.policy_name == policy].set_index("episode_id").final_total_after_tax_wealth
            diff = paired_difference(x, hold, ids)
            win, tie, loss = paired_counts(diff)
            mean_ci = bootstrap_ci(diff, indices)
            median_ci = bootstrap_ci(diff, indices, median=True)
            summary_rows.append({"rate_suffix": rate, "annual_cash_rate": float(frame.annual_cash_rate.iloc[0]),
                                 "policy_name": policy, "episodes": len(ids),
                                 "mean_final_total_after_tax_wealth": float(x.mean()),
                                 "median_final_total_after_tax_wealth": float(x.median()),
                                 "mean_excess_vs_same_rate_hold": float(diff.mean()),
                                 "median_excess_vs_same_rate_hold": float(np.median(diff)),
                                 "paired_wins": win, "paired_ties": tie, "paired_losses": loss,
                                 "mean_excess_ci95_low": mean_ci[0], "mean_excess_ci95_high": mean_ci[1],
                                 "median_excess_ci95_low": median_ci[0], "median_excess_ci95_high": median_ci[1]})
    for policy in POLICIES:
        by_rate = {rate: frames[rate].loc[frames[rate].policy_name == policy]
                   .set_index("episode_id").final_total_after_tax_wealth for rate in RATES}
        for i, low in enumerate(RATES):
            for high in RATES[i + 1:]:
                diff = paired_difference(by_rate[high], by_rate[low], ids)
                win, tie, loss = paired_counts(diff)
                ci = bootstrap_ci(diff, indices)
                cross_rows.append({"policy_name": policy, "lower_rate_suffix": low,
                                   "higher_rate_suffix": high, "episodes": len(ids),
                                   "mean_wealth_difference_higher_minus_lower": float(diff.mean()),
                                   "median_wealth_difference_higher_minus_lower": float(np.median(diff)),
                                   "paired_wins_higher": win, "paired_ties": tie,
                                   "paired_losses_higher": loss, "mean_difference_ci95_low": ci[0],
                                   "mean_difference_ci95_high": ci[1]})
    return pd.DataFrame(summary_rows), pd.DataFrame(cross_rows)


def behavior(frames: dict[str, pd.DataFrame], manifests: dict[str, dict], ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    behavior_rows = []
    action_rows = []
    for rate in RATES:
        frame = frames[rate]
        entries = {entry["policy_name"]: entry for entry in manifests[rate]["step_rollouts"]}
        for policy in POLICIES:
            x = frame.loc[frame.policy_name == policy]
            sold = x.first_sale_days_from_start.dropna()
            behavior_rows.append({
                "rate_suffix": rate, "annual_cash_rate": float(x.annual_cash_rate.iloc[0]),
                "policy_name": policy, "episodes": len(x),
                "episodes_with_discretionary_sale": len(sold),
                "share_with_discretionary_sale": float(len(sold) / len(x)),
                "mean_first_sale_days_if_sold": float(sold.mean()) if len(sold) else np.nan,
                "median_first_sale_days_if_sold": float(sold.median()) if len(sold) else np.nan,
                "mean_discretionary_sale_count": float(x.discretionary_sale_count.mean()),
                "mean_discretionary_sale_fraction": float(x.discretionary_sale_fraction.mean()),
                "mean_remaining_fraction_before_terminal": float(x.remaining_fraction_before_terminal.mean()),
                "share_early_episode_termination": float(x.early_episode_termination.mean()),
                "mean_gross_sale_proceeds": float(x.gross_sale_proceeds_cumulative.mean()),
                "mean_cash_principal": float(x.cash_principal.mean()),
                "mean_gross_interest": float(x.gross_interest.mean()),
                "mean_settled_interest_tax": float(x.settled_interest_tax.mean()),
                "mean_net_interest": float(x.net_interest.mean()),
                "mean_equity_tax": float(x.equity_tax.mean()),
            })
            path = ROOT / entries[policy]["path"]
            steps = pd.read_parquet(path, columns=["episode_id", "action", "discretionary_sale_fraction_executed", "done"])
            if len(steps) != entries[policy]["rows"] or steps.loc[steps.done, "episode_id"].tolist() != ids:
                raise AssertionError(f"{rate}/{policy}: step terminal coverage mismatch.")
            if not steps.action.isin(range(5)).all():
                raise AssertionError(f"{rate}/{policy}: unknown action.")
            counts = steps.action.value_counts()
            action_rows.append({"rate_suffix": rate, "annual_cash_rate": float(x.annual_cash_rate.iloc[0]),
                                "policy_name": policy, "decision_steps": len(steps),
                                **{f"action_{j}_share": float(counts.get(j, 0) / len(steps)) for j in range(5)},
                                "executed_sale_step_share": float((steps.discretionary_sale_fraction_executed > 0).mean()),
                                "mean_executed_original_fraction_per_step": float(steps.discretionary_sale_fraction_executed.mean())})
    return pd.DataFrame(behavior_rows), pd.DataFrame(action_rows)


def figures(summary: pd.DataFrame, behavior_table: pd.DataFrame, actions: pd.DataFrame, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    rate_labels = ["7", "10", "10.5", "12", "13.75", "15"]
    positions = np.arange(len(RATES))
    colors = {DQN: "#125a78", HOLD: "#8c4d16", "sell_immediately": "#3b916b",
              "sell_half_then_hold": "#8064a2", "sell_quarters_over_time": "#bd6660", "random_policy": "#777777"}
    names = {DQN: "Frozen DQN", HOLD: "Hold to terminal", "sell_immediately": "Sell immediately",
             "sell_half_then_hold": "Half then hold", "sell_quarters_over_time": "Quarters", "random_policy": "Random"}
    files = []
    def save(fig: plt.Figure, stem: str) -> None:
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = directory / f"{stem}.{extension}"
            fig.savefig(path, dpi=190, bbox_inches="tight")
            files.append(path)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for policy in POLICIES:
        x = summary.loc[summary.policy_name == policy].set_index("rate_suffix").loc[list(RATES)]
        ax.plot(positions, x.mean_final_total_after_tax_wealth, marker="o", linewidth=2.5 if policy in (DQN, HOLD) else 1.3,
                alpha=1 if policy in (DQN, HOLD) else 0.75, label=names[policy], color=colors[policy])
    ax.set(xlabel="Fixed annual cash rate", ylabel="Mean final normalized after-tax wealth",
           title="Brazil-inspired treatment: final wealth at original terminal date")
    ax.set_xticks(positions, [f"{x}%" for x in rate_labels])
    ax.grid(alpha=0.22)
    ax.legend(ncol=2, fontsize=8)
    save(fig, "final_wealth_by_rate")

    dqn = summary.loc[summary.policy_name == DQN].set_index("rate_suffix").loc[list(RATES)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    mean = dqn.mean_excess_vs_same_rate_hold.to_numpy()
    lower = dqn.mean_excess_ci95_low.to_numpy()
    upper = dqn.mean_excess_ci95_high.to_numpy()
    ax.errorbar(positions, mean, yerr=[mean - lower, upper - mean], fmt="o-", capsize=4,
                linewidth=2, color=colors[DQN], label="Paired mean and 95% bootstrap CI")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set(xlabel="Fixed annual cash rate", ylabel="DQN minus hold, normalized wealth",
           title="Frozen DQN versus same-rate hold (paired episodes)")
    ax.set_xticks(positions, [f"{x}%" for x in rate_labels])
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    save(fig, "dqn_excess_vs_hold")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    panels = [("mean_first_sale_days_if_sold", "First sale day, among episodes with a sale"),
              ("mean_discretionary_sale_fraction", "Mean discretionary fraction sold"),
              ("mean_net_interest", "Mean net cash interest"),
              ("mean_equity_tax", "Mean equity tax")]
    for ax, (column, title) in zip(axes.flat, panels):
        for policy in (DQN, HOLD, "sell_immediately", "sell_half_then_hold"):
            x = behavior_table.loc[behavior_table.policy_name == policy].set_index("rate_suffix").loc[list(RATES)]
            values = x[column].to_numpy()
            if np.isfinite(values).any():
                ax.plot(positions, values, marker="o", label=names[policy], color=colors[policy])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xticks(positions, [f"{x}%" for x in rate_labels])
    axes[1, 0].set_xlabel("Fixed annual cash rate")
    axes[1, 1].set_xlabel("Fixed annual cash rate")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=8)
    save(fig, "sale_and_cash_behavior")

    dqn_actions = actions.loc[actions.policy_name == DQN].set_index("rate_suffix").loc[list(RATES)]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bottom = np.zeros(len(RATES))
    for j, label in enumerate(("Hold", "Sell 25%", "Sell 50%", "Sell 75%", "Sell 100%")):
        values = dqn_actions[f"action_{j}_share"].to_numpy()
        ax.bar(rate_labels, values, bottom=bottom, label=label)
        bottom += values
    ax.set(xlabel="Fixed annual cash rate (%)", ylabel="Share of DQN decision steps",
           title="Frozen DQN action mix across rates")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=8)
    save(fig, "dqn_action_shares")
    return files


def markdown_report(summary: pd.DataFrame, cross: pd.DataFrame, behavior_table: pd.DataFrame,
                    actions: pd.DataFrame, reconcile: pd.DataFrame) -> str:
    dqn = summary.loc[summary.policy_name == DQN].set_index("rate_suffix").loc[list(RATES)]
    hold = summary.loc[summary.policy_name == HOLD].set_index("rate_suffix").loc[list(RATES)]
    dqn_behavior = behavior_table.loc[behavior_table.policy_name == DQN].set_index("rate_suffix").loc[list(RATES)]
    lines = ["# Brazil-inspired sensitivity: locked test analysis", "",
             "## Scope and method", "",
             "All numbers below use only the six hashed Phase 7 test outputs, the same 1,519 ordered episode IDs, and the six validation-selected seed-42 DQNs. Wealth is normalized by original position basis and measured at each episode's original terminal date. The first/subsequent-sale Q-value margins remain 0.070/0.020. Within each rate, DQN excess is paired with hold on the same episode ID. Percentile bootstrap intervals resample the 1,519 paired IDs 4,000 times with seed 20260925; they measure episode sampling uncertainty conditional on the frozen models and historical paths.", "",
             "## Primary comparison: frozen DQN versus same-rate hold", "",
             "| Cash rate | DQN mean wealth | Hold mean wealth | Mean excess (95% paired CI) | Median excess | Wins / ties / losses |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for rate in RATES:
        a, b = dqn.loc[rate], hold.loc[rate]
        lines.append(f"| {a.annual_cash_rate:.2%} | {a.mean_final_total_after_tax_wealth:.4f} | {b.mean_final_total_after_tax_wealth:.4f} | {a.mean_excess_vs_same_rate_hold:+.4f} [{a.mean_excess_ci95_low:+.4f}, {a.mean_excess_ci95_high:+.4f}] | {a.median_excess_vs_same_rate_hold:+.4f} | {int(a.paired_wins)} / {int(a.paired_ties)} / {int(a.paired_losses)} |")
    lines += ["", "The paired median-excess intervals are in `tables/policy_summary.csv`. Figures `figures/final_wealth_by_rate.*` and `figures/dqn_excess_vs_hold.*` show the performance paths and paired uncertainty.", "",
              "## Final normalized wealth for every policy", "",
              "Each cell is mean / median at the original terminal date. `tables/policy_summary.csv` retains unrounded values and paired comparisons.", "",
              "| Cash rate | Frozen DQN | Hold | Immediate | Half then hold | Quarters | Random |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for rate in RATES:
        group = summary.loc[summary.rate_suffix == rate].set_index("policy_name")
        cells = [f"{group.loc[policy, 'mean_final_total_after_tax_wealth']:.4f} / {group.loc[policy, 'median_final_total_after_tax_wealth']:.4f}" for policy in POLICIES]
        lines.append(f"| {group.iloc[0].annual_cash_rate:.2%} | " + " | ".join(cells) + " |")
    lines += ["", "## Across-rate and behavior checks", "",
              "`tables/cross_rate_paired.csv` contains all 15 rate pairs for each of the six policies. Each difference is higher-rate wealth minus lower-rate wealth on the same episode ID, with wins, ties, losses and a paired 95% bootstrap interval for the mean. These are comparisons among separately trained policies under a combined scenario change; they do not isolate a tax-only effect.", "",
              "| Policy | Paired mean wealth difference, 15% minus 7% (95% CI) | Wins / ties / losses at 15% |",
              "| --- | ---: | ---: |"]
    for policy in POLICIES:
        c = cross.loc[(cross.policy_name == policy) & (cross.lower_rate_suffix == RATES[0]) & (cross.higher_rate_suffix == RATES[-1])].iloc[0]
        lines.append(f"| {policy.replace('_', ' ')} | {c.mean_wealth_difference_higher_minus_lower:+.4f} [{c.mean_difference_ci95_low:+.4f}, {c.mean_difference_ci95_high:+.4f}] | {int(c.paired_wins_higher)} / {int(c.paired_ties)} / {int(c.paired_losses_higher)} |")
    lines += ["", "| Cash rate | DQN sale share | Mean first sale day if sold | Mean discretionary fraction | Mean remaining fraction before terminal | Mean gross interest | Mean interest tax | Mean net interest |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for rate in RATES:
        a = dqn_behavior.loc[rate]
        lines.append(f"| {a.annual_cash_rate:.2%} | {a.share_with_discretionary_sale:.1%} | {a.mean_first_sale_days_if_sold:.1f} | {a.mean_discretionary_sale_fraction:.3f} | {a.mean_remaining_fraction_before_terminal:.3f} | {a.mean_gross_interest:.4f} | {a.mean_settled_interest_tax:.4f} | {a.mean_net_interest:.4f} |")
    low, high = dqn_behavior.loc[RATES[0]], dqn_behavior.loc[RATES[-1]]
    cross_dqn = cross.loc[(cross.policy_name == DQN) & (cross.lower_rate_suffix == RATES[0]) & (cross.higher_rate_suffix == RATES[-1])].iloc[0]
    lines += ["", f"Across the 7%–15% endpoints, the frozen DQN's paired mean wealth difference is {cross_dqn.mean_wealth_difference_higher_minus_lower:+.4f} (95% CI [{cross_dqn.mean_difference_ci95_low:+.4f}, {cross_dqn.mean_difference_ci95_high:+.4f}]). Its mean net interest changes from {low.mean_net_interest:.4f} to {high.mean_net_interest:.4f}; mean discretionary fraction sold changes from {low.mean_discretionary_sale_fraction:.3f} to {high.mean_discretionary_sale_fraction:.3f}. The gross-interest, settled-interest-tax, net-interest, gross-sale-proceeds, equity-tax, sale-count, first-sale, inventory, early-termination and action-share breakdowns for every policy are in `tables/behavior_summary.csv` and `tables/action_shares.csv`.", "",
              "`figures/sale_and_cash_behavior.*` shows selected behavior measures, and `figures/dqn_action_shares.*` shows the five DQN action shares. Each figure has PNG and SVG versions.", "",
              "## Interpretation and limits", "",
              "The treatments combine a flat 15% tax on positive equity gains with a rate-specific cash account. Higher cash rates can increase the value of earlier sale proceeds; the observed policy and wealth differences also reflect separate training at each rate. These results support a sensitivity comparison for the combined treatment, not a tax-only or causal estimate. The cash rate is fixed within an episode, and the study uses historical paths of U.S.-listed stocks in USD without BRL conversion.", "",
              f"The [Banco Central Focus report with reference date 11 September 2026]({FOCUS_URL}) lists end-year Selic medians of 13.75% (2026), 12% (2027), 10.5% (2028), and 10% (2029). The 7% scenario is a hypothetical lower bound, not a Focus forecast. The [Banco Central June 2025 Copom communication]({HISTORICAL_URL}) documents 15% as a historical policy-rate anchor, not a current Focus projection. These rates are scenario inputs, not modeled Brazilian cash instruments.", "",
              "There is one training seed per rate. The 36-input feedforward policy omits remaining inventory and cash-lot history, so its observation is partially observable. Checkpoints were selected using the predetermined first 500 validation IDs, not the 1,519 test IDs. Brazil training used gamma = 1.0; the historical U.S. C-lite v5 run used 0.99. The paired episode bootstrap does not account for overlapping ticker/time paths, seed variation, or market-regime uncertainty.", "",
              "## Original U.S. control (separate metric)", "",
              "The original U.S. C-lite v5 control remains in `runs/train_reward_c_lite_v5_full/`. Its reported `after_tax_total_value` is a tax-adjusted PnL measure; Brazil `final_total_after_tax_wealth` includes recovered basis and cash interest. No U.S. output was used as a Phase 8 numerical input, and the two metrics are not plotted on one scale.", "",
              "## Reproducibility and reconciliation", "",
              "Run `py -3.13 scripts/build_brazil_sensitivity_phase8.py` from the repository root. The builder verifies the access record, checkpoint/config/data/state hashes, all six evaluation manifests and output hashes, ordered episode coverage, terminal step coverage, and accounting identities before writing this analysis. `analysis_manifest.json` records every input and generated output hash. `tables/reconciliation.csv` contains the maximum row-level residuals for net interest, wealth, sale proceeds, base reward, sale fractions and same-rate hold excess. All residuals must be at most 1e-8. No model was retrained or evaluated by this builder.", ""]
    return "\n".join(lines)


def build(output: Path = OUT) -> dict:
    access, ids, frames, manifests = verified_inputs()
    reconcile = reconciliation(frames)
    policy_summary, cross_rate = comparisons(frames, ids)
    behavior_table, action_table = behavior(frames, manifests, ids)
    if len(policy_summary) != 36 or len(cross_rate) != 90 or len(behavior_table) != 36 or len(action_table) != 36:
        raise AssertionError("Analysis table completeness failed.")
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    products: list[Path] = []
    for name, table in (("policy_summary", policy_summary), ("dqn_vs_hold", policy_summary.loc[policy_summary.policy_name == DQN]),
                        ("cross_rate_paired", cross_rate), ("behavior_summary", behavior_table),
                        ("action_shares", action_table), ("reconciliation", reconcile)):
        path = table_dir / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.12g")
        products.append(path)
    products += figures(policy_summary, behavior_table, action_table, output / "figures")
    report_path = output / "summary.md"
    report_path.write_text(markdown_report(policy_summary, cross_rate, behavior_table, action_table, reconcile), encoding="utf-8")
    products.append(report_path)
    source_manifests = []
    for rate in RATES:
        path = ROOT / f"runs/brazil_sensitivity_rate_{rate}_v1/phase7_test/evaluation_manifest.json"
        source_manifests.append({"rate_suffix": rate, "path": path.relative_to(ROOT).as_posix(),
                                 "sha256": sha256(path), "episode_metrics": manifests[rate]["episode_metrics"],
                                 "step_rollouts": manifests[rate]["step_rollouts"],
                                 "frozen_checkpoint_sha256": manifests[rate]["frozen_checkpoint_sha256"],
                                 "effective_config_sha256": manifests[rate]["effective_config_sha256"],
                                 "scenario_parquet_sha256": manifests[rate]["scenario_parquet_sha256"],
                                 "state_freeze_sha256": manifests[rate]["state_freeze_sha256"]})
    manifest = {
        "phase": 8, "created_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "scripts/build_brazil_sensitivity_phase8.py",
        "builder_sha256": sha256(ROOT / "scripts/build_brazil_sensitivity_phase8.py"),
        "source": "locked_phase7_test_outputs_only",
        "test_access_path": "runs/brazil_phase7_test_access/test_access_checkpoint.json",
        "test_access_sha256": sha256(ROOT / "runs/brazil_phase7_test_access/test_access_checkpoint.json"),
        "ordered_test_id_sha256": ordered_id_hash(ids), "episodes_per_policy_rate": len(ids),
        "rates": list(RATES), "policies": list(POLICIES),
        "bootstrap": {"method": "paired_episode_percentile", "seed": BOOTSTRAP_SEED,
                      "replicates": BOOTSTRAP_REPS, "confidence_level": 0.95,
                      "tie_tolerance": TIE_TOL},
        "primary_focus_source": FOCUS_URL, "historical_15pct_source": HISTORICAL_URL,
        "source_evaluation_manifests": source_manifests,
        "outputs": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                     "bytes": path.stat().st_size} for path in products],
        "reconciliation_max_abs_residual": float(reconcile.filter(like="max_").to_numpy().max()),
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    result = build(args.output_dir.resolve())
    print(f"Phase 8 complete: {len(result['source_evaluation_manifests'])} frozen rates, "
          f"{len(result['outputs'])} outputs; maximum accounting residual "
          f"{result['reconciliation_max_abs_residual']:.3g}")


if __name__ == "__main__":
    main()
