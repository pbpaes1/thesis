"""Manual smoke-test runner for the thesis tax-aware environment.

This script is intended for console inspection (not unit testing).
It runs a few hardcoded trajectories and prints per-step bookkeeping output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.tax_profiles import load_tax_profile
from src.environment.tax_aware_env import TaxAwareEnv


STATE_COLUMNS_PATH = PROJECT_ROOT / "data" / "freeze" / "v1" / "allowed_state_columns_v1.json"
PARQUET_PATH = PROJECT_ROOT / "data" / "episodes" / "drl_episodes.parquet"
TAX_PROFILE_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "individual_tax_profiles_v1.yaml"
)

STANDARD_PROFILE: dict[str, Any] = load_tax_profile(
    TAX_PROFILE_CONFIG_PATH,
    "mass_affluent_individual",
)
HIGH_INCOME_PROFILE: dict[str, Any] = load_tax_profile(
    TAX_PROFILE_CONFIG_PATH,
    "high_income_individual",
)

TRAJECTORIES: list[tuple[str, list[float]]] = [
    ("Trajectory A (Hold, Hold, Sell 100%)", [0.0, 0.0, 1.0]),
    ("Trajectory B (Sell 50%, Hold, Sell 50%)", [0.5, 0.0, 0.5]),
    ("Trajectory C (Sell 75%, Sell 50%)", [0.75, 0.5]),
]

SCENARIOS: list[tuple[str, bool, bool]] = [
    ("Short-term positive-gain scenario", True, True),
    ("Long-term positive-gain scenario", False, True),
]


def load_state_columns(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen state schema JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cols = payload.get("allowed_state_columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError(
            "Invalid state schema JSON: expected non-empty 'allowed_state_columns' list."
        )
    return [str(c) for c in cols]


def make_env(
    parquet_path: Path,
    state_columns: list[str],
    tax_profile: dict[str, Any],
) -> TaxAwareEnv:
    return TaxAwareEnv(
        parquet_path=parquet_path,
        state_columns=state_columns,
        tax_config=tax_profile,
    )


def action_index_for_fraction(env: TaxAwareEnv, fraction: float) -> int:
    for idx, value in enumerate(env.action_fractions):
        if np.isclose(float(value), float(fraction), atol=1e-12):
            return idx
    raise ValueError(
        f"Action fraction {fraction} is not in env.action_fractions={env.action_fractions}."
    )


def find_candidate_episode_row(
    env: TaxAwareEnv,
    *,
    min_steps: int,
    short_term: bool | None,
    positive_gain: bool | None,
) -> tuple[str, int] | None:
    env._load_episode_index()
    if env._df is None:
        raise RuntimeError("Environment dataframe is not loaded.")

    ranked: list[tuple[tuple[float, int], str, int]] = []

    for episode_id, row_idx in env._episode_index.items():
        ep = env._df.iloc[row_idx].reset_index(drop=True)
        if len(ep) < min_steps or ep.empty:
            continue

        pnl = pd.to_numeric(ep["unrealized_gains_pct"], errors="coerce")
        dates = pd.to_datetime(ep["date"], errors="coerce")
        transitions = pd.to_datetime(ep["tax_transition_date"], errors="coerce")
        max_start_row = len(ep) - min_steps
        if max_start_row < 0:
            continue

        mask = pd.Series(True, index=ep.index)
        mask &= dates.notna() & transitions.notna() & pnl.notna()
        mask &= ep.index <= max_start_row

        if short_term is True:
            mask &= dates < transitions
        elif short_term is False:
            mask &= dates >= transitions

        if positive_gain is True:
            mask &= pnl > 0.0
        elif positive_gain is False:
            mask &= pnl <= 0.0

        candidate_rows = np.flatnonzero(mask.to_numpy())
        for row_pos in candidate_rows:
            row_pnl = float(pnl.iloc[row_pos])
            remaining_rows = int(len(ep) - row_pos)
            score = (row_pnl, remaining_rows)
            ranked.append((score, str(episode_id), int(row_pos)))

    if not ranked:
        return None

    ranked.sort(key=lambda x: (-x[0][0], -x[0][1], x[1], x[2]))
    _, best_episode_id, best_row_pos = ranked[0]
    return best_episode_id, best_row_pos


def _fmt_float(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(v):
        return "nan"
    return f"{v:.6f}"


def _fmt_date(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "invalid_date"
    return ts.strftime("%Y-%m-%d")


def _regime_label(date_value: Any, transition_value: Any) -> str:
    date_ts = pd.to_datetime(date_value, errors="coerce")
    transition_ts = pd.to_datetime(transition_value, errors="coerce")
    if pd.isna(date_ts) or pd.isna(transition_ts):
        return "unknown"
    return "short_term" if date_ts < transition_ts else "long_term"


def _advance_to_row(env: TaxAwareEnv, row_pos: int) -> bool:
    if row_pos <= 0:
        return True
    hold_idx = action_index_for_fraction(env, 0.0)
    for _ in range(row_pos):
        _, _, done, _, _ = env.step(hold_idx)
        if done and env._current_row_ptr < row_pos:
            return False
    return env._current_row_ptr == row_pos


def run_trajectory_from_row(
    env: TaxAwareEnv,
    *,
    episode_id: str,
    row_pos: int,
    scenario_label: str,
    trajectory_name: str,
    trajectory_fractions: list[float],
) -> None:
    initial_obs, _ = env.reset(episode_id=episode_id)
    if env._current_episode_df is None:
        raise RuntimeError("Active episode dataframe not available after reset.")

    if not _advance_to_row(env, row_pos):
        print(
            f"WARNING: could not advance to row_pos={row_pos} in episode "
            f"'{episode_id}'. Skipping trajectory."
        )
        return

    current_obs = env._get_observation()
    start_row = env._current_episode_df.iloc[row_pos]
    start_regime = _regime_label(start_row["date"], start_row["tax_transition_date"])

    print("\n" + "=" * 120)
    print(trajectory_name)
    print(f"scenario             : {scenario_label}")
    print(f"episode_id           : {episode_id}")
    print(f"start_row_pos        : {row_pos}")
    print(f"start_row_date       : {_fmt_date(start_row['date'])}")
    print(f"start_row_regime     : {start_regime}")
    print(f"tax_profile_name     : {env.tax_config.get('profile_name', 'unspecified_individual')}")
    print(f"initial_obs_shape    : {initial_obs.shape}")
    print(f"start_obs_shape      : {current_obs.shape}")
    print(f"action_fractions_map : {dict(enumerate(env.action_fractions))}")

    ep_len = len(env._current_episode_df)
    rows_remaining = ep_len - row_pos
    if rows_remaining <= 0:
        print(
            "WARNING: no rows remaining from selected start. Skipping trajectory."
        )
        return
    if rows_remaining < len(trajectory_fractions):
        print(
            f"WARNING: rows remaining from start ({rows_remaining}) are shorter than "
            f"trajectory steps ({len(trajectory_fractions)}). Running until done."
        )

    action_indices = [action_index_for_fraction(env, frac) for frac in trajectory_fractions]

    header = (
        f"{'step':>4} {'date':<10} {'a_idx':>5} {'req_frac':>10} {'exe_frac':>10} "
        f"{'sold':>10} {'remain':>10} {'regime':<11} {'rate':>9} {'full_pnl':>11} "
        f"{'pre_tax':>11} {'tax_paid':>11} {'after_tax':>11} {'cum_pre':>11} "
        f"{'cum_after':>11} {'done':>6}"
    )
    print(header)
    print("-" * len(header))

    for step_no, action_idx in enumerate(action_indices, start=1):
        _, _, done, _, info = env.step(action_idx)
        print(
            f"{step_no:>4d} "
            f"{_fmt_date(info.get('date')):<10} "
            f"{int(info.get('action', -1)):>5d} "
            f"{_fmt_float(info.get('action_fraction_requested')):>10} "
            f"{_fmt_float(info.get('action_fraction_executed')):>10} "
            f"{_fmt_float(info.get('sold_fraction')):>10} "
            f"{_fmt_float(info.get('remaining_fraction')):>10} "
            f"{str(info.get('tax_regime', 'n/a')):<11} "
            f"{_fmt_float(info.get('applicable_tax_rate')):>9} "
            f"{_fmt_float(info.get('full_position_pnl')):>11} "
            f"{_fmt_float(info.get('realized_pre_tax_increment')):>11} "
            f"{_fmt_float(info.get('tax_paid')):>11} "
            f"{_fmt_float(info.get('realized_after_tax_increment')):>11} "
            f"{_fmt_float(info.get('cum_realized_pre_tax_pnl')):>11} "
            f"{_fmt_float(info.get('cum_realized_after_tax_pnl')):>11} "
            f"{str(bool(done)):>6}"
        )

        if done and step_no < len(action_indices):
            print(
                f"INFO: done=True reached at step {step_no}; "
                "remaining planned actions skipped."
            )
            break


def main() -> None:
    state_columns = load_state_columns(STATE_COLUMNS_PATH)

    profiles = [STANDARD_PROFILE, HIGH_INCOME_PROFILE]
    min_required_steps = max(len(path) for _, path in TRAJECTORIES)

    for profile in profiles:
        env = make_env(PARQUET_PATH, state_columns, profile)
        print("\n" + "#" * 120)
        print(f"Profile: {profile['profile_name']}")

        for scenario_label, is_short_term, needs_positive_gain in SCENARIOS:
            candidate = find_candidate_episode_row(
                env,
                min_steps=min_required_steps,
                short_term=is_short_term,
                positive_gain=needs_positive_gain,
            )
            if candidate is None:
                # Fallback for datasets where qualifying rows appear only at terminal step.
                candidate = find_candidate_episode_row(
                    env,
                    min_steps=1,
                    short_term=is_short_term,
                    positive_gain=needs_positive_gain,
                )
                if candidate is not None:
                    print(
                        f"\nWARNING: using fallback min_steps=1 for '{scenario_label}' "
                        "because no longer rollout window was found."
                    )
            if candidate is None:
                print(
                    f"\nWARNING: no candidate found for '{scenario_label}' "
                    f"under profile '{profile['profile_name']}'. Skipping scenario."
                )
                continue

            episode_id, row_pos = candidate
            if env._df is None:
                raise RuntimeError("Environment dataframe is not loaded.")
            scenario_row = env._df.iloc[env._episode_index[episode_id]].reset_index(drop=True).iloc[row_pos]

            print("\n" + "-" * 120)
            print(f"Scenario: {scenario_label}")
            print(f"episode_id           : {episode_id}")
            print(f"row_pos              : {row_pos}")
            print(f"sale_start_date      : {_fmt_date(scenario_row['date'])}")
            print(f"sale_start_regime    : {_regime_label(scenario_row['date'], scenario_row['tax_transition_date'])}")
            print(f"tax_profile_name     : {profile['profile_name']}")

            for name, fractions in TRAJECTORIES:
                run_trajectory_from_row(
                    env,
                    episode_id=episode_id,
                    row_pos=row_pos,
                    scenario_label=scenario_label,
                    trajectory_name=name,
                    trajectory_fractions=fractions,
                )


if __name__ == "__main__":
    main()
