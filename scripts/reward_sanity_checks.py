"""Direct-run Reward A sanity checks on synthetic episodes.

Run from the project root:
    python scripts/reward_sanity_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.tax_aware_env import TaxAwareEnv


STANDARD_PROFILE = {
    "profile_name": "mass_affluent_individual",
    "short_term_rate": 0.24,
    "long_term_rate": 0.15,
    "niit_rate": 0.0,
    "apply_niit": False,
}

TAX_FREE_PROFILE = {
    "profile_name": "tax_free",
    "short_term_rate": 0.0,
    "long_term_rate": 0.0,
    "niit_rate": 0.0,
    "apply_niit": False,
}


def approx_equal(a: Any, b: Any, tol: float = 1e-10) -> bool:
    return abs(float(a) - float(b)) <= tol


def assert_approx(label: str, actual: Any, expected: Any, tol: float = 1e-10) -> None:
    assert approx_equal(actual, expected, tol), (
        f"{label}: actual={float(actual):.12f}, expected={float(expected):.12f}"
    )


def make_synthetic_env(
    dates: list[str],
    gains: list[float],
    transition_date: str,
    tax_profile: dict[str, Any],
) -> TaxAwareEnv:
    if len(dates) != len(gains):
        raise ValueError("dates and gains must have the same length.")

    temp_dir = TemporaryDirectory()
    parquet_path = Path(temp_dir.name) / "synthetic_episode.parquet"
    df = pd.DataFrame(
        {
            "episode_id": ["episode_1"] * len(dates),
            "date": pd.to_datetime(dates),
            "tax_transition_date": pd.to_datetime(
                [transition_date] * len(dates)
            ),
            "unrealized_gains_pct": gains,
            "feature": np.arange(len(dates), dtype=float),
        }
    )
    df.to_parquet(parquet_path)

    env = TaxAwareEnv(
        parquet_path=parquet_path,
        state_columns=["feature"],
        tax_config=tax_profile,
    )
    env._reward_sanity_temp_dir = temp_dir
    return env


def action_index_for_fraction(env: TaxAwareEnv, target_fraction: float) -> int:
    for idx, value in enumerate(env.action_fractions):
        if approx_equal(value, target_fraction):
            return idx
    raise ValueError(
        f"Action fraction {target_fraction} not found in {env.action_fractions}."
    )


def assert_reward_identity(step_result: dict[str, Any]) -> None:
    reward = step_result["reward"]
    info = step_result["info"]

    assert info["reward_version"] == "A_after_tax_total_value_change"
    assert_approx("reward info alias", reward, info["reward"])
    assert_approx("reward_A", reward, info["reward_A"])
    assert_approx(
        "reward identity",
        reward,
        info["after_tax_total_value"] - info["previous_after_tax_total_value"],
    )
    assert info["remaining_fraction"] >= -1e-12
    assert info["sold_fraction"] <= 1.0 + 1e-12
    assert_approx(
        "after-tax total identity",
        info["after_tax_total_value"],
        info["cum_realized_after_tax_pnl"]
        + info["after_tax_liquidation_value_remaining"],
    )


def run_actions(env: TaxAwareEnv, actions: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, action_fraction in enumerate(actions):
        action_idx = action_index_for_fraction(env, action_fraction)
        _, reward, done, truncated, info = env.step(action_idx)
        row = {
            "step": step,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "info": info,
        }
        assert_reward_identity(row)
        rows.append(row)
        if done:
            break
    return rows


def _fmt_value(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if value is None:
        return "None"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def print_scenario_summary(name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{name}")
    fields = [
        "scenario_name",
        "step",
        "action_fraction_requested",
        "action_fraction_executed",
        "reward",
        "previous_after_tax_total_value",
        "after_tax_total_value",
        "realized_after_tax_increment",
        "after_tax_liquidation_value_remaining",
        "cum_realized_after_tax_pnl",
        "sold_fraction",
        "remaining_fraction",
        "tax_regime",
        "after_tax_liquidation_tax_regime",
        "terminal_liquidation_executed",
    ]
    print(" | ".join(fields))
    for row in rows:
        info = row["info"]
        values = {
            "scenario_name": name,
            "step": row["step"],
            "reward": row["reward"],
            **info,
        }
        print(" | ".join(_fmt_value(values[field]) for field in fields))


def scenario_immediate_sell() -> None:
    name = "scenario_immediate_sell"
    env = make_synthetic_env(
        dates=["2020-01-01", "2020-01-02", "2020-01-03"],
        gains=[0.30, 0.30, 0.30],
        transition_date="2020-01-10",
        tax_profile=STANDARD_PROFILE,
    )
    _, reset_info = env.reset()
    assert_approx("initial after-tax value", reset_info["after_tax_total_value"], 0.228)

    rows = run_actions(env, [1.0])
    print_scenario_summary(name, rows)
    info = rows[0]["info"]

    assert_approx("executed fraction", info["action_fraction_executed"], 1.0)
    assert_approx("applicable tax rate", info["applicable_tax_rate"], 0.24)
    assert_approx("realized pre-tax increment", info["realized_pre_tax_increment"], 0.30)
    assert_approx("tax paid", info["tax_paid"], 0.072)
    assert_approx(
        "realized after-tax increment",
        info["realized_after_tax_increment"],
        0.228,
    )
    assert_approx("after-tax total value", info["after_tax_total_value"], 0.228)
    assert_approx("reward", rows[0]["reward"], 0.0)
    assert_approx("remaining fraction", info["remaining_fraction"], 0.0)
    assert_approx("sold fraction", info["sold_fraction"], 1.0)
    assert info["terminal_liquidation_executed"] is False
    assert rows[0]["done"] is True


def scenario_safe_waiting_until_long_term() -> None:
    name = "scenario_safe_waiting_until_long_term"
    env = make_synthetic_env(
        dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
        gains=[0.30, 0.30, 0.30, 0.30],
        transition_date="2020-01-03",
        tax_profile=STANDARD_PROFILE,
    )
    _, reset_info = env.reset()
    assert_approx("initial after-tax value", reset_info["after_tax_total_value"], 0.228)

    rows = run_actions(env, [0.0, 0.0, 0.0])
    print_scenario_summary(name, rows)

    assert_approx("step 0 reward", rows[0]["reward"], 0.0)
    assert_approx("step 1 reward", rows[1]["reward"], 0.0)

    long_term_rows = [
        row
        for row in rows
        if row["info"]["after_tax_liquidation_tax_regime"] == "long_term"
    ]
    assert long_term_rows, "Expected at least one long-term liquidation valuation step."
    first_long_term = long_term_rows[0]
    assert_approx("first long-term reward", first_long_term["reward"], 0.027)
    assert_approx(
        "first long-term executed fraction",
        first_long_term["info"]["action_fraction_executed"],
        0.0,
    )


def scenario_crash_while_waiting() -> None:
    name = "scenario_crash_while_waiting"
    env = make_synthetic_env(
        dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
        gains=[0.30, 0.25, -0.10, -0.20],
        transition_date="2020-01-10",
        tax_profile=STANDARD_PROFILE,
    )
    _, reset_info = env.reset()
    assert_approx("initial after-tax value", reset_info["after_tax_total_value"], 0.228)

    rows = run_actions(env, [0.0, 0.0, 0.0])
    print_scenario_summary(name, rows)

    assert_approx("step 0 reward", rows[0]["reward"], 0.0)
    assert_approx("step 1 after-tax value", rows[1]["info"]["after_tax_total_value"], 0.19)
    assert_approx("step 1 reward", rows[1]["reward"], -0.038)
    assert any(row["reward"] < 0.0 for row in rows), "Expected at least one negative reward."

    final_info = rows[-1]["info"]
    assert final_info["terminal_liquidation_executed"] is True
    assert_approx(
        "terminal final-row PnL",
        final_info["terminal_liquidation_full_position_pnl"],
        -0.20,
    )
    assert_approx("terminal tax paid", final_info["terminal_liquidation_tax_paid"], 0.0)
    assert_approx(
        "terminal after-tax increment",
        final_info["terminal_liquidation_after_tax_increment"],
        -0.20,
    )
    assert_approx("final after-tax total value", final_info["after_tax_total_value"], -0.20)


def scenario_partial_exit_then_wait() -> None:
    name = "scenario_partial_exit_then_wait"
    env = make_synthetic_env(
        dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
        gains=[0.30, 0.35, 0.35, 0.40],
        transition_date="2020-01-03",
        tax_profile=STANDARD_PROFILE,
    )
    _, reset_info = env.reset()
    assert_approx("initial after-tax value", reset_info["after_tax_total_value"], 0.228)

    rows = run_actions(env, [0.5, 0.0, 0.5])
    print_scenario_summary(name, rows)

    first = rows[0]["info"]
    second = rows[1]["info"]
    final = rows[2]["info"]

    assert_approx("first sale realized after-tax", first["realized_after_tax_increment"], 0.114)
    assert_approx("first sale remaining liquidation", first["after_tax_liquidation_value_remaining"], 0.114)
    assert_approx("first sale total", first["after_tax_total_value"], 0.228)
    assert_approx("first sale reward", rows[0]["reward"], 0.0)

    assert_approx("hold cumulative realized", second["cum_realized_after_tax_pnl"], 0.114)
    assert_approx("hold remaining liquidation", second["after_tax_liquidation_value_remaining"], 0.133)
    assert_approx("hold total", second["after_tax_total_value"], 0.247)
    assert_approx("hold reward", rows[1]["reward"], 0.019)
    assert rows[1]["reward"] > 0.0

    assert final["tax_regime"] == "long_term"
    assert_approx("final executed fraction", final["action_fraction_executed"], 0.5)
    assert_approx("final tax rate", final["applicable_tax_rate"], 0.15)
    assert_approx(
        "final realized after-tax increment",
        final["realized_after_tax_increment"],
        0.14875,
    )
    assert_approx("final cumulative realized", final["cum_realized_after_tax_pnl"], 0.26275)
    assert_approx("final remaining fraction", final["remaining_fraction"], 0.0)
    assert_approx("final total", final["after_tax_total_value"], 0.26275)
    assert_approx("final reward", rows[2]["reward"], 0.01575)
    assert final["terminal_liquidation_executed"] is False


def scenario_oversell_clamp() -> None:
    name = "scenario_oversell_clamp"
    env = make_synthetic_env(
        dates=["2020-01-01", "2020-01-02", "2020-01-03"],
        gains=[0.20, 0.20, 0.20],
        transition_date="2020-01-10",
        tax_profile=STANDARD_PROFILE,
    )
    env.reset()

    rows = run_actions(env, [0.75, 0.50])
    print_scenario_summary(name, rows)

    first = rows[0]["info"]
    final = rows[1]["info"]

    assert_approx("first requested fraction", first["action_fraction_requested"], 0.75)
    assert_approx("first executed fraction", first["action_fraction_executed"], 0.75)
    assert_approx("first remaining fraction", first["remaining_fraction"], 0.25)

    assert_approx("second requested fraction", final["action_fraction_requested"], 0.50)
    assert_approx("second executed fraction", final["action_fraction_executed"], 0.25)
    assert_approx("final remaining fraction", final["remaining_fraction"], 0.0)
    assert_approx("final sold fraction", final["sold_fraction"], 1.0)
    assert final["sold_fraction"] <= 1.0 + 1e-12
    assert final["terminal_liquidation_executed"] is False
    assert rows[1]["done"] is True


def scenario_terminal_liquidation_uses_final_row_pnl() -> None:
    name = "scenario_terminal_liquidation_uses_final_row_pnl"
    env = make_synthetic_env(
        dates=["2020-01-01", "2020-01-02", "2020-01-03"],
        gains=[0.10, 0.10, 0.30],
        transition_date="2020-01-10",
        tax_profile=STANDARD_PROFILE,
    )
    _, reset_info = env.reset()
    assert_approx("initial after-tax value", reset_info["after_tax_total_value"], 0.076)

    rows = run_actions(env, [0.0, 0.0])
    print_scenario_summary(name, rows)

    final = rows[-1]["info"]
    assert final["terminal_liquidation_executed"] is True
    assert final["sale_row_ptr"] == 1
    assert final["terminal_liquidation_row_ptr"] == 2
    assert_approx("sale row PnL", final["full_position_pnl"], 0.10)
    assert_approx(
        "terminal final-row PnL",
        final["terminal_liquidation_full_position_pnl"],
        0.30,
    )
    assert_approx("terminal tax rate", final["terminal_liquidation_tax_rate"], 0.15)
    assert_approx(
        "terminal pre-tax increment",
        final["terminal_liquidation_pre_tax_increment"],
        0.30,
    )
    assert_approx("terminal tax paid", final["terminal_liquidation_tax_paid"], 0.045)
    assert_approx(
        "terminal after-tax increment",
        final["terminal_liquidation_after_tax_increment"],
        0.255,
    )
    assert_approx("final after-tax total value", final["after_tax_total_value"], 0.255)
    assert_approx("final reward", rows[-1]["reward"], 0.179)


def main() -> None:
    scenario_immediate_sell()
    scenario_safe_waiting_until_long_term()
    scenario_crash_while_waiting()
    scenario_partial_exit_then_wait()
    scenario_oversell_clamp()
    scenario_terminal_liquidation_uses_final_row_pnl()
    print("ALL REWARD SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
