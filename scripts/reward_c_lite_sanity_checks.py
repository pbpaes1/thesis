"""Direct-run Reward C-lite sanity checks on synthetic episodes.

Run from the project root:
    python scripts/reward_c_lite_sanity_checks.py
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

from src.environment.tax_aware_env import (  # noqa: E402
    REWARD_A_VERSION,
    REWARD_C_LITE_VERSION,
    TaxAwareEnv,
)


OUTPUT_DIR = PROJECT_ROOT / "runs" / "reward_c_lite_sanity_checks"
LAMBDA_COOLDOWN = 0.0025
COOLDOWN_DAYS = 20
TAX_FREE_PROFILE: dict[str, Any] = {
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


def reward_config() -> dict[str, Any]:
    return {
        "version": REWARD_C_LITE_VERSION,
        "use_environment_reward": True,
        "assert_reward_version": True,
        "expected_info_reward_version": REWARD_C_LITE_VERSION,
        "base_reward_version": REWARD_A_VERSION,
        "use_drawdown_penalty": False,
        "use_cooldown_penalty": True,
        "use_explicit_tax_saving_bonus": False,
        "use_reward_clipping": False,
        "use_reward_normalization": False,
        "cooldown_penalty": {
            "enabled": True,
            "lambda_cooldown": LAMBDA_COOLDOWN,
            "cooldown_days": COOLDOWN_DAYS,
            "scale_by_executed_fraction": True,
            "penalize_first_sale": False,
        },
    }


def make_synthetic_env(dates: list[str], gains: list[float] | None = None) -> TaxAwareEnv:
    if gains is None:
        gains = [0.20] * len(dates)
    if len(dates) != len(gains):
        raise ValueError("dates and gains must have the same length.")

    temp_dir = TemporaryDirectory()
    parquet_path = Path(temp_dir.name) / "synthetic_episode.parquet"
    df = pd.DataFrame(
        {
            "episode_id": ["episode_1"] * len(dates),
            "date": pd.to_datetime(dates),
            "tax_transition_date": pd.to_datetime(["2030-01-01"] * len(dates)),
            "unrealized_gains_pct": gains,
            "feature": np.arange(len(dates), dtype=float),
        }
    )
    df.to_parquet(parquet_path)

    env = TaxAwareEnv(
        parquet_path=parquet_path,
        state_columns=["feature"],
        tax_config=TAX_FREE_PROFILE,
        reward_config=reward_config(),
    )
    env._reward_c_lite_sanity_temp_dir = temp_dir
    return env


def action_index_for_fraction(env: TaxAwareEnv, target_fraction: float) -> int:
    for idx, value in enumerate(env.action_fractions):
        if approx_equal(value, target_fraction):
            return idx
    raise ValueError(
        f"Action fraction {target_fraction} not found in {env.action_fractions}."
    )


def step_fraction(env: TaxAwareEnv, fraction: float) -> dict[str, Any]:
    action_idx = action_index_for_fraction(env, fraction)
    _, reward, done, truncated, info = env.step(action_idx)
    assert truncated is False
    assert_reward_c_lite_identity(reward, info)
    return {"reward": reward, "done": done, "info": info}


def assert_reward_c_lite_identity(reward: float, info: dict[str, Any]) -> None:
    assert info["reward_version"] == REWARD_C_LITE_VERSION
    assert_approx("reward alias", reward, info["reward"])
    assert_approx(
        "reward A identity",
        info["reward_A"],
        info["after_tax_total_value"] - info["previous_after_tax_total_value"],
    )
    assert_approx(
        "reward C-lite identity",
        info["reward_C_lite"],
        info["reward_A"] - info["cooldown_penalty"],
    )
    assert_approx("returned reward", reward, info["reward_C_lite"])
    assert_approx(
        "after-tax total identity",
        info["after_tax_total_value"],
        info["cum_realized_after_tax_pnl"]
        + info["after_tax_liquidation_value_remaining"],
    )


def expected_penalty(executed_fraction: float, days_since: int) -> float:
    return (
        LAMBDA_COOLDOWN
        * executed_fraction
        * max(0, COOLDOWN_DAYS - days_since)
        / COOLDOWN_DAYS
    )


def result_row(name: str, info: dict[str, Any], *, passed: bool = True) -> dict[str, Any]:
    return {
        "scenario_name": name,
        "passed": passed,
        "reward": info["reward"],
        "reward_A": info["reward_A"],
        "reward_C_lite": info["reward_C_lite"],
        "cooldown_penalty": info["cooldown_penalty"],
        "cooldown_penalty_applied": info["cooldown_penalty_applied"],
        "days_since_last_sale": info["days_since_last_sale"],
        "action_fraction_requested": info["action_fraction_requested"],
        "action_fraction_executed": info["action_fraction_executed"],
        "sale_count": info["sale_count"],
        "terminal_liquidation_executed": info["terminal_liquidation_executed"],
        "assumption": (
            "automatic terminal liquidation is not penalized because it is not "
            "agent-requested"
            if name == "automatic_terminal_liquidation_not_penalized"
            else ""
        ),
    }


def scenario_single_sale_no_penalty() -> dict[str, Any]:
    name = "single_sale_no_penalty"
    env = make_synthetic_env(["2020-01-01", "2020-01-06", "2020-01-07"])
    env.reset()
    row = step_fraction(env, 0.50)
    info = row["info"]
    assert info["previous_sale_exists"] is False
    assert info["days_since_last_sale"] is None
    assert_approx("first sale penalty", info["cooldown_penalty"], 0.0)
    assert_approx("first sale reward", row["reward"], info["reward_A"])
    return result_row(name, info)


def scenario_repeated_sale_inside_cooldown() -> dict[str, Any]:
    name = "repeated_sale_inside_cooldown"
    env = make_synthetic_env(["2020-01-01", "2020-01-06", "2020-01-07"])
    env.reset()
    step_fraction(env, 0.25)
    row = step_fraction(env, 0.50)
    info = row["info"]
    penalty = expected_penalty(0.50, 5)
    assert info["cooldown_penalty"] > 0.0
    assert info["cooldown_penalty_applied"] is True
    assert_approx("inside cooldown penalty", info["cooldown_penalty"], penalty)
    assert_approx("C-lite below A", info["reward_C_lite"], info["reward_A"] - penalty)
    return result_row(name, info)


def scenario_repeated_sale_outside_cooldown() -> dict[str, Any]:
    name = "repeated_sale_outside_cooldown"
    env = make_synthetic_env(["2020-01-01", "2020-02-01", "2020-02-03"])
    env.reset()
    step_fraction(env, 0.25)
    row = step_fraction(env, 0.50)
    info = row["info"]
    assert info["days_since_last_sale"] == 31
    assert_approx("outside cooldown penalty", info["cooldown_penalty"], 0.0)
    return result_row(name, info)


def scenario_hold_between_sales_does_not_reset() -> dict[str, Any]:
    name = "hold_between_sales_does_not_reset"
    env = make_synthetic_env(
        ["2020-01-01", "2020-01-06", "2020-01-10", "2020-01-13"]
    )
    env.reset()
    first = step_fraction(env, 0.25)["info"]
    hold = step_fraction(env, 0.0)["info"]
    row = step_fraction(env, 0.25)
    info = row["info"]
    assert hold["sale_count"] == first["sale_count"]
    assert hold["last_sale_date_after_step"] == first["last_sale_date_after_step"]
    assert info["days_since_last_sale"] == 9
    assert info["cooldown_penalty"] > 0.0
    return result_row(name, info)


def scenario_oversell_clamp_penalty_uses_executed_fraction() -> dict[str, Any]:
    name = "oversell_clamp_penalty_uses_executed_fraction"
    env = make_synthetic_env(["2020-01-01", "2020-01-06", "2020-01-07"])
    env.reset()
    step_fraction(env, 0.75)
    row = step_fraction(env, 0.50)
    info = row["info"]
    penalty = expected_penalty(0.25, 5)
    assert_approx("oversell requested", info["action_fraction_requested"], 0.50)
    assert_approx("oversell executed", info["action_fraction_executed"], 0.25)
    assert_approx("oversell cooldown penalty", info["cooldown_penalty"], penalty)
    return result_row(name, info)


def scenario_automatic_terminal_liquidation_not_penalized() -> dict[str, Any]:
    name = "automatic_terminal_liquidation_not_penalized"
    env = make_synthetic_env(["2020-01-01", "2020-01-06", "2020-01-07"])
    env.reset()
    first = step_fraction(env, 0.25)["info"]
    row = step_fraction(env, 0.0)
    info = row["info"]
    assert row["done"] is True
    assert info["terminal_liquidation_executed"] is True
    assert_approx("terminal action execution", info["action_fraction_executed"], 0.0)
    assert_approx("terminal cooldown penalty", info["cooldown_penalty"], 0.0)
    assert info["sale_count"] == first["sale_count"]
    assert info["last_sale_date_after_step"] == first["last_sale_date_after_step"]
    return result_row(name, info)


def main() -> None:
    scenarios = [
        scenario_single_sale_no_penalty,
        scenario_repeated_sale_inside_cooldown,
        scenario_repeated_sale_outside_cooldown,
        scenario_hold_between_sales_does_not_reset,
        scenario_oversell_clamp_penalty_uses_executed_fraction,
        scenario_automatic_terminal_liquidation_not_penalized,
    ]
    rows = [scenario() for scenario in scenarios]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "reward_c_lite_sanity_check_results.csv"
    summary_path = OUTPUT_DIR / "reward_c_lite_sanity_check_summary.txt"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Reward C-lite Sanity Check Summary\n")
        handle.write(f"reward_version: {REWARD_C_LITE_VERSION}\n")
        handle.write(f"lambda_cooldown: {LAMBDA_COOLDOWN}\n")
        handle.write(f"cooldown_days: {COOLDOWN_DAYS}\n")
        handle.write(
            "terminal_liquidation_assumption: automatic terminal liquidation is "
            "not penalized because it is not agent-requested\n"
        )
        handle.write(f"num_scenarios: {len(rows)}\n")
        handle.write("status: passed\n")
    print("REWARD C-LITE SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
