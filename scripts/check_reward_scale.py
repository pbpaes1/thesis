"""Reward scale stability check for Reward A.

Run from the project root:
    python scripts/check_reward_scale.py
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


MAX_EPISODES = 500
SCHEMA_PATH = Path("data/freeze/v1/allowed_state_columns_v1.json")
PARQUET_PATH = Path("data/episodes/drl_episodes.parquet")
OUTPUT_DIR = Path("reports/reward_scale")
TAX_PROFILE_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "individual_tax_profiles_v1.yaml"
)
TAX_PROFILE_NAMES = (
    "tax_free",
    "mass_affluent_individual",
    "high_income_individual",
)

TAX_PROFILES = tuple(
    load_tax_profile(TAX_PROFILE_CONFIG_PATH, profile_name)
    for profile_name in TAX_PROFILE_NAMES
)

POLICY_NAMES = (
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "sell_quarters_over_time",
    "random_policy",
)

GROUP_COLUMNS = ["profile_name", "policy_name"]


def approx_equal(a: Any, b: Any, tol: float = 1e-10) -> bool:
    return abs(float(a) - float(b)) <= tol


def assert_approx(label: str, actual: Any, expected: Any, tol: float = 1e-10) -> None:
    assert approx_equal(actual, expected, tol), (
        f"{label}: actual={float(actual):.12f}, expected={float(expected):.12f}"
    )


def load_state_columns(schema_path: Path) -> list[str]:
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    state_columns = schema.get("allowed_state_columns")
    if not isinstance(state_columns, list) or not state_columns:
        raise ValueError(
            f"Schema file {schema_path} must contain a non-empty "
            "'allowed_state_columns' list."
        )
    if not all(isinstance(column, str) for column in state_columns):
        raise ValueError("'allowed_state_columns' must contain strings only.")
    return state_columns


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


def action_index_for_fraction(env: TaxAwareEnv, target_fraction: float) -> int:
    for idx, value in enumerate(env.action_fractions):
        if approx_equal(value, target_fraction):
            return idx
    raise ValueError(
        f"Action fraction {target_fraction} not found in {env.action_fractions}."
    )


def get_policy_action_fraction(
    policy_name: str,
    step_idx: int,
    rng: np.random.Generator,
    env: TaxAwareEnv,
) -> float:
    if policy_name == "hold_to_terminal":
        return 0.0

    if policy_name == "sell_immediately":
        return 1.0 if step_idx == 0 else 0.0

    if policy_name == "sell_half_then_hold":
        return 0.5 if step_idx == 0 else 0.0

    if policy_name == "sell_quarters_over_time":
        return 0.25

    if policy_name == "random_policy":
        return float(rng.choice(env.action_fractions))

    raise ValueError(f"Unknown policy_name: {policy_name}")


def assert_step_reward_identity(reward: float, info: dict[str, Any]) -> None:
    assert info["reward_version"] == "A_after_tax_total_value_change"
    assert_approx("reward info alias", reward, info["reward"])
    assert_approx("reward_A", reward, info["reward_A"])
    assert_approx(
        "reward identity",
        reward,
        info["after_tax_total_value"] - info["previous_after_tax_total_value"],
    )
    assert_approx(
        "after-tax total identity",
        info["after_tax_total_value"],
        info["cum_realized_after_tax_pnl"]
        + info["after_tax_liquidation_value_remaining"],
    )
    assert info["remaining_fraction"] >= -1e-12
    assert info["sold_fraction"] <= 1.0 + 1e-12
    assert np.isfinite(reward)


def run_episode(
    env: TaxAwareEnv,
    episode_id: str,
    policy_name: str,
    profile_name: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, reset_info = env.reset(episode_id=episode_id)
    initial_after_tax_total_value = float(reset_info["after_tax_total_value"])

    step_rows: list[dict[str, Any]] = []
    done = False
    step_idx = 0
    max_steps = len(env._current_episode_df) + 5

    while not done:
        if step_idx >= max_steps:
            raise RuntimeError(
                f"Episode {episode_id} exceeded max_steps={max_steps} "
                f"under policy {policy_name}."
            )

        action_fraction = get_policy_action_fraction(
            policy_name=policy_name,
            step_idx=step_idx,
            rng=rng,
            env=env,
        )
        action_idx = action_index_for_fraction(env, action_fraction)
        _, reward, done, truncated, info = env.step(action_idx)

        assert truncated is False
        assert_step_reward_identity(reward, info)

        step_rows.append(
            {
                "profile_name": profile_name,
                "policy_name": policy_name,
                "episode_id": episode_id,
                "step": step_idx,
                "date": info["date"],
                "reward": float(reward),
                "reward_A": float(info["reward_A"]),
                "previous_after_tax_total_value": float(
                    info["previous_after_tax_total_value"]
                ),
                "after_tax_total_value": float(info["after_tax_total_value"]),
                "realized_after_tax_increment": float(
                    info["realized_after_tax_increment"]
                ),
                "after_tax_liquidation_value_remaining": float(
                    info["after_tax_liquidation_value_remaining"]
                ),
                "cum_realized_after_tax_pnl": float(
                    info["cum_realized_after_tax_pnl"]
                ),
                "full_position_pnl": float(info["full_position_pnl"]),
                "action_fraction_requested": float(
                    info["action_fraction_requested"]
                ),
                "action_fraction_executed": float(
                    info["action_fraction_executed"]
                ),
                "sold_fraction": float(info["sold_fraction"]),
                "remaining_fraction": float(info["remaining_fraction"]),
                "tax_regime": info["tax_regime"],
                "after_tax_liquidation_tax_regime": info[
                    "after_tax_liquidation_tax_regime"
                ],
                "terminal_liquidation_executed": bool(
                    info["terminal_liquidation_executed"]
                ),
                "done": bool(done),
            }
        )
        step_idx += 1

    episode_total_reward = sum(row["reward"] for row in step_rows)
    final_after_tax_total_value = step_rows[-1]["after_tax_total_value"]
    episode_steps = len(step_rows)

    assert_approx(
        "episode reward telescoping identity",
        episode_total_reward,
        final_after_tax_total_value - initial_after_tax_total_value,
        tol=1e-10,
    )

    episode_row = {
        "profile_name": profile_name,
        "policy_name": policy_name,
        "episode_id": episode_id,
        "episode_total_reward": episode_total_reward,
        "episode_initial_after_tax_total_value": initial_after_tax_total_value,
        "episode_final_after_tax_total_value": final_after_tax_total_value,
        "episode_steps": episode_steps,
        "episode_terminal_liquidation_executed": step_rows[-1][
            "terminal_liquidation_executed"
        ],
        "episode_final_remaining_fraction": step_rows[-1]["remaining_fraction"],
        "episode_final_sold_fraction": step_rows[-1]["sold_fraction"],
    }
    return step_rows, episode_row


def summarize_numeric(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    abs_numeric = numeric.abs()
    return {
        "count": float(numeric.count()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std()),
        "min": float(numeric.min()),
        "p01": float(numeric.quantile(0.01)),
        "p05": float(numeric.quantile(0.05)),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.quantile(0.50)),
        "p75": float(numeric.quantile(0.75)),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
        "abs_mean": float(abs_numeric.mean()),
        "abs_p95": float(abs_numeric.quantile(0.95)),
        "abs_p99": float(abs_numeric.quantile(0.99)),
    }


def _summary_table(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    stat_columns = [
        "count",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "p99",
        "max",
        "abs_mean",
        "abs_p95",
        "abs_p99",
    ]
    if df.empty:
        return pd.DataFrame(
            columns=[
                *GROUP_COLUMNS,
                *stat_columns,
            ]
        )

    rows: list[dict[str, Any]] = []
    for group_values, group_df in df.groupby(GROUP_COLUMNS, sort=False):
        profile_name, policy_name = group_values
        rows.append(
            {
                "profile_name": profile_name,
                "policy_name": policy_name,
                **summarize_numeric(group_df[value_column]),
            }
        )
    return pd.DataFrame(rows, columns=[*GROUP_COLUMNS, *stat_columns])


def build_summary_text(
    steps_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    warnings: list[str],
) -> str:
    step_summary = _summary_table(steps_df, "reward")
    episode_summary = _summary_table(episodes_df, "episode_total_reward")

    with pd.option_context(
        "display.max_columns",
        None,
        "display.max_rows",
        None,
        "display.width",
        240,
        "display.float_format",
        "{:.10f}".format,
    ):
        step_summary_text = step_summary.to_string(index=False)
        episode_summary_text = episode_summary.to_string(index=False)

    warning_lines = (
        [f"- {warning}" for warning in warnings]
        if warnings
        else ["No stability warnings triggered."]
    )

    lines = [
        "Step-level reward summary by profile and policy",
        step_summary_text,
        "",
        "Episode-level total reward summary by profile and policy",
        episode_summary_text,
        "",
        "Warnings",
        *warning_lines,
        "",
        "Saved outputs to reports/reward_scale/",
    ]
    return "\n".join(lines)


def _build_stability_warnings(
    steps_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
) -> tuple[list[str], bool]:
    warnings: list[str] = []

    step_rewards = pd.to_numeric(steps_df["reward"], errors="coerce")
    episode_rewards = pd.to_numeric(
        episodes_df["episode_total_reward"],
        errors="coerce",
    )
    step_finite = np.isfinite(step_rewards.to_numpy(dtype=float))
    episode_finite = np.isfinite(episode_rewards.to_numpy(dtype=float))

    has_fatal_warning = False
    if not step_finite.all():
        warnings.append("Any step reward is NaN or infinite.")
        has_fatal_warning = True
    if not episode_finite.all():
        warnings.append("Any episode total reward is NaN or infinite.")
        has_fatal_warning = True

    finite_step_rewards = step_rewards[step_finite]
    finite_episode_rewards = episode_rewards[episode_finite]

    if not finite_step_rewards.empty and finite_step_rewards.abs().max() > 5.0:
        warnings.append("max(abs(step reward)) > 5.0")
    if (
        not finite_episode_rewards.empty
        and finite_episode_rewards.abs().max() > 5.0
    ):
        warnings.append("max(abs(episode_total_reward)) > 5.0")
    if not finite_step_rewards.empty and (finite_step_rewards.abs() > 1.0).mean() > 0.01:
        warnings.append("More than 1% of step rewards have abs(reward) > 1.0")
    if (
        not finite_episode_rewards.empty
        and (finite_episode_rewards.abs() > 1.0).mean() > 0.01
    ):
        warnings.append(
            "More than 1% of episode total rewards have "
            "abs(episode_total_reward) > 1.0"
        )

    return warnings, has_fatal_warning


def main() -> None:
    print("Reward Scale Stability Check")

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema path not found: {SCHEMA_PATH}")
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet path not found: {PARQUET_PATH}")

    state_columns = load_state_columns(SCHEMA_PATH)
    rng = np.random.default_rng(42)

    all_step_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    loaded_episode_count: int | None = None

    profile_names = ", ".join(profile["profile_name"] for profile in TAX_PROFILES)
    policy_names = ", ".join(POLICY_NAMES)

    for profile in TAX_PROFILES:
        env = make_env(PARQUET_PATH, state_columns, profile)
        env._load_episode_index()
        episode_ids = sorted(env._episode_index.keys())[:MAX_EPISODES]

        if loaded_episode_count is None:
            loaded_episode_count = len(episode_ids)
            print(f"Loaded {loaded_episode_count} episodes")
            print(f"Running profiles: {profile_names}")
            print(f"Running policies: {policy_names}")
            print()

        profile_name = str(profile["profile_name"])
        for policy_name in POLICY_NAMES:
            for episode_id in episode_ids:
                step_rows, episode_row = run_episode(
                    env=env,
                    episode_id=episode_id,
                    policy_name=policy_name,
                    profile_name=profile_name,
                    rng=rng,
                )
                all_step_rows.extend(step_rows)
                all_episode_rows.append(episode_row)

    steps_df = pd.DataFrame(all_step_rows)
    episodes_df = pd.DataFrame(all_episode_rows)
    warnings, has_fatal_warning = _build_stability_warnings(steps_df, episodes_df)
    summary_text = build_summary_text(steps_df, episodes_df, warnings)

    print(summary_text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    steps_df.to_csv(OUTPUT_DIR / "reward_scale_steps.csv", index=False)
    episodes_df.to_csv(OUTPUT_DIR / "reward_scale_episodes.csv", index=False)

    full_console_summary_text = "\n".join(
        [
            "Reward Scale Stability Check",
            f"Loaded {loaded_episode_count} episodes",
            f"Running profiles: {profile_names}",
            f"Running policies: {policy_names}",
            "",
            summary_text,
            "REWARD SCALE CHECK COMPLETE",
        ]
    )
    (OUTPUT_DIR / "reward_scale_summary.txt").write_text(
        full_console_summary_text + "\n",
        encoding="utf-8",
    )

    if has_fatal_warning:
        raise AssertionError("Non-finite rewards found.")

    print("REWARD SCALE CHECK COMPLETE")


if __name__ == "__main__":
    main()
