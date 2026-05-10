"""Synthetic tests for Reward C-lite cooldown penalty behavior."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest

import numpy as np
import pandas as pd

from src.environment.tax_aware_env import (
    REWARD_A_VERSION,
    REWARD_C_LITE_VERSION,
    TaxAwareEnv,
)


TAX_FREE_PROFILE: dict[str, Any] = {
    "profile_name": "tax_free",
    "short_term_rate": 0.0,
    "long_term_rate": 0.0,
    "niit_rate": 0.0,
    "apply_niit": False,
}

LAMBDA_COOLDOWN = 0.0025
COOLDOWN_DAYS = 20


class TestRewardCLite(unittest.TestCase):
    def _make_env(
        self,
        *,
        dates: list[str],
        gains: list[float] | None = None,
        reward_version: str = REWARD_C_LITE_VERSION,
    ) -> TaxAwareEnv:
        if gains is None:
            gains = [0.20] * len(dates)
        if len(dates) != len(gains):
            self.fail("dates and gains must have the same length.")

        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
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

        reward_config: dict[str, Any] = {
            "version": reward_version,
            "use_environment_reward": True,
            "assert_reward_version": True,
            "expected_info_reward_version": reward_version,
            "use_drawdown_penalty": False,
            "use_cooldown_penalty": reward_version == REWARD_C_LITE_VERSION,
            "use_explicit_tax_saving_bonus": False,
            "use_reward_clipping": False,
            "use_reward_normalization": False,
        }
        if reward_version == REWARD_C_LITE_VERSION:
            reward_config.update(
                {
                    "base_reward_version": REWARD_A_VERSION,
                    "cooldown_penalty": {
                        "enabled": True,
                        "lambda_cooldown": LAMBDA_COOLDOWN,
                        "cooldown_days": COOLDOWN_DAYS,
                        "scale_by_executed_fraction": True,
                        "penalize_first_sale": False,
                    },
                }
            )

        return TaxAwareEnv(
            parquet_path=parquet_path,
            state_columns=["feature"],
            tax_config=TAX_FREE_PROFILE,
            reward_config=reward_config,
        )

    def _action_index_for_fraction(self, env: TaxAwareEnv, target: float) -> int:
        for idx, value in enumerate(env.action_fractions):
            if np.isclose(value, target, atol=1e-12):
                return idx
        self.fail(f"Action fraction {target} not found in action_fractions.")
        return -1

    def _step_fraction(self, env: TaxAwareEnv, fraction: float) -> tuple[float, dict[str, Any]]:
        _, reward, _, _, info = env.step(self._action_index_for_fraction(env, fraction))
        return reward, info

    def _expected_penalty(self, executed_fraction: float, days_since: int) -> float:
        return (
            LAMBDA_COOLDOWN
            * executed_fraction
            * max(0, COOLDOWN_DAYS - days_since)
            / COOLDOWN_DAYS
        )

    def test_reward_a_backward_compatibility(self) -> None:
        env = self._make_env(
            dates=["2020-01-01", "2020-01-06", "2020-01-07"],
            reward_version=REWARD_A_VERSION,
        )
        env.reset()
        reward, info = self._step_fraction(env, 0.50)

        self.assertEqual(info["reward_version"], REWARD_A_VERSION)
        self.assertAlmostEqual(reward, info["reward_A"], places=12)
        self.assertAlmostEqual(info["reward"], info["reward_A"], places=12)
        self.assertAlmostEqual(info["reward_C_lite"], info["reward_A"], places=12)
        self.assertAlmostEqual(info["cooldown_penalty"], 0.0, places=12)

    def test_first_sale_has_no_cooldown_penalty(self) -> None:
        env = self._make_env(dates=["2020-01-01", "2020-01-06", "2020-01-07"])
        env.reset()
        reward, info = self._step_fraction(env, 0.50)

        self.assertEqual(info["reward_version"], REWARD_C_LITE_VERSION)
        self.assertFalse(info["previous_sale_exists"])
        self.assertIsNone(info["days_since_last_sale"])
        self.assertAlmostEqual(info["cooldown_penalty"], 0.0, places=12)
        self.assertFalse(info["cooldown_penalty_applied"])
        self.assertAlmostEqual(reward, info["reward_A"], places=12)
        self.assertAlmostEqual(info["reward_C_lite"], info["reward_A"], places=12)

    def test_second_sale_inside_cooldown_is_penalized(self) -> None:
        env = self._make_env(dates=["2020-01-01", "2020-01-06", "2020-01-07"])
        env.reset()
        self._step_fraction(env, 0.25)
        reward, info = self._step_fraction(env, 0.50)

        expected_penalty = self._expected_penalty(0.50, 5)
        self.assertTrue(info["previous_sale_exists"])
        self.assertEqual(info["days_since_last_sale"], 5)
        self.assertTrue(info["cooldown_penalty_applied"])
        self.assertAlmostEqual(info["cooldown_penalty"], expected_penalty, places=12)
        self.assertAlmostEqual(
            info["reward_C_lite"],
            info["reward_A"] - expected_penalty,
            places=12,
        )
        self.assertAlmostEqual(reward, info["reward_C_lite"], places=12)

    def test_second_sale_outside_cooldown_is_not_penalized(self) -> None:
        env = self._make_env(dates=["2020-01-01", "2020-02-01", "2020-02-03"])
        env.reset()
        self._step_fraction(env, 0.25)
        reward, info = self._step_fraction(env, 0.50)

        self.assertEqual(info["days_since_last_sale"], 31)
        self.assertAlmostEqual(info["cooldown_penalty"], 0.0, places=12)
        self.assertFalse(info["cooldown_penalty_applied"])
        self.assertAlmostEqual(reward, info["reward_A"], places=12)

    def test_hold_does_not_change_sale_state_or_penalize(self) -> None:
        env = self._make_env(
            dates=["2020-01-01", "2020-01-06", "2020-01-10", "2020-01-13"]
        )
        env.reset()
        _, first_info = self._step_fraction(env, 0.25)
        reward, hold_info = self._step_fraction(env, 0.0)

        self.assertAlmostEqual(hold_info["action_fraction_executed"], 0.0, places=12)
        self.assertAlmostEqual(hold_info["cooldown_penalty"], 0.0, places=12)
        self.assertEqual(hold_info["sale_count"], first_info["sale_count"])
        self.assertEqual(
            hold_info["last_sale_date_after_step"],
            first_info["last_sale_date_after_step"],
        )
        self.assertAlmostEqual(reward, hold_info["reward_A"], places=12)

    def test_oversell_clamp_penalty_uses_executed_fraction(self) -> None:
        env = self._make_env(dates=["2020-01-01", "2020-01-06", "2020-01-07"])
        env.reset()
        self._step_fraction(env, 0.75)
        _, info = self._step_fraction(env, 0.50)

        expected_penalty = self._expected_penalty(0.25, 5)
        self.assertAlmostEqual(info["action_fraction_requested"], 0.50, places=12)
        self.assertAlmostEqual(info["action_fraction_executed"], 0.25, places=12)
        self.assertAlmostEqual(info["cooldown_penalty"], expected_penalty, places=12)

    def test_automatic_terminal_liquidation_is_not_penalized(self) -> None:
        env = self._make_env(dates=["2020-01-01", "2020-01-06", "2020-01-07"])
        env.reset()
        _, first_info = self._step_fraction(env, 0.25)
        reward, terminal_info = self._step_fraction(env, 0.0)

        self.assertTrue(terminal_info["terminal_liquidation_executed"])
        self.assertAlmostEqual(terminal_info["action_fraction_executed"], 0.0, places=12)
        self.assertAlmostEqual(terminal_info["cooldown_penalty"], 0.0, places=12)
        self.assertEqual(terminal_info["sale_count"], first_info["sale_count"])
        self.assertEqual(
            terminal_info["last_sale_date_after_step"],
            first_info["last_sale_date_after_step"],
        )
        self.assertAlmostEqual(reward, terminal_info["reward_A"], places=12)


if __name__ == "__main__":
    unittest.main()
