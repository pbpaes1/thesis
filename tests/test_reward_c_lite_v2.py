"""Synthetic tests for Reward C-lite v2 transaction and cooldown penalties."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest

import numpy as np
import pandas as pd
import yaml

from src.environment.tax_aware_env import (
    REWARD_A_VERSION,
    REWARD_C_LITE_VERSION,
    REWARD_C_LITE_V2_VERSION,
    TaxAwareEnv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAMBDA_TRANSACTION = 0.0010
LAMBDA_COOLDOWN = 0.0025
COOLDOWN_DAYS = 20

TAX_FREE_PROFILE: dict[str, Any] = {
    "profile_name": "tax_free",
    "short_term_rate": 0.0,
    "long_term_rate": 0.0,
    "niit_rate": 0.0,
    "apply_niit": False,
}


class TestRewardCLiteV2(unittest.TestCase):
    def _make_env(
        self,
        *,
        dates: list[str],
        gains: list[float] | None = None,
        reward_version: str = REWARD_C_LITE_V2_VERSION,
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
            "base_reward_version": REWARD_A_VERSION,
            "use_drawdown_penalty": False,
            "use_transaction_penalty": reward_version == REWARD_C_LITE_V2_VERSION,
            "use_cooldown_penalty": reward_version
            in {REWARD_C_LITE_VERSION, REWARD_C_LITE_V2_VERSION},
            "use_explicit_tax_saving_bonus": False,
            "use_reward_clipping": False,
            "use_reward_normalization": False,
        }
        if reward_version == REWARD_C_LITE_V2_VERSION:
            reward_config["transaction_penalty"] = {
                "enabled": True,
                "lambda_transaction": LAMBDA_TRANSACTION,
                "scale_by_executed_fraction": True,
                "apply_to_first_sale": True,
                "apply_to_automatic_terminal_liquidation": False,
            }
        if reward_version in {REWARD_C_LITE_VERSION, REWARD_C_LITE_V2_VERSION}:
            reward_config["cooldown_penalty"] = {
                "enabled": True,
                "lambda_cooldown": LAMBDA_COOLDOWN,
                "cooldown_days": COOLDOWN_DAYS,
                "scale_by_executed_fraction": True,
                "penalize_first_sale": False,
                "apply_to_automatic_terminal_liquidation": False,
            }

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

    def _step_fraction(
        self,
        env: TaxAwareEnv,
        fraction: float,
    ) -> tuple[float, dict[str, Any]]:
        _, reward, _, _, info = env.step(self._action_index_for_fraction(env, fraction))
        return reward, info

    def _cooldown_penalty(self, executed_fraction: float, days_since: int) -> float:
        return (
            LAMBDA_COOLDOWN
            * executed_fraction
            * max(0, COOLDOWN_DAYS - days_since)
            / COOLDOWN_DAYS
        )

    def _transaction_penalty(self, executed_fraction: float) -> float:
        return LAMBDA_TRANSACTION * executed_fraction

    def test_reward_a_backward_compatibility(self) -> None:
        env = self._make_env(
            dates=["2020-01-01", "2020-01-06", "2020-01-07"],
            reward_version=REWARD_A_VERSION,
        )
        env.reset()
        reward, info = self._step_fraction(env, 0.50)

        self.assertEqual(info["reward_version"], REWARD_A_VERSION)
        self.assertAlmostEqual(reward, info["reward_A"], places=12)
        self.assertAlmostEqual(info["transaction_penalty"], 0.0, places=12)
        self.assertAlmostEqual(info["cooldown_penalty"], 0.0, places=12)

    def test_c_lite_v1_backward_compatibility(self) -> None:
        env = self._make_env(
            dates=["2020-01-01", "2020-01-06", "2020-01-07"],
            reward_version=REWARD_C_LITE_VERSION,
        )
        env.reset()
        self._step_fraction(env, 0.25)
        reward, info = self._step_fraction(env, 0.50)

        self.assertEqual(info["reward_version"], REWARD_C_LITE_VERSION)
        self.assertAlmostEqual(info["transaction_penalty"], 0.0, places=12)
        self.assertGreater(info["cooldown_penalty"], 0.0)
        self.assertAlmostEqual(
            reward,
            info["reward_A"] - info["cooldown_penalty"],
            places=12,
        )

    def test_first_discretionary_sale_has_transaction_but_no_cooldown(self) -> None:
        env = self._make_env(dates=["2020-01-01", "2020-01-06", "2020-01-07"])
        env.reset()
        reward, info = self._step_fraction(env, 0.50)

        expected_transaction = self._transaction_penalty(0.50)
        self.assertEqual(info["reward_version"], REWARD_C_LITE_V2_VERSION)
        self.assertFalse(info["previous_sale_exists"])
        self.assertAlmostEqual(info["transaction_penalty"], expected_transaction)
        self.assertTrue(info["transaction_penalty_applied"])
        self.assertAlmostEqual(info["cooldown_penalty"], 0.0, places=12)
        self.assertFalse(info["cooldown_penalty_applied"])
        self.assertAlmostEqual(
            info["reward_C_lite_v2"],
            info["reward_A"] - expected_transaction,
            places=12,
        )
        self.assertAlmostEqual(reward, info["reward_C_lite_v2"], places=12)

    def test_repeated_sale_inside_cooldown_has_both_penalties(self) -> None:
        env = self._make_env(dates=["2020-01-01", "2020-01-06", "2020-01-07"])
        env.reset()
        self._step_fraction(env, 0.25)
        reward, info = self._step_fraction(env, 0.50)

        transaction = self._transaction_penalty(0.50)
        cooldown = self._cooldown_penalty(0.50, 5)
        self.assertEqual(info["days_since_last_sale"], 5)
        self.assertAlmostEqual(info["transaction_penalty"], transaction, places=12)
        self.assertAlmostEqual(info["cooldown_penalty"], cooldown, places=12)
        self.assertTrue(info["transaction_penalty_applied"])
        self.assertTrue(info["cooldown_penalty_applied"])
        self.assertAlmostEqual(
            info["reward_C_lite_v2"],
            info["reward_A"] - transaction - cooldown,
            places=12,
        )
        self.assertAlmostEqual(reward, info["reward_C_lite_v2"], places=12)

    def test_repeated_sale_outside_cooldown_has_transaction_only(self) -> None:
        env = self._make_env(dates=["2020-01-01", "2020-02-01", "2020-02-03"])
        env.reset()
        self._step_fraction(env, 0.25)
        reward, info = self._step_fraction(env, 0.50)

        transaction = self._transaction_penalty(0.50)
        self.assertEqual(info["days_since_last_sale"], 31)
        self.assertAlmostEqual(info["transaction_penalty"], transaction, places=12)
        self.assertAlmostEqual(info["cooldown_penalty"], 0.0, places=12)
        self.assertAlmostEqual(reward, info["reward_A"] - transaction, places=12)

    def test_hold_does_not_change_sale_state_or_penalize(self) -> None:
        env = self._make_env(
            dates=["2020-01-01", "2020-01-06", "2020-01-10", "2020-01-13"]
        )
        env.reset()
        _, first_info = self._step_fraction(env, 0.25)
        reward, hold_info = self._step_fraction(env, 0.0)

        self.assertAlmostEqual(hold_info["action_fraction_executed"], 0.0, places=12)
        self.assertAlmostEqual(hold_info["transaction_penalty"], 0.0, places=12)
        self.assertAlmostEqual(hold_info["cooldown_penalty"], 0.0, places=12)
        self.assertEqual(hold_info["sale_count"], first_info["sale_count"])
        self.assertEqual(
            hold_info["last_sale_date_after_step"],
            first_info["last_sale_date_after_step"],
        )
        self.assertAlmostEqual(reward, hold_info["reward_A"], places=12)

    def test_oversell_clamp_uses_executed_fraction_for_both_penalties(self) -> None:
        env = self._make_env(dates=["2020-01-01", "2020-01-06", "2020-01-07"])
        env.reset()
        self._step_fraction(env, 0.75)
        _, info = self._step_fraction(env, 0.50)

        expected_transaction = self._transaction_penalty(0.25)
        expected_cooldown = self._cooldown_penalty(0.25, 5)
        self.assertAlmostEqual(info["action_fraction_requested"], 0.50, places=12)
        self.assertAlmostEqual(info["action_fraction_executed"], 0.25, places=12)
        self.assertAlmostEqual(
            info["transaction_penalty"],
            expected_transaction,
            places=12,
        )
        self.assertAlmostEqual(info["cooldown_penalty"], expected_cooldown, places=12)

    def test_automatic_terminal_liquidation_is_not_penalized(self) -> None:
        env = self._make_env(dates=["2020-01-01", "2020-01-06", "2020-01-07"])
        env.reset()
        _, first_info = self._step_fraction(env, 0.25)
        reward, terminal_info = self._step_fraction(env, 0.0)

        self.assertTrue(terminal_info["terminal_liquidation_executed"])
        self.assertTrue(terminal_info["is_automatic_terminal_liquidation"])
        self.assertAlmostEqual(
            terminal_info["action_fraction_executed"],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(terminal_info["transaction_penalty"], 0.0, places=12)
        self.assertAlmostEqual(terminal_info["cooldown_penalty"], 0.0, places=12)
        self.assertEqual(terminal_info["sale_count"], first_info["sale_count"])
        self.assertAlmostEqual(reward, terminal_info["reward_A"], places=12)

    def test_v2_gamma_config_is_one(self) -> None:
        with (PROJECT_ROOT / "configs" / "train_reward_c_lite_v2.yaml").open(
            "r",
            encoding="utf-8",
        ) as handle:
            config = yaml.safe_load(handle)
        self.assertEqual(config["training"]["discount_factor_gamma"], 1.0)

    def test_v2_exploration_probabilities(self) -> None:
        with (PROJECT_ROOT / "configs" / "train_reward_c_lite_v2.yaml").open(
            "r",
            encoding="utf-8",
        ) as handle:
            config = yaml.safe_load(handle)
        probabilities = config["exploration"]["action_probabilities"]
        self.assertEqual(probabilities, [0.65, 0.20, 0.10, 0.04, 0.01])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
        self.assertEqual(
            len(probabilities),
            len(config["action_space"]["action_fractions"]),
        )


if __name__ == "__main__":
    unittest.main()
