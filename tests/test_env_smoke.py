"""Smoke tests for the TaxAwareEnv Step 4 accounting contract."""

from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.environment.tax_aware_env import TaxAwareEnv


TAX_FREE_PROFILE: dict[str, Any] = {
    "profile_name": "tax_free",
    "short_term_rate": 0.0,
    "long_term_rate": 0.0,
    "niit_rate": 0.0,
    "apply_niit": False,
}

STANDARD_PROFILE: dict[str, Any] = {
    "profile_name": "mass_affluent_individual",
    "short_term_rate": 0.24,
    "long_term_rate": 0.15,
    "niit_rate": 0.0,
    "apply_niit": False,
}

HIGH_INCOME_PROFILE: dict[str, Any] = {
    "profile_name": "high_income_individual",
    "short_term_rate": 0.35,
    "long_term_rate": 0.15,
    "niit_rate": 0.038,
    "apply_niit": True,
}


class TestTaxAwareEnvSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.schema_path = cls.project_root / "data" / "freeze" / "v1" / "allowed_state_columns_v1.json"
        cls.parquet_path = cls.project_root / "data" / "episodes" / "drl_episodes.parquet"

        if not cls.schema_path.exists():
            raise unittest.SkipTest(
                f"Missing frozen schema JSON: {cls.schema_path}"
            )
        if not cls.parquet_path.exists():
            raise unittest.SkipTest(
                f"Missing episode parquet dataset: {cls.parquet_path}"
            )

        with cls.schema_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        cls.state_columns = payload["allowed_state_columns"]

    def _make_env(self, tax_profile: dict[str, Any]) -> TaxAwareEnv:
        return TaxAwareEnv(
            parquet_path=self.parquet_path,
            state_columns=self.state_columns,
            tax_config=tax_profile,
        )

    def setUp(self) -> None:
        self.env = self._make_env(TAX_FREE_PROFILE)

    def _action_index_for_fraction(self, env: TaxAwareEnv, target: float) -> int:
        for idx, value in enumerate(env.action_fractions):
            if np.isclose(value, target, atol=1e-12):
                return idx
        self.fail(f"Action fraction {target} not found in action_fractions.")
        return -1

    def _find_episode_row(
        self,
        env: TaxAwareEnv,
        *,
        short_term: bool | None,
        positive_gain: bool | None,
        avoid_terminal_step: bool = False,
    ) -> tuple[str, int, pd.Series]:
        env._load_episode_index()
        if env._df is None:
            self.fail("Environment dataframe did not load.")

        for episode_id, row_idx in env._episode_index.items():
            ep = env._df.iloc[row_idx].reset_index(drop=True)
            if ep.empty:
                continue

            date_series = pd.to_datetime(ep["date"], errors="coerce")
            transition_series = pd.to_datetime(ep["tax_transition_date"], errors="coerce")
            mask = pd.Series(True, index=ep.index)

            if short_term is True:
                mask &= date_series < transition_series
            elif short_term is False:
                mask &= date_series >= transition_series

            if positive_gain is True:
                mask &= ep["unrealized_gains_pct"] > 0.0
            elif positive_gain is False:
                mask &= ep["unrealized_gains_pct"] <= 0.0

            # Final rows are not directly actionable under current traversal
            # behavior because reaching them also returns done=True.
            if len(ep) > 1:
                mask &= ep.index < (len(ep) - 1)
            else:
                continue
            if avoid_terminal_step:
                # Some tests assert cumulative realized accounting for the
                # explicit action only, so they need a non-terminal sale step.
                if len(ep) > 2:
                    mask &= ep.index < (len(ep) - 2)
                else:
                    continue

            positions = np.flatnonzero(mask.to_numpy())
            if positions.size > 0:
                row_pos = int(positions[0])
                return episode_id, row_pos, ep.iloc[row_pos]

        self.skipTest(
            "No suitable episode/row found for requested selection criteria."
        )
        raise AssertionError("unreachable")

    def _run_sale_at_row(
        self,
        env: TaxAwareEnv,
        *,
        episode_id: str,
        row_pos: int,
        sell_fraction: float,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        env.reset(episode_id=episode_id)
        hold_idx = self._action_index_for_fraction(env, 0.0)
        for _ in range(row_pos):
            _, _, done, _, _ = env.step(hold_idx)
            if done:
                self.fail(
                    f"Episode '{episode_id}' terminated before reaching row {row_pos}."
                )

        sell_idx = self._action_index_for_fraction(env, sell_fraction)
        return env.step(sell_idx)

    def _assert_increment_formula(self, info: dict[str, Any]) -> None:
        expected_pre_tax = (
            info["action_fraction_executed"] * info["full_position_pnl"]
        )
        self.assertAlmostEqual(
            info["realized_pre_tax_increment"], expected_pre_tax, places=12
        )

        if info["realized_pre_tax_increment"] > 0.0:
            expected_tax_paid = (
                info["realized_pre_tax_increment"] * info["applicable_tax_rate"]
            )
        else:
            expected_tax_paid = 0.0
        self.assertAlmostEqual(info["tax_paid"], expected_tax_paid, places=12)

        self.assertAlmostEqual(
            info["realized_after_tax_increment"],
            info["realized_pre_tax_increment"] - info["tax_paid"],
            places=12,
        )

    def _assert_reward_a_formula(self, reward: float, info: dict[str, Any]) -> None:
        self.assertEqual(
            info["reward_version"],
            "A_after_tax_total_value_change",
        )
        self.assertAlmostEqual(info["reward"], reward, places=12)
        self.assertAlmostEqual(info["reward_A"], reward, places=12)
        self.assertAlmostEqual(info["reward_C_lite"], reward, places=12)
        self.assertAlmostEqual(info["cooldown_penalty"], 0.0, places=12)
        self.assertAlmostEqual(
            reward,
            info["after_tax_total_value"]
            - info["previous_after_tax_total_value"],
            places=12,
        )
        self.assertAlmostEqual(
            info["after_tax_total_value"],
            info["cum_realized_after_tax_pnl"]
            + info["after_tax_liquidation_value_remaining"],
            places=12,
        )

        remaining_pre_tax_value = info[
            "after_tax_liquidation_pre_tax_value_remaining"
        ]
        if remaining_pre_tax_value > 0.0:
            expected_tax_drag = (
                remaining_pre_tax_value * info["after_tax_liquidation_tax_rate"]
            )
        else:
            expected_tax_drag = 0.0
        self.assertAlmostEqual(
            info["after_tax_liquidation_tax_drag_remaining"],
            expected_tax_drag,
            places=12,
        )
        self.assertAlmostEqual(
            info["after_tax_liquidation_value_remaining"],
            remaining_pre_tax_value - expected_tax_drag,
            places=12,
        )

    def test_reset_and_step_smoke(self) -> None:
        obs, reset_info = self.env.reset()

        self.assertEqual(obs.shape, (len(self.state_columns),))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(self.env._current_row_ptr, 0)
        self.assertIsNotNone(self.env._current_episode_id)
        self.assertIsNotNone(self.env._current_episode_df)
        self.assertEqual(reset_info["episode_id"], self.env._current_episode_id)
        self.assertEqual(reset_info["current_row_ptr"], 0)
        self.assertIn("date", reset_info)
        self.assertEqual(reset_info["sold_fraction"], 0.0)
        self.assertEqual(reset_info["remaining_fraction"], 1.0)
        self.assertEqual(reset_info["cum_realized_pre_tax_pnl"], 0.0)
        self.assertEqual(reset_info["cum_realized_after_tax_pnl"], 0.0)
        self.assertIn("tax_profile_name", reset_info)
        self.assertEqual(
            reset_info["after_tax_total_value"],
            reset_info["previous_after_tax_total_value"],
        )
        self.assertAlmostEqual(
            reset_info["after_tax_total_value"],
            reset_info["cum_realized_after_tax_pnl"]
            + reset_info["after_tax_liquidation_value_remaining"],
            places=12,
        )

        episode_len = len(self.env._current_episode_df)
        self.assertGreaterEqual(episode_len, 1)

        next_obs, reward, done, truncated, info = self.env.step(0)

        self.assertEqual(next_obs.shape, (len(self.state_columns),))
        self.assertEqual(next_obs.dtype, np.float32)
        self._assert_reward_a_formula(reward, info)
        self.assertFalse(truncated)

        expected_ptr = 1 if episode_len > 1 else 0
        expected_done = episode_len == 1
        self.assertEqual(self.env._current_row_ptr, expected_ptr)
        self.assertEqual(done, expected_done)

        self.assertEqual(info["episode_id"], self.env._current_episode_id)
        self.assertEqual(info["current_row_ptr"], self.env._current_row_ptr)
        self.assertIn("date", info)
        self.assertEqual(info["action"], 0)
        self.assertEqual(info["remaining_fraction"], 1.0)

    def test_action_bookkeeping_hold(self) -> None:
        self.env.reset()
        hold_idx = self._action_index_for_fraction(self.env, 0.0)
        _, reward, _, truncated, info = self.env.step(hold_idx)

        self._assert_reward_a_formula(reward, info)
        self.assertFalse(truncated)
        self.assertAlmostEqual(info["action_fraction_requested"], 0.0, places=12)
        self.assertAlmostEqual(info["action_fraction_executed"], 0.0, places=12)
        self.assertAlmostEqual(info["sold_fraction"], 0.0, places=12)
        self.assertAlmostEqual(info["remaining_fraction"], 1.0, places=12)
        self.assertAlmostEqual(self.env._sold_fraction, 0.0, places=12)
        self.assertAlmostEqual(self.env._remaining_fraction, 1.0, places=12)

    def test_action_bookkeeping_sell_50pct(self) -> None:
        self.env.reset()
        sell_50_idx = self._action_index_for_fraction(self.env, 0.5)
        _, reward, _, truncated, info = self.env.step(sell_50_idx)

        self._assert_reward_a_formula(reward, info)
        self.assertFalse(truncated)
        self.assertAlmostEqual(info["action_fraction_requested"], 0.5, places=12)
        self.assertAlmostEqual(info["action_fraction_executed"], 0.5, places=12)
        self.assertAlmostEqual(info["sold_fraction"], 0.5, places=12)
        self.assertAlmostEqual(info["remaining_fraction"], 0.5, places=12)
        self.assertAlmostEqual(self.env._sold_fraction, 0.5, places=12)
        self.assertAlmostEqual(self.env._remaining_fraction, 0.5, places=12)
        self._assert_increment_formula(info)

    def test_action_bookkeeping_oversell_prevention(self) -> None:
        self.env.reset()
        sell_75_idx = self._action_index_for_fraction(self.env, 0.75)
        sell_50_idx = self._action_index_for_fraction(self.env, 0.5)

        _, _, _, _, info_first = self.env.step(sell_75_idx)
        _, reward_second, done_second, truncated_second, info_second = self.env.step(sell_50_idx)

        self._assert_reward_a_formula(reward_second, info_second)
        self.assertFalse(truncated_second)
        self.assertAlmostEqual(info_first["action_fraction_requested"], 0.75, places=12)
        self.assertAlmostEqual(info_first["action_fraction_executed"], 0.75, places=12)
        self.assertAlmostEqual(info_second["action_fraction_requested"], 0.5, places=12)
        self.assertAlmostEqual(info_second["action_fraction_executed"], 0.25, places=12)
        self.assertAlmostEqual(info_second["sold_fraction"], 1.0, places=12)
        self.assertAlmostEqual(info_second["remaining_fraction"], 0.0, places=12)
        self.assertAlmostEqual(self.env._sold_fraction, 1.0, places=12)
        self.assertAlmostEqual(self.env._remaining_fraction, 0.0, places=12)

        # Current terminal condition also marks done on full liquidation.
        self.assertTrue(done_second)

    def test_tax_accounting_short_term_positive_standard_profile(self) -> None:
        env = self._make_env(STANDARD_PROFILE)
        episode_id, row_pos, row = self._find_episode_row(
            env,
            short_term=True,
            positive_gain=True,
            avoid_terminal_step=True,
        )
        _, reward, _, truncated, info = self._run_sale_at_row(
            env, episode_id=episode_id, row_pos=row_pos, sell_fraction=0.5
        )

        self._assert_reward_a_formula(reward, info)
        self.assertFalse(truncated)
        self.assertEqual(info["tax_profile_name"], "mass_affluent_individual")
        self.assertEqual(info["tax_regime"], "short_term")
        self.assertAlmostEqual(info["applicable_tax_rate"], 0.24, places=12)
        self.assertAlmostEqual(info["full_position_pnl"], float(row["unrealized_gains_pct"]), places=12)
        self.assertAlmostEqual(info["realized_pre_tax_increment"], 0.5 * info["full_position_pnl"], places=12)
        self.assertAlmostEqual(
            info["tax_paid"],
            info["realized_pre_tax_increment"] * 0.24,
            places=12,
        )
        self.assertAlmostEqual(
            info["realized_after_tax_increment"],
            info["realized_pre_tax_increment"] - info["tax_paid"],
            places=12,
        )
        self.assertAlmostEqual(
            info["cum_realized_pre_tax_pnl"],
            info["realized_pre_tax_increment"],
            places=12,
        )
        self.assertAlmostEqual(
            info["cum_realized_after_tax_pnl"],
            info["realized_after_tax_increment"],
            places=12,
        )

    def test_tax_accounting_long_term_positive_standard_profile(self) -> None:
        env = self._make_env(STANDARD_PROFILE)
        episode_id, row_pos, _ = self._find_episode_row(
            env, short_term=False, positive_gain=True
        )
        _, _, _, truncated, info = self._run_sale_at_row(
            env, episode_id=episode_id, row_pos=row_pos, sell_fraction=0.5
        )

        self.assertFalse(truncated)
        self.assertEqual(info["tax_regime"], "long_term")
        self.assertAlmostEqual(info["applicable_tax_rate"], 0.15, places=12)
        self._assert_increment_formula(info)

    def test_tax_accounting_high_income_profile_with_niit(self) -> None:
        env = self._make_env(HIGH_INCOME_PROFILE)
        episode_id, row_pos, _ = self._find_episode_row(
            env, short_term=False, positive_gain=True
        )
        _, _, _, truncated, info = self._run_sale_at_row(
            env, episode_id=episode_id, row_pos=row_pos, sell_fraction=0.5
        )

        self.assertFalse(truncated)
        self.assertEqual(info["tax_profile_name"], "high_income_individual")
        self.assertEqual(info["tax_regime"], "long_term")
        self.assertAlmostEqual(info["applicable_tax_rate"], 0.188, places=12)
        self._assert_increment_formula(info)

    def test_tax_accounting_tax_free_profile(self) -> None:
        env = self._make_env(TAX_FREE_PROFILE)
        episode_id, row_pos, _ = self._find_episode_row(
            env, short_term=None, positive_gain=True
        )
        _, _, _, truncated, info = self._run_sale_at_row(
            env, episode_id=episode_id, row_pos=row_pos, sell_fraction=0.5
        )

        self.assertFalse(truncated)
        self.assertAlmostEqual(info["applicable_tax_rate"], 0.0, places=12)
        self.assertAlmostEqual(info["tax_paid"], 0.0, places=12)
        self.assertAlmostEqual(
            info["realized_after_tax_increment"],
            info["realized_pre_tax_increment"],
            places=12,
        )
        self._assert_increment_formula(info)

    def test_tax_accounting_non_positive_gain_has_zero_tax(self) -> None:
        env = self._make_env(STANDARD_PROFILE)
        episode_id, row_pos, _ = self._find_episode_row(
            env, short_term=None, positive_gain=False
        )
        _, _, _, truncated, info = self._run_sale_at_row(
            env, episode_id=episode_id, row_pos=row_pos, sell_fraction=0.5
        )

        self.assertFalse(truncated)
        self.assertLessEqual(info["realized_pre_tax_increment"], 0.0)
        self.assertAlmostEqual(info["tax_paid"], 0.0, places=12)
        self.assertAlmostEqual(
            info["realized_after_tax_increment"],
            info["realized_pre_tax_increment"],
            places=12,
        )
        self._assert_increment_formula(info)

    def test_oversell_with_tax_accounting_cumulative_consistency(self) -> None:
        env = self._make_env(STANDARD_PROFILE)
        env.reset()

        sell_75_idx = self._action_index_for_fraction(env, 0.75)
        sell_50_idx = self._action_index_for_fraction(env, 0.5)

        _, _, _, _, info_first = env.step(sell_75_idx)
        _, reward_second, done_second, truncated_second, info_second = env.step(sell_50_idx)

        self.assertFalse(truncated_second)
        self._assert_reward_a_formula(reward_second, info_second)
        self.assertAlmostEqual(info_first["action_fraction_executed"], 0.75, places=12)
        self.assertAlmostEqual(info_second["action_fraction_requested"], 0.5, places=12)
        self.assertAlmostEqual(info_second["action_fraction_executed"], 0.25, places=12)
        self.assertAlmostEqual(info_second["sold_fraction"], 1.0, places=12)
        self.assertAlmostEqual(info_second["remaining_fraction"], 0.0, places=12)
        self.assertTrue(done_second)

        self._assert_increment_formula(info_first)
        self._assert_increment_formula(info_second)

        expected_cum_pre = (
            info_first["realized_pre_tax_increment"]
            + info_second["realized_pre_tax_increment"]
        )
        expected_cum_after = (
            info_first["realized_after_tax_increment"]
            + info_second["realized_after_tax_increment"]
        )
        self.assertAlmostEqual(
            info_second["cum_realized_pre_tax_pnl"], expected_cum_pre, places=12
        )
        self.assertAlmostEqual(
            info_second["cum_realized_after_tax_pnl"], expected_cum_after, places=12
        )


class TestTaxAwareEnvRewardA(unittest.TestCase):
    def _make_synthetic_env(
        self,
        *,
        dates: list[str],
        gains: list[float],
        transition_date: str,
        tax_profile: dict[str, Any],
    ) -> TaxAwareEnv:
        if len(dates) != len(gains):
            self.fail("Synthetic dates and gains must have the same length.")

        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        parquet_path = Path(temp_dir.name) / "synthetic_episodes.parquet"
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
        return TaxAwareEnv(
            parquet_path=parquet_path,
            state_columns=["feature"],
            tax_config=tax_profile,
        )

    def _action_index_for_fraction(self, env: TaxAwareEnv, target: float) -> int:
        for idx, value in enumerate(env.action_fractions):
            if np.isclose(value, target, atol=1e-12):
                return idx
        self.fail(f"Action fraction {target} not found in action_fractions.")
        return -1

    def _assert_reward_a_formula(self, reward: float, info: dict[str, Any]) -> None:
        self.assertEqual(
            info["reward_version"],
            "A_after_tax_total_value_change",
        )
        self.assertAlmostEqual(info["reward"], reward, places=12)
        self.assertAlmostEqual(info["reward_A"], reward, places=12)
        self.assertAlmostEqual(info["reward_C_lite"], reward, places=12)
        self.assertAlmostEqual(info["cooldown_penalty"], 0.0, places=12)
        self.assertAlmostEqual(
            reward,
            info["after_tax_total_value"]
            - info["previous_after_tax_total_value"],
            places=12,
        )
        self.assertAlmostEqual(
            info["after_tax_total_value"],
            info["cum_realized_after_tax_pnl"]
            + info["after_tax_liquidation_value_remaining"],
            places=12,
        )

    def test_holding_rewards_after_tax_position_value_changes(self) -> None:
        env = self._make_synthetic_env(
            dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
            gains=[0.10, 0.20, 0.05, 0.05],
            transition_date="2020-01-10",
            tax_profile=STANDARD_PROFILE,
        )
        hold_idx = self._action_index_for_fraction(env, 0.0)

        _, reset_info = env.reset()
        self.assertAlmostEqual(reset_info["after_tax_total_value"], 0.10 * 0.76)

        _, reward_0, done_0, _, info_0 = env.step(hold_idx)
        self._assert_reward_a_formula(reward_0, info_0)
        self.assertAlmostEqual(reward_0, 0.0, places=12)
        self.assertFalse(done_0)

        _, reward_1, done_1, _, info_1 = env.step(hold_idx)
        self._assert_reward_a_formula(reward_1, info_1)
        self.assertAlmostEqual(reward_1, 0.20 * 0.76 - 0.10 * 0.76, places=12)
        self.assertGreater(reward_1, 0.0)
        self.assertFalse(done_1)

        _, reward_2, done_2, _, info_2 = env.step(hold_idx)
        self._assert_reward_a_formula(reward_2, info_2)
        self.assertLess(reward_2, 0.0)
        self.assertTrue(done_2)

    def test_crossing_tax_transition_changes_after_tax_liquidation_value(self) -> None:
        env = self._make_synthetic_env(
            dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
            gains=[0.10, 0.10, 0.10, 0.10],
            transition_date="2020-01-02",
            tax_profile=STANDARD_PROFILE,
        )
        hold_idx = self._action_index_for_fraction(env, 0.0)

        env.reset()
        env.step(hold_idx)
        _, reward, done, _, info = env.step(hold_idx)

        self._assert_reward_a_formula(reward, info)
        self.assertEqual(info["after_tax_liquidation_tax_regime"], "long_term")
        self.assertAlmostEqual(reward, 0.10 * 0.85 - 0.10 * 0.76, places=12)
        self.assertGreater(reward, 0.0)
        self.assertFalse(done)

    def test_partial_sale_moves_value_without_double_counting(self) -> None:
        env = self._make_synthetic_env(
            dates=["2020-01-01", "2020-01-02", "2020-01-03"],
            gains=[0.20, 0.20, 0.20],
            transition_date="2020-01-10",
            tax_profile=STANDARD_PROFILE,
        )
        sell_50_idx = self._action_index_for_fraction(env, 0.5)

        env.reset()
        _, reward, done, _, info = env.step(sell_50_idx)

        self._assert_reward_a_formula(reward, info)
        self.assertAlmostEqual(reward, 0.0, places=12)
        self.assertAlmostEqual(info["cum_realized_after_tax_pnl"], 0.5 * 0.20 * 0.76)
        self.assertAlmostEqual(
            info["after_tax_liquidation_value_remaining"],
            0.5 * 0.20 * 0.76,
        )
        self.assertFalse(done)

    def test_long_term_sale_uses_profile_effective_rate(self) -> None:
        env = self._make_synthetic_env(
            dates=["2020-01-02", "2020-01-03", "2020-01-06"],
            gains=[0.20, 0.20, 0.20],
            transition_date="2020-01-01",
            tax_profile=HIGH_INCOME_PROFILE,
        )
        sell_50_idx = self._action_index_for_fraction(env, 0.5)

        env.reset()
        _, reward, done, _, info = env.step(sell_50_idx)

        self._assert_reward_a_formula(reward, info)
        self.assertEqual(info["tax_regime"], "long_term")
        self.assertAlmostEqual(info["applicable_tax_rate"], 0.188, places=12)
        self.assertAlmostEqual(info["realized_pre_tax_increment"], 0.10, places=12)
        self.assertAlmostEqual(info["tax_paid"], 0.0188, places=12)
        self.assertAlmostEqual(
            info["realized_after_tax_increment"],
            0.0812,
            places=12,
        )
        self.assertFalse(done)

    def test_terminal_liquidation_uses_long_term_treatment(self) -> None:
        terminal_profile = {
            "profile_name": "terminal_profile",
            "short_term_rate": 0.40,
            "long_term_rate": 0.10,
            "niit_rate": 0.0,
            "apply_niit": False,
        }
        env = self._make_synthetic_env(
            dates=["2020-01-01", "2020-01-02", "2020-01-03"],
            gains=[0.10, 0.10, 0.10],
            transition_date="2020-01-10",
            tax_profile=terminal_profile,
        )
        hold_idx = self._action_index_for_fraction(env, 0.0)

        _, reset_info = env.reset()
        self.assertAlmostEqual(reset_info["after_tax_total_value"], 0.10 * 0.60)
        env.step(hold_idx)
        _, reward, done, _, info = env.step(hold_idx)

        self._assert_reward_a_formula(reward, info)
        self.assertTrue(done)
        self.assertTrue(info["terminal_liquidation_executed"])
        self.assertEqual(info["terminal_liquidation_tax_regime"], "long_term")
        self.assertAlmostEqual(info["terminal_liquidation_tax_rate"], 0.10)
        self.assertAlmostEqual(info["terminal_liquidation_pre_tax_increment"], 0.10)
        self.assertAlmostEqual(info["terminal_liquidation_tax_paid"], 0.01)
        self.assertAlmostEqual(info["terminal_liquidation_after_tax_increment"], 0.09)
        self.assertAlmostEqual(reward, 0.10 * 0.90 - 0.10 * 0.60, places=12)
        self.assertAlmostEqual(info["remaining_fraction"], 0.0, places=12)
        self.assertAlmostEqual(info["after_tax_liquidation_value_remaining"], 0.0)
        self.assertAlmostEqual(info["after_tax_total_value"], 0.09)

    def test_terminal_liquidation_uses_final_row_pnl(self) -> None:
        terminal_profile = {
            "profile_name": "terminal_profile",
            "short_term_rate": 0.40,
            "long_term_rate": 0.10,
            "niit_rate": 0.0,
            "apply_niit": False,
        }
        env = self._make_synthetic_env(
            dates=["2020-01-01", "2020-01-02", "2020-01-03"],
            gains=[0.10, 0.10, 0.30],
            transition_date="2020-01-10",
            tax_profile=terminal_profile,
        )
        hold_idx = self._action_index_for_fraction(env, 0.0)

        _, reset_info = env.reset()
        self.assertAlmostEqual(reset_info["after_tax_total_value"], 0.10 * 0.60)
        env.step(hold_idx)
        _, reward, done, _, info = env.step(hold_idx)

        self._assert_reward_a_formula(reward, info)
        self.assertTrue(done)
        self.assertEqual(info["sale_row_ptr"], 1)
        self.assertEqual(info["terminal_liquidation_row_ptr"], 2)
        self.assertAlmostEqual(info["full_position_pnl"], 0.10, places=12)
        self.assertAlmostEqual(
            info["terminal_liquidation_full_position_pnl"],
            0.30,
            places=12,
        )
        self.assertAlmostEqual(
            info["terminal_liquidation_pre_tax_increment"],
            0.30,
            places=12,
        )
        self.assertAlmostEqual(info["terminal_liquidation_tax_paid"], 0.03)
        self.assertAlmostEqual(
            info["terminal_liquidation_after_tax_increment"],
            0.27,
            places=12,
        )
        self.assertAlmostEqual(reward, 0.30 * 0.90 - 0.10 * 0.60, places=12)
        self.assertAlmostEqual(info["after_tax_total_value"], 0.27, places=12)


if __name__ == "__main__":
    unittest.main()
