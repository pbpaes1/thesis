"""Smoke tests for the TaxAwareEnv Step 4 accounting contract."""

from __future__ import annotations

import json
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

            # Prefer non-terminal rows so traversal-to-row is always feasible.
            if len(ep) > 1:
                mask &= ep.index < (len(ep) - 1)

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

        episode_len = len(self.env._current_episode_df)
        self.assertGreaterEqual(episode_len, 1)

        next_obs, reward, done, truncated, info = self.env.step(0)

        self.assertEqual(next_obs.shape, (len(self.state_columns),))
        self.assertEqual(next_obs.dtype, np.float32)
        self.assertEqual(reward, 0.0)
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

        self.assertEqual(reward, 0.0)
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

        self.assertEqual(reward, 0.0)
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

        self.assertEqual(reward_second, 0.0)
        self.assertFalse(truncated_second)
        self.assertAlmostEqual(info_first["action_fraction_requested"], 0.75, places=12)
        self.assertAlmostEqual(info_first["action_fraction_executed"], 0.75, places=12)
        self.assertAlmostEqual(info_second["action_fraction_requested"], 0.5, places=12)
        self.assertAlmostEqual(info_second["action_fraction_executed"], 0.25, places=12)
        self.assertAlmostEqual(info_second["sold_fraction"], 1.0, places=12)
        self.assertAlmostEqual(info_second["remaining_fraction"], 0.0, places=12)
        self.assertAlmostEqual(self.env._sold_fraction, 1.0, places=12)
        self.assertAlmostEqual(self.env._remaining_fraction, 0.0, places=12)

        # Step 4 placeholder terminal condition also marks done on full liquidation.
        self.assertTrue(done_second)

    def test_tax_accounting_short_term_positive_standard_profile(self) -> None:
        env = self._make_env(STANDARD_PROFILE)
        episode_id, row_pos, row = self._find_episode_row(
            env, short_term=True, positive_gain=True
        )
        _, reward, _, truncated, info = self._run_sale_at_row(
            env, episode_id=episode_id, row_pos=row_pos, sell_fraction=0.5
        )

        self.assertEqual(reward, 0.0)
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
        _, _, done_second, truncated_second, info_second = env.step(sell_50_idx)

        self.assertFalse(truncated_second)
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


if __name__ == "__main__":
    unittest.main()
