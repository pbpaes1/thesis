from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from scripts.evaluate_reward_a_baselines import evaluate_policy_on_episodes
from src.environment.tax_aware_env import REWARD_A_VERSION, TaxAwareEnv


class TestBaselineTerminalTaxMetrics(unittest.TestCase):
    def _make_env(self, tax_profile: dict[str, Any]) -> TaxAwareEnv:
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        parquet_path = Path(self.temp_dir.name) / "episodes.parquet"
        df = pd.DataFrame(
            {
                "episode_id": ["episode_1", "episode_1", "episode_1"],
                "date": pd.to_datetime(
                    ["2020-01-01", "2020-01-02", "2020-01-03"]
                ),
                "tax_transition_date": pd.to_datetime(
                    ["2020-01-10", "2020-01-10", "2020-01-10"]
                ),
                "unrealized_gains_pct": [0.10, 0.10, 0.30],
                "feature": [0.0, 1.0, 2.0],
            }
        )
        df.to_parquet(parquet_path)
        return TaxAwareEnv(
            parquet_path=parquet_path,
            state_columns=["feature"],
            tax_config=tax_profile,
        )

    def test_hold_to_terminal_effective_tax_rate_includes_terminal_liquidation(
        self,
    ) -> None:
        env = self._make_env(
            {
                "profile_name": "terminal_profile",
                "short_term_rate": 0.40,
                "long_term_rate": 0.10,
                "niit_rate": 0.0,
                "apply_niit": False,
            }
        )

        episode_rows, step_rows = evaluate_policy_on_episodes(
            env=env,
            policy_name="hold_to_terminal",
            episode_ids=["episode_1"],
            split_name="test",
            expected_reward_version=REWARD_A_VERSION,
            rng=np.random.default_rng(123),
            q_net=None,
            device=None,  # Not used by non-DQN baseline policies.
        )

        self.assertEqual(len(episode_rows), 1)
        episode = episode_rows[0]
        self.assertTrue(episode["episode_terminal_liquidation_executed"])
        self.assertFalse(episode["episode_cut_occurred"])
        self.assertAlmostEqual(
            episode["pct_episode_position_sold_short_term"],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            episode["pct_episode_position_sold_long_term"],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            episode["terminal_liquidation_fraction"],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            episode["total_positive_taxable_pre_tax_increment"],
            0.30,
            places=12,
        )
        self.assertAlmostEqual(episode["total_tax_paid"], 0.03, places=12)
        self.assertAlmostEqual(
            episode["total_terminal_liquidation_tax_paid"],
            0.03,
            places=12,
        )
        self.assertAlmostEqual(
            episode["mean_effective_tax_rate_on_sales"],
            0.10,
            places=12,
        )

        terminal_step = step_rows[-1]
        self.assertTrue(terminal_step["terminal_liquidation_executed"])
        self.assertAlmostEqual(
            terminal_step["terminal_liquidation_tax_rate"],
            0.10,
            places=12,
        )
        self.assertAlmostEqual(
            terminal_step["terminal_liquidation_pre_tax_increment"],
            0.30,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
