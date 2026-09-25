"""Phase 5 trainer and validation-only benchmark boundaries."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.evaluate_reward_a_baselines import evaluate_brazil_validation
from scripts.run_brazil_phase5_smoke import accounting_anchors
from scripts.train_dqn_reward_a_full import (
    _validate_config_for_environment_reward, assert_reward_info,
    build_episode_splits, load_yaml, make_env, run_validation_policy, QNetwork,
)


ROOT = Path(__file__).resolve().parents[1]


class TestBrazilPhase5Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_yaml(ROOT / "configs/brazil_sensitivity_rate_0700_v1.yaml")

    def test_all_rates_have_independent_accounting_anchors(self) -> None:
        cases = accounting_anchors()
        self.assertEqual(len(cases), 18)
        self.assertEqual({case["policy"] for case in cases}, {"hold", "immediate", "staggered"})

    def test_trainer_wires_scenario_state_and_saved_splits(self) -> None:
        _validate_config_for_environment_reward(self.config)
        env = make_env(self.config)
        self.assertEqual(env.economic_scenario["name"], "brazil_inspired_v1")
        self.assertEqual(len(env.state_columns), 36)
        splits = build_episode_splits(env, self.config)
        self.assertEqual({key: len(value) for key, value in splits.items()},
                         {"train": 7083, "validation": 1517, "test": 1519})
        obs, _ = env.reset(splits["train"][0])
        self.assertEqual(obs.shape, (36,))
        self.assertTrue(np.isfinite(obs).all())
        _, reward, done, _, info = env.step(4)
        self.assertTrue(done)
        self.assertTrue(np.isfinite(reward))
        assert_reward_info(reward, info, self.config["reward"]["version"])
        self.assertGreaterEqual(info["terminal_valuation_date"], info["date"])

    def test_changed_saved_split_is_rejected(self) -> None:
        config = deepcopy(self.config)
        with TemporaryDirectory() as temp:
            saved = pd.read_csv(ROOT / config["splits"]["source_csv"], dtype=str)
            train_rows = saved.index[saved["split"] == "train"][:2]
            saved.loc[train_rows, "episode_id"] = saved.loc[train_rows[::-1], "episode_id"].to_numpy()
            path = Path(temp) / "altered.csv"
            saved.to_csv(path, index=False)
            config["splits"]["source_csv"] = str(path)
            with self.assertRaisesRegex(ValueError, "split differs"):
                build_episode_splits(make_env(config), config)

    def test_validation_uses_wealth_and_fixed_first_sale_margin(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "one.parquet"
            pd.DataFrame({"episode_id": ["a"], "date": pd.to_datetime(["2021-01-01"]),
                          "adj_close": [100.0], "simulated_purchase_price": [100.0],
                          "feature": [0.0]}).to_parquet(path)
            config = deepcopy(self.config)
            config["environment"]["parquet_path"] = str(path)
            schema = Path(temp) / "state.json"
            schema.write_text('{"allowed_state_columns": ["feature"]}', encoding="utf-8")
            config["environment"]["state_schema_path"] = str(schema)
            env = make_env(config)
            net = QNetwork(1, 5, [], "relu")
            with torch.no_grad():
                net.net[-1].weight.zero_()
                net.net[-1].bias.copy_(torch.tensor([0.0, 0.03, 0.0, 0.0, 0.0]))
            rows, metrics = run_validation_policy(env, net, ["a"], config["reward"]["version"],
                                                   torch.device("cpu"), fixed_margins=(0.07, 0.02))
            self.assertEqual(int(rows.iloc[0]["action_idx"]), 0)
            self.assertEqual(metrics["mean_final_total_after_tax_wealth"], 1.0)
            self.assertNotIn("mean_final_after_tax_total_value", metrics)

    def test_benchmark_rejects_us_and_has_no_test_selection(self) -> None:
        us = load_yaml(ROOT / "configs/train_reward_c_lite_v5.yaml")
        with self.assertRaisesRegex(ValueError, "requires economic_scenario"):
            evaluate_brazil_validation(us, max_episodes=1)


if __name__ == "__main__":
    unittest.main()
