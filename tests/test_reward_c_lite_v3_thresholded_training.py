"""Tests for Reward C-lite v3 thresholded training behavior policy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml

from scripts.train_dqn_reward_a_full import (
    _validate_config_for_environment_reward,
    load_exploration_action_probabilities,
    resolve_training_exploitation_settings,
    select_thresholded_greedy_action,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FixedQNetwork(torch.nn.Module):
    def __init__(self, q_values: list[float]) -> None:
        super().__init__()
        self.register_buffer(
            "fixed_q_values",
            torch.tensor(q_values, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fixed_q_values.unsqueeze(0).repeat(x.shape[0], 1)


class TestRewardCLiteV3ThresholdedTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SimpleNamespace(action_fractions=(0.0, 0.25, 0.5, 0.75, 1.0))
        self.obs = np.array([0.0], dtype=np.float32)
        self.rng = np.random.default_rng(42)
        self.device = torch.device("cpu")

    def _select(self, q_values: list[float], margin: float) -> int:
        q_net = FixedQNetwork(q_values).to(self.device)
        return select_thresholded_greedy_action(
            q_net=q_net,
            obs=self.obs,
            num_actions=len(q_values),
            rng=self.rng,
            device=self.device,
            env=self.env,
            margin=margin,
        )

    def test_thresholded_greedy_returns_hold_when_sell_is_inside_margin(self) -> None:
        action_idx = self._select([1.0, 1.005, 1.019, 0.75, 0.50], margin=0.020)

        self.assertEqual(action_idx, 0)

    def test_thresholded_greedy_returns_best_sell_when_sell_exceeds_margin(self) -> None:
        action_idx = self._select([1.0, 1.005, 1.25, 1.05, 0.50], margin=0.020)

        self.assertEqual(action_idx, 2)

    def test_thresholded_greedy_uses_strict_greater_than(self) -> None:
        action_idx = self._select([1.0, 2.0, 0.50, 0.25, 0.10], margin=1.0)

        self.assertEqual(action_idx, 0)

    def test_v3_config_loads_with_thresholded_training_policy(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "train_reward_c_lite_v3.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        self.assertEqual(config["run"]["name"], "train_reward_c_lite_v3_full")
        self.assertFalse(config["reward"]["use_transaction_penalty"])
        self.assertFalse(config["reward"]["use_drawdown_penalty"])
        self.assertFalse(config["reward"]["use_explicit_tax_saving_bonus"])
        self.assertEqual(config["training"]["exploitation_policy"], "thresholded_greedy")
        self.assertAlmostEqual(
            config["training"]["thresholded_greedy"]["margin"],
            0.020,
            places=12,
        )
        self.assertEqual(
            resolve_training_exploitation_settings(config),
            ("thresholded_greedy", True, 0.020),
        )
        _validate_config_for_environment_reward(config)

    def test_v3_exploration_probabilities_sum_to_one(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "train_reward_c_lite_v3.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        probabilities = config["exploration"]["action_probabilities"]
        self.assertEqual(probabilities, [0.50, 0.20, 0.15, 0.10, 0.05])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
        loaded = load_exploration_action_probabilities(
            config,
            num_actions=len(config["action_space"]["action_fractions"]),
        )
        np.testing.assert_allclose(loaded, probabilities)

    def test_v3_keeps_v1_tax_action_and_split_schema(self) -> None:
        v1_path = PROJECT_ROOT / "configs" / "train_reward_c_lite_v1.yaml"
        v3_path = PROJECT_ROOT / "configs" / "train_reward_c_lite_v3.yaml"
        with v1_path.open("r", encoding="utf-8") as handle:
            v1_config = yaml.safe_load(handle)
        with v3_path.open("r", encoding="utf-8") as handle:
            v3_config = yaml.safe_load(handle)

        self.assertEqual(v3_config["environment"], v1_config["environment"])
        self.assertEqual(v3_config["tax_profile"], v1_config["tax_profile"])
        self.assertEqual(v3_config["action_space"], v1_config["action_space"])
        self.assertEqual(v3_config["splits"], v1_config["splits"])


if __name__ == "__main__":
    unittest.main()
