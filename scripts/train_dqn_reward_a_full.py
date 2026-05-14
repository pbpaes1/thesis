"""Full DQN training script for environment-provided rewards.

This entrypoint keeps the Reward A script name for compatibility, but the
reward version is read from the YAML config and checked against environment
step info.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError(
        "pyyaml is required to run scripts/train_dqn_reward_a_full.py. "
        "Install it with: pip install pyyaml"
    ) from exc

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError(
        "PyTorch is required to run scripts/train_dqn_reward_a_full.py. "
        "Install torch before running the full DQN script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.tax_profiles import resolve_tax_profile_from_config  # noqa: E402
from src.environment.tax_aware_env import (  # noqa: E402
    REWARD_A_VERSION,
    REWARD_C_LITE_VERSION,
    REWARD_C_LITE_V2_VERSION,
    TaxAwareEnv,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "train_reward_a_v3.yaml"
REWARD_ASSERT_TOL = 1e-8

TRAIN_METRIC_COLUMNS = [
    "epoch_idx",
    "epoch_number",
    "global_train_episode_idx",
    "global_step",
    "train_episode_idx",
    "episode_id",
    "step_in_episode",
    "epsilon",
    "exploitation_policy",
    "thresholded_greedy_margin",
    "was_thresholded_to_hold",
    "loss",
    "replay_buffer_size",
    "mean_q_value",
    "max_q_value",
    "min_q_value",
    "reward",
    "done",
]

EVAL_METRIC_COLUMNS = [
    "evaluation_episode_idx",
    "epoch_idx",
    "epoch_number",
    "global_train_episode_idx",
    "global_step",
    "num_validation_episodes",
    "mean_episode_total_reward",
    "median_episode_total_reward",
    "min_episode_total_reward",
    "max_episode_total_reward",
    "mean_final_after_tax_total_value",
    "median_final_after_tax_total_value",
    "mean_episode_length",
    "terminal_liquidation_frequency",
    "full_liquidation_frequency",
]

TRAIN_ROLLOUT_COLUMNS = [
    "split",
    "epoch_idx",
    "epoch_number",
    "global_train_episode_idx",
    "episode_idx",
    "episode_id",
    "step_in_episode",
    "global_step",
    "date",
    "epsilon",
    "action_idx",
    "action_fraction_requested",
    "action_fraction_executed",
    "raw_greedy_action_idx",
    "raw_greedy_action_fraction",
    "selected_action_idx",
    "selected_action_fraction",
    "hold_q",
    "best_sell_q",
    "q_margin",
    "thresholded_greedy_margin",
    "was_thresholded_to_hold",
    "reward",
    "reward_A",
    "reward_C_lite",
    "reward_C_lite_v2",
    "transaction_penalty",
    "transaction_penalty_applied",
    "cooldown_penalty",
    "cooldown_penalty_applied",
    "days_since_last_sale",
    "sale_count",
    "last_sale_date_before_step",
    "last_sale_date_after_step",
    "after_tax_total_value",
    "previous_after_tax_total_value",
    "realized_after_tax_increment",
    "cum_realized_after_tax_pnl",
    "after_tax_liquidation_value_remaining",
    "sold_fraction",
    "remaining_fraction",
    "tax_regime",
    "after_tax_liquidation_tax_regime",
    "is_automatic_terminal_liquidation",
    "terminal_liquidation_executed",
    "done",
    "loss",
]

VALIDATION_ROLLOUT_COLUMNS = [
    "split",
    "evaluation_episode_idx",
    "global_step",
    "episode_idx",
    "episode_id",
    "step_in_episode",
    "date",
    "action_idx",
    "action_fraction_requested",
    "action_fraction_executed",
    "reward",
    "reward_A",
    "reward_C_lite",
    "reward_C_lite_v2",
    "transaction_penalty",
    "transaction_penalty_applied",
    "cooldown_penalty",
    "cooldown_penalty_applied",
    "days_since_last_sale",
    "sale_count",
    "last_sale_date_before_step",
    "last_sale_date_after_step",
    "after_tax_total_value",
    "previous_after_tax_total_value",
    "realized_after_tax_increment",
    "cum_realized_after_tax_pnl",
    "after_tax_liquidation_value_remaining",
    "sold_fraction",
    "remaining_fraction",
    "tax_regime",
    "after_tax_liquidation_tax_regime",
    "is_automatic_terminal_liquidation",
    "terminal_liquidation_executed",
    "done",
]


def _require(config: dict, path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing required config value: {path}")
        current = current[key]
    return current


def _project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.8f}"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping at top level: {path}")
    return payload


def save_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_state_columns(schema_path: Path) -> list[str]:
    with schema_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    columns = payload.get("allowed_state_columns") if isinstance(payload, dict) else payload
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        raise ValueError(
            "State schema must contain a list of strings under "
            "'allowed_state_columns'."
        )
    if not columns:
        raise ValueError(f"State schema has no allowed state columns: {schema_path}")
    return columns


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_config)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but CUDA is not available.")
    return device


def make_env(config: dict) -> TaxAwareEnv:
    env_config = _require(config, "environment")
    parquet_path = _project_path(_require(env_config, "parquet_path"))
    schema_path = _project_path(_require(env_config, "state_schema_path"))
    state_columns = load_state_columns(schema_path)
    action_fractions = _require(config, "action_space.action_fractions")
    tax_config = resolve_tax_profile_from_config(config, base_dir=PROJECT_ROOT)
    seed = int(_require(config, "training.seed"))

    return TaxAwareEnv(
        parquet_path=parquet_path,
        state_columns=state_columns,
        action_fractions=action_fractions,
        tax_config=tax_config,
        reward_config=config.get("reward"),
        seed=seed,
    )


def action_index_for_fraction(env: TaxAwareEnv, target_fraction: float) -> int:
    for idx, action_fraction in enumerate(env.action_fractions):
        if approx_equal(float(action_fraction), float(target_fraction)):
            return idx
    raise ValueError(
        f"Action fraction {target_fraction} is not present in "
        f"env.action_fractions={env.action_fractions}."
    )


def approx_equal(a: float, b: float, tol: float = REWARD_ASSERT_TOL) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def assert_reward_info(
    reward: float,
    info: dict,
    expected_reward_version: str,
) -> None:
    assert info["reward_version"] == expected_reward_version, (
        f"Unexpected reward_version={info['reward_version']!r}; "
        f"expected {expected_reward_version!r}."
    )
    assert approx_equal(reward, info["reward"]), "reward != info['reward']"
    reward_A = float(info["reward_A"])
    transaction_penalty = float(info.get("transaction_penalty", 0.0) or 0.0)
    cooldown_penalty = float(info.get("cooldown_penalty", 0.0) or 0.0)
    reward_C_lite = info.get("reward_C_lite")
    if reward_C_lite is not None:
        reward_C_lite = float(reward_C_lite)
    reward_C_lite_v2 = info.get("reward_C_lite_v2")
    if reward_C_lite_v2 is not None:
        reward_C_lite_v2 = float(reward_C_lite_v2)
    assert approx_equal(
        reward_A,
        info["after_tax_total_value"] - info["previous_after_tax_total_value"],
    ), "Reward A identity failed."
    assert approx_equal(
        info["after_tax_total_value"],
        info["cum_realized_after_tax_pnl"]
        + info["after_tax_liquidation_value_remaining"],
    ), "After-tax total value decomposition failed."
    if expected_reward_version == REWARD_A_VERSION:
        assert approx_equal(reward, reward_A), "Reward A run returned non-A reward."
        assert approx_equal(transaction_penalty, 0.0), (
            "Reward A run produced a transaction penalty."
        )
        assert approx_equal(cooldown_penalty, 0.0), (
            "Reward A run produced a cooldown penalty."
        )
        if reward_C_lite is not None:
            assert approx_equal(reward_C_lite, reward_A), (
                "Reward A run has inconsistent reward_C_lite alias."
            )
        if reward_C_lite_v2 is not None:
            assert approx_equal(reward_C_lite_v2, reward_A), (
                "Reward A run has inconsistent reward_C_lite_v2 alias."
            )
    elif expected_reward_version == REWARD_C_LITE_VERSION:
        assert reward_C_lite is not None, "Reward C-lite missing info['reward_C_lite']."
        assert approx_equal(transaction_penalty, 0.0), (
            "Reward C-lite v1 run produced a transaction penalty."
        )
        assert approx_equal(reward, reward_C_lite), (
            "Reward C-lite run returned non-C-lite reward."
        )
        assert approx_equal(reward_C_lite, reward_A - cooldown_penalty), (
            "Reward C-lite identity failed."
        )
        if reward_C_lite_v2 is not None:
            assert approx_equal(reward_C_lite_v2, reward_A - cooldown_penalty), (
                "Reward C-lite v1 has inconsistent reward_C_lite_v2 alias."
            )
    elif expected_reward_version == REWARD_C_LITE_V2_VERSION:
        assert reward_C_lite_v2 is not None, (
            "Reward C-lite v2 missing info['reward_C_lite_v2']."
        )
        assert reward_C_lite is not None, (
            "Reward C-lite v2 missing info['reward_C_lite']."
        )
        assert approx_equal(reward, reward_C_lite_v2), (
            "Reward C-lite v2 run returned non-v2 reward."
        )
        assert approx_equal(reward_C_lite, reward_A - cooldown_penalty), (
            "Reward C-lite v2 has inconsistent reward_C_lite diagnostic."
        )
        assert approx_equal(
            reward_C_lite_v2,
            reward_A - transaction_penalty - cooldown_penalty,
        ), "Reward C-lite v2 identity failed."
    else:
        raise AssertionError(f"Unsupported expected reward version: {expected_reward_version}")
    assert np.isfinite(reward), "Reward is not finite."


def build_episode_splits(env: TaxAwareEnv, config: dict) -> dict[str, list[str]]:
    split_config = _require(config, "splits")
    method = str(_require(split_config, "method"))
    shuffle_episodes = bool(_require(split_config, "shuffle_episodes"))
    if method != "chronological_by_episode":
        raise ValueError(
            "Full training script only supports splits.method="
            "'chronological_by_episode'."
        )
    if shuffle_episodes:
        raise ValueError("Full training script requires shuffle_episodes=false.")

    env._load_episode_index()
    if env._df is None:
        raise RuntimeError("Environment dataframe is not loaded.")

    episode_start_dates: list[tuple[str, pd.Timestamp]] = []

    for episode_id, row_idx in env._episode_index.items():
        episode_df = env._df.iloc[row_idx]
        if episode_df.empty:
            continue

        first_date = pd.to_datetime(episode_df["date"], errors="coerce").min()
        if pd.isna(first_date):
            raise ValueError(f"Episode {episode_id} has invalid or missing dates.")

        episode_start_dates.append((episode_id, first_date))

    episode_start_dates = sorted(
        episode_start_dates,
        key=lambda item: (item[1], item[0]),
    )

    episode_ids = [episode_id for episode_id, _ in episode_start_dates]
    n_total = len(episode_ids)
    if n_total == 0:
        raise ValueError("No episodes found for train/validation/test splitting.")

    train_fraction = float(_require(split_config, "train_fraction"))
    validation_fraction = float(_require(split_config, "validation_fraction"))
    test_fraction = float(_require(split_config, "test_fraction"))
    if any(
        fraction < 0.0
        for fraction in (train_fraction, validation_fraction, test_fraction)
    ):
        raise ValueError("Split fractions must be non-negative.")

    n_train = int(n_total * train_fraction)
    n_validation = int(n_total * validation_fraction)
    train_ids = episode_ids[:n_train]
    validation_ids = episode_ids[n_train : n_train + n_validation]
    test_ids = episode_ids[n_train + n_validation :]

    if not train_ids:
        raise ValueError("Training split is empty.")
    if validation_fraction > 0.0 and not validation_ids:
        raise ValueError(
            "Validation split is empty despite validation_fraction > 0."
        )

    combined = train_ids + validation_ids + test_ids
    if len(combined) != n_total or set(combined) != set(episode_ids):
        raise AssertionError("Episode split coverage failed.")
    if len(combined) != len(set(combined)):
        raise AssertionError("An episode appears in more than one split.")

    return {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_layers: list[int],
        activation: str,
    ) -> None:
        super().__init__()
        if activation != "relu":
            raise ValueError(f"Unsupported activation {activation!r}; expected 'relu'.")

        layers: list[nn.Module] = []
        input_dim = obs_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self.capacity = int(capacity)
        self._buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.capacity
        )
        self._rng = np.random.default_rng(seed)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append(
            (
                np.asarray(obs, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32).copy(),
                bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from "
                f"replay buffer with {len(self._buffer)} rows."
            )
        indices = self._rng.choice(len(self._buffer), size=batch_size, replace=False)
        samples = [self._buffer[int(idx)] for idx in indices]
        obs, actions, rewards, next_obs, dones = zip(*samples)
        return (
            np.stack(obs).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_obs).astype(np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


@dataclass(frozen=True)
class TrainingActionSelection:
    action_idx: int
    raw_greedy_action_idx: int | None
    raw_greedy_action_fraction: float | None
    selected_action_idx: int
    selected_action_fraction: float
    hold_q: float | None
    best_sell_q: float | None
    q_margin: float | None
    thresholded_greedy_margin: float | None
    was_thresholded_to_hold: bool


def load_exploration_action_probabilities(
    config: dict,
    num_actions: int,
) -> np.ndarray | None:
    exploration_config = config.get("exploration")
    if exploration_config is None:
        return None
    if not isinstance(exploration_config, dict):
        raise ValueError("exploration must be a mapping when provided.")

    policy = exploration_config.get("random_action_policy", "uniform")
    if policy == "uniform":
        return None
    if policy != "hold_biased":
        raise ValueError(
            "Unsupported exploration.random_action_policy="
            f"{policy!r}; expected 'uniform' or 'hold_biased'."
        )

    raw_probs = exploration_config.get("action_probabilities")
    if raw_probs is None:
        raise ValueError(
            "exploration.action_probabilities is required for hold_biased exploration."
        )

    try:
        probs = np.asarray(raw_probs, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "exploration.action_probabilities must be numeric."
        ) from exc

    if probs.shape != (num_actions,):
        raise ValueError(
            "exploration.action_probabilities must have shape "
            f"({num_actions},), got {probs.shape}."
        )
    if not np.isfinite(probs).all():
        raise ValueError(
            "exploration.action_probabilities must contain only finite values."
        )
    if (probs < 0.0).any():
        raise ValueError(
            "exploration.action_probabilities must not contain negative values."
        )
    if not np.isclose(float(probs.sum()), 1.0):
        raise ValueError(
            "exploration.action_probabilities must sum to 1.0, got "
            f"{float(probs.sum()):.12f}."
        )

    return probs


def _compute_q_values(
    q_net: QNetwork,
    obs: np.ndarray,
    num_actions: int,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values_tensor = q_net(obs_tensor)
        if q_values_tensor.shape != (1, num_actions):
            raise AssertionError(
                f"QNetwork output shape {tuple(q_values_tensor.shape)} does not match "
                f"(1, {num_actions})."
            )
        if not torch.isfinite(q_values_tensor).all():
            raise AssertionError("Q-values contain NaN or infinite values.")
        q_values = q_values_tensor.squeeze(0).detach().cpu().numpy().astype(float)
    if q_values.shape != (num_actions,):
        raise AssertionError(
            f"Q-values shape {q_values.shape} does not match ({num_actions},)."
        )
    if not np.isfinite(q_values).all():
        raise AssertionError("Q-values contain NaN or infinite values.")
    return q_values


def _thresholded_greedy_details(
    q_values: np.ndarray,
    num_actions: int,
    env: TaxAwareEnv,
    margin: float,
    first_sale_margin: float | None = None,
) -> dict[str, int | float | bool]:
    if margin < 0.0:
        raise ValueError(f"Threshold margin must be non-negative, got {margin}.")
    active_margin = thresholded_greedy_active_margin(
        env=env,
        margin=margin,
        first_sale_margin=first_sale_margin,
    )
    if q_values.shape != (num_actions,):
        raise AssertionError(
            f"Q-values shape {q_values.shape} does not match ({num_actions},)."
        )
    if not np.isfinite(q_values).all():
        raise AssertionError("Q-values contain NaN or infinite values.")

    hold_idx = action_index_for_fraction(env, 0.0)
    sell_indices = [idx for idx in range(num_actions) if idx != hold_idx]
    if not sell_indices:
        raise ValueError("Thresholded greedy policy requires at least one sell action.")

    best_sell_idx = max(sell_indices, key=lambda idx: float(q_values[idx]))
    hold_q = float(q_values[hold_idx])
    best_sell_q = float(q_values[best_sell_idx])
    raw_greedy_action_idx = int(np.argmax(q_values))
    selected_action_idx = (
        int(best_sell_idx)
        if best_sell_q > hold_q + active_margin
        else int(hold_idx)
    )
    was_thresholded_to_hold = bool(
        raw_greedy_action_idx != hold_idx and selected_action_idx == hold_idx
    )

    return {
        "hold_idx": int(hold_idx),
        "best_sell_idx": int(best_sell_idx),
        "raw_greedy_action_idx": raw_greedy_action_idx,
        "selected_action_idx": selected_action_idx,
        "hold_q": hold_q,
        "best_sell_q": best_sell_q,
        "q_margin": float(best_sell_q - hold_q),
        "thresholded_greedy_margin": active_margin,
        "was_thresholded_to_hold": was_thresholded_to_hold,
    }


def is_before_first_discretionary_sale(env: TaxAwareEnv) -> bool:
    remaining_fraction = getattr(env, "_remaining_fraction", None)
    if remaining_fraction is not None:
        return bool(np.isclose(float(remaining_fraction), 1.0, rtol=0.0, atol=1e-12))

    sold_fraction = getattr(env, "_sold_fraction", None)
    if sold_fraction is not None:
        return bool(np.isclose(float(sold_fraction), 0.0, rtol=0.0, atol=1e-12))

    sale_count = getattr(env, "_sale_count", None)
    if sale_count is not None:
        return int(sale_count) == 0

    return False


def thresholded_greedy_active_margin(
    env: TaxAwareEnv,
    margin: float,
    first_sale_margin: float | None = None,
) -> float:
    if margin < 0.0:
        raise ValueError(f"Threshold margin must be non-negative, got {margin}.")
    if first_sale_margin is None:
        return float(margin)
    if first_sale_margin < 0.0:
        raise ValueError(
            f"First-sale threshold margin must be non-negative, got {first_sale_margin}."
        )
    if is_before_first_discretionary_sale(env):
        return float(first_sale_margin)
    return float(margin)


def select_thresholded_greedy_action(
    q_net: QNetwork,
    obs: np.ndarray,
    num_actions: int,
    rng: np.random.Generator,
    device: torch.device,
    env: TaxAwareEnv,
    margin: float,
    first_sale_margin: float | None = None,
) -> int:
    _ = rng
    q_values = _compute_q_values(q_net, obs, num_actions, device)
    details = _thresholded_greedy_details(
        q_values,
        num_actions,
        env,
        margin,
        first_sale_margin=first_sale_margin,
    )
    return int(details["selected_action_idx"])


def select_training_action(
    q_net: QNetwork,
    obs: np.ndarray,
    epsilon: float,
    num_actions: int,
    rng: np.random.Generator,
    device: torch.device,
    env: TaxAwareEnv,
    exploitation_policy: str,
    thresholded_greedy_margin: float | None,
    thresholded_greedy_first_sale_margin: float | None = None,
    exploration_action_probabilities: np.ndarray | None = None,
) -> TrainingActionSelection:
    active_thresholded_greedy_margin = (
        thresholded_greedy_active_margin(
            env=env,
            margin=thresholded_greedy_margin,
            first_sale_margin=thresholded_greedy_first_sale_margin,
        )
        if thresholded_greedy_margin is not None
        else None
    )

    if rng.random() < epsilon:
        if exploration_action_probabilities is not None:
            action_idx = int(rng.choice(num_actions, p=exploration_action_probabilities))
        else:
            action_idx = int(rng.integers(0, num_actions))
        action_fraction = float(env.action_fractions[action_idx])
        return TrainingActionSelection(
            action_idx=action_idx,
            raw_greedy_action_idx=None,
            raw_greedy_action_fraction=None,
            selected_action_idx=action_idx,
            selected_action_fraction=action_fraction,
            hold_q=None,
            best_sell_q=None,
            q_margin=None,
            thresholded_greedy_margin=active_thresholded_greedy_margin,
            was_thresholded_to_hold=False,
        )

    q_values = _compute_q_values(q_net, obs, num_actions, device)
    raw_greedy_action_idx = int(np.argmax(q_values))
    raw_greedy_action_fraction = float(env.action_fractions[raw_greedy_action_idx])

    if exploitation_policy == "thresholded_greedy":
        if thresholded_greedy_margin is None:
            raise ValueError(
                "training.thresholded_greedy.margin is required when "
                "training.exploitation_policy='thresholded_greedy'."
            )
        details = _thresholded_greedy_details(
            q_values,
            num_actions,
            env,
            thresholded_greedy_margin,
            first_sale_margin=thresholded_greedy_first_sale_margin,
        )
        selected_action_idx = int(details["selected_action_idx"])
        return TrainingActionSelection(
            action_idx=selected_action_idx,
            raw_greedy_action_idx=raw_greedy_action_idx,
            raw_greedy_action_fraction=raw_greedy_action_fraction,
            selected_action_idx=selected_action_idx,
            selected_action_fraction=float(env.action_fractions[selected_action_idx]),
            hold_q=float(details["hold_q"]),
            best_sell_q=float(details["best_sell_q"]),
            q_margin=float(details["q_margin"]),
            thresholded_greedy_margin=float(details["thresholded_greedy_margin"]),
            was_thresholded_to_hold=bool(details["was_thresholded_to_hold"]),
        )

    if exploitation_policy != "greedy":
        raise ValueError(
            f"Unsupported training.exploitation_policy={exploitation_policy!r}; "
            "expected 'greedy' or 'thresholded_greedy'."
        )

    return TrainingActionSelection(
        action_idx=raw_greedy_action_idx,
        raw_greedy_action_idx=raw_greedy_action_idx,
        raw_greedy_action_fraction=raw_greedy_action_fraction,
        selected_action_idx=raw_greedy_action_idx,
        selected_action_fraction=raw_greedy_action_fraction,
        hold_q=None,
        best_sell_q=None,
        q_margin=None,
        thresholded_greedy_margin=thresholded_greedy_margin,
        was_thresholded_to_hold=False,
    )


def select_epsilon_greedy_action(
    q_net: QNetwork,
    obs: np.ndarray,
    epsilon: float,
    num_actions: int,
    rng: np.random.Generator,
    device: torch.device,
    exploration_action_probabilities: np.ndarray | None = None,
) -> int:
    if rng.random() < epsilon:
        if exploration_action_probabilities is not None:
            return int(rng.choice(num_actions, p=exploration_action_probabilities))
        return int(rng.integers(0, num_actions))
    q_values = _compute_q_values(q_net, obs, num_actions, device)
    return int(np.argmax(q_values))


def select_greedy_action(
    q_net: QNetwork,
    obs: np.ndarray,
    num_actions: int,
    device: torch.device,
) -> int:
    q_values = _compute_q_values(q_net, obs, num_actions, device)
    return int(np.argmax(q_values))


def compute_epsilon(
    global_step: int,
    start: float,
    end: float,
    decay_steps: int,
) -> float:
    if decay_steps <= 0:
        return float(end)
    progress = min(max(global_step, 0) / float(decay_steps), 1.0)
    return float(start + progress * (end - start))


def optimize_dqn(
    q_net: QNetwork,
    target_net: QNetwork,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    gradient_clip_norm: float | None,
) -> tuple[float, float, float, float] | None:
    if len(replay_buffer) < batch_size:
        return None

    obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)
    obs_batch = torch.as_tensor(obs, dtype=torch.float32, device=device)
    action_batch = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    reward_batch = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_obs_batch = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    done_batch = torch.as_tensor(dones, dtype=torch.float32, device=device)

    q_values = q_net(obs_batch)
    if not torch.isfinite(q_values).all():
        raise AssertionError("Q-values contain NaN or infinite values.")
    q_selected = q_values.gather(1, action_batch).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_obs_batch)
        if not torch.isfinite(next_q_values).all():
            raise AssertionError("Target Q-values contain NaN or infinite values.")
        target = reward_batch + gamma * (1.0 - done_batch) * next_q_values.max(
            dim=1
        ).values

    loss = nn.MSELoss()(q_selected, target)
    if not torch.isfinite(loss):
        raise AssertionError("DQN loss is NaN or infinite.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if gradient_clip_norm is not None:
        nn.utils.clip_grad_norm_(q_net.parameters(), float(gradient_clip_norm))
    optimizer.step()

    q_numpy = q_values.detach().cpu().numpy()
    return (
        float(loss.detach().cpu().item()),
        float(q_numpy.mean()),
        float(q_numpy.max()),
        float(q_numpy.min()),
    )


def run_validation_policy(
    env: TaxAwareEnv,
    q_net: QNetwork,
    episode_ids: list[str],
    expected_reward_version: str,
    device: torch.device,
    max_eval_episodes: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    if max_eval_episodes is not None:
        if max_eval_episodes <= 0:
            raise ValueError("max_eval_episodes must be positive when provided.")
        episode_ids = episode_ids[:max_eval_episodes]
    if not episode_ids:
        raise ValueError("Validation requested with no validation episodes.")

    was_training = q_net.training
    q_net.eval()
    num_actions = len(env.action_fractions)
    obs_dim = len(env.state_columns)
    rollout_rows: list[dict[str, Any]] = []
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    final_values: list[float] = []
    terminal_liquidation_flags: list[bool] = []
    full_liquidation_flags: list[bool] = []

    try:
        for episode_idx, episode_id in enumerate(episode_ids, start=1):
            obs, _reset_info = env.reset(episode_id=episode_id)
            if obs.shape != (obs_dim,):
                raise AssertionError(
                    f"Validation observation shape {obs.shape} does not match "
                    f"expected {(obs_dim,)}."
                )
            if not np.isfinite(obs).all():
                raise AssertionError(
                    "Validation observation contains NaN or infinite values."
                )

            done = False
            step_in_episode = 0
            episode_total_reward = 0.0
            max_steps = len(env._current_episode_df) + 5
            last_info: dict[str, Any] | None = None

            while not done:
                if step_in_episode > max_steps:
                    raise RuntimeError(
                        f"Validation safety cap exceeded for episode_id={episode_id}: "
                        f"max_steps={max_steps}."
                    )

                action_idx = select_greedy_action(q_net, obs, num_actions, device)
                next_obs, reward, done, truncated, info = env.step(action_idx)
                if truncated is not False:
                    raise AssertionError(
                        "TaxAwareEnv returned truncated=True during validation."
                    )
                assert_reward_info(reward, info, expected_reward_version)
                if not np.isfinite(reward):
                    raise AssertionError("Validation reward is NaN or infinite.")
                if next_obs.shape != (obs_dim,):
                    raise AssertionError(
                        f"Validation next observation shape {next_obs.shape} "
                        f"does not match expected {(obs_dim,)}."
                    )
                if not np.isfinite(next_obs).all():
                    raise AssertionError(
                        "Validation next observation contains NaN or infinite values."
                    )

                rollout_rows.append(
                    {
                        "split": "validation",
                        "evaluation_episode_idx": None,
                        "global_step": None,
                        "episode_idx": episode_idx,
                        "episode_id": episode_id,
                        "step_in_episode": step_in_episode,
                        "date": info.get("date"),
                        "action_idx": action_idx,
                        "action_fraction_requested": info.get(
                            "action_fraction_requested"
                        ),
                        "action_fraction_executed": info.get(
                            "action_fraction_executed"
                        ),
                        "reward": reward,
                        "reward_A": info.get("reward_A"),
                        "reward_C_lite": info.get("reward_C_lite"),
                        "reward_C_lite_v2": info.get("reward_C_lite_v2"),
                        "transaction_penalty": info.get("transaction_penalty"),
                        "transaction_penalty_applied": info.get(
                            "transaction_penalty_applied"
                        ),
                        "cooldown_penalty": info.get("cooldown_penalty"),
                        "cooldown_penalty_applied": info.get(
                            "cooldown_penalty_applied"
                        ),
                        "days_since_last_sale": info.get("days_since_last_sale"),
                        "sale_count": info.get("sale_count"),
                        "last_sale_date_before_step": info.get(
                            "last_sale_date_before_step"
                        ),
                        "last_sale_date_after_step": info.get(
                            "last_sale_date_after_step"
                        ),
                        "after_tax_total_value": info.get("after_tax_total_value"),
                        "previous_after_tax_total_value": info.get(
                            "previous_after_tax_total_value"
                        ),
                        "realized_after_tax_increment": info.get(
                            "realized_after_tax_increment"
                        ),
                        "cum_realized_after_tax_pnl": info.get(
                            "cum_realized_after_tax_pnl"
                        ),
                        "after_tax_liquidation_value_remaining": info.get(
                            "after_tax_liquidation_value_remaining"
                        ),
                        "sold_fraction": info.get("sold_fraction"),
                        "remaining_fraction": info.get("remaining_fraction"),
                        "tax_regime": info.get("tax_regime"),
                        "after_tax_liquidation_tax_regime": info.get(
                            "after_tax_liquidation_tax_regime"
                        ),
                        "is_automatic_terminal_liquidation": info.get(
                            "is_automatic_terminal_liquidation"
                        ),
                        "terminal_liquidation_executed": info.get(
                            "terminal_liquidation_executed"
                        ),
                        "done": done,
                    }
                )

                episode_total_reward += float(reward)
                last_info = info
                obs = next_obs
                step_in_episode += 1

            if last_info is None:
                raise RuntimeError(f"Validation episode {episode_id} had no steps.")
            episode_rewards.append(episode_total_reward)
            episode_lengths.append(step_in_episode)
            final_values.append(float(last_info["after_tax_total_value"]))
            terminal_liquidation_flags.append(
                bool(last_info.get("terminal_liquidation_executed", False))
            )
            full_liquidation_flags.append(
                approx_equal(float(last_info.get("remaining_fraction", 0.0)), 0.0)
            )
    finally:
        if was_training:
            q_net.train()

    metrics = {
        "num_validation_episodes": len(episode_rewards),
        "mean_episode_total_reward": float(np.mean(episode_rewards)),
        "median_episode_total_reward": float(np.median(episode_rewards)),
        "min_episode_total_reward": float(np.min(episode_rewards)),
        "max_episode_total_reward": float(np.max(episode_rewards)),
        "mean_final_after_tax_total_value": float(np.mean(final_values)),
        "median_final_after_tax_total_value": float(np.median(final_values)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "terminal_liquidation_frequency": float(np.mean(terminal_liquidation_flags)),
        "full_liquidation_frequency": float(np.mean(full_liquidation_flags)),
    }
    return (
        pd.DataFrame(rollout_rows, columns=VALIDATION_ROLLOUT_COLUMNS),
        metrics,
    )


def _checkpoint_payload(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    config: dict,
    obs_dim: int,
    num_actions: int,
    expected_reward_version: str,
    global_step: int,
    train_episode_idx: int,
    exploration_random_action_policy: str = "uniform",
    exploration_action_probabilities: list[float] | None = None,
    epoch_idx: int | None = None,
    epoch_number: int | None = None,
    global_train_episode_idx: int | None = None,
    best_model_metric: str | None = None,
    best_metric_value: float | None = None,
) -> dict:
    resolved_tax_profile = resolve_tax_profile_from_config(
        config,
        base_dir=PROJECT_ROOT,
    )
    return {
        "model_state_dict": q_net.state_dict(),
        "target_model_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "resolved_tax_profile": resolved_tax_profile,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "reward_version": expected_reward_version,
        "global_step": global_step,
        "train_episode_idx": train_episode_idx,
        "epoch_idx": epoch_idx,
        "epoch_number": epoch_number,
        "global_train_episode_idx": global_train_episode_idx,
        "best_model_metric": best_model_metric,
        "best_metric_value": best_metric_value,
        "exploration_random_action_policy": exploration_random_action_policy,
        "exploration_action_probabilities": exploration_action_probabilities,
    }


def _validate_config_for_environment_reward(config: dict) -> None:
    algorithm_name = str(_require(config, "algorithm.name")).lower()
    framework = str(_require(config, "algorithm.framework")).lower()
    optimizer_name = str(_require(config, "training.optimizer")).lower()
    if algorithm_name != "dqn":
        raise ValueError(
            f"Full script only supports algorithm.name='dqn', got {algorithm_name!r}."
        )
    if framework != "torch":
        raise ValueError(
            f"Full script only supports algorithm.framework='torch', got {framework!r}."
        )
    if optimizer_name != "adam":
        raise ValueError(
            f"Full script only supports training.optimizer='adam', got {optimizer_name!r}."
        )

    reward_version = str(_require(config, "reward.version"))
    expected_reward_version = str(_require(config, "reward.expected_info_reward_version"))
    if reward_version != expected_reward_version:
        raise ValueError(
            "Config reward.version and reward.expected_info_reward_version "
            f"must match, got {reward_version!r} and {expected_reward_version!r}."
        )
    supported_reward_versions = {
        REWARD_A_VERSION,
        REWARD_C_LITE_VERSION,
        REWARD_C_LITE_V2_VERSION,
    }
    if reward_version not in supported_reward_versions:
        raise ValueError(
            f"Unsupported reward.version={reward_version!r}; expected "
            f"one of {sorted(supported_reward_versions)}."
        )
    if not bool(_require(config, "reward.use_environment_reward")):
        raise ValueError("Full training requires use_environment_reward=true.")
    excluded_flags = [
        "reward.use_drawdown_penalty",
        "reward.use_explicit_tax_saving_bonus",
        "reward.use_reward_clipping",
        "reward.use_reward_normalization",
    ]
    enabled_flags = [flag for flag in excluded_flags if bool(_require(config, flag))]
    if enabled_flags:
        raise ValueError(
            "Full training must not enable unsupported reward options: "
            f"{enabled_flags}"
        )
    use_cooldown_penalty = bool(_require(config, "reward.use_cooldown_penalty"))
    use_transaction_penalty = bool(
        _get_nested(config, "reward.use_transaction_penalty", False)
    )
    if reward_version == REWARD_A_VERSION and (
        use_cooldown_penalty or use_transaction_penalty
    ):
        raise ValueError(
            "Reward A full training requires transaction and cooldown penalties off."
        )
    if reward_version in {REWARD_C_LITE_VERSION, REWARD_C_LITE_V2_VERSION}:
        if str(_require(config, "reward.base_reward_version")) != REWARD_A_VERSION:
            raise ValueError("Reward C-lite requires base_reward_version=Reward A.")
        if not use_cooldown_penalty:
            raise ValueError("Reward C-lite requires use_cooldown_penalty=true.")
        cooldown_config = _require(config, "reward.cooldown_penalty")
        if not bool(_require(cooldown_config, "enabled")):
            raise ValueError("Reward C-lite requires cooldown_penalty.enabled=true.")
        if float(_require(cooldown_config, "lambda_cooldown")) < 0.0:
            raise ValueError("lambda_cooldown must be non-negative.")
        if int(_require(cooldown_config, "cooldown_days")) <= 0:
            raise ValueError("cooldown_days must be positive.")
        if bool(_require(cooldown_config, "penalize_first_sale")):
            raise ValueError("Reward C-lite must not penalize the first sale.")
        if bool(
            _get_nested(
                cooldown_config,
                "apply_to_automatic_terminal_liquidation",
                False,
            )
        ):
            raise ValueError(
                "Reward C-lite must not penalize automatic terminal liquidation."
            )
    if reward_version == REWARD_C_LITE_VERSION and use_transaction_penalty:
        raise ValueError("Reward C-lite v1 requires use_transaction_penalty=false.")
    if reward_version == REWARD_C_LITE_V2_VERSION:
        if not use_transaction_penalty:
            raise ValueError("Reward C-lite v2 requires use_transaction_penalty=true.")
        transaction_config = _require(config, "reward.transaction_penalty")
        if not bool(_require(transaction_config, "enabled")):
            raise ValueError(
                "Reward C-lite v2 requires transaction_penalty.enabled=true."
            )
        if float(_require(transaction_config, "lambda_transaction")) < 0.0:
            raise ValueError("lambda_transaction must be non-negative.")
        if not bool(_require(transaction_config, "scale_by_executed_fraction")):
            raise ValueError(
                "Reward C-lite v2 requires transaction penalties to scale by "
                "executed fraction."
            )
        if not bool(_require(transaction_config, "apply_to_first_sale")):
            raise ValueError("Reward C-lite v2 must penalize the first sale.")
        if bool(_require(transaction_config, "apply_to_automatic_terminal_liquidation")):
            raise ValueError(
                "Reward C-lite v2 must not penalize automatic terminal liquidation."
            )
    resolve_training_exploitation_settings(config)


def _get_nested(config: dict, path: str, default: Any = None) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_training_exploitation_settings(
    config: dict,
) -> tuple[str, bool, float | None]:
    exploitation_policy = str(
        _get_nested(config, "training.exploitation_policy", "greedy")
    )
    if exploitation_policy not in {"greedy", "thresholded_greedy"}:
        raise ValueError(
            f"Unsupported training.exploitation_policy={exploitation_policy!r}; "
            "expected 'greedy' or 'thresholded_greedy'."
        )

    thresholded_config = _get_nested(config, "training.thresholded_greedy", {}) or {}
    if not isinstance(thresholded_config, dict):
        raise ValueError("training.thresholded_greedy must be a mapping when provided.")

    thresholded_enabled = bool(
        thresholded_config.get("enabled", exploitation_policy == "thresholded_greedy")
    )
    thresholded_margin: float | None = None
    if exploitation_policy == "thresholded_greedy":
        if not thresholded_enabled:
            raise ValueError(
                "training.thresholded_greedy.enabled must be true when "
                "training.exploitation_policy='thresholded_greedy'."
            )
        if "margin" not in thresholded_config:
            raise ValueError(
                "training.thresholded_greedy.margin is required when "
                "training.exploitation_policy='thresholded_greedy'."
            )
        thresholded_margin = float(thresholded_config["margin"])
        if thresholded_margin < 0.0:
            raise ValueError("training.thresholded_greedy.margin must be non-negative.")
        hold_action_fraction = float(
            thresholded_config.get("hold_action_fraction", 0.0)
        )
        if not approx_equal(hold_action_fraction, 0.0):
            raise ValueError(
                "Only hold_action_fraction=0.0 is supported for thresholded greedy."
            )
        if not bool(thresholded_config.get("use_strict_greater_than", True)):
            raise ValueError(
                "Thresholded greedy training uses a strict '>' comparison; "
                "set use_strict_greater_than=true."
            )
        resolve_thresholded_greedy_first_sale_margin(config, thresholded_margin)

    return exploitation_policy, thresholded_enabled, thresholded_margin


def resolve_thresholded_greedy_first_sale_margin(
    config: dict,
    thresholded_greedy_margin: float | None,
) -> float | None:
    if thresholded_greedy_margin is None:
        return None

    thresholded_config = _get_nested(config, "training.thresholded_greedy", {}) or {}
    if not isinstance(thresholded_config, dict):
        raise ValueError("training.thresholded_greedy must be a mapping when provided.")

    if "first_sale_margin" not in thresholded_config:
        return float(thresholded_greedy_margin)

    first_sale_margin = float(thresholded_config["first_sale_margin"])
    if first_sale_margin < 0.0:
        raise ValueError(
            "training.thresholded_greedy.first_sale_margin must be non-negative."
        )
    return first_sale_margin


def _is_better_metric(
    current_value: float,
    best_value: float | None,
    mode: str,
) -> bool:
    if not np.isfinite(current_value):
        return False
    if best_value is None:
        return True
    if mode == "max":
        return current_value > best_value
    if mode == "min":
        return current_value < best_value
    raise ValueError(f"Unsupported training.best_model_mode={mode!r}.")


def write_best_validation_summary(
    path: Path,
    *,
    best_model_metric: str,
    best_model_mode: str,
    best_metric_value: float,
    best_epoch_idx: int,
    best_epoch_number: int,
    best_global_train_episode_idx: int,
    best_global_step: int,
    model_path: Path,
) -> None:
    payload = {
        "best_model_metric": best_model_metric,
        "best_model_mode": best_model_mode,
        "best_metric_value": best_metric_value,
        "best_epoch_idx": best_epoch_idx,
        "best_epoch_number": best_epoch_number,
        "best_global_train_episode_idx": best_global_train_episode_idx,
        "best_global_step": best_global_step,
        "model_path": _relative_project_path(model_path),
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write("DQN Best Validation Model Summary\n")
        for key, value in payload.items():
            handle.write(f"{key}: {value}\n")


def train_full(config: dict, config_path: Path = CONFIG_PATH) -> dict:
    _validate_config_for_environment_reward(config)

    seed = int(_require(config, "training.seed"))
    gamma = float(_require(config, "training.discount_factor_gamma"))
    learning_rate = float(_require(config, "training.learning_rate"))
    gradient_clip_norm = _require(config, "training.gradient_clip_norm")
    target_update_frequency_steps = int(
        _require(config, "algorithm.target_network.update_frequency_steps")
    )
    batch_size = int(_require(config, "algorithm.replay_buffer.batch_size"))
    replay_capacity = int(_require(config, "algorithm.replay_buffer.capacity"))
    epsilon_start = float(_require(config, "training.epsilon_schedule.start"))
    epsilon_end = float(_require(config, "training.epsilon_schedule.end"))
    epsilon_decay_steps = int(_require(config, "training.epsilon_schedule.decay_steps"))
    checkpoint_frequency_episodes = int(
        _require(config, "training.checkpoint_frequency_episodes")
    )
    evaluation_frequency_episodes = int(
        _require(config, "evaluation.evaluation_frequency_episodes")
    )
    evaluate_during_training = bool(
        _require(config, "evaluation.evaluate_during_training")
    )
    max_eval_episodes = _require(config, "evaluation").get("max_eval_episodes")
    if max_eval_episodes is not None:
        max_eval_episodes = int(max_eval_episodes)
    max_episodes_train = _require(config, "data").get("max_episodes_train")
    if max_episodes_train is not None:
        max_episodes_train = int(max_episodes_train)
        if max_episodes_train <= 0:
            raise ValueError("data.max_episodes_train must be positive when set.")
    num_epochs = int(_get_nested(config, "training.num_epochs", 1))
    if num_epochs <= 0:
        raise ValueError("training.num_epochs must be positive.")
    save_best_validation_model = bool(
        _get_nested(config, "training.save_best_validation_model", False)
    )
    best_model_metric = str(
        _get_nested(
            config,
            "training.best_model_metric",
            "mean_final_after_tax_total_value",
        )
    )
    best_model_mode = str(_get_nested(config, "training.best_model_mode", "max"))
    if best_model_mode not in {"max", "min"}:
        raise ValueError("training.best_model_mode must be 'max' or 'min'.")
    (
        exploitation_policy,
        thresholded_greedy_enabled,
        thresholded_greedy_margin,
    ) = resolve_training_exploitation_settings(config)
    thresholded_greedy_first_sale_margin = resolve_thresholded_greedy_first_sale_margin(
        config,
        thresholded_greedy_margin,
    )

    expected_reward_version = str(_require(config, "reward.expected_info_reward_version"))
    base_reward_version = str(
        _get_nested(config, "reward.base_reward_version", REWARD_A_VERSION)
    )
    transaction_config = _get_nested(config, "reward.transaction_penalty", {}) or {}
    transaction_penalty_enabled = bool(
        _get_nested(config, "reward.use_transaction_penalty", False)
    )
    lambda_transaction = transaction_config.get("lambda_transaction")
    cooldown_config = _get_nested(config, "reward.cooldown_penalty", {}) or {}
    cooldown_penalty_enabled = bool(
        _get_nested(config, "reward.use_cooldown_penalty", False)
    )
    lambda_cooldown = cooldown_config.get("lambda_cooldown")
    cooldown_days = cooldown_config.get("cooldown_days")
    run_name = str(_require(config, "run.name"))
    device = resolve_device(str(_require(config, "training.device")))
    output_dir = _project_path(_require(config, "logging.output_dir"))
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resolved_tax_profile = resolve_tax_profile_from_config(
        config,
        base_dir=PROJECT_ROOT,
    )
    configured_action_fractions = _require(config, "action_space.action_fractions")
    exploration_action_probabilities = load_exploration_action_probabilities(
        config=config,
        num_actions=len(configured_action_fractions),
    )
    exploration_random_action_policy = (
        "uniform"
        if exploration_action_probabilities is None
        else str(config["exploration"]["random_action_policy"])
    )
    exploration_action_probabilities_list = (
        exploration_action_probabilities.tolist()
        if exploration_action_probabilities is not None
        else None
    )
    print(
        "DQN Full Training\n"
        f"config={_relative_project_path(config_path)}\n"
        f"run_name={run_name}\n"
        f"reward_version={expected_reward_version}\n"
        f"output_dir={_relative_project_path(output_dir)}\n"
        f"gamma={gamma}\n"
        f"num_epochs={num_epochs}\n"
        f"exploitation_policy={exploitation_policy}\n"
        f"thresholded_greedy_enabled={thresholded_greedy_enabled}\n"
        f"thresholded_greedy_margin={thresholded_greedy_margin}\n"
        f"thresholded_greedy_first_sale_margin={thresholded_greedy_first_sale_margin}\n"
        f"exploration_policy={exploration_random_action_policy}\n"
        f"epsilon_decay_steps={epsilon_decay_steps}"
    )

    set_seeds(seed)
    rng = np.random.default_rng(seed)
    env = make_env(config)
    state_columns = load_state_columns(
        _project_path(_require(config, "environment.state_schema_path"))
    )
    obs_dim = len(state_columns)

    for fraction in configured_action_fractions:
        action_index_for_fraction(env, float(fraction))

    env._load_episode_index()
    splits = build_episode_splits(env, config)
    full_train_ids = splits["train"]
    train_ids = (
        full_train_ids
        if max_episodes_train is None
        else full_train_ids[:max_episodes_train]
    )
    if not train_ids:
        raise ValueError("No training episodes selected for full training.")
    validation_ids = splits["validation"]
    test_ids = splits["test"]
    n_total = sum(len(ids) for ids in splits.values())

    split_rows = [
        {"episode_id": episode_id, "split": split_name}
        for split_name in ("train", "validation", "test")
        for episode_id in splits[split_name]
    ]
    episode_splits_path = output_dir / "episode_splits.csv"
    pd.DataFrame(split_rows, columns=["episode_id", "split"]).to_csv(
        episode_splits_path,
        index=False,
    )

    hidden_layers = list(_require(config, "algorithm.policy_network.hidden_layers"))
    activation = str(_require(config, "algorithm.policy_network.activation"))
    num_actions = len(env.action_fractions)
    if num_actions != len(configured_action_fractions):
        raise AssertionError(
            "Environment action count does not match configured action_fractions."
        )

    q_net = QNetwork(obs_dim, num_actions, hidden_layers, activation).to(device)
    target_net = QNetwork(obs_dim, num_actions, hidden_layers, activation).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    with torch.no_grad():
        dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device=device)
        dummy_q_values = q_net(dummy_obs)
        assert dummy_q_values.shape == (1, num_actions), (
            f"Expected dummy Q-value shape (1, {num_actions}), got "
            f"{tuple(dummy_q_values.shape)}."
        )
        assert torch.isfinite(dummy_q_values).all(), (
            "Initial dummy Q-values contain NaN or infinite values."
        )

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_capacity, seed=seed)

    train_metric_rows: list[dict[str, Any]] = []
    eval_metric_rows: list[dict[str, Any]] = []
    train_rollout_rows: list[dict[str, Any]] = []
    validation_rollout_frames: list[pd.DataFrame] = []
    losses: list[float] = []
    train_episode_rewards: list[float] = []
    train_episode_lengths: list[int] = []
    nan_reward_count = 0
    inf_reward_count = 0
    nan_loss_count = 0
    inf_loss_count = 0
    global_step = 0
    optimization_steps = 0
    final_epsilon = epsilon_start
    last_loss: float | None = None
    thresholded_to_hold_count = 0
    thresholded_exploitation_step_count = 0

    config_used_path = output_dir / "config_used.yaml"
    save_yaml(config, config_used_path)

    total_training_episode_passes = len(train_ids) * num_epochs
    global_train_episode_idx = 0
    best_validation_model_path = output_dir / "best_validation_model.pt"
    best_validation_summary_path = output_dir / "best_validation_summary.txt"
    best_validation_metric_value: float | None = None
    best_validation_epoch_idx: int | None = None
    best_validation_epoch_number: int | None = None
    best_validation_global_train_episode_idx: int | None = None
    best_validation_global_step: int | None = None

    for epoch_idx in range(num_epochs):
        epoch_number = epoch_idx + 1
        for train_episode_idx, episode_id in enumerate(train_ids, start=1):
            global_train_episode_idx += 1
            obs, _reset_info = env.reset(episode_id=episode_id)
            if obs.shape != (obs_dim,):
                raise AssertionError(
                    f"Observation shape {obs.shape} does not match expected {(obs_dim,)}."
                )
            if not np.isfinite(obs).all():
                raise AssertionError("Initial observation contains NaN or infinite values.")

            done = False
            step_in_episode = 0
            episode_total_reward = 0.0
            max_steps = len(env._current_episode_df) + 5
            last_info: dict[str, Any] | None = None
            episode_last_loss: float | None = None

            while not done:
                if step_in_episode > max_steps:
                    raise RuntimeError(
                        f"Safety cap exceeded for epoch={epoch_number} "
                        f"episode_id={episode_id}: max_steps={max_steps}."
                    )

                epsilon = compute_epsilon(
                    global_step,
                    epsilon_start,
                    epsilon_end,
                    epsilon_decay_steps,
                )
                final_epsilon = epsilon
                action_selection = select_training_action(
                    q_net=q_net,
                    obs=obs,
                    epsilon=epsilon,
                    num_actions=num_actions,
                    rng=rng,
                    device=device,
                    env=env,
                    exploitation_policy=exploitation_policy,
                    thresholded_greedy_margin=thresholded_greedy_margin,
                    thresholded_greedy_first_sale_margin=(
                        thresholded_greedy_first_sale_margin
                    ),
                    exploration_action_probabilities=exploration_action_probabilities,
                )
                action_idx = action_selection.action_idx
                if (
                    exploitation_policy == "thresholded_greedy"
                    and action_selection.raw_greedy_action_idx is not None
                ):
                    thresholded_exploitation_step_count += 1
                if action_selection.was_thresholded_to_hold:
                    thresholded_to_hold_count += 1
                next_obs, reward, done, truncated, info = env.step(action_idx)
                if truncated is not False:
                    raise AssertionError(
                        "TaxAwareEnv returned truncated=True in training."
                    )
                assert_reward_info(reward, info, expected_reward_version)
                if np.isnan(reward):
                    nan_reward_count += 1
                if np.isinf(reward):
                    inf_reward_count += 1
                if not np.isfinite(reward):
                    raise AssertionError("Reward is NaN or infinite.")
                if next_obs.shape != (obs_dim,):
                    raise AssertionError(
                        f"Next observation shape {next_obs.shape} does not match "
                        f"expected {(obs_dim,)}."
                    )
                if not np.isfinite(next_obs).all():
                    raise AssertionError(
                        "Next observation contains NaN or infinite values."
                    )

                replay_buffer.push(obs, action_idx, reward, next_obs, done)
                optimize_result = optimize_dqn(
                    q_net=q_net,
                    target_net=target_net,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device,
                    gradient_clip_norm=gradient_clip_norm,
                )
                loss: float | None = None
                if optimize_result is not None:
                    loss, mean_q, max_q, min_q = optimize_result
                    if np.isnan(loss):
                        nan_loss_count += 1
                    if np.isinf(loss):
                        inf_loss_count += 1
                    if not np.isfinite(loss):
                        raise AssertionError("DQN loss is NaN or infinite.")
                    losses.append(loss)
                    last_loss = loss
                    episode_last_loss = loss
                    optimization_steps += 1
                    train_metric_rows.append(
                        {
                            "epoch_idx": epoch_idx,
                            "epoch_number": epoch_number,
                            "global_train_episode_idx": global_train_episode_idx,
                            "global_step": global_step,
                            "train_episode_idx": train_episode_idx,
                            "episode_id": episode_id,
                            "step_in_episode": step_in_episode,
                            "epsilon": epsilon,
                            "exploitation_policy": exploitation_policy,
                            "thresholded_greedy_margin": (
                                action_selection.thresholded_greedy_margin
                            ),
                            "was_thresholded_to_hold": (
                                action_selection.was_thresholded_to_hold
                            ),
                            "loss": loss,
                            "replay_buffer_size": len(replay_buffer),
                            "mean_q_value": mean_q,
                            "max_q_value": max_q,
                            "min_q_value": min_q,
                            "reward": reward,
                            "done": done,
                        }
                    )

                if (
                    target_update_frequency_steps > 0
                    and global_step > 0
                    and global_step % target_update_frequency_steps == 0
                ):
                    target_net.load_state_dict(q_net.state_dict())

                train_rollout_rows.append(
                    {
                        "split": "train",
                        "epoch_idx": epoch_idx,
                        "epoch_number": epoch_number,
                        "global_train_episode_idx": global_train_episode_idx,
                        "episode_idx": train_episode_idx,
                        "episode_id": episode_id,
                        "step_in_episode": step_in_episode,
                        "global_step": global_step,
                        "date": info.get("date"),
                        "epsilon": epsilon,
                        "action_idx": action_idx,
                        "action_fraction_requested": info.get(
                            "action_fraction_requested"
                        ),
                        "action_fraction_executed": info.get(
                            "action_fraction_executed"
                        ),
                        "raw_greedy_action_idx": (
                            action_selection.raw_greedy_action_idx
                        ),
                        "raw_greedy_action_fraction": (
                            action_selection.raw_greedy_action_fraction
                        ),
                        "selected_action_idx": action_selection.selected_action_idx,
                        "selected_action_fraction": (
                            action_selection.selected_action_fraction
                        ),
                        "hold_q": action_selection.hold_q,
                        "best_sell_q": action_selection.best_sell_q,
                        "q_margin": action_selection.q_margin,
                        "thresholded_greedy_margin": (
                            action_selection.thresholded_greedy_margin
                        ),
                        "was_thresholded_to_hold": (
                            action_selection.was_thresholded_to_hold
                        ),
                        "reward": reward,
                        "reward_A": info.get("reward_A"),
                        "reward_C_lite": info.get("reward_C_lite"),
                        "reward_C_lite_v2": info.get("reward_C_lite_v2"),
                        "transaction_penalty": info.get("transaction_penalty"),
                        "transaction_penalty_applied": info.get(
                            "transaction_penalty_applied"
                        ),
                        "cooldown_penalty": info.get("cooldown_penalty"),
                        "cooldown_penalty_applied": info.get(
                            "cooldown_penalty_applied"
                        ),
                        "days_since_last_sale": info.get("days_since_last_sale"),
                        "sale_count": info.get("sale_count"),
                        "last_sale_date_before_step": info.get(
                            "last_sale_date_before_step"
                        ),
                        "last_sale_date_after_step": info.get(
                            "last_sale_date_after_step"
                        ),
                        "after_tax_total_value": info.get("after_tax_total_value"),
                        "previous_after_tax_total_value": info.get(
                            "previous_after_tax_total_value"
                        ),
                        "realized_after_tax_increment": info.get(
                            "realized_after_tax_increment"
                        ),
                        "cum_realized_after_tax_pnl": info.get(
                            "cum_realized_after_tax_pnl"
                        ),
                        "after_tax_liquidation_value_remaining": info.get(
                            "after_tax_liquidation_value_remaining"
                        ),
                        "sold_fraction": info.get("sold_fraction"),
                        "remaining_fraction": info.get("remaining_fraction"),
                        "tax_regime": info.get("tax_regime"),
                        "after_tax_liquidation_tax_regime": info.get(
                            "after_tax_liquidation_tax_regime"
                        ),
                        "is_automatic_terminal_liquidation": info.get(
                            "is_automatic_terminal_liquidation"
                        ),
                        "terminal_liquidation_executed": info.get(
                            "terminal_liquidation_executed"
                        ),
                        "done": done,
                        "loss": loss,
                    }
                )

                episode_total_reward += float(reward)
                last_info = info
                obs = next_obs
                step_in_episode += 1
                global_step += 1

            train_episode_rewards.append(episode_total_reward)
            train_episode_lengths.append(step_in_episode)
            final_value = (
                float(last_info["after_tax_total_value"])
                if last_info is not None
                else float("nan")
            )
            print(
                f"epoch={epoch_number}/{num_epochs} "
                f"episode={train_episode_idx}/{len(train_ids)} "
                f"global_episode={global_train_episode_idx} "
                f"episode_id={episode_id} "
                f"steps={step_in_episode} "
                f"total_reward={episode_total_reward:.8f} "
                f"final_value={final_value:.8f} "
                f"epsilon={final_epsilon:.6f} "
                f"last_loss={_format_optional_float(episode_last_loss)}"
            )

            should_checkpoint = (
                checkpoint_frequency_episodes > 0
                and global_train_episode_idx % checkpoint_frequency_episodes == 0
            ) or global_train_episode_idx == total_training_episode_passes
            if should_checkpoint:
                checkpoint_path = (
                    checkpoint_dir
                    / f"checkpoint_global_episode_{global_train_episode_idx:06d}.pt"
                )
                torch.save(
                    _checkpoint_payload(
                        q_net,
                        target_net,
                        optimizer,
                        config,
                        obs_dim,
                        num_actions,
                        expected_reward_version,
                        global_step,
                        train_episode_idx,
                        exploration_random_action_policy,
                        exploration_action_probabilities_list,
                        epoch_idx=epoch_idx,
                        epoch_number=epoch_number,
                        global_train_episode_idx=global_train_episode_idx,
                    ),
                    checkpoint_path,
                )

            should_validate = (
                evaluate_during_training
                and validation_ids
                and evaluation_frequency_episodes > 0
                and (
                    global_train_episode_idx % evaluation_frequency_episodes == 0
                    or global_train_episode_idx == total_training_episode_passes
                )
            )
            if should_validate:
                validation_rollouts, validation_metrics = run_validation_policy(
                    env=env,
                    q_net=q_net,
                    episode_ids=validation_ids,
                    expected_reward_version=expected_reward_version,
                    device=device,
                    max_eval_episodes=max_eval_episodes,
                )
                validation_rollouts["evaluation_episode_idx"] = (
                    global_train_episode_idx
                )
                validation_rollouts["global_step"] = global_step
                validation_rollout_frames.append(validation_rollouts)
                eval_row = {
                    "evaluation_episode_idx": global_train_episode_idx,
                    "epoch_idx": epoch_idx,
                    "epoch_number": epoch_number,
                    "global_train_episode_idx": global_train_episode_idx,
                    "global_step": global_step,
                    **validation_metrics,
                }
                eval_metric_rows.append(eval_row)
                print(
                    f"validation_at_global_episode={global_train_episode_idx} "
                    f"epoch={epoch_number} "
                    f"global_step={global_step} "
                    f"mean_reward={validation_metrics['mean_episode_total_reward']:.8f} "
                    "mean_final_value="
                    f"{validation_metrics['mean_final_after_tax_total_value']:.8f} "
                    "terminal_liquidation_frequency="
                    f"{validation_metrics['terminal_liquidation_frequency']:.8f}"
                )

                if save_best_validation_model:
                    if best_model_metric not in validation_metrics:
                        available_metrics = ", ".join(sorted(validation_metrics))
                        raise KeyError(
                            f"Best model metric {best_model_metric!r} is not in "
                            f"validation metrics. Available: {available_metrics}"
                        )
                    current_metric_value = float(validation_metrics[best_model_metric])
                    if _is_better_metric(
                        current_metric_value,
                        best_validation_metric_value,
                        best_model_mode,
                    ):
                        best_validation_metric_value = current_metric_value
                        best_validation_epoch_idx = epoch_idx
                        best_validation_epoch_number = epoch_number
                        best_validation_global_train_episode_idx = (
                            global_train_episode_idx
                        )
                        best_validation_global_step = global_step
                        torch.save(
                            _checkpoint_payload(
                                q_net,
                                target_net,
                                optimizer,
                                config,
                                obs_dim,
                                num_actions,
                                expected_reward_version,
                                global_step,
                                train_episode_idx,
                                exploration_random_action_policy,
                                exploration_action_probabilities_list,
                                epoch_idx=epoch_idx,
                                epoch_number=epoch_number,
                                global_train_episode_idx=global_train_episode_idx,
                                best_model_metric=best_model_metric,
                                best_metric_value=best_validation_metric_value,
                            ),
                            best_validation_model_path,
                        )
                        write_best_validation_summary(
                            best_validation_summary_path,
                            best_model_metric=best_model_metric,
                            best_model_mode=best_model_mode,
                            best_metric_value=best_validation_metric_value,
                            best_epoch_idx=epoch_idx,
                            best_epoch_number=epoch_number,
                            best_global_train_episode_idx=global_train_episode_idx,
                            best_global_step=global_step,
                            model_path=best_validation_model_path,
                        )
                        print(
                            "NEW BEST VALIDATION MODEL saved "
                            f"metric={best_model_metric} "
                            f"value={best_validation_metric_value:.8f}"
                        )

    target_net.load_state_dict(q_net.state_dict())
    final_checkpoint_path = (
        checkpoint_dir / f"checkpoint_global_episode_{global_train_episode_idx:06d}.pt"
    )
    torch.save(
        _checkpoint_payload(
            q_net,
            target_net,
            optimizer,
            config,
            obs_dim,
            num_actions,
            expected_reward_version,
            global_step,
            len(train_ids),
            exploration_random_action_policy,
            exploration_action_probabilities_list,
            epoch_idx=num_epochs - 1,
            epoch_number=num_epochs,
            global_train_episode_idx=global_train_episode_idx,
            best_model_metric=best_model_metric,
            best_metric_value=best_validation_metric_value,
        ),
        final_checkpoint_path,
    )

    train_metrics = pd.DataFrame(train_metric_rows, columns=TRAIN_METRIC_COLUMNS)
    eval_metrics = pd.DataFrame(eval_metric_rows, columns=EVAL_METRIC_COLUMNS)
    train_episode_rollouts = pd.DataFrame(
        train_rollout_rows,
        columns=TRAIN_ROLLOUT_COLUMNS,
    )
    if validation_rollout_frames:
        validation_episode_rollouts = pd.concat(
            validation_rollout_frames,
            ignore_index=True,
        )[VALIDATION_ROLLOUT_COLUMNS]
    else:
        validation_episode_rollouts = pd.DataFrame(
            columns=VALIDATION_ROLLOUT_COLUMNS
        )

    train_metrics_path = output_dir / "train_metrics.csv"
    eval_metrics_path = output_dir / "eval_metrics.csv"
    train_rollouts_path = output_dir / "train_episode_rollouts.csv"
    validation_rollouts_path = output_dir / "validation_episode_rollouts.csv"
    final_model_path = output_dir / "final_model.pt"
    summary_path = output_dir / "training_summary.txt"

    train_metrics.to_csv(train_metrics_path, index=False)
    eval_metrics.to_csv(eval_metrics_path, index=False)
    train_episode_rollouts.to_csv(train_rollouts_path, index=False)
    validation_episode_rollouts.to_csv(validation_rollouts_path, index=False)

    final_payload = _checkpoint_payload(
        q_net,
        target_net,
        optimizer,
        config,
        obs_dim,
        num_actions,
        expected_reward_version,
        global_step,
        len(train_ids),
        exploration_random_action_policy,
        exploration_action_probabilities_list,
        epoch_idx=num_epochs - 1,
        epoch_number=num_epochs,
        global_train_episode_idx=global_train_episode_idx,
        best_model_metric=best_model_metric,
        best_metric_value=best_validation_metric_value,
    )
    torch.save(final_payload, final_model_path)

    final_validation = eval_metric_rows[-1] if eval_metric_rows else {}
    summary = {
        "config_path": _relative_project_path(config_path),
        "run_name": run_name,
        "reward_version": expected_reward_version,
        "base_reward_version": base_reward_version,
        "discount_factor_gamma": gamma,
        "exploitation_policy": exploitation_policy,
        "thresholded_greedy_enabled": thresholded_greedy_enabled,
        "thresholded_greedy_margin": thresholded_greedy_margin,
        "thresholded_greedy_first_sale_margin": thresholded_greedy_first_sale_margin,
        "transaction_penalty_enabled": transaction_penalty_enabled,
        "lambda_transaction": lambda_transaction,
        "cooldown_penalty_enabled": cooldown_penalty_enabled,
        "lambda_cooldown": lambda_cooldown,
        "cooldown_days": cooldown_days,
        "tax_profile_name": resolved_tax_profile["profile_name"],
        "resolved_tax_profile": resolved_tax_profile,
        "num_total_episodes": n_total,
        "num_train_episodes": len(train_ids),
        "num_validation_episodes": len(validation_ids),
        "num_test_episodes": len(test_ids),
        "num_epochs": num_epochs,
        "total_training_episode_passes": total_training_episode_passes,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "total_environment_steps": global_step,
        "num_optimization_steps": optimization_steps,
        "thresholded_exploitation_step_count": thresholded_exploitation_step_count,
        "thresholded_to_hold_count": thresholded_to_hold_count,
        "final_epsilon": final_epsilon,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "last_loss": float(last_loss) if last_loss is not None else None,
        "min_loss": float(np.min(losses)) if losses else None,
        "max_loss": float(np.max(losses)) if losses else None,
        "mean_train_episode_reward": float(np.mean(train_episode_rewards))
        if train_episode_rewards
        else None,
        "median_train_episode_reward": float(np.median(train_episode_rewards))
        if train_episode_rewards
        else None,
        "mean_train_episode_length": float(np.mean(train_episode_lengths))
        if train_episode_lengths
        else None,
        "nan_reward_count": nan_reward_count,
        "inf_reward_count": inf_reward_count,
        "nan_loss_count": nan_loss_count,
        "inf_loss_count": inf_loss_count,
        "final_validation_mean_episode_reward": final_validation.get(
            "mean_episode_total_reward"
        ),
        "final_validation_median_episode_reward": final_validation.get(
            "median_episode_total_reward"
        ),
        "final_validation_mean_final_after_tax_total_value": final_validation.get(
            "mean_final_after_tax_total_value"
        ),
        "final_validation_terminal_liquidation_frequency": final_validation.get(
            "terminal_liquidation_frequency"
        ),
        "best_model_metric": best_model_metric,
        "best_model_mode": best_model_mode,
        "best_validation_metric_value": best_validation_metric_value,
        "best_validation_epoch_number": best_validation_epoch_number,
        "best_validation_global_train_episode_idx": (
            best_validation_global_train_episode_idx
        ),
        "final_model_path": _relative_project_path(final_model_path),
        "best_validation_model_path": (
            _relative_project_path(best_validation_model_path)
            if best_validation_metric_value is not None
            else None
        ),
        "train_metrics_path": _relative_project_path(train_metrics_path),
        "eval_metrics_path": _relative_project_path(eval_metrics_path),
        "train_episode_rollouts_path": _relative_project_path(train_rollouts_path),
        "validation_episode_rollouts_path": _relative_project_path(
            validation_rollouts_path
        ),
        "episode_splits_path": _relative_project_path(episode_splits_path),
        "exploration_random_action_policy": exploration_random_action_policy,
        "exploration_action_probabilities": exploration_action_probabilities_list,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("DQN Full Training Summary\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")

    summary["config_used_path"] = _relative_project_path(config_used_path)
    summary["summary_path"] = _relative_project_path(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN using the environment reward from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=(
            "Training config path. Defaults to "
            f"{_relative_project_path(CONFIG_PATH)}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _project_path(args.config)
    config = load_yaml(config_path)
    train_full(config, config_path=config_path)
    print(f"DQN FULL TRAINING COMPLETE reward_version={config['reward']['version']}")


if __name__ == "__main__":
    main()
