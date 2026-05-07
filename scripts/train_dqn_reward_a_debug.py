"""Debug DQN training pipeline for frozen Reward A.

This script verifies that a minimal DQN agent can interact with TaxAwareEnv
using the frozen Reward A contract. It is intentionally small and debug-only:
it does not implement baseline evaluation, validation splits, simulation, or
full training.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError(
        "pyyaml is required to run scripts/train_dqn_reward_a_debug.py. "
        "Install it with: pip install pyyaml"
    ) from exc

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError(
        "PyTorch is required to run scripts/train_dqn_reward_a_debug.py. "
        "Install torch before running the debug DQN script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.tax_aware_env import TaxAwareEnv  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "train_reward_a_v1.yaml"
REWARD_ASSERT_TOL = 1e-8

TRAIN_METRIC_COLUMNS = [
    "global_step",
    "episode_idx",
    "episode_id",
    "step_in_episode",
    "epsilon",
    "loss",
    "replay_buffer_size",
    "mean_q_value",
    "max_q_value",
    "min_q_value",
    "reward",
    "done",
]

ROLLOUT_COLUMNS = [
    "episode_idx",
    "episode_id",
    "step_in_episode",
    "global_step",
    "date",
    "epsilon",
    "action_idx",
    "action_fraction_requested",
    "action_fraction_executed",
    "reward",
    "reward_A",
    "after_tax_total_value",
    "previous_after_tax_total_value",
    "realized_after_tax_increment",
    "cum_realized_after_tax_pnl",
    "after_tax_liquidation_value_remaining",
    "sold_fraction",
    "remaining_fraction",
    "tax_regime",
    "after_tax_liquidation_tax_regime",
    "terminal_liquidation_executed",
    "done",
    "loss",
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


def _approx_equal(left: float, right: float, *, tol: float = REWARD_ASSERT_TOL) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


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

    if isinstance(payload, dict):
        columns = payload.get("allowed_state_columns")
    else:
        columns = payload

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
    tax_config = _require(config, "tax_profile")
    seed = int(_require(config, "training.seed"))

    return TaxAwareEnv(
        parquet_path=parquet_path,
        state_columns=state_columns,
        action_fractions=action_fractions,
        tax_config=tax_config,
        seed=seed,
    )


def action_index_for_fraction(env: TaxAwareEnv, target_fraction: float) -> int:
    for idx, action_fraction in enumerate(env.action_fractions):
        if _approx_equal(float(action_fraction), float(target_fraction)):
            return idx
    raise ValueError(
        f"Action fraction {target_fraction} is not present in "
        f"env.action_fractions={env.action_fractions}."
    )


def assert_reward_info(
    reward: float,
    info: dict,
    expected_reward_version: str,
) -> None:
    assert info["reward_version"] == expected_reward_version, (
        f"Unexpected reward_version={info['reward_version']!r}; "
        f"expected {expected_reward_version!r}."
    )
    assert _approx_equal(reward, info["reward"]), "reward != info['reward']"
    assert _approx_equal(reward, info["reward_A"]), "reward != info['reward_A']"
    assert _approx_equal(
        reward,
        info["after_tax_total_value"] - info["previous_after_tax_total_value"],
    ), "Reward A identity failed."
    assert _approx_equal(
        info["after_tax_total_value"],
        info["cum_realized_after_tax_pnl"]
        + info["after_tax_liquidation_value_remaining"],
    ), "After-tax total value decomposition failed."
    assert np.isfinite(reward), "Reward is not finite."


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


def select_action(
    q_net: QNetwork,
    obs: np.ndarray,
    epsilon: float,
    num_actions: int,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(obs_tensor)
        if q_values.shape != (1, num_actions):
            raise AssertionError(
                f"QNetwork output shape {tuple(q_values.shape)} does not match "
                f"(1, {num_actions})."
            )
        if not torch.isfinite(q_values).all():
            raise AssertionError("Q-values contain NaN or infinite values.")
        if rng.random() < epsilon:
            return int(rng.integers(0, num_actions))
        return int(torch.argmax(q_values, dim=1).item())


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
) -> float | None:
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

    return float(loss.detach().cpu().item())


def _q_stats(q_net: QNetwork, obs: np.ndarray, device: torch.device) -> tuple[float, float, float]:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(obs_tensor)
        if not torch.isfinite(q_values).all():
            raise AssertionError("Q-values contain NaN or infinite values.")
    q_numpy = q_values.detach().cpu().numpy()
    return float(q_numpy.mean()), float(q_numpy.max()), float(q_numpy.min())


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.8f}"


def _checkpoint_payload(
    q_net: QNetwork,
    target_net: QNetwork,
    config: dict,
    obs_dim: int,
    num_actions: int,
    expected_reward_version: str,
    global_step: int,
    episode_idx: int | None,
) -> dict:
    return {
        "model_state_dict": q_net.state_dict(),
        "target_model_state_dict": target_net.state_dict(),
        "config": config,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "reward_version": expected_reward_version,
        "global_step": global_step,
        "episode_idx": episode_idx,
    }


def _final_model_payload(
    q_net: QNetwork,
    target_net: QNetwork,
    config: dict,
    obs_dim: int,
    num_actions: int,
    expected_reward_version: str,
) -> dict:
    return {
        "model_state_dict": q_net.state_dict(),
        "target_model_state_dict": target_net.state_dict(),
        "config": config,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "reward_version": expected_reward_version,
    }


def train_debug(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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
    debug_episodes = int(_require(config, "training.debug.episodes"))
    checkpoint_frequency_episodes = int(
        _require(config, "training.checkpoint_frequency_episodes")
    )
    algorithm_name = str(_require(config, "algorithm.name")).lower()
    framework = str(_require(config, "algorithm.framework")).lower()
    optimizer_name = str(_require(config, "training.optimizer")).lower()
    if algorithm_name != "dqn":
        raise ValueError(f"Debug script only supports algorithm.name='dqn', got {algorithm_name!r}.")
    if framework != "torch":
        raise ValueError(f"Debug script only supports algorithm.framework='torch', got {framework!r}.")
    if optimizer_name != "adam":
        raise ValueError(f"Debug script only supports training.optimizer='adam', got {optimizer_name!r}.")

    reward_version = str(_require(config, "reward.version"))
    expected_reward_version = str(_require(config, "reward.expected_info_reward_version"))
    if reward_version != expected_reward_version:
        raise ValueError(
            "Config reward.version and reward.expected_info_reward_version "
            f"must match, got {reward_version!r} and {expected_reward_version!r}."
        )
    device = resolve_device(str(_require(config, "training.device")))

    debug_dir = _project_path(_require(config, "logging.output_dir")) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = debug_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(seed)
    rng = np.random.default_rng(seed)
    env = make_env(config)
    state_columns = load_state_columns(
        _project_path(_require(config, "environment.state_schema_path"))
    )
    obs_dim = len(state_columns)
    num_actions = len(env.action_fractions)

    for fraction in _require(config, "action_space.action_fractions"):
        action_index_for_fraction(env, float(fraction))

    env._load_episode_index()
    episode_ids = sorted(env._episode_index.keys())[:debug_episodes]
    if not episode_ids:
        raise ValueError("No debug episodes available from environment episode index.")

    hidden_layers = list(_require(config, "algorithm.policy_network.hidden_layers"))
    activation = str(_require(config, "algorithm.policy_network.activation"))
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

    metrics_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    losses: list[float] = []
    nan_reward_count = 0
    inf_reward_count = 0
    nan_loss_count = 0
    inf_loss_count = 0
    checkpoint_paths: list[str] = []
    global_step = 0
    optimization_steps = 0
    final_epsilon = epsilon_start

    for episode_idx, episode_id in enumerate(episode_ids, start=1):
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
        last_loss: float | None = None

        while not done:
            if step_in_episode > max_steps:
                raise RuntimeError(
                    f"Safety cap exceeded for episode_id={episode_id}: "
                    f"max_steps={max_steps}."
                )

            epsilon = compute_epsilon(
                global_step,
                epsilon_start,
                epsilon_end,
                epsilon_decay_steps,
            )
            final_epsilon = epsilon
            action_idx = select_action(
                q_net,
                obs,
                epsilon,
                num_actions,
                rng,
                device,
            )
            next_obs, reward, done, truncated, info = env.step(action_idx)
            if truncated is not False:
                raise AssertionError("TaxAwareEnv returned truncated=True in debug run.")
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
                raise AssertionError("Next observation contains NaN or infinite values.")

            replay_buffer.push(obs, action_idx, reward, next_obs, done)
            loss = optimize_dqn(
                q_net=q_net,
                target_net=target_net,
                replay_buffer=replay_buffer,
                optimizer=optimizer,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
                gradient_clip_norm=gradient_clip_norm,
            )
            if loss is not None:
                if np.isnan(loss):
                    nan_loss_count += 1
                if np.isinf(loss):
                    inf_loss_count += 1
                if not np.isfinite(loss):
                    raise AssertionError("DQN loss is NaN or infinite.")
                losses.append(loss)
                optimization_steps += 1
                last_loss = loss
                mean_q, max_q, min_q = _q_stats(q_net, obs, device)
                metrics_rows.append(
                    {
                        "global_step": global_step,
                        "episode_idx": episode_idx,
                        "episode_id": episode_id,
                        "step_in_episode": step_in_episode,
                        "epsilon": epsilon,
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

            rollout_rows.append(
                {
                    "episode_idx": episode_idx,
                    "episode_id": episode_id,
                    "step_in_episode": step_in_episode,
                    "global_step": global_step,
                    "date": info.get("date"),
                    "epsilon": epsilon,
                    "action_idx": action_idx,
                    "action_fraction_requested": info.get(
                        "action_fraction_requested"
                    ),
                    "action_fraction_executed": info.get("action_fraction_executed"),
                    "reward": reward,
                    "reward_A": info.get("reward_A"),
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

        episode_rewards.append(episode_total_reward)
        episode_lengths.append(step_in_episode)
        final_value = (
            float(last_info["after_tax_total_value"]) if last_info is not None else float("nan")
        )
        print(
            f"episode={episode_idx}/{len(episode_ids)} "
            f"episode_id={episode_id} "
            f"steps={step_in_episode} "
            f"total_reward={episode_total_reward:.8f} "
            f"final_value={final_value:.8f} "
            f"epsilon={final_epsilon:.6f} "
            f"last_loss={_format_optional_float(last_loss)}"
        )

        should_checkpoint = (
            checkpoint_frequency_episodes > 0
            and episode_idx % checkpoint_frequency_episodes == 0
        ) or episode_idx == len(episode_ids)
        if should_checkpoint:
            checkpoint_path = (
                checkpoint_dir / f"checkpoint_episode_{episode_idx:04d}.pt"
            )
            torch.save(
                _checkpoint_payload(
                    q_net,
                    target_net,
                    config,
                    obs_dim,
                    num_actions,
                    expected_reward_version,
                    global_step,
                    episode_idx,
                ),
                checkpoint_path,
            )
            checkpoint_paths.append(str(checkpoint_path.relative_to(PROJECT_ROOT)))

    target_net.load_state_dict(q_net.state_dict())

    train_metrics = pd.DataFrame(metrics_rows, columns=TRAIN_METRIC_COLUMNS)
    episode_rollouts = pd.DataFrame(rollout_rows, columns=ROLLOUT_COLUMNS)

    config_copy_path = debug_dir / "debug_config_used.yaml"
    train_metrics_path = debug_dir / "train_metrics.csv"
    episode_rollouts_path = debug_dir / "episode_rollouts.csv"
    model_path = debug_dir / "final_debug_model.pt"
    summary_path = debug_dir / "debug_training_summary.txt"

    save_yaml(config, config_copy_path)
    train_metrics.to_csv(train_metrics_path, index=False)
    episode_rollouts.to_csv(episode_rollouts_path, index=False)
    torch.save(
        _final_model_payload(
            q_net,
            target_net,
            config,
            obs_dim,
            num_actions,
            expected_reward_version,
        ),
        model_path,
    )

    summary = {
        "reward_version": expected_reward_version,
        "num_debug_episodes": len(episode_ids),
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "total_environment_steps": global_step,
        "num_optimization_steps": optimization_steps,
        "final_epsilon": final_epsilon,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "last_loss": float(losses[-1]) if losses else None,
        "min_loss": float(np.min(losses)) if losses else None,
        "max_loss": float(np.max(losses)) if losses else None,
        "mean_episode_reward": float(np.mean(episode_rewards))
        if episode_rewards
        else None,
        "mean_episode_length": float(np.mean(episode_lengths))
        if episode_lengths
        else None,
        "nan_reward_count": nan_reward_count,
        "inf_reward_count": inf_reward_count,
        "nan_loss_count": nan_loss_count,
        "inf_loss_count": inf_loss_count,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "train_metrics_path": str(train_metrics_path.relative_to(PROJECT_ROOT)),
        "episode_rollouts_path": str(
            episode_rollouts_path.relative_to(PROJECT_ROOT)
        ),
        "checkpoint_paths": checkpoint_paths,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("DQN Reward A Debug Training Summary\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")

    summary["debug_config_used_path"] = str(config_copy_path.relative_to(PROJECT_ROOT))
    summary["summary_path"] = str(summary_path.relative_to(PROJECT_ROOT))
    return train_metrics, episode_rollouts, summary


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    train_debug(config)
    print("DQN REWARD A DEBUG TRAINING COMPLETE")


if __name__ == "__main__":
    main()
