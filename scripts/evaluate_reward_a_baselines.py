"""Evaluate trained DQN and simple baseline liquidation policies."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError(
        "PyYAML is required to run scripts/evaluate_reward_a_baselines.py. "
        "Install it with: pip install pyyaml"
    ) from exc

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - exercised only when missing.
    raise ImportError(
        "PyTorch is required to run scripts/evaluate_reward_a_baselines.py. "
        "Install torch before running the Reward A baseline evaluator."
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
MAX_VALIDATION_EPISODES = None
MAX_TEST_EPISODES = None

POLICY_NAMES = [
    "trained_dqn_greedy",
    "trained_dqn_thresholded_margin_0p001",
    "trained_dqn_thresholded_margin_0p005",
    "trained_dqn_thresholded_margin_0p010",
    "trained_dqn_thresholded_margin_0p015",
    "trained_dqn_thresholded_margin_0p020",
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
    "sell_quarters_over_time",
    "random_policy",
]

THRESHOLDED_DQN_MARGINS = {
    "trained_dqn_thresholded_margin_0p001": 0.001,
    "trained_dqn_thresholded_margin_0p005": 0.005,
    "trained_dqn_thresholded_margin_0p010": 0.010,
    "trained_dqn_thresholded_margin_0p015": 0.015,
    "trained_dqn_thresholded_margin_0p020": 0.020,
}


def _format_margin_token(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "p")


def _parse_margin_token(token: str) -> float:
    return float(token.replace("p", "."))


def first_sale_thresholded_policy_name(
    first_sale_margin: float,
    margin: float,
) -> str:
    return (
        "trained_dqn_first_sale_margin_"
        f"{_format_margin_token(first_sale_margin)}"
        "_normal_"
        f"{_format_margin_token(margin)}"
    )


STEP_ROLLOUT_COLUMNS = [
    "split",
    "policy_name",
    "episode_id",
    "step_in_episode",
    "date",
    "tax_transition_date",
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
    "realized_pre_tax_increment",
    "tax_paid",
    "realized_after_tax_increment",
    "cum_realized_after_tax_pnl",
    "after_tax_liquidation_value_remaining",
    "sold_fraction",
    "remaining_fraction",
    "tax_regime",
    "applicable_tax_rate",
    "after_tax_liquidation_tax_regime",
    "is_automatic_terminal_liquidation",
    "terminal_liquidation_executed",
    "terminal_liquidation_tax_rate",
    "terminal_liquidation_pre_tax_increment",
    "terminal_liquidation_tax_paid",
    "terminal_liquidation_after_tax_increment",
    "done",
]

EPISODE_METRIC_COLUMNS = [
    "split",
    "policy_name",
    "episode_id",
    "episode_total_reward",
    "episode_total_reward_A",
    "episode_total_reward_C_lite",
    "episode_total_reward_C_lite_v2",
    "episode_total_transaction_penalty",
    "episode_total_cooldown_penalty",
    "transaction_penalty_frequency",
    "mean_transaction_penalty_when_applied",
    "cooldown_penalty_frequency",
    "mean_cooldown_penalty_when_applied",
    "episode_initial_after_tax_total_value",
    "episode_final_after_tax_total_value",
    "episode_realized_after_tax_pnl",
    "episode_steps",
    "episode_terminal_liquidation_executed",
    "episode_final_remaining_fraction",
    "episode_final_sold_fraction",
    "episode_full_liquidation",
    "episode_cut_occurred",
    "first_cut_step",
    "first_cut_date",
    "first_cut_fraction_executed",
    "days_to_first_sale",
    "steps_to_first_sale",
    "first_sale_date",
    "first_sale_before_tax_transition",
    "pct_episode_position_sold_short_term",
    "pct_episode_position_sold_long_term",
    "total_position_sold_short_term",
    "total_position_sold_long_term",
    "mean_effective_tax_rate_on_sales",
    "total_tax_paid",
    "total_positive_taxable_pre_tax_increment",
    "terminal_liquidation_fraction",
    "total_terminal_liquidation_tax_paid",
    "total_transaction_penalty",
    "total_cooldown_penalty",
    "num_discretionary_sales",
    "num_cooldown_penalized_sales",
    "sold_before_tax_transition_flag",
    "reached_tax_transition_before_first_sale_flag",
]

SUMMARY_COLUMNS = [
    "split",
    "policy_name",
    "num_episodes",
    "mean_episode_total_reward",
    "mean_episode_total_reward_A",
    "mean_episode_total_reward_C_lite",
    "mean_episode_total_reward_C_lite_v2",
    "mean_episode_total_transaction_penalty",
    "mean_episode_total_cooldown_penalty",
    "median_episode_total_reward",
    "std_episode_total_reward",
    "min_episode_total_reward",
    "max_episode_total_reward",
    "p05_episode_total_reward",
    "p95_episode_total_reward",
    "mean_final_after_tax_total_value",
    "median_final_after_tax_total_value",
    "mean_realized_after_tax_pnl",
    "median_realized_after_tax_pnl",
    "mean_episode_steps",
    "median_episode_steps",
    "terminal_liquidation_frequency",
    "full_liquidation_frequency",
    "cut_frequency",
    "sell_immediately_frequency",
    "mean_first_cut_step",
    "median_first_cut_step",
    "mean_first_cut_fraction_executed",
    "average_days_to_first_sale",
    "median_days_to_first_sale",
    "average_steps_to_first_sale",
    "pct_episodes_with_first_sale_before_tax_transition",
    "pct_episodes_sold_before_tax_transition",
    "mean_pct_position_sold_short_term",
    "mean_pct_position_sold_long_term",
    "mean_effective_tax_rate",
    "mean_total_tax_paid",
    "mean_transaction_penalty",
    "mean_cooldown_penalty",
    "mean_num_discretionary_sales",
    "mean_num_cooldown_penalized_sales",
    "mean_excess_value_vs_sell_immediately",
    "mean_excess_value_vs_hold_to_terminal",
]

SUMMARY_CONTEXT: dict[str, Any] = {}


def _require(config: dict, path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing required config value: {path}")
        current = current[key]
    return current


def _relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_config)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but CUDA is not available.")
    return device


def _validate_reward_config(config: dict) -> None:
    reward_version = str(_require(config, "reward.version"))
    expected_reward_version = str(_require(config, "reward.expected_info_reward_version"))
    if expected_reward_version != reward_version:
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
        raise ValueError("Baseline evaluation requires use_environment_reward=true.")

    excluded_flags = [
        "reward.use_drawdown_penalty",
        "reward.use_explicit_tax_saving_bonus",
        "reward.use_reward_clipping",
        "reward.use_reward_normalization",
    ]
    enabled_flags = [flag for flag in excluded_flags if bool(_require(config, flag))]
    if enabled_flags:
        raise ValueError(
            "Baseline evaluation must not enable unsupported reward options: "
            f"{enabled_flags}"
        )
    use_cooldown_penalty = bool(_require(config, "reward.use_cooldown_penalty"))
    use_transaction_penalty = bool(
        config.get("reward", {}).get("use_transaction_penalty", False)
    )
    if reward_version == REWARD_A_VERSION and (
        use_cooldown_penalty or use_transaction_penalty
    ):
        raise ValueError(
            "Reward A evaluation requires transaction and cooldown penalties off."
        )
    if reward_version in {REWARD_C_LITE_VERSION, REWARD_C_LITE_V2_VERSION}:
        if str(_require(config, "reward.base_reward_version")) != REWARD_A_VERSION:
            raise ValueError("Reward C-lite requires base_reward_version=Reward A.")
        if not use_cooldown_penalty:
            raise ValueError("Reward C-lite requires use_cooldown_penalty=true.")
        cooldown_config = _require(config, "reward.cooldown_penalty")
        if not bool(_require(cooldown_config, "enabled")):
            raise ValueError("Reward C-lite requires cooldown_penalty.enabled=true.")
        if int(_require(cooldown_config, "cooldown_days")) <= 0:
            raise ValueError("cooldown_days must be positive.")
        if float(_require(cooldown_config, "lambda_cooldown")) < 0.0:
            raise ValueError("lambda_cooldown must be non-negative.")
        if bool(_require(cooldown_config, "penalize_first_sale")):
            raise ValueError("Reward C-lite must not penalize the first sale.")
        if bool(
            cooldown_config.get("apply_to_automatic_terminal_liquidation", False)
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

    resolve_tax_profile_from_config(config, base_dir=PROJECT_ROOT)


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


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def make_env(config: dict) -> TaxAwareEnv:
    env_config = _require(config, "environment")
    parquet_path = resolve_project_path(_require(env_config, "parquet_path"))
    schema_path = resolve_project_path(_require(env_config, "state_schema_path"))
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


def assert_tax_profiles_match(
    saved_tax_profile: dict[str, Any],
    current_tax_profile: dict[str, Any],
    *,
    artifact_path: Path,
) -> None:
    fields = (
        "profile_name",
        "short_term_rate",
        "long_term_rate",
        "niit_rate",
        "apply_niit",
    )
    mismatches: list[str] = []
    for field in fields:
        saved_value = saved_tax_profile.get(field)
        current_value = current_tax_profile.get(field)
        if saved_value is None or current_value is None:
            if saved_value != current_value:
                mismatches.append(
                    f"{field}: saved={saved_value!r}, current={current_value!r}"
                )
            continue
        if isinstance(current_value, bool):
            if not isinstance(saved_value, bool) or saved_value != current_value:
                mismatches.append(
                    f"{field}: saved={saved_value!r}, current={current_value!r}"
                )
        elif isinstance(current_value, (int, float)):
            try:
                saved_number = float(saved_value)
                current_number = float(current_value)
            except (TypeError, ValueError):
                mismatches.append(
                    f"{field}: saved={saved_value!r}, current={current_value!r}"
                )
                continue
            if not approx_equal(saved_number, current_number):
                mismatches.append(
                    f"{field}: saved={saved_value!r}, current={current_value!r}"
                )
        elif saved_value != current_value:
            mismatches.append(
                f"{field}: saved={saved_value!r}, current={current_value!r}"
            )

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Trained model tax profile does not match the current evaluation "
            f"config for {artifact_path}: {mismatch_text}. Rerun training before "
            "evaluating baselines."
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
        assert reward_C_lite is not None, "Reward C-lite v2 missing reward_C_lite."
        assert reward_C_lite_v2 is not None, (
            "Reward C-lite v2 missing info['reward_C_lite_v2']."
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


def action_index_for_fraction(env: TaxAwareEnv, target_fraction: float) -> int:
    for idx, action_fraction in enumerate(env.action_fractions):
        if approx_equal(float(action_fraction), float(target_fraction)):
            return idx
    raise ValueError(
        f"Action fraction {target_fraction} is not present in "
        f"env.action_fractions={env.action_fractions}."
    )


def load_episode_splits(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Episode split file not found: {path}")

    splits_df = pd.read_csv(path)
    required_columns = {"episode_id", "split"}
    missing_columns = required_columns - set(splits_df.columns)
    if missing_columns:
        raise ValueError(
            f"Episode split file is missing required columns: {sorted(missing_columns)}"
        )
    if splits_df["episode_id"].isna().any() or splits_df["split"].isna().any():
        raise ValueError("Episode split file contains missing episode_id or split.")

    splits_df = splits_df.copy()
    splits_df["episode_id"] = splits_df["episode_id"].astype(str)
    splits_df["split"] = splits_df["split"].astype(str)

    splits = {
        split_name: group["episode_id"].tolist()
        for split_name, group in splits_df.groupby("split", sort=False)
    }
    if not splits.get("validation"):
        raise ValueError("Validation split is empty in episode_splits.csv.")
    if not splits.get("test"):
        raise ValueError("Test split is empty in episode_splits.csv.")
    return splits


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
        input_dim = int(obs_dim)
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, int(num_actions)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_trained_q_network(
    config: dict,
    model_path: Path,
    obs_dim: int,
    num_actions: int,
    device: torch.device,
) -> QNetwork:
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model artifact not found: {model_path}")

    hidden_layers = list(_require(config, "algorithm.policy_network.hidden_layers"))
    activation = str(_require(config, "algorithm.policy_network.activation"))
    expected_reward_version = str(_require(config, "reward.expected_info_reward_version"))
    q_net = QNetwork(obs_dim, num_actions, hidden_layers, activation).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - for older torch versions.
        checkpoint = torch.load(model_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected model artifact to contain a dict: {model_path}")
    required_keys = {"model_state_dict", "obs_dim", "num_actions", "reward_version"}
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(
            f"Model artifact is missing required keys: {sorted(missing_keys)}"
        )

    assert checkpoint["reward_version"] == expected_reward_version, (
        f"Saved reward_version={checkpoint['reward_version']!r}; "
        f"expected {expected_reward_version!r}."
    )
    assert int(checkpoint["obs_dim"]) == int(obs_dim), (
        f"Saved obs_dim={checkpoint['obs_dim']}; current obs_dim={obs_dim}."
    )
    assert int(checkpoint["num_actions"]) == int(num_actions), (
        "Saved num_actions="
        f"{checkpoint['num_actions']}; current num_actions={num_actions}."
    )
    current_tax_profile = resolve_tax_profile_from_config(
        config,
        base_dir=PROJECT_ROOT,
    )
    saved_tax_profile = checkpoint.get("resolved_tax_profile")
    if saved_tax_profile is None:
        saved_config = checkpoint.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError(
                "Model artifact does not include a saved tax profile or config: "
                f"{model_path}"
            )
        saved_tax_profile = resolve_tax_profile_from_config(
            saved_config,
            base_dir=PROJECT_ROOT,
        )
    if not isinstance(saved_tax_profile, dict):
        raise ValueError(
            f"Model artifact resolved_tax_profile must be a mapping: {model_path}"
        )
    assert_tax_profiles_match(
        saved_tax_profile=saved_tax_profile,
        current_tax_profile=current_tax_profile,
        artifact_path=model_path,
    )

    q_net.load_state_dict(checkpoint["model_state_dict"])
    q_net.eval()

    with torch.no_grad():
        dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device=device)
        dummy_q_values = q_net(dummy_obs)
        assert dummy_q_values.shape == (1, num_actions), (
            f"Expected dummy Q-value shape (1, {num_actions}), got "
            f"{tuple(dummy_q_values.shape)}."
        )
        assert torch.isfinite(dummy_q_values).all(), (
            "Loaded Q-network produced NaN or infinite values for dummy input."
        )

    return q_net


def select_dqn_greedy_action(
    q_net: QNetwork,
    obs: np.ndarray,
    num_actions: int,
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
        return int(torch.argmax(q_values, dim=1).item())


def threshold_margin_for_policy(policy_name: str) -> float | None:
    if policy_name in THRESHOLDED_DQN_MARGINS:
        return THRESHOLDED_DQN_MARGINS[policy_name]
    if policy_name.startswith("trained_dqn_thresholded"):
        known = ", ".join(sorted(THRESHOLDED_DQN_MARGINS))
        raise ValueError(
            f"Unknown thresholded DQN policy {policy_name!r}. Known policies: {known}"
        )
    return None


def first_sale_threshold_settings_for_policy(
    policy_name: str,
) -> tuple[float, float] | None:
    prefix = "trained_dqn_first_sale_margin_"
    normal_separator = "_normal_"
    if not policy_name.startswith(prefix):
        return None

    remainder = policy_name.removeprefix(prefix)
    if normal_separator not in remainder:
        raise ValueError(
            f"Unknown first-sale thresholded DQN policy {policy_name!r}."
        )
    first_sale_token, margin_token = remainder.split(normal_separator, maxsplit=1)
    first_sale_margin = _parse_margin_token(first_sale_token)
    margin = _parse_margin_token(margin_token)
    if margin < 0.0 or first_sale_margin < 0.0:
        raise ValueError(
            f"First-sale thresholded DQN policy {policy_name!r} has a negative margin."
        )
    return margin, first_sale_margin


def threshold_settings_for_policy(
    policy_name: str,
) -> tuple[float, float | None] | None:
    margin = threshold_margin_for_policy(policy_name)
    if margin is not None:
        return margin, None
    first_sale_settings = first_sale_threshold_settings_for_policy(policy_name)
    if first_sale_settings is not None:
        return first_sale_settings
    return None


def resolve_evaluation_policy_names(config: dict) -> list[str]:
    policy_names = list(POLICY_NAMES)
    evaluation_config = config.get("evaluation", {})
    if not isinstance(evaluation_config, dict):
        raise ValueError("evaluation must be a mapping when provided.")

    first_sale_config = evaluation_config.get("first_sale_margin_policies", {}) or {}
    if not isinstance(first_sale_config, dict):
        raise ValueError(
            "evaluation.first_sale_margin_policies must be a mapping when provided."
        )
    if bool(first_sale_config.get("enabled", False)):
        margin = float(first_sale_config.get("margin", 0.020))
        if margin < 0.0:
            raise ValueError(
                "evaluation.first_sale_margin_policies.margin must be non-negative."
            )
        first_sale_margins = first_sale_config.get("first_sale_margins")
        if first_sale_margins is None:
            raise ValueError(
                "evaluation.first_sale_margin_policies.first_sale_margins is "
                "required when enabled=true."
            )
        if not isinstance(first_sale_margins, list):
            raise ValueError(
                "evaluation.first_sale_margin_policies.first_sale_margins must be a list."
            )
        for first_sale_margin in first_sale_margins:
            first_sale_margin = float(first_sale_margin)
            if first_sale_margin < 0.0:
                raise ValueError(
                    "evaluation.first_sale_margin_policies.first_sale_margins "
                    "must be non-negative."
                )
            policy_name = first_sale_thresholded_policy_name(
                first_sale_margin=first_sale_margin,
                margin=margin,
            )
            if policy_name not in policy_names:
                policy_names.append(policy_name)

    return policy_names


def is_before_first_discretionary_sale(env: TaxAwareEnv) -> bool:
    sold_fraction = getattr(env, "_sold_fraction", None)
    if sold_fraction is not None:
        return bool(np.isclose(float(sold_fraction), 0.0, rtol=0.0, atol=1e-12))

    remaining_fraction = getattr(env, "_remaining_fraction", None)
    if remaining_fraction is not None:
        return bool(np.isclose(float(remaining_fraction), 1.0, rtol=0.0, atol=1e-12))

    sale_count = getattr(env, "_sale_count", None)
    if sale_count is not None:
        return int(sale_count) == 0

    return False


def active_threshold_margin(
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


def select_dqn_thresholded_greedy_action(
    q_net: QNetwork,
    obs: np.ndarray,
    num_actions: int,
    device: torch.device,
    env: TaxAwareEnv,
    margin: float,
    first_sale_margin: float | None = None,
) -> int:
    if margin < 0.0:
        raise ValueError(f"Threshold margin must be non-negative, got {margin}.")
    active_margin = active_threshold_margin(
        env=env,
        margin=margin,
        first_sale_margin=first_sale_margin,
    )

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

        q_values = q_values_tensor.squeeze(0).detach().cpu().numpy()

    hold_idx = action_index_for_fraction(env, 0.0)
    sell_indices = [idx for idx in range(num_actions) if idx != hold_idx]
    if not sell_indices:
        raise ValueError("Thresholded greedy policy requires at least one sell action.")

    best_sell_idx = max(sell_indices, key=lambda idx: float(q_values[idx]))
    hold_q = float(q_values[hold_idx])
    best_sell_q = float(q_values[best_sell_idx])

    if best_sell_q > hold_q + active_margin:
        return int(best_sell_idx)
    return int(hold_idx)


def baseline_action_fraction(
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
    raise ValueError(
        f"Policy {policy_name!r} does not use baseline_action_fraction()."
    )


def evaluate_policy_on_episodes(
    env: TaxAwareEnv,
    policy_name: str,
    episode_ids: list[str],
    split_name: str,
    expected_reward_version: str,
    rng: np.random.Generator,
    q_net: QNetwork | None,
    device: torch.device,
    policy_names: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    valid_policy_names = POLICY_NAMES if policy_names is None else policy_names
    threshold_settings = threshold_settings_for_policy(policy_name)
    if policy_name not in valid_policy_names and threshold_settings is None:
        raise ValueError(f"Unknown policy_name={policy_name!r}.")
    if (
        policy_name == "trained_dqn_greedy" or threshold_settings is not None
    ) and q_net is None:
        raise ValueError(f"{policy_name} requires a loaded QNetwork.")

    obs_dim = len(env.state_columns)
    num_actions = len(env.action_fractions)
    episode_metric_rows: list[dict[str, Any]] = []
    step_rollout_rows: list[dict[str, Any]] = []

    for episode_id in episode_ids:
        obs, reset_info = env.reset(episode_id=episode_id)
        if obs.shape != (obs_dim,):
            raise AssertionError(
                f"Observation shape {obs.shape} does not match expected {(obs_dim,)}."
            )
        if not np.isfinite(obs).all():
            raise AssertionError(
                f"Initial observation contains NaN or infinite values for "
                f"episode_id={episode_id}."
            )

        episode_initial_after_tax_total_value = float(
            reset_info["after_tax_total_value"]
        )
        episode_start_date = pd.to_datetime(reset_info["date"], errors="coerce")
        episode_tax_transition_date = pd.to_datetime(
            reset_info["tax_transition_date"],
            errors="coerce",
        )
        if pd.isna(episode_start_date) or pd.isna(episode_tax_transition_date):
            raise ValueError(
                f"Invalid episode date or tax_transition_date for episode_id={episode_id}."
            )
        if not np.isfinite(episode_initial_after_tax_total_value):
            raise AssertionError(
                f"Initial after-tax total value is not finite for episode_id={episode_id}."
            )

        done = False
        step_in_episode = 0
        episode_total_reward = 0.0
        episode_total_reward_A = 0.0
        episode_total_reward_C_lite = 0.0
        episode_total_reward_C_lite_v2 = 0.0
        episode_total_transaction_penalty = 0.0
        episode_total_cooldown_penalty = 0.0
        transaction_penalty_steps = 0
        transaction_penalties_applied: list[float] = []
        cooldown_penalty_steps = 0
        cooldown_penalties_applied: list[float] = []
        total_position_sold_short_term = 0.0
        total_position_sold_long_term = 0.0
        total_tax_paid = 0.0
        total_discretionary_sale_tax_paid = 0.0
        total_positive_discretionary_sale_pre_tax = 0.0
        total_positive_terminal_liquidation_pre_tax = 0.0
        total_terminal_liquidation_tax_paid = 0.0
        terminal_liquidation_fraction = 0.0
        num_discretionary_sales = 0
        num_cooldown_penalized_sales = 0
        if env._current_episode_df is None:
            raise RuntimeError("Environment did not set _current_episode_df on reset.")
        max_steps = len(env._current_episode_df) + 5
        last_info: dict[str, Any] | None = None
        first_cut_step: int | None = None
        first_cut_date: Any | None = None
        first_cut_fraction_executed: float | None = None
        episode_cut_occurred = False

        while not done:
            if step_in_episode > max_steps:
                raise RuntimeError(
                    f"Safety cap exceeded for split={split_name} "
                    f"policy={policy_name} episode_id={episode_id}: "
                    f"max_steps={max_steps}."
                )

            if policy_name == "trained_dqn_greedy":
                action_idx = select_dqn_greedy_action(
                    q_net=q_net,
                    obs=obs,
                    num_actions=num_actions,
                    device=device,
                )
            elif threshold_settings is not None:
                threshold_margin, first_sale_margin = threshold_settings
                action_idx = select_dqn_thresholded_greedy_action(
                    q_net=q_net,
                    obs=obs,
                    num_actions=num_actions,
                    device=device,
                    env=env,
                    margin=threshold_margin,
                    first_sale_margin=first_sale_margin,
                )
            else:
                action_fraction = baseline_action_fraction(
                    policy_name=policy_name,
                    step_idx=step_in_episode,
                    rng=rng,
                    env=env,
                )
                action_idx = action_index_for_fraction(env, action_fraction)

            next_obs, reward, done, truncated, info = env.step(action_idx)
            if truncated is not False:
                raise AssertionError("TaxAwareEnv returned truncated=True.")
            assert_reward_info(reward, info, expected_reward_version)
            if not np.isfinite(reward):
                raise AssertionError(
                    f"Reward is NaN or infinite for split={split_name} "
                    f"policy={policy_name} episode_id={episode_id} "
                    f"step={step_in_episode}."
                )
            if next_obs.shape != (obs_dim,):
                raise AssertionError(
                    f"Next observation shape {next_obs.shape} does not match "
                    f"expected {(obs_dim,)}."
                )
            if not np.isfinite(next_obs).all():
                raise AssertionError(
                    f"Next observation contains NaN or infinite values for "
                    f"split={split_name} policy={policy_name} "
                    f"episode_id={episode_id} step={step_in_episode}."
                )

            action_fraction_executed = float(info["action_fraction_executed"])
            step_reward_A = float(info.get("reward_A", reward))
            step_reward_C_lite = float(info.get("reward_C_lite", step_reward_A))
            step_reward_C_lite_v2 = float(
                info.get("reward_C_lite_v2", step_reward_C_lite)
            )
            step_transaction_penalty = float(
                info.get("transaction_penalty", 0.0) or 0.0
            )
            step_transaction_applied = bool(
                info.get("transaction_penalty_applied", False)
            )
            step_cooldown_penalty = float(info.get("cooldown_penalty", 0.0) or 0.0)
            step_cooldown_applied = bool(info.get("cooldown_penalty_applied", False))
            if step_transaction_applied:
                transaction_penalty_steps += 1
                transaction_penalties_applied.append(step_transaction_penalty)
            if step_cooldown_applied:
                cooldown_penalty_steps += 1
                cooldown_penalties_applied.append(step_cooldown_penalty)
                num_cooldown_penalized_sales += 1
            if action_fraction_executed > 0.0 and not episode_cut_occurred:
                episode_cut_occurred = True
                first_cut_step = step_in_episode
                first_cut_date = info.get("date")
                first_cut_fraction_executed = action_fraction_executed
            if action_fraction_executed > 0.0:
                num_discretionary_sales += 1
                tax_regime = str(info.get("tax_regime"))
                if tax_regime == "short_term":
                    total_position_sold_short_term += action_fraction_executed
                elif tax_regime == "long_term":
                    total_position_sold_long_term += action_fraction_executed
                realized_pre_tax_increment = float(
                    info.get("realized_pre_tax_increment", 0.0) or 0.0
                )
                action_tax_paid = float(info.get("tax_paid", 0.0) or 0.0)
                total_discretionary_sale_tax_paid += action_tax_paid
                if realized_pre_tax_increment > 0.0:
                    total_positive_discretionary_sale_pre_tax += (
                        realized_pre_tax_increment
                    )
            terminal_tax_paid = float(
                info.get("terminal_liquidation_tax_paid", 0.0) or 0.0
            )
            terminal_pre_tax_increment = float(
                info.get("terminal_liquidation_pre_tax_increment", 0.0) or 0.0
            )
            if bool(info.get("terminal_liquidation_executed", False)):
                terminal_fraction = float(
                    max(
                        0.0,
                        1.0
                        - total_position_sold_short_term
                        - total_position_sold_long_term,
                    )
                )
                terminal_liquidation_fraction += terminal_fraction
                total_position_sold_long_term += terminal_fraction
                total_terminal_liquidation_tax_paid += terminal_tax_paid
                if terminal_pre_tax_increment > 0.0:
                    total_positive_terminal_liquidation_pre_tax += (
                        terminal_pre_tax_increment
                    )
            total_tax_paid += float(info.get("tax_paid", 0.0) or 0.0)
            total_tax_paid += terminal_tax_paid

            step_rollout_rows.append(
                {
                    "split": split_name,
                    "policy_name": policy_name,
                    "episode_id": episode_id,
                    "step_in_episode": step_in_episode,
                    "date": info.get("date"),
                    "tax_transition_date": info.get("tax_transition_date"),
                    "action_idx": int(action_idx),
                    "action_fraction_requested": info.get(
                        "action_fraction_requested"
                    ),
                    "action_fraction_executed": action_fraction_executed,
                    "reward": float(reward),
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
                    "realized_pre_tax_increment": info.get(
                        "realized_pre_tax_increment"
                    ),
                    "tax_paid": info.get("tax_paid"),
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
                    "applicable_tax_rate": info.get("applicable_tax_rate"),
                    "after_tax_liquidation_tax_regime": info.get(
                        "after_tax_liquidation_tax_regime"
                    ),
                    "is_automatic_terminal_liquidation": info.get(
                        "is_automatic_terminal_liquidation"
                    ),
                    "terminal_liquidation_executed": info.get(
                        "terminal_liquidation_executed"
                    ),
                    "terminal_liquidation_tax_rate": info.get(
                        "terminal_liquidation_tax_rate"
                    ),
                    "terminal_liquidation_pre_tax_increment": info.get(
                        "terminal_liquidation_pre_tax_increment"
                    ),
                    "terminal_liquidation_tax_paid": info.get(
                        "terminal_liquidation_tax_paid"
                    ),
                    "terminal_liquidation_after_tax_increment": info.get(
                        "terminal_liquidation_after_tax_increment"
                    ),
                    "done": bool(done),
                }
            )

            episode_total_reward += float(reward)
            episode_total_reward_A += step_reward_A
            episode_total_reward_C_lite += step_reward_C_lite
            episode_total_reward_C_lite_v2 += step_reward_C_lite_v2
            episode_total_transaction_penalty += step_transaction_penalty
            episode_total_cooldown_penalty += step_cooldown_penalty
            last_info = info
            obs = next_obs
            step_in_episode += 1

        if last_info is None:
            raise RuntimeError(
                f"No steps were executed for split={split_name} "
                f"policy={policy_name} episode_id={episode_id}."
            )

        episode_final_after_tax_total_value = float(
            last_info["after_tax_total_value"]
        )
        assert approx_equal(
            episode_total_reward_A,
            episode_final_after_tax_total_value
            - episode_initial_after_tax_total_value,
        ), (
            "Episode Reward A telescoping failed for "
            f"split={split_name} policy={policy_name} episode_id={episode_id}."
        )
        if expected_reward_version == REWARD_A_VERSION:
            assert approx_equal(episode_total_reward, episode_total_reward_A), (
                "Reward A episode total differs from summed Reward A."
            )
        elif expected_reward_version == REWARD_C_LITE_VERSION:
            assert approx_equal(
                episode_total_reward,
                episode_total_reward_A - episode_total_cooldown_penalty,
            ), (
                "Reward C-lite episode total does not equal Reward A minus "
                "cooldown penalties."
            )
            assert approx_equal(episode_total_reward, episode_total_reward_C_lite), (
                "Reward C-lite episode total differs from summed reward_C_lite."
            )
        elif expected_reward_version == REWARD_C_LITE_V2_VERSION:
            assert approx_equal(
                episode_total_reward,
                episode_total_reward_A
                - episode_total_transaction_penalty
                - episode_total_cooldown_penalty,
            ), (
                "Reward C-lite v2 episode total does not equal Reward A minus "
                "transaction and cooldown penalties."
            )
            assert approx_equal(
                episode_total_reward,
                episode_total_reward_C_lite_v2,
            ), (
                "Reward C-lite v2 episode total differs from summed "
                "reward_C_lite_v2."
            )

        episode_final_remaining_fraction = float(last_info["remaining_fraction"])
        episode_final_sold_fraction = float(last_info["sold_fraction"])
        transaction_penalty_frequency = (
            float(transaction_penalty_steps / step_in_episode)
            if step_in_episode > 0
            else 0.0
        )
        mean_transaction_penalty_when_applied = (
            float(np.mean(transaction_penalties_applied))
            if transaction_penalties_applied
            else 0.0
        )
        cooldown_penalty_frequency = (
            float(cooldown_penalty_steps / step_in_episode)
            if step_in_episode > 0
            else 0.0
        )
        mean_cooldown_penalty_when_applied = (
            float(np.mean(cooldown_penalties_applied))
            if cooldown_penalties_applied
            else 0.0
        )
        first_sale_date = first_cut_date
        first_sale_timestamp = (
            pd.to_datetime(first_sale_date, errors="coerce")
            if first_sale_date is not None
            else pd.NaT
        )
        if first_sale_date is not None and pd.isna(first_sale_timestamp):
            raise ValueError(
                f"Invalid first sale date for episode_id={episode_id}: "
                f"{first_sale_date!r}."
            )
        days_to_first_sale = (
            int((first_sale_timestamp - episode_start_date).days)
            if first_sale_date is not None
            else None
        )
        steps_to_first_sale = first_cut_step
        first_sale_before_tax_transition = (
            bool(first_sale_timestamp < episode_tax_transition_date)
            if first_sale_date is not None
            else False
        )
        terminal_or_last_date_value = last_info.get("terminal_liquidation_date")
        if terminal_or_last_date_value is None or pd.isna(terminal_or_last_date_value):
            terminal_or_last_date_value = last_info.get("date")
        terminal_or_last_date = pd.to_datetime(
            terminal_or_last_date_value,
            errors="coerce",
        )
        reached_tax_transition_before_first_sale_flag = (
            bool(first_sale_timestamp >= episode_tax_transition_date)
            if first_sale_date is not None
            else bool(terminal_or_last_date >= episode_tax_transition_date)
        )
        sold_before_tax_transition_flag = bool(total_position_sold_short_term > 0.0)
        total_positive_taxable_pre_tax_increment = float(
            total_positive_discretionary_sale_pre_tax
            + total_positive_terminal_liquidation_pre_tax
        )
        mean_effective_tax_rate_on_sales = (
            float(
                total_tax_paid / total_positive_taxable_pre_tax_increment
            )
            if total_positive_taxable_pre_tax_increment > 0.0
            else 0.0
        )
        episode_metric_rows.append(
            {
                "split": split_name,
                "policy_name": policy_name,
                "episode_id": episode_id,
                "episode_total_reward": float(episode_total_reward),
                "episode_total_reward_A": float(episode_total_reward_A),
                "episode_total_reward_C_lite": float(episode_total_reward_C_lite),
                "episode_total_reward_C_lite_v2": float(
                    episode_total_reward_C_lite_v2
                ),
                "episode_total_transaction_penalty": float(
                    episode_total_transaction_penalty
                ),
                "episode_total_cooldown_penalty": float(
                    episode_total_cooldown_penalty
                ),
                "transaction_penalty_frequency": transaction_penalty_frequency,
                "mean_transaction_penalty_when_applied": (
                    mean_transaction_penalty_when_applied
                ),
                "cooldown_penalty_frequency": cooldown_penalty_frequency,
                "mean_cooldown_penalty_when_applied": (
                    mean_cooldown_penalty_when_applied
                ),
                "episode_initial_after_tax_total_value": (
                    episode_initial_after_tax_total_value
                ),
                "episode_final_after_tax_total_value": (
                    episode_final_after_tax_total_value
                ),
                "episode_realized_after_tax_pnl": float(
                    last_info["cum_realized_after_tax_pnl"]
                ),
                "episode_steps": int(step_in_episode),
                "episode_terminal_liquidation_executed": bool(
                    last_info["terminal_liquidation_executed"]
                ),
                "episode_final_remaining_fraction": episode_final_remaining_fraction,
                "episode_final_sold_fraction": episode_final_sold_fraction,
                "episode_full_liquidation": bool(
                    episode_final_remaining_fraction == 0.0
                ),
                "episode_cut_occurred": bool(episode_cut_occurred),
                "first_cut_step": first_cut_step,
                "first_cut_date": first_cut_date,
                "first_cut_fraction_executed": first_cut_fraction_executed,
                "days_to_first_sale": days_to_first_sale,
                "steps_to_first_sale": steps_to_first_sale,
                "first_sale_date": first_sale_date,
                "first_sale_before_tax_transition": (
                    first_sale_before_tax_transition
                ),
                "pct_episode_position_sold_short_term": float(
                    total_position_sold_short_term
                ),
                "pct_episode_position_sold_long_term": float(
                    total_position_sold_long_term
                ),
                "total_position_sold_short_term": float(
                    total_position_sold_short_term
                ),
                "total_position_sold_long_term": float(
                    total_position_sold_long_term
                ),
                "mean_effective_tax_rate_on_sales": (
                    mean_effective_tax_rate_on_sales
                ),
                "total_tax_paid": float(total_tax_paid),
                "total_positive_taxable_pre_tax_increment": (
                    total_positive_taxable_pre_tax_increment
                ),
                "terminal_liquidation_fraction": float(
                    terminal_liquidation_fraction
                ),
                "total_terminal_liquidation_tax_paid": float(
                    total_terminal_liquidation_tax_paid
                ),
                "total_transaction_penalty": float(
                    episode_total_transaction_penalty
                ),
                "total_cooldown_penalty": float(episode_total_cooldown_penalty),
                "num_discretionary_sales": int(num_discretionary_sales),
                "num_cooldown_penalized_sales": int(num_cooldown_penalized_sales),
                "sold_before_tax_transition_flag": sold_before_tax_transition_flag,
                "reached_tax_transition_before_first_sale_flag": (
                    reached_tax_transition_before_first_sale_flag
                ),
            }
        )

    return episode_metric_rows, step_rollout_rows


def summarize_by_policy(episode_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if episode_metrics_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    metrics = episode_metrics_df.copy()
    metrics["first_cut_step"] = pd.to_numeric(
        metrics["first_cut_step"],
        errors="coerce",
    )
    metrics["first_cut_fraction_executed"] = pd.to_numeric(
        metrics["first_cut_fraction_executed"],
        errors="coerce",
    )
    for column in [
        "days_to_first_sale",
        "steps_to_first_sale",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "mean_effective_tax_rate_on_sales",
        "total_tax_paid",
        "total_transaction_penalty",
        "total_cooldown_penalty",
        "num_discretionary_sales",
        "num_cooldown_penalized_sales",
    ]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics["first_sale_before_tax_transition_bool"] = (
        metrics["first_sale_before_tax_transition"].fillna(False).astype(bool)
    )
    metrics["sold_before_tax_transition_bool"] = (
        metrics["sold_before_tax_transition_flag"].fillna(False).astype(bool)
    )
    metrics["sell_immediately"] = metrics["first_cut_step"].eq(0)

    summary_df = (
        metrics.groupby(["split", "policy_name"], sort=True)
        .agg(
            num_episodes=("episode_total_reward", "size"),
            mean_episode_total_reward=("episode_total_reward", "mean"),
            mean_episode_total_reward_A=("episode_total_reward_A", "mean"),
            mean_episode_total_reward_C_lite=(
                "episode_total_reward_C_lite",
                "mean",
            ),
            mean_episode_total_reward_C_lite_v2=(
                "episode_total_reward_C_lite_v2",
                "mean",
            ),
            mean_episode_total_transaction_penalty=(
                "episode_total_transaction_penalty",
                "mean",
            ),
            mean_episode_total_cooldown_penalty=(
                "episode_total_cooldown_penalty",
                "mean",
            ),
            median_episode_total_reward=("episode_total_reward", "median"),
            std_episode_total_reward=("episode_total_reward", "std"),
            min_episode_total_reward=("episode_total_reward", "min"),
            max_episode_total_reward=("episode_total_reward", "max"),
            p05_episode_total_reward=(
                "episode_total_reward",
                lambda series: series.quantile(0.05),
            ),
            p95_episode_total_reward=(
                "episode_total_reward",
                lambda series: series.quantile(0.95),
            ),
            mean_final_after_tax_total_value=(
                "episode_final_after_tax_total_value",
                "mean",
            ),
            median_final_after_tax_total_value=(
                "episode_final_after_tax_total_value",
                "median",
            ),
            mean_realized_after_tax_pnl=("episode_realized_after_tax_pnl", "mean"),
            median_realized_after_tax_pnl=(
                "episode_realized_after_tax_pnl",
                "median",
            ),
            mean_episode_steps=("episode_steps", "mean"),
            median_episode_steps=("episode_steps", "median"),
            terminal_liquidation_frequency=(
                "episode_terminal_liquidation_executed",
                "mean",
            ),
            full_liquidation_frequency=("episode_full_liquidation", "mean"),
            cut_frequency=("episode_cut_occurred", "mean"),
            sell_immediately_frequency=("sell_immediately", "mean"),
            mean_first_cut_step=("first_cut_step", "mean"),
            median_first_cut_step=("first_cut_step", "median"),
            mean_first_cut_fraction_executed=(
                "first_cut_fraction_executed",
                "mean",
            ),
            average_days_to_first_sale=("days_to_first_sale", "mean"),
            median_days_to_first_sale=("days_to_first_sale", "median"),
            average_steps_to_first_sale=("steps_to_first_sale", "mean"),
            pct_episodes_with_first_sale_before_tax_transition=(
                "first_sale_before_tax_transition_bool",
                "mean",
            ),
            pct_episodes_sold_before_tax_transition=(
                "sold_before_tax_transition_bool",
                "mean",
            ),
            mean_pct_position_sold_short_term=(
                "pct_episode_position_sold_short_term",
                "mean",
            ),
            mean_pct_position_sold_long_term=(
                "pct_episode_position_sold_long_term",
                "mean",
            ),
            mean_effective_tax_rate=(
                "mean_effective_tax_rate_on_sales",
                "mean",
            ),
            mean_total_tax_paid=("total_tax_paid", "mean"),
            mean_transaction_penalty=("total_transaction_penalty", "mean"),
            mean_cooldown_penalty=("total_cooldown_penalty", "mean"),
            mean_num_discretionary_sales=("num_discretionary_sales", "mean"),
            mean_num_cooldown_penalized_sales=(
                "num_cooldown_penalized_sales",
                "mean",
            ),
        )
        .reset_index()
    )
    for baseline_policy, output_column in [
        ("sell_immediately", "mean_excess_value_vs_sell_immediately"),
        ("hold_to_terminal", "mean_excess_value_vs_hold_to_terminal"),
    ]:
        baseline_values = summary_df.loc[
            summary_df["policy_name"].eq(baseline_policy),
            ["split", "mean_final_after_tax_total_value"],
        ].set_index("split")["mean_final_after_tax_total_value"]
        summary_df[output_column] = (
            summary_df["mean_final_after_tax_total_value"]
            - summary_df["split"].map(baseline_values)
        )
    return summary_df[SUMMARY_COLUMNS]


def build_summary_text(
    summary_df: pd.DataFrame,
    episode_metrics_df: pd.DataFrame,
) -> str:
    def _best_policy(split_name: str) -> str:
        split_summary = summary_df[summary_df["split"] == split_name]
        if split_summary.empty:
            return "None"
        best_idx = split_summary["mean_final_after_tax_total_value"].idxmax()
        return str(split_summary.loc[best_idx, "policy_name"])

    num_validation_episodes = int(
        episode_metrics_df.loc[
            episode_metrics_df["split"] == "validation",
            "episode_id",
        ].nunique()
    )
    num_test_episodes = int(
        episode_metrics_df.loc[
            episode_metrics_df["split"] == "test",
            "episode_id",
        ].nunique()
    )
    policies_evaluated = SUMMARY_CONTEXT.get("policies_evaluated", POLICY_NAMES)
    summary_table = summary_df.to_string(index=False)

    lines = [
        "Reward Baseline Evaluation Summary",
        f"reward_version: {SUMMARY_CONTEXT.get('reward_version', 'unknown')}",
        f"tax_profile_name: {SUMMARY_CONTEXT.get('tax_profile_name', 'unknown')}",
        f"resolved_tax_profile: {SUMMARY_CONTEXT.get('resolved_tax_profile', 'unknown')}",
        f"model_path: {SUMMARY_CONTEXT.get('model_path', 'unknown')}",
        f"episode_splits_path: {SUMMARY_CONTEXT.get('episode_splits_path', 'unknown')}",
        f"num_validation_episodes: {num_validation_episodes}",
        f"num_test_episodes: {num_test_episodes}",
        "policies_evaluated: " + ", ".join(str(policy) for policy in policies_evaluated),
        "best_validation_policy_by_mean_final_after_tax_total_value: "
        + _best_policy("validation"),
        "best_test_policy_by_mean_final_after_tax_total_value: "
        + _best_policy("test"),
        "summary_table:",
        summary_table,
        "",
    ]
    return "\n".join(lines)


def _apply_episode_cap(
    episode_ids: list[str],
    cap: int | None,
    split_name: str,
) -> list[str]:
    if cap is None:
        return episode_ids
    if int(cap) <= 0:
        raise ValueError(f"MAX_{split_name.upper()}_EPISODES must be positive or None.")
    return episode_ids[: int(cap)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DQN and baseline policies for an environment reward."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=(
            "Evaluation config path. Defaults to "
            f"{_relative_project_path(CONFIG_PATH)}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    print("Reward Baseline Evaluation")

    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)
    _validate_reward_config(config)
    policy_names = resolve_evaluation_policy_names(config)
    expected_reward_version = str(_require(config, "reward.expected_info_reward_version"))
    output_dir = resolve_project_path(_require(config, "logging.output_dir"))
    baselines_dir = output_dir / "baselines"
    best_validation_model_path = output_dir / "best_validation_model.pt"
    final_model_path = output_dir / "final_model.pt"
    model_path = (
        best_validation_model_path
        if best_validation_model_path.exists()
        else final_model_path
    )
    episode_splits_path = output_dir / "episode_splits.csv"

    env = make_env(config)
    resolved_tax_profile = resolve_tax_profile_from_config(
        config,
        base_dir=PROJECT_ROOT,
    )
    state_columns = load_state_columns(
        resolve_project_path(_require(config, "environment.state_schema_path"))
    )
    obs_dim = len(state_columns)
    num_actions = len(env.action_fractions)
    for action_fraction in _require(config, "action_space.action_fractions"):
        action_index_for_fraction(env, float(action_fraction))

    device = _resolve_device(str(_require(config, "training.device")))
    q_net = load_trained_q_network(
        config=config,
        model_path=model_path,
        obs_dim=obs_dim,
        num_actions=num_actions,
        device=device,
    )
    print(f"Loaded trained model: {_relative_project_path(model_path)}")

    splits = load_episode_splits(episode_splits_path)
    print(f"Loaded episode splits: {_relative_project_path(episode_splits_path)}")

    validation_ids = _apply_episode_cap(
        splits["validation"],
        MAX_VALIDATION_EPISODES,
        "validation",
    )
    test_ids = _apply_episode_cap(splits["test"], MAX_TEST_EPISODES, "test")
    if not validation_ids:
        raise ValueError("No validation episodes selected for baseline evaluation.")
    if not test_ids:
        raise ValueError("No test episodes selected for baseline evaluation.")

    baselines_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(
        {
            "source_config_path": _relative_project_path(config_path),
            "model_path": _relative_project_path(model_path),
            "episode_splits_path": _relative_project_path(episode_splits_path),
            "output_dir": _relative_project_path(baselines_dir),
            "reward_version": expected_reward_version,
            "max_validation_episodes": MAX_VALIDATION_EPISODES,
            "max_test_episodes": MAX_TEST_EPISODES,
            "policies_evaluated": policy_names,
            "resolved_tax_profile": resolved_tax_profile,
            "training_config": config,
        },
        baselines_dir / "baseline_config_used.yaml",
    )

    rng = np.random.default_rng(42)
    episode_metric_rows: list[dict[str, Any]] = []
    step_rollout_rows: list[dict[str, Any]] = []

    for policy_name in policy_names:
        for split_name, episode_ids in (
            ("validation", validation_ids),
            ("test", test_ids),
        ):
            print(
                f"Evaluating split={split_name} "
                f"policy={policy_name} episodes={len(episode_ids)}"
            )
            policy_episode_rows, policy_step_rows = evaluate_policy_on_episodes(
                env=env,
                policy_name=policy_name,
                episode_ids=episode_ids,
                split_name=split_name,
                expected_reward_version=expected_reward_version,
                rng=rng,
                q_net=q_net,
                device=device,
                policy_names=policy_names,
            )
            episode_metric_rows.extend(policy_episode_rows)
            step_rollout_rows.extend(policy_step_rows)

    episode_metrics_df = pd.DataFrame(
        episode_metric_rows,
        columns=EPISODE_METRIC_COLUMNS,
    )
    step_rollouts_df = pd.DataFrame(
        step_rollout_rows,
        columns=STEP_ROLLOUT_COLUMNS,
    )
    summary_df = summarize_by_policy(episode_metrics_df)

    episode_metrics_df.to_csv(
        baselines_dir / "baseline_episode_metrics.csv",
        index=False,
    )
    step_rollouts_df.to_csv(
        baselines_dir / "baseline_step_rollouts.csv",
        index=False,
    )
    summary_df.to_csv(
        baselines_dir / "baseline_summary_by_policy.csv",
        index=False,
    )

    SUMMARY_CONTEXT.update(
        {
            "reward_version": expected_reward_version,
            "tax_profile_name": resolved_tax_profile["profile_name"],
            "resolved_tax_profile": resolved_tax_profile,
            "model_path": _relative_project_path(model_path),
            "episode_splits_path": _relative_project_path(episode_splits_path),
            "policies_evaluated": policy_names,
        }
    )
    summary_text = build_summary_text(summary_df, episode_metrics_df)
    with (baselines_dir / "baseline_evaluation_summary.txt").open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(summary_text)

    print(f"BASELINE EVALUATION COMPLETE reward_version={expected_reward_version}")


if __name__ == "__main__":
    main()
