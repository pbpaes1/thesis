from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_quant_analysis_phase_1_3 import (
    ANNUALIZATION_FACTOR,
    DAILY_RISK_FREE_RATE,
    SHARPE_ANNUALIZATION_FACTOR,
    build_eaat_sharpe_metrics,
)


def _policy_df(policy_name: str = "policy") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_name": [policy_name],
            "policy_order": [0],
        }
    )


def _stock_path(
    returns: np.ndarray,
    *,
    episode_id: str = "E1",
    start: str = "2020-01-01",
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(returns) + 1)
    return pd.DataFrame(
        {
            "episode_id": episode_id,
            "date": dates,
            "stock_day": np.arange(len(returns) + 1),
            "stock_return": [np.nan, *returns.tolist()],
            "horizon_scope": "full_day0_to_terminal",
        }
    )


def _metrics(
    rollouts: pd.DataFrame,
    stock_paths: pd.DataFrame,
    *,
    policy_name: str = "policy",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return build_eaat_sharpe_metrics(
        rollouts,
        stock_paths,
        _policy_df(policy_name),
        source_path=Path("synthetic"),
        expected_splits=["test"],
    )


def test_hold_to_terminal_eaat_and_ta_eaat_single_tranche() -> None:
    returns = np.linspace(-0.001, 0.002, 252)
    stock_paths = _stock_path(returns)
    terminal_return = 0.40
    dates = stock_paths["date"].tolist()
    rollouts = pd.DataFrame(
        [
            {
                "split": "test",
                "policy_name": "policy",
                "episode_id": "E1",
                "step_in_episode": 0,
                "date": dates[-1],
                "action_fraction_executed": 0.0,
                "remaining_fraction": 0.0,
                "terminal_liquidation_executed": True,
                "realized_after_tax_increment": 0.0,
                "terminal_liquidation_after_tax_increment": terminal_return,
            }
        ]
    )

    _, episodes, tranches = _metrics(rollouts, stock_paths)
    episode = episodes.iloc[0]

    assert np.isclose(episode["EAAT_terminal_after_tax_wealth"], 1.0 + terminal_return)
    assert len(tranches) == 1
    assert tranches.iloc[0]["tranche_type"] == "terminal_liquidation"
    assert np.isclose(
        episode["TA_EAAT_annualized_after_tax_return"],
        (1.0 + terminal_return) ** (ANNUALIZATION_FACTOR / 252) - 1.0,
    )


def test_partial_sale_eaat_terminal_wealth_and_ta_components() -> None:
    returns = np.linspace(-0.002, 0.003, 252)
    stock_paths = _stock_path(returns)
    dates = stock_paths["date"].tolist()
    rollouts = pd.DataFrame(
        [
            {
                "split": "test",
                "policy_name": "policy",
                "episode_id": "E1",
                "step_in_episode": 0,
                "date": dates[189],
                "action_fraction_executed": 0.5,
                "remaining_fraction": 0.5,
                "terminal_liquidation_executed": False,
                "realized_after_tax_increment": 0.5 * 0.35,
                "terminal_liquidation_after_tax_increment": 0.0,
            },
            {
                "split": "test",
                "policy_name": "policy",
                "episode_id": "E1",
                "step_in_episode": 1,
                "date": dates[251],
                "action_fraction_executed": 0.0,
                "remaining_fraction": 0.0,
                "terminal_liquidation_executed": True,
                "realized_after_tax_increment": 0.0,
                "terminal_liquidation_after_tax_increment": 0.5 * 0.40,
            },
        ]
    )

    _, episodes, _ = _metrics(rollouts, stock_paths)
    episode = episodes.iloc[0]
    expected_wealth = (
        0.5 * 1.35 * (1.0 + DAILY_RISK_FREE_RATE) ** (252 - 189)
        + 0.5 * 1.40
    )
    expected_ta_return = (
        0.5 * (1.35 ** (ANNUALIZATION_FACTOR / 189) - 1.0)
        + 0.5 * (1.40 ** (ANNUALIZATION_FACTOR / 252) - 1.0)
    )
    expected_ta_volatility = (
        0.5 * np.std(returns[:189], ddof=1) * SHARPE_ANNUALIZATION_FACTOR
        + 0.5 * np.std(returns[:252], ddof=1) * SHARPE_ANNUALIZATION_FACTOR
    )

    assert np.isclose(episode["EAAT_terminal_after_tax_wealth"], expected_wealth)
    assert np.isclose(
        episode["TA_EAAT_annualized_after_tax_return"],
        expected_ta_return,
    )
    assert np.isclose(
        episode["TA_EAAT_annualized_volatility"],
        expected_ta_volatility,
    )


def test_full_liquidation_early_eaat_returns_cash_after_sale() -> None:
    returns = np.linspace(-0.003, 0.004, 20)
    stock_paths = _stock_path(returns)
    dates = stock_paths["date"].tolist()
    rollouts = pd.DataFrame(
        [
            {
                "split": "test",
                "policy_name": "policy",
                "episode_id": "E1",
                "step_in_episode": 0,
                "date": dates[5],
                "action_fraction_executed": 1.0,
                "remaining_fraction": 0.0,
                "terminal_liquidation_executed": False,
                "realized_after_tax_increment": 0.10,
                "terminal_liquidation_after_tax_increment": 0.0,
            }
        ]
    )

    _, episodes, _ = _metrics(rollouts, stock_paths)
    expected_policy_returns = np.asarray(
        [*returns[:5].tolist(), *([DAILY_RISK_FREE_RATE] * 15)]
    )
    expected_volatility = (
        np.std(expected_policy_returns, ddof=1) * SHARPE_ANNUALIZATION_FACTOR
    )

    assert np.isclose(
        episodes.iloc[0]["EAAT_annualized_volatility"],
        expected_volatility,
    )


def test_zero_volatility_sharpes_are_nan() -> None:
    returns = np.zeros(10)
    stock_paths = _stock_path(returns)
    dates = stock_paths["date"].tolist()
    rollouts = pd.DataFrame(
        [
            {
                "split": "test",
                "policy_name": "policy",
                "episode_id": "E1",
                "step_in_episode": 0,
                "date": dates[-1],
                "action_fraction_executed": 0.0,
                "remaining_fraction": 0.0,
                "terminal_liquidation_executed": True,
                "realized_after_tax_increment": 0.0,
                "terminal_liquidation_after_tax_increment": 0.10,
            }
        ]
    )

    _, episodes, _ = _metrics(rollouts, stock_paths)

    assert np.isnan(episodes.iloc[0]["EAAT_Sharpe"])
    assert np.isnan(episodes.iloc[0]["TA_EAAT_Sharpe"])
