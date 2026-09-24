"""Synthetic, basis-normalized Brazil trajectories and a frozen U.S. fixture."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import yaml

from src.config.tax_profiles import resolve_economic_scenario_from_config
from src.environment.tax_aware_env import (
    BRAZIL_BASE_REWARD_VERSION,
    BRAZIL_C_LITE_REWARD_VERSION,
    REWARD_A_VERSION,
    TaxAwareEnv,
)


ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-10


class TestBrazilEnvAccounting(unittest.TestCase):
    def _env(
        self,
        dates: list[str],
        prices: list[float],
        *,
        reward_version: str = BRAZIL_C_LITE_REWARD_VERSION,
        rate: float = 0.10,
        basis: list[float] | None = None,
    ) -> TaxAwareEnv:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "episode.parquet"
        pd.DataFrame({
            "episode_id": ["sample"] * len(dates),
            "date": pd.to_datetime(dates),
            "adj_close": prices,
            "simulated_purchase_price": basis if basis is not None else [100.0] * len(dates),
            "feature": [float(i) for i in range(len(dates))],
        }).to_parquet(path, index=False)
        with (ROOT / "configs" / "brazil_sensitivity_rate_1000_v1.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["economic_scenario"]["cash_account"]["annual_gross_rate"] = rate
        scenario = resolve_economic_scenario_from_config(config)
        reward = dict(config["reward"])
        reward["version"] = reward_version
        reward["base_reward_version"] = BRAZIL_BASE_REWARD_VERSION
        return TaxAwareEnv(
            path, ["feature"], reward_config=reward, economic_scenario=scenario,
        )

    def _run(self, env: TaxAwareEnv, actions: list[int]) -> tuple[dict, list[dict], list[float]]:
        _, reset = env.reset("sample")
        infos = []
        rewards = []
        for index, action in enumerate(actions):
            _, reward, done, truncated, info = env.step(action)
            self.assertEqual(done, index == len(actions) - 1)
            self.assertFalse(truncated)
            self.assertAlmostEqual(reward, info["shaped_reward"], delta=TOL)
            infos.append(info)
            rewards.append(reward)
        self.assertAlmostEqual(
            sum(info["base_reward"] for info in infos),
            infos[-1]["total_after_tax_wealth"] - reset["total_after_tax_wealth"],
            delta=TOL,
        )
        with self.assertRaisesRegex(RuntimeError, "complete"):
            env.step(0)
        return reset, infos, rewards

    def test_hold_to_terminal_has_no_cash_interest(self) -> None:
        env = self._env(["2021-01-01", "2021-01-04"], [120.0, 120.0])
        reset, infos, _ = self._run(env, [0, 0])
        self.assertAlmostEqual(reset["total_after_tax_wealth"], 1.17, delta=TOL)
        final = infos[-1]
        self.assertAlmostEqual(final["total_after_tax_wealth"], 1.17, delta=TOL)
        self.assertEqual(final["cash_interest_gross_cumulative"], 0.0)
        self.assertEqual(final["discretionary_sale_count"], 0)
        self.assertEqual(final["mandatory_sale_count"], 1)
        self.assertIsNone(final["first_sale_date"])
        self.assertAlmostEqual(final["mandatory_after_tax_proceeds"], 1.17, delta=TOL)
        self.assertAlmostEqual(final["equity_tax_cumulative"], 0.03, delta=TOL)
        self.assertEqual(final["remaining_inventory_value"], 0.0)

    def test_early_full_sale_accrues_to_original_terminal(self) -> None:
        env = self._env(["2021-01-01", "2021-01-04", "2021-01-05"], [120.0] * 3)
        _, infos, _ = self._run(env, [4])
        self.assertEqual([info["action_fraction_executed"] for info in infos], [1.0])
        self.assertEqual(infos[-1]["discretionary_sale_count"], 1)
        self.assertEqual(infos[-1]["mandatory_sale_count"], 0)
        self.assertEqual(str(infos[-1]["date"].date()), "2021-01-01")
        self.assertEqual(str(infos[-1]["terminal_valuation_date"].date()), "2021-01-05")
        self.assertEqual(infos[-1]["current_row_ptr"], 0)
        gross = 1.17 * 1.10 ** (2 / 252)
        interest = gross - 1.17
        expected = gross - 0.225 * interest
        self.assertAlmostEqual(infos[-1]["cash_interest_gross_cumulative"], interest, delta=TOL)
        self.assertAlmostEqual(infos[-1]["fixed_income_tax_step"], 0.225 * interest, delta=TOL)
        self.assertAlmostEqual(infos[-1]["total_after_tax_wealth"], expected, delta=TOL)
        self.assertAlmostEqual(infos[-1]["cash_balance"], expected, delta=TOL)
        self.assertEqual(infos[-1]["cash_lot_count"], 0)

    def test_equal_terminal_wealth_has_equal_undiscounted_base_return(self) -> None:
        dates = ["2021-01-01", "2021-01-02", "2021-01-03"]
        cash_growth = 1.10 ** (2 / 252)
        early_wealth = 1.0 + (1.0 - 0.225) * (cash_growth - 1.0)
        terminal_price = 100.0 * (1.0 + (early_wealth - 1.0) / (1.0 - 0.15))
        prices = [100.0, 100.0, terminal_price]

        early_reset, early_infos, _ = self._run(self._env(dates, prices), [4])
        hold_reset, hold_infos, _ = self._run(self._env(dates, prices), [0, 0, 0])
        early_final = early_infos[-1]
        hold_final = hold_infos[-1]

        self.assertEqual((len(early_infos), len(hold_infos)), (1, 3))
        self.assertEqual(str(early_final["date"].date()), "2021-01-01")
        self.assertEqual(str(early_final["terminal_valuation_date"].date()), "2021-01-03")
        self.assertAlmostEqual(early_final["cash_interest_gross_step"], cash_growth - 1.0, delta=TOL)
        self.assertAlmostEqual(early_final["cash_interest_gross_cumulative"], cash_growth - 1.0, delta=TOL)
        self.assertAlmostEqual(early_final["fixed_income_tax_step"], 0.225 * (cash_growth - 1.0), delta=TOL)
        self.assertAlmostEqual(early_final["fixed_income_tax_cumulative"], early_final["fixed_income_tax_step"], delta=TOL)
        self.assertEqual(hold_final["cash_interest_gross_cumulative"], 0.0)
        self.assertEqual(hold_final["mandatory_sale_count"], 1)
        self.assertAlmostEqual(early_final["total_after_tax_wealth"], early_wealth, delta=TOL)
        self.assertAlmostEqual(hold_final["total_after_tax_wealth"], early_wealth, delta=TOL)
        early_base_return = sum(info["base_reward"] for info in early_infos)
        hold_base_return = sum(info["base_reward"] for info in hold_infos)
        self.assertAlmostEqual(early_base_return, early_wealth - early_reset["total_after_tax_wealth"], delta=TOL)
        self.assertAlmostEqual(hold_base_return, early_wealth - hold_reset["total_after_tax_wealth"], delta=TOL)
        self.assertAlmostEqual(early_base_return, hold_base_return, delta=TOL)

    def test_early_full_sale_with_two_lots_settles_at_original_horizon(self) -> None:
        env = self._env(["2021-01-01", "2021-01-03", "2021-07-01"], [120.0] * 3)
        reset, infos, rewards = self._run(env, [1, 3])
        first_principal = 0.25 * (1.20 - 0.15 * 0.20)
        second_principal = 0.75 * (1.20 - 0.15 * 0.20)
        first_gross = first_principal * 1.10 ** (2 / 252)
        second_gross = second_principal * 1.10 ** (1 / 252)
        first_interest = first_gross - first_principal
        second_interest = second_gross - second_principal
        settled_tax = 0.20 * first_interest + 0.225 * second_interest
        expected_wealth = first_gross + second_gross - settled_tax
        final = infos[-1]
        self.assertEqual([info["date"].date().isoformat() for info in infos], ["2021-01-01", "2021-01-03"])
        self.assertEqual(str(final["terminal_valuation_date"].date()), "2021-07-01")
        self.assertEqual(final["current_row_ptr"], 1)
        self.assertEqual(final["discretionary_sale_count"], 2)
        self.assertEqual(final["mandatory_sale_count"], 0)
        self.assertEqual(final["first_sale_days_from_start"], 0)
        self.assertEqual(final["remaining_fraction_before_terminal"], 0.0)
        self.assertEqual(final["remaining_fraction"], 0.0)
        self.assertEqual(final["fixed_income_tax_estimated_liability"], 0.0)
        self.assertEqual(final["cash_lot_count"], 0)
        self.assertAlmostEqual(final["cash_principal_cumulative"], first_principal + second_principal, delta=TOL)
        self.assertAlmostEqual(final["cash_interest_gross_cumulative"], first_interest + second_interest, delta=TOL)
        self.assertAlmostEqual(final["fixed_income_tax_step"], settled_tax, delta=TOL)
        self.assertAlmostEqual(final["fixed_income_tax_cumulative"], settled_tax, delta=TOL)
        self.assertAlmostEqual(final["cash_balance"], expected_wealth, delta=TOL)
        self.assertAlmostEqual(final["total_after_tax_wealth"], expected_wealth, delta=TOL)
        self.assertAlmostEqual(final["base_reward"], expected_wealth - infos[0]["total_after_tax_wealth"], delta=TOL)
        self.assertAlmostEqual(sum(info["base_reward"] for info in infos), expected_wealth - reset["total_after_tax_wealth"], delta=TOL)
        self.assertAlmostEqual(rewards[-1], final["base_reward"] - final["cooldown_penalty"], delta=TOL)

    def test_partial_repeated_sales_and_cooldown_shape_only_reward(self) -> None:
        env = self._env(["2021-01-01", "2021-01-02", "2021-01-03"], [120.0] * 3)
        _, infos, rewards = self._run(env, [1, 1, 0])
        first_gross = 0.2925 * 1.10 ** (2 / 252)
        second_gross = 0.2925 * 1.10 ** (1 / 252)
        expected = (
            first_gross - 0.225 * (first_gross - 0.2925)
            + second_gross - 0.225 * (second_gross - 0.2925)
            + 0.585
        )
        self.assertAlmostEqual(infos[-1]["total_after_tax_wealth"], expected, delta=TOL)
        self.assertEqual(infos[-1]["discretionary_sale_count"], 2)
        self.assertEqual(infos[-1]["mandatory_sale_count"], 1)
        self.assertEqual(infos[1]["cash_lot_count"], 2)
        self.assertAlmostEqual(infos[-1]["cash_principal_cumulative"], 0.585, delta=TOL)
        self.assertAlmostEqual(infos[-1]["equity_tax_cumulative"], 0.03, delta=TOL)
        penalty = 0.0025 * 0.25 * 19 / 20
        self.assertAlmostEqual(infos[1]["cooldown_penalty"], penalty, delta=TOL)
        self.assertAlmostEqual(rewards[1], infos[1]["base_reward"] - penalty, delta=TOL)
        self.assertAlmostEqual(sum(rewards), expected - 1.17 - penalty, delta=TOL)

    def test_loss_sale_has_no_equity_credit(self) -> None:
        env = self._env(["2021-01-01", "2021-01-02"], [80.0, 80.0])
        _, infos, _ = self._run(env, [2, 0])
        self.assertAlmostEqual(infos[0]["discretionary_cash_deposit"], 0.4, delta=TOL)
        self.assertAlmostEqual(infos[-1]["mandatory_after_tax_proceeds"], 0.4, delta=TOL)
        self.assertEqual(infos[-1]["equity_tax_cumulative"], 0.0)
        self.assertGreater(infos[-1]["cash_interest_gross_cumulative"], 0.0)

    def test_separate_lots_use_181_and_180_calendar_day_tiers(self) -> None:
        env = self._env(["2021-01-01", "2021-01-02", "2021-07-01"], [120.0] * 3)
        _, infos, _ = self._run(env, [1, 1, 0])
        gross_181 = 0.2925 * 1.10 ** (2 / 252)
        gross_180 = 0.2925 * 1.10 ** (1 / 252)
        expected_tax = 0.20 * (gross_181 - 0.2925) + 0.225 * (gross_180 - 0.2925)
        self.assertAlmostEqual(infos[-1]["fixed_income_tax_step"], expected_tax, delta=TOL)
        self.assertAlmostEqual(infos[-1]["total_after_tax_wealth"], gross_181 + gross_180 - expected_tax + 0.585, delta=TOL)
        self.assertGreater(infos[-1]["cash_interest_gross_cumulative"], expected_tax)

    def test_open_lot_marks_tax_liability_at_180_to_181_boundary(self) -> None:
        env = self._env(
            ["2021-01-01", "2021-06-30", "2021-07-01", "2021-07-02"],
            [120.0] * 4,
        )
        _, infos, _ = self._run(env, [2, 0, 0, 0])
        gross_day_180 = 0.585 * 1.10 ** (1 / 252)
        gross_day_181 = 0.585 * 1.10 ** (2 / 252)
        self.assertAlmostEqual(
            infos[1]["fixed_income_tax_estimated_liability"],
            0.225 * (gross_day_180 - 0.585), delta=TOL,
        )
        self.assertAlmostEqual(
            infos[2]["fixed_income_tax_estimated_liability"],
            0.20 * (gross_day_181 - 0.585), delta=TOL,
        )
        self.assertEqual(infos[1]["fixed_income_tax_cumulative"], 0.0)
        self.assertEqual(infos[2]["fixed_income_tax_cumulative"], 0.0)
        expected_day_181_wealth = gross_day_181 - 0.20 * (gross_day_181 - 0.585) + 0.585
        self.assertAlmostEqual(infos[2]["total_after_tax_wealth"], expected_day_181_wealth, delta=TOL)
        self.assertAlmostEqual(
            infos[2]["base_reward"],
            infos[2]["total_after_tax_wealth"] - infos[1]["total_after_tax_wealth"],
            delta=TOL,
        )
        self.assertEqual(infos[-1]["fixed_income_tax_estimated_liability"], 0.0)
        self.assertGreater(infos[-1]["fixed_income_tax_cumulative"], 0.0)

    def test_final_row_discretionary_sale_precedes_mandatory_remainder(self) -> None:
        env = self._env(["2021-01-01", "2021-01-04"], [120.0, 120.0])
        _, infos, _ = self._run(env, [0, 2])
        final = infos[-1]
        self.assertEqual(final["discretionary_sale_count"], 1)
        self.assertEqual(final["mandatory_sale_count"], 1)
        self.assertEqual(final["first_sale_days_from_start"], 3)
        self.assertEqual(str(final["first_sale_date"].date()), "2021-01-04")
        self.assertAlmostEqual(final["discretionary_cash_deposit"], 0.585, delta=TOL)
        self.assertAlmostEqual(final["mandatory_after_tax_proceeds"], 0.585, delta=TOL)
        self.assertAlmostEqual(final["total_after_tax_wealth"], 1.17, delta=TOL)
        self.assertEqual(final["cash_interest_gross_cumulative"], 0.0)
        self.assertEqual(final["fixed_income_tax_cumulative"], 0.0)
        self.assertAlmostEqual(final["remaining_fraction_before_terminal"], 0.5, delta=TOL)

    def test_one_row_final_sale_has_zero_accrual(self) -> None:
        env = self._env(["2021-01-01"], [120.0])
        _, infos, _ = self._run(env, [4])
        final = infos[-1]
        self.assertEqual(final["discretionary_sale_count"], 1)
        self.assertEqual(final["mandatory_sale_count"], 0)
        self.assertEqual(final["cash_interest_gross_cumulative"], 0.0)
        self.assertAlmostEqual(final["total_after_tax_wealth"], 1.17, delta=TOL)

    def test_brazil_validation_and_reward_selection(self) -> None:
        env = self._env(["2021-01-01"], [120.0], reward_version=BRAZIL_BASE_REWARD_VERSION)
        _, infos, rewards = self._run(env, [0])
        self.assertAlmostEqual(rewards[0], infos[0]["base_reward"], delta=TOL)
        for dates, prices, basis, message in [
            (["2021-01-01", "2021-01-01"], [120.0, 120.0], None, "strictly increasing"),
            (["2021-01-01"], [0.0], None, "adj_close"),
            (["2021-01-01", "2021-01-02"], [120.0, 120.0], [100.0, 101.0], "constant"),
        ]:
            with self.subTest(message=message):
                bad = self._env(dates, prices, basis=basis)
                with self.assertRaisesRegex(ValueError, message):
                    bad.reset()

        with (ROOT / "configs" / "brazil_sensitivity_rate_1000_v1.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        scenario = resolve_economic_scenario_from_config(config)
        with self.assertRaisesRegex(ValueError, "selected together"):
            TaxAwareEnv("unused.parquet", ["feature"], reward_config={"version": REWARD_A_VERSION}, economic_scenario=scenario)
        with self.assertRaisesRegex(ValueError, "selected together"):
            TaxAwareEnv("unused.parquet", ["feature"], reward_config={"version": BRAZIL_C_LITE_REWARD_VERSION})

    def test_us_literal_regression(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "us.parquet"
        pd.DataFrame({
            "episode_id": ["us"] * 3,
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "tax_transition_date": pd.to_datetime(["2020-01-10"] * 3),
            "unrealized_gains_pct": [0.20] * 3,
            "feature": [0.0, 1.0, 2.0],
        }).to_parquet(path, index=False)
        env = TaxAwareEnv(
            path, ["feature"],
            tax_config={"profile_name": "fixture", "short_term_rate": 0.24, "long_term_rate": 0.15, "niit_rate": 0.0, "apply_niit": False},
            reward_config={"version": REWARD_A_VERSION},
        )
        _, reset = env.reset("us")
        self.assertAlmostEqual(reset["after_tax_total_value"], 0.152, delta=TOL)
        _, first_reward, first_done, _, first = env.step(2)
        self.assertFalse(first_done)
        self.assertAlmostEqual(first_reward, 0.0, delta=TOL)
        self.assertAlmostEqual(first["after_tax_total_value"], 0.152, delta=TOL)
        _, second_reward, second_done, _, second = env.step(0)
        self.assertTrue(second_done)
        self.assertAlmostEqual(second_reward, 0.009, delta=TOL)
        self.assertAlmostEqual(second["after_tax_total_value"], 0.161, delta=TOL)
        self.assertNotIn("total_after_tax_wealth", second)


if __name__ == "__main__":
    unittest.main()
