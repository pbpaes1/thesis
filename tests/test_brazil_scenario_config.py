"""Configuration-only checks; scenario data and the v2 freeze do not exist yet."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import yaml

from src.config.tax_profiles import (
    resolve_economic_scenario_from_config,
    resolve_tax_profile_from_config,
)


ROOT = Path(__file__).resolve().parents[1]
RATE_GRID = {"0700": 0.07, "1000": 0.10, "1050": 0.105, "1200": 0.12, "1375": 0.1375, "1500": 0.15}


def load_config(name: str) -> dict:
    with (ROOT / "configs" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TestBrazilScenarioConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("brazil_sensitivity_rate_1000_v1.yaml")

    def test_all_six_configs_are_isolated_and_fixed(self) -> None:
        output_dirs: set[str] = set()
        for suffix, expected_rate in RATE_GRID.items():
            with self.subTest(suffix=suffix):
                config = load_config(f"brazil_sensitivity_rate_{suffix}_v1.yaml")
                scenario = resolve_economic_scenario_from_config(config)
                self.assertIsNotNone(scenario)
                assert scenario is not None
                self.assertEqual(scenario["cash_account"]["annual_gross_rate"], expected_rate)
                self.assertEqual(scenario["equity_tax"]["rate"], 0.15)
                self.assertEqual(config["training"]["seed"], 42)
                self.assertEqual(config["training"]["discount_factor_gamma"], 1.0)
                self.assertEqual(config["reproducibility"]["python_seed"], 42)
                self.assertEqual(config["training"]["best_model_metric"], "mean_final_total_after_tax_wealth")
                self.assertEqual(config["evaluation"]["fixed_decision_margins"], {"first_sale": 0.070, "subsequent_sale": 0.020})
                self.assertNotIn("first_sale_margin_policies", config["evaluation"])
                self.assertNotIn("tax_profile", config)
                self.assertTrue(config["environment"]["terminal_conditions"]["done_on_full_liquidation"])
                self.assertEqual(config["environment"]["parquet_path"], "data/episodes/drl_episodes_brazil_v1.parquet")
                self.assertEqual(config["environment"]["state_schema_path"], "data/freeze/v2/allowed_state_columns_v2.json")
                output_dir = config["logging"]["output_dir"]
                self.assertEqual(output_dir, f"runs/brazil_sensitivity_rate_{suffix}_v1")
                self.assertNotIn(output_dir, output_dirs)
                output_dirs.add(output_dir)
                self.assertTrue(all(path.startswith(output_dir + "/") for path in config["logging"]["csv_logs"].values()))

    def test_missing_and_invalid_fields_name_the_path(self) -> None:
        changes = [
            (("economic_scenario", "equity_tax", "rate"), None, "economic_scenario.equity_tax.rate"),
            (("economic_scenario", "equity_tax", "rate"), 1.1, "economic_scenario.equity_tax.rate"),
            (("economic_scenario", "equity_tax", "loss_credit"), True, "economic_scenario.equity_tax.loss_credit"),
            (("economic_scenario", "cash_account", "annual_gross_rate"), 0.0, "economic_scenario.cash_account.annual_gross_rate"),
            (("economic_scenario", "cash_account", "annual_gross_rate"), 0.11, "economic_scenario.cash_account.annual_gross_rate"),
            (("economic_scenario", "cash_account", "compounding"), "simple", "economic_scenario.cash_account.compounding"),
            (("economic_scenario", "cash_account", "market_days_per_year"), 365, "economic_scenario.cash_account.market_days_per_year"),
            (("economic_scenario", "cash_account", "interest_tax", "tiers"), [], "economic_scenario.cash_account.interest_tax.tiers"),
            (("economic_scenario", "terminal_horizon"), "tax_transition", "economic_scenario.terminal_horizon"),
            (("economic_scenario", "fx_conversion"), True, "economic_scenario.fx_conversion"),
        ]
        for path, value, expected_message in changes:
            with self.subTest(path=path, value=value):
                config = deepcopy(self.config)
                parent = config
                for key in path[:-1]:
                    parent = parent[key]
                if value is None:
                    parent.pop(path[-1])
                else:
                    parent[path[-1]] = value
                with self.assertRaisesRegex(ValueError, expected_message):
                    resolve_economic_scenario_from_config(config)

    def test_tier_boundary_and_contradictory_legacy_fields_fail(self) -> None:
        for field in ("tax_profile", "short_term_rate", "long_term_rate"):
            with self.subTest(field=field):
                config = deepcopy(self.config)
                config[field] = {"profile_name": "legacy"} if field == "tax_profile" else 0.15
                with self.assertRaisesRegex(ValueError, field):
                    resolve_economic_scenario_from_config(config)

        config = deepcopy(self.config)
        config["economic_scenario"]["equity_tax"]["short_term_rate"] = 0.15
        with self.assertRaisesRegex(ValueError, "economic_scenario.equity_tax.short_term_rate"):
            resolve_economic_scenario_from_config(config)

        config = deepcopy(self.config)
        config["economic_scenario"]["cash_account"]["interest_tax"]["tiers"][0]["max_calendar_days"] = 181
        with self.assertRaisesRegex(ValueError, r"tiers\[0\].max_calendar_days"):
            resolve_economic_scenario_from_config(config)

    def test_us_configs_resolve_without_scenario(self) -> None:
        us_config = load_config("train_reward_c_lite_v5.yaml")
        self.assertEqual(us_config["training"]["discount_factor_gamma"], 0.99)
        self.assertIsNone(resolve_economic_scenario_from_config(us_config))
        self.assertEqual(
            resolve_tax_profile_from_config(us_config, base_dir=ROOT),
            {
                "profile_name": "mass_affluent_individual",
                "short_term_rate": 0.1919,
                "long_term_rate": 0.1018,
                "niit_rate": 0.0,
                "apply_niit": False,
            },
        )

        equal_rates = deepcopy(us_config)
        equal_rates["tax_profile"] = {
            "profile_name": "equal_us_rates",
            "short_term_rate": 0.15,
            "long_term_rate": 0.15,
            "niit_rate": 0.0,
            "apply_niit": False,
        }
        self.assertIsNone(resolve_economic_scenario_from_config(equal_rates))
        self.assertEqual(resolve_tax_profile_from_config(equal_rates)["short_term_rate"], 0.15)


if __name__ == "__main__":
    unittest.main()
