"""Phase 4 date derivation, v2 freeze, and control-universe checks."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.generate_brazil_episodes_v1 import (
    NEW_TIME,
    OLD_TIME,
    build_allowed_columns,
    compare_control_splits,
    derive_terminal_horizon,
    file_hash,
    id_list_hash,
    validate_config_targets,
    validate_scenario_frame,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_episodes() -> pd.DataFrame:
    frame = pd.DataFrame({
        "episode_id": ["a", "a", "a", "b", "b"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-04", "2021-01-06", "2021-02-01", "2021-02-02"]),
        "tax_transition_date": pd.to_datetime(["2021-12-31"] * 5),
        OLD_TIME: [0.99, 0.98, 0.97, 0.9, 0.89],
        "adj_close": [120.0] * 5,
        "simulated_purchase_price": [100.0] * 5,
    })
    for i in range(35):
        frame[f"f{i}"] = float(i)
    return frame


class TestBrazilTerminalHorizonState(unittest.TestCase):
    def setUp(self) -> None:
        self.allowed = [f"f{i}" for i in range(35)] + [NEW_TIME]

    def test_clock_uses_actual_last_row_not_tax_transition(self) -> None:
        source = synthetic_episodes()
        original = source.copy(deep=True)
        scenario = derive_terminal_horizon(source)
        self.assertEqual(scenario[NEW_TIME].tolist(), [5 / 365, 2 / 365, 0.0, 1 / 365, 0.0])
        self.assertEqual(scenario[OLD_TIME].tolist(), original[OLD_TIME].tolist())
        pd.testing.assert_frame_equal(source, original)
        pd.testing.assert_frame_equal(scenario[original.columns], original)
        validate_scenario_frame(scenario, self.allowed)

    def test_invalid_order_missingness_duplicates_and_horizon_fail(self) -> None:
        cases = [
            (lambda f: f.loc.__setitem__(1, f.loc[0]), "Duplicate episode/date"),
            (lambda f: f.loc.__setitem__(1, pd.Series({**f.loc[1].to_dict(), "date": pd.Timestamp("2020-12-31")})), "strictly increasing"),
            (lambda f: f.loc.__setitem__((0, "date"), pd.NaT), "dates"),
            (lambda f: f.loc.__setitem__((0, "adj_close"), float("nan")), "adj_close"),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                source = synthetic_episodes()
                mutate(source)
                with self.assertRaisesRegex(ValueError, message):
                    derive_terminal_horizon(source)

        for position, value, message in [
            (0, 4 / 365, "do not match"),
            (1, 0.5, "non-increasing"),
            (2, 0.1, "non-increasing|final row"),
            (0, float("nan"), "finite"),
        ]:
            with self.subTest(position=position, value=value):
                scenario = derive_terminal_horizon(synthetic_episodes())
                scenario.loc[position, NEW_TIME] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_scenario_frame(scenario, self.allowed)

    def test_leakage_and_v1_swap(self) -> None:
        with (ROOT / "data/freeze/v1/allowed_state_columns_v1.json").open(encoding="utf-8") as handle:
            v1 = json.load(handle)
        v2 = build_allowed_columns(v1)
        self.assertEqual(len(v2), 36)
        self.assertNotIn(OLD_TIME, v2)
        self.assertIn(NEW_TIME, v2)
        self.assertEqual([c for c in v2 if c != NEW_TIME], [c for c in v1["allowed_state_columns"] if c != OLD_TIME])
        self.assertEqual(v2.index(NEW_TIME), v1["allowed_state_columns"].index(OLD_TIME))

        scenario = derive_terminal_horizon(synthetic_episodes())
        leaking = self.allowed.copy()
        leaking[0] = OLD_TIME
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_scenario_frame(scenario, leaking)
        leaking[0] = "cash_balance"
        scenario["cash_balance"] = 1.0
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_scenario_frame(scenario, leaking)

    def test_split_comparison_rejects_changed_membership(self) -> None:
        dates = pd.date_range("2021-01-01", periods=10, freq="D")
        frame = pd.DataFrame({"episode_id": [f"ep{i}" for i in range(10)], "date": dates})
        expected = [(f"ep{i}", "train" if i < 7 else "validation" if i == 7 else "test") for i in range(10)]
        with TemporaryDirectory() as temp:
            path = Path(temp) / "splits.csv"
            pd.DataFrame(expected, columns=["episode_id", "split"]).to_csv(path, index=False)
            details = compare_control_splits(frame, path)
            self.assertEqual({key: value["count"] for key, value in details.items()}, {"train": 7, "validation": 1, "test": 2})
            self.assertEqual(details["train"]["ordered_id_sha256"], id_list_hash([f"ep{i}" for i in range(7)]))
            changed = expected.copy()
            changed[7] = ("ep7", "test")
            pd.DataFrame(changed, columns=["episode_id", "split"]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "validation"):
                compare_control_splits(frame, path)

    def test_generated_artifacts_match_control(self) -> None:
        scenario_path = ROOT / "data/episodes/drl_episodes_brazil_v1.parquet"
        v2_dir = ROOT / "data/freeze/v2"
        with (v2_dir / "allowed_state_columns_v2.json").open(encoding="utf-8") as handle:
            allowed_payload = json.load(handle)
        with (v2_dir / "excluded_columns_v2.json").open(encoding="utf-8") as handle:
            excluded_payload = json.load(handle)
        allowed = allowed_payload["allowed_state_columns"]
        excluded = [item["column_name"] for item in excluded_payload["excluded_columns"]]
        self.assertEqual(len(allowed), 36)
        self.assertEqual(len(excluded), 22)
        self.assertIn(OLD_TIME, excluded)
        self.assertNotIn(OLD_TIME, allowed)
        self.assertNotIn("cash_balance", allowed)
        self.assertEqual(set(allowed).intersection(excluded), set())
        original = pd.read_parquet(ROOT / "data/episodes/drl_episodes.parquet", columns=["episode_id", "date"])
        scenario = pd.read_parquet(scenario_path, columns=["episode_id", "date", NEW_TIME])
        pd.testing.assert_frame_equal(scenario[["episode_id", "date"]], original)
        self.assertEqual(len(scenario), 973658)
        self.assertEqual(scenario.episode_id.nunique(), 10119)
        self.assertTrue((scenario.groupby("episode_id", sort=False).tail(1)[NEW_TIME] == 0).all())
        self.assertEqual(set(scenario.episode_id), set(pd.read_csv(ROOT / "runs/train_reward_c_lite_v5_full/episode_splits.csv")["episode_id"]))
        compare_control_splits(scenario, ROOT / "runs/train_reward_c_lite_v5_full/episode_splits.csv")
        summary = (v2_dir / "state_freeze_v2_summary.md").read_text(encoding="utf-8")
        self.assertIn(file_hash(scenario_path), summary)
        self.assertIn("partially observable", summary)
        validate_config_targets(ROOT / "configs")


if __name__ == "__main__":
    unittest.main()
