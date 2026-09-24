"""Independent arithmetic fixtures for the pure Brazil v1 accounting helpers."""

from __future__ import annotations

import unittest

from src.accounting.brazil_v1 import (
    after_tax_lot_value,
    after_tax_sale_deposit,
    calculate_sale,
    gross_lot_value,
    interest_tax_rate,
    positive_gain_equity_tax,
    proportional_sale,
)


TOL = 1e-10


class TestBrazilFlatTaxCashAccount(unittest.TestCase):
    def test_gain_and_loss_sale_proceeds_include_basis(self) -> None:
        proceeds, basis = proportional_sale(1.0, 1.2)
        self.assertAlmostEqual(proceeds, 1.2, delta=TOL)
        self.assertAlmostEqual(basis, 1.0, delta=TOL)
        self.assertAlmostEqual(positive_gain_equity_tax(proceeds, basis), 0.03, delta=TOL)
        self.assertAlmostEqual(after_tax_sale_deposit(proceeds, 0.03), 1.17, delta=TOL)
        gain_sale = calculate_sale(1.0, 1.2)
        self.assertAlmostEqual(gain_sale.cash_deposit, 1.17, delta=TOL)
        self.assertAlmostEqual(gain_sale.equity_tax, 0.03, delta=TOL)

        loss_sale = calculate_sale(1.0, 0.8)
        self.assertAlmostEqual(loss_sale.realized_gain, -0.2, delta=TOL)
        self.assertEqual(loss_sale.equity_tax, 0.0)
        self.assertAlmostEqual(loss_sale.cash_deposit, 0.8, delta=TOL)

    def test_full_year_cash_growth_and_tax(self) -> None:
        self.assertAlmostEqual(gross_lot_value(1.17, 0.10, 252), 1.287, delta=TOL)
        lot = after_tax_lot_value(1.17, 0.10, 252, 365)
        self.assertAlmostEqual(lot.principal, 1.17, delta=TOL)
        self.assertAlmostEqual(lot.gross_interest, 0.117, delta=TOL)
        self.assertAlmostEqual(lot.interest_tax, 0.0234, delta=TOL)
        self.assertAlmostEqual(lot.after_tax_value, 1.2636, delta=TOL)

    def test_calendar_day_tiers_with_same_market_age(self) -> None:
        at_180 = after_tax_lot_value(1.17, 0.10, 126, 180)
        at_181 = after_tax_lot_value(1.17, 0.10, 126, 181)
        self.assertEqual(interest_tax_rate(180), 0.225)
        self.assertEqual(interest_tax_rate(181), 0.20)
        self.assertAlmostEqual(at_180.gross_value, 1.227106352359, delta=TOL)
        self.assertAlmostEqual(at_180.after_tax_value, 1.214257423078, delta=TOL)
        self.assertAlmostEqual(at_181.after_tax_value, 1.215685081887, delta=TOL)
        self.assertAlmostEqual(at_181.after_tax_value - at_180.after_tax_value, 0.001427658809, delta=TOL)

    def test_partial_sale_and_terminal_sale(self) -> None:
        half = calculate_sale(0.5, 1.2)
        self.assertAlmostEqual(half.gross_proceeds, 0.6, delta=TOL)
        self.assertAlmostEqual(half.allocated_basis, 0.5, delta=TOL)
        self.assertAlmostEqual(half.equity_tax, 0.015, delta=TOL)
        self.assertAlmostEqual(half.cash_deposit, 0.585, delta=TOL)
        invested_half = after_tax_lot_value(0.585, 0.10, 252, 365)
        unsold_half_at_terminal = 0.585  # 0.6 proceeds - 0.015 tax; no interest.
        self.assertAlmostEqual(invested_half.after_tax_value + unsold_half_at_terminal, 1.2168, delta=TOL)

        terminal_sale = calculate_sale(1.0, 1.2)
        terminal_lot = after_tax_lot_value(terminal_sale.cash_deposit, 0.10, 0, 0)
        self.assertEqual(terminal_lot.gross_interest, 0.0)
        self.assertEqual(terminal_lot.interest_tax, 0.0)
        self.assertAlmostEqual(terminal_lot.after_tax_value, 1.17, delta=TOL)

    def test_invalid_pure_inputs_fail(self) -> None:
        invalid = [
            (lambda: proportional_sale(-0.1, 1.2), "executed_fraction"),
            (lambda: proportional_sale(0.5, 0.0), "price_ratio"),
            (lambda: proportional_sale(0.5, float("nan")), "price_ratio"),
            (lambda: positive_gain_equity_tax(0.5, -0.1), "allocated_basis"),
            (lambda: after_tax_sale_deposit(0.5, 0.6), "equity_tax"),
            (lambda: gross_lot_value(0.0, 0.10, 1), "principal"),
            (lambda: gross_lot_value(1.0, 0.0, 1), "annual_gross_rate"),
            (lambda: gross_lot_value(1.0, 0.10, -1), "market_intervals"),
            (lambda: interest_tax_rate(-1), "calendar_days_held"),
        ]
        for call, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    call()


if __name__ == "__main__":
    unittest.main()
