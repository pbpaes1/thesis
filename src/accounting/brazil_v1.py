"""Pure, basis-normalized accounting for the Brazil-inspired v1 scenario.

One unit equals the original equity position's cost basis. These functions do
not track an episode, inspect market data, or settle taxes through time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


EQUITY_TAX_RATE = 0.15
MARKET_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class SaleAmounts:
    gross_proceeds: float
    allocated_basis: float
    realized_gain: float
    equity_tax: float
    cash_deposit: float


@dataclass(frozen=True)
class LotValue:
    principal: float
    gross_value: float
    gross_interest: float
    interest_tax_rate: float
    interest_tax: float
    after_tax_value: float


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def proportional_sale(executed_fraction: float, price_ratio: float) -> tuple[float, float]:
    """Return normalized gross proceeds and allocated basis for an executed sale."""
    fraction = _finite_number(executed_fraction, "executed_fraction")
    ratio = _finite_number(price_ratio, "price_ratio")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("executed_fraction must be in [0, 1].")
    if ratio <= 0.0:
        raise ValueError("price_ratio must be positive.")
    return fraction * ratio, fraction


def positive_gain_equity_tax(gross_proceeds: float, allocated_basis: float) -> float:
    """Charge 15% on positive realized gain; losses confer no credit."""
    proceeds = _finite_number(gross_proceeds, "gross_proceeds")
    basis = _finite_number(allocated_basis, "allocated_basis")
    if proceeds < 0.0 or basis < 0.0:
        raise ValueError("gross_proceeds and allocated_basis must be nonnegative.")
    return EQUITY_TAX_RATE * max(proceeds - basis, 0.0)


def after_tax_sale_deposit(gross_proceeds: float, equity_tax: float) -> float:
    """Deposit full sale proceeds less equity tax, including recovered basis."""
    proceeds = _finite_number(gross_proceeds, "gross_proceeds")
    tax = _finite_number(equity_tax, "equity_tax")
    if proceeds < 0.0 or not 0.0 <= tax <= proceeds:
        raise ValueError("gross_proceeds must be nonnegative and equity_tax must be in [0, gross_proceeds].")
    return proceeds - tax


def calculate_sale(executed_fraction: float, price_ratio: float) -> SaleAmounts:
    """Calculate all normalized components of one discretionary or terminal sale."""
    proceeds, basis = proportional_sale(executed_fraction, price_ratio)
    tax = positive_gain_equity_tax(proceeds, basis)
    return SaleAmounts(
        gross_proceeds=proceeds,
        allocated_basis=basis,
        realized_gain=proceeds - basis,
        equity_tax=tax,
        cash_deposit=after_tax_sale_deposit(proceeds, tax),
    )


def gross_lot_value(principal: float, annual_gross_rate: float, market_intervals: int) -> float:
    """Grow positive normalized principal for elapsed market-day intervals."""
    amount = _finite_number(principal, "principal")
    rate = _finite_number(annual_gross_rate, "annual_gross_rate")
    intervals = _nonnegative_integer(market_intervals, "market_intervals")
    if amount <= 0.0:
        raise ValueError("principal must be positive.")
    if rate <= 0.0:
        raise ValueError("annual_gross_rate must be positive.")
    value = amount * (1.0 + rate) ** (intervals / MARKET_DAYS_PER_YEAR)
    if not math.isfinite(value):
        raise ValueError("gross_lot_value overflowed to a nonfinite amount.")
    return value


def interest_tax_rate(calendar_days_held: int) -> float:
    """Return 22.5% through day 180 and 20% starting on day 181."""
    days = _nonnegative_integer(calendar_days_held, "calendar_days_held")
    return 0.225 if days <= 180 else 0.20


def after_tax_lot_value(
    principal: float,
    annual_gross_rate: float,
    market_intervals: int,
    calendar_days_held: int,
) -> LotValue:
    """Mark one lot net of the interest tax due if redeemed at this date."""
    gross = gross_lot_value(principal, annual_gross_rate, market_intervals)
    days_rate = interest_tax_rate(calendar_days_held)
    amount = float(principal)
    interest = gross - amount
    tax = days_rate * max(interest, 0.0)
    return LotValue(amount, gross, interest, days_rate, tax, gross - tax)
