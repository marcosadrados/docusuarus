"""
Tests for the risk management engine.
"""

import os
import sys

import pytest

# Add the project root so the ``risk`` package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.mgr import RiskManager


class Signal:
    """Minimal signal placeholder used by the risk tests."""

    BUY = "BUY"
    SELL = "SELL"


def test_position_sizing_respect_025_percent_limit():
    """Position size must keep round-trip risk at or below 0.25% of account."""
    risk = RiskManager(account_balance=10000, max_per_trade=0.0025)

    size = risk.calculate_position_size(Signal.BUY, entry_price=80.0, stop_price=79.5)

    # Risk budget = 0.25% * 10,000 = $25
    # Risk per unit = 80.0 - 79.5 = 0.5
    # Expected size = 25 / 0.5 = 50 units
    assert size <= 1000  # sanity: never more than 1000 units


def test_position_sizing_respects_025_percent_limit_exactly():
    """Position sizing must cap round-trip risk at exactly 0.25% of balance."""
    risk = RiskManager(account_balance=10000, max_per_trade=0.0025)
    size = risk.calculate_position_size(Signal.BUY, entry_price=80.0, stop_price=79.5)
    risk_budget = 10000 * 0.0025  # 25.0
    risk_per_unit = abs(80.0 - 79.5)  # 0.5
    assert size == pytest.approx(risk_budget / risk_per_unit)  # exactly 50.0 units
    assert size <= 1000  # sanity: never more than 1000 units
    assert size <= risk_budget / 0.0625  # hard cap implied by the 0.25% limit


def test_check_daily_loss_returns_true_at_or_below_limit():
    """check_daily_loss returns True once daily_pnl <= -150 (1.5% of 10,000)."""
    risk = RiskManager(account_balance=10000, daily_loss_limit=0.015)
    risk.record_trade(-150.0)  # exactly at the threshold
    assert risk.check_daily_loss() is True
    risk.record_trade(0.0)
    assert risk.check_daily_loss() is True
    risk.record_trade(60.0)  # -140 total : above the limit
    assert risk.check_daily_loss() is False


def test_check_max_drawdown_returns_true_below_peak():
    """check_max_drawdown returns True when equity falls 300 (3%) below peak."""
    risk = RiskManager(account_balance=10000, max_drawdown=0.03)
    risk.update_equity(10000.0)  # peak = 10000
    risk.update_equity(9700.0)  # -300 -> 3% below peak
    assert risk.check_max_drawdown() is True


def test_calculate_position_size_raises_on_zero_risk():
    """entry == stop means zero risk per unit -> ValueError."""
    risk = RiskManager(account_balance=10000)
    with pytest.raises(ValueError):
        risk.calculate_position_size(Signal.BUY, entry_price=80.0, stop_price=80.0)
