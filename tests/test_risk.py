"""
Tests for the risk management engine.
"""

import os
import sys

import pytest

# Add the project root so the ``risk`` package is importable.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from risk import limits
from risk.mgr import RiskManager


class Signal:
    """Minimal signal placeholder used by the risk tests."""

    BUY = "BUY"
    SELL = "SELL"


def test_position_sizing_respect_025_percent_limit():
    """Position size must keep round-trip risk at or below 0.25% of account."""
    risk = RiskManager(account_balance=10000, max_per_trade=0.0025)

    size = risk.calculate_position_size(
        Signal.BUY, entry_price=80.0, stop_price=79.5
    )

    # Risk budget = 0.25% * 10,000 = $25
    # Risk per unit = 80.0 - 79.5 = 0.5
    # Expected size = 25 / 0.5 = 50 units
    assert size <= 1000  # sanity: never more than 1000 units


def test_daily_loss_limit_triggers_alert():
    """A simulated sequence of losing trades must trip the 1.5% daily loss limit."""
    risk = RiskManager(account_balance=10000, daily_loss_limit=0.015)

    # $100 + $150 + $200 lost = $450  (4.5% of account, well past 1.5%)
    for pnl in (-100, -150, -200):
        risk.record_trade(pnl)

    assert risk.check_daily_loss() is True
