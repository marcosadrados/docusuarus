"""Tests for the comparative performance metrics module."""

import pytest

from backtest.metrics import compute


def test_compute_empty():
    m = compute([])
    assert m["total_trades"] == 0
    assert m["profit_factor"] == 0.0
    assert m["sqn"] == 0.0


def test_compute_basic_totals():
    trades = [
        {"side": "BUY", "gross_pnl": 1.0, "pnl": 0.8},
        {"side": "BUY", "gross_pnl": 0.5, "pnl": 0.4},
        {"side": "SELL", "gross_pnl": -0.6, "pnl": -0.7},
    ]
    m = compute(trades)
    assert m["total_trades"] == 3
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["profit_factor"] == pytest.approx(1.2 / 0.7)
    assert m["avg_win"] == pytest.approx(0.6)
    assert m["avg_loss"] == pytest.approx(0.7)
    assert m["expectancy"] == pytest.approx((0.8 + 0.4 - 0.7) / 3)
    assert m["sqn"] > 0
    assert m["net_pnl"] == pytest.approx(0.5)


def test_compute_with_wins_and_losses_only():
    trades = [
        {"side": "BUY", "gross_pnl": 0.5, "pnl": 0.5},
        {"side": "SELL", "gross_pnl": -0.5, "pnl": -0.5},
    ]
    m = compute(trades)
    assert m["win_rate"] == 0.5
    assert m["profit_factor"] == 1.0
    assert m["avg_win"] == 0.5
    assert m["avg_loss"] == 0.5
