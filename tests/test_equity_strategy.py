"""Tests for the US-equities scalping strategy (VWAP-pullback + volume-accrual).

Assertions are written against the contract used by the rest of the signal
engine (``signals.strategy.Candle``, ``signals.strategy.Signal``) and by the
backtest runner (``signals.strategy.EquityStrategy``), so the equity strategy
is exercised identically to the futures ``MarketTrendStrategy``.
"""

import os
import sys


# Project root so ``signals`` is importable from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.strategy import Candle, Signal  # noqa: E402
from signals.strategy.equity import EquityStrategy  # noqa: E402


def make_candles(n=60, base_price=150.0, step=0.01, volume=1_500_000.0):
    """Deterministic ascending candles around ``base_price``."""
    candles = []
    for i in range(n):
        open_price = base_price + i * step
        close_price = open_price + 0.05
        candles.append(
            Candle(
                timestamp=1700000000 + i * 60,
                open=open_price,
                high=open_price + 0.1,
                low=open_price - 0.1,
                close=close_price,
                volume=volume,
            )
        )
    return candles


def _window(pullbacks=0, rebounds=0, base_volume=1_500_000.0, spike_volume=4_000_000.0):
    """Build a trailing window that ends exactly at a pullback or rebound."""
    # (candles is a list of Candle, used immediately below)
    n = 30
    candles = []
    for i in range(n):
        # Start from a downtrend so the VWAP sits above the current price.
        close = 150.5 - 0.05 * i
        if pullbacks > 0 and i == (n - 1):
            close = 148.5
        if rebounds > 0 and i == (n - 1):
            close = 151.0
        candles.append(
            Candle(
                timestamp=1700000000 + i * 60,
                open=close - 0.01,
                high=close + 0.02,
                low=close - 0.02,
                close=close,
                volume=spike_volume if i == n - 1 else base_volume,
            )
        )
    return candles


def test_buy_signal_fires_on_vwap_pullback_with_accumulation():
    """Price below VWAP + volume above the window average => BUY."""
    strategy = EquityStrategy(
        lookback=20, pullback=0.003, volume_multiple=1.5, symbol="AAPL"
    )
    candles = _window(pullbacks=1)
    signals = strategy.generate(candles)

    assert len(signals) == 1
    s = signals[0]
    assert isinstance(s, Signal)
    assert s.direction == "BUY"
    assert 0.0 < s.confidence <= 1.0
    assert s.reason
    assert s.timestamp > 0


def test_sell_signal_fires_on_vwap_rebound_with_accumulation():
    """Price above VWAP + volume above the window average => SELL."""
    strategy = EquityStrategy(
        lookback=20, pullback=0.003, volume_multiple=1.5, symbol="AAPL"
    )
    candles = _window(rebounds=1)
    signals = strategy.generate(candles)

    assert len(signals) == 1
    s = signals[0]
    assert isinstance(s, Signal)
    assert s.direction == "SELL"
    assert 0.0 < s.confidence <= 1.0
    assert s.reason


def test_hold_when_price_range_bound():
    """Price near VWAP with only average volume => HOLD (no confirmation)."""
    strategy = EquityStrategy(
        lookback=20, pullback=0.003, volume_multiple=1.5, symbol="AAPL"
    )
    candles = make_candles(n=30, base_price=150.0, step=0.01)
    signals = strategy.generate(candles)

    assert len(signals) == 1
    s = signals[0]
    assert s.direction == "HOLD"
    assert s.reason == "No VWAP signal"


def test_not_enough_data_returns_empty():
    """Strategy refuses to trade before the lookback window is warm."""
    strategy = EquityStrategy(
        lookback=20, pullback=0.003, volume_multiple=1.5, symbol="AAPL"
    )
    candles = make_candles(n=19)
    signals = strategy.generate(candles)
    assert signals == []


def test_signal_fields_are_valid():
    """Every equity signal carries well-scoped, in-range fields."""
    strategy = EquityStrategy(
        lookback=20, pullback=0.003, volume_multiple=1.5, symbol="AAPL"
    )
    candles = _window(pullbacks=1)
    signals = strategy.generate(candles)

    for signal in signals:
        assert signal.direction in ("BUY", "SELL", "HOLD")
        assert 0.0 <= signal.confidence <= 1.0
        assert 0.0 <= signal.strength <= 1.0
        assert isinstance(signal.reason, str) and signal.reason
        assert signal.timestamp > 0
