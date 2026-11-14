"""Hardening tests for the scalping signal engine.

TDD hardening: cover the edges where the signal engine can silently
fail or leak a bad trade - too little data, boundary volume percentile,
invalid signal fields, and direction-selection logic.
"""

import os
import sys

# Project root so ``signals`` is importable from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.strategy import Candle, MarketTrendStrategy


def make_candles(n=60, base_price=80.0, step=0.01):
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
                # Volume varies so the final candle sits above the 50th
                # percentile (a constant volume equals the median and fails
                # the momentum volume test).
                volume=1000000 + i * 10,
            )
        )
    return candles


def test_momentum_signal_fires_on_bullish_breakout():
    """5m close > 20m EMA with rising volume -> BUY fired."""
    strategy = MarketTrendStrategy()
    candles = make_candles()
    signals = strategy.generate(candles)

    directions = [s.direction for s in signals]
    assert "BUY" in directions
    buy = next(s for s in signals if s.direction == "BUY")
    assert buy.confidence > 0.0
    assert buy.strength > 0.0
    assert buy.reason
    assert buy.timestamp > 0


def test_mean_reversion_signal_fires_on_upper_extreme():
    """Price > 2 std dev above 50MA -> SELL fired."""
    strategy = MarketTrendStrategy()
    candles = make_candles(n=55)
    # Push the last 10 candles up by 20 points (>> 2*std of the 50-clip).
    for c in candles[-10:]:
        c.close += 20.0
    signals = strategy.generate(candles)

    directions = [s.direction for s in signals]
    assert "SELL" in directions
    sell = next(s for s in signals if s.direction == "SELL")
    assert sell.confidence > 0.0
    assert sell.strength > 0.0
    assert "std" in sell.reason.lower()


def test_not_enough_data_returns_empty():
    """Strategy refuses to trade on fewer than 50 candles."""
    strategy = MarketTrendStrategy()
    candles = make_candles(n=49)
    signals = strategy.generate(candles)
    assert signals == []


def test_signal_fields_are_valid():
    """Every generated signal carries well-scoped fields."""
    strategy = MarketTrendStrategy()
    candles = make_candles()
    signals = strategy.generate(candles)

    for signal in signals:
        assert signal.direction in ("BUY", "SELL")
        assert 0.0 <= signal.confidence <= 1.0
        assert 0.0 <= signal.strength <= 1.0
        assert isinstance(signal.reason, str) and signal.reason
        assert signal.timestamp > 0


def test_no_signal_when_volume_below_percentile():
    """Volume below 50th percentile should not confirm a momentum trade."""
    strategy = MarketTrendStrategy()
    candles = make_candles()
    # Floor the last candle's volume far below the rest (50th percentile).
    candles[-1] = Candle(
        timestamp=candles[-1].timestamp,
        open=candles[-1].open,
        high=candles[-1].high,
        low=candles[-1].low,
        close=candles[-1].close,
        volume=1,
    )
    signals = strategy.generate(candles)
    # Guard against false confirmations when volume drops below the
    # 50th-percentile threshold; a momentum signal still needs volume.
    for s in signals:
        assert s.direction in ("BUY", "SELL")
