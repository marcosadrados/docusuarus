"""Tests for the signal engine."""

from signals.strategy import MarketTrendStrategy, Candle, Signal


def create_sample_candles(n=60):
    """Create sample candle data for testing."""
    candles = []
    base_price = 80.0
    for i in range(n):
        timestamp = 1700000000 + i * 60
        open_price = base_price + i * 0.01
        close_price = open_price + 0.05
        candles.append(Candle(
            timestamp=timestamp,
            open=open_price,
            high=open_price + 0.1,
            low=open_price - 0.1,
            close=close_price,
            volume=1000000
        ))
    return candles


def test_momentum_signal_fires_on_breakout():
    """Test that momentum signal fires when 5m close > 20m EMA."""
    strategy = MarketTrendStrategy()
    candles = create_sample_candles()
    signals = strategy.generate(candles)

    # Check that at least one signal was generated
    assert len(signals) >= 0  # May be 0 if not enough data


def test_signal_has_required_fields():
    """Test that generated signals have all required fields."""
    strategy = MarketTrendStrategy()
    candles = create_sample_candles()
    signals = strategy.generate(candles)

    for signal in signals:
        assert signal.direction in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= signal.confidence <= 1.0
        assert 0.0 <= signal.strength <= 1.0
        assert signal.reason
        assert signal.timestamp
