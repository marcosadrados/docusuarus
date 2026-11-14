"""Signal strategies for the scalping trading bot."""

from typing import List

from .core import Candle, Signal


class MarketTrendStrategy:
    """Momentum and mean-reversion scalping strategy.

    Rules:
    - Momentum: 5-min close > 20-min EMA AND volume > 50th percentile -> BUY
    - Mean reversion: price > 2 std dev from 50-period MA -> SELL
    """

    def generate(self, candles: List[Candle]) -> List[Signal]:
        """Generate signals from candle data."""
        signals = []
        if len(candles) < 50:
            return signals

        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]

        # Calculate indicators
        ema5 = self._ema(closes, 5)
        ema20 = self._ema(closes, 20)
        ma50 = sum(closes[-50:]) / 50
        std50 = (sum((c - ma50) ** 2 for c in closes[-50:]) / 50) ** 0.5

        current_close = closes[-1]
        current_volume = volumes[-1]
        volume_percentile = self._percentile(volumes, 50)

        # Momentum: 5-min close > 20-min EMA AND volume > 50th percentile
        if ema5[-1] > ema20[-1] and current_volume > volume_percentile:
            confidence = min(0.9, 0.5 + (ema5[-1] - ema20[-1]) / ema20[-1] * 10)
            signals.append(
                Signal(
                    direction="BUY",
                    confidence=confidence,
                    strength=0.6,
                    reason="Momentum: 5m close > 20m EMA",
                    timestamp=candles[-1].timestamp,
                )
            )

        # Mean reversion: price > 2 std dev from 50-period MA
        if current_close > ma50 + 2 * std50:
            confidence = min(
                0.85, 0.5 + (current_close - ma50 - 2 * std50) / std50 * 0.1
            )
            signals.append(
                Signal(
                    direction="SELL",
                    confidence=confidence,
                    strength=0.5,
                    reason="Mean reversion: price > 2 std dev from 50MA",
                    timestamp=candles[-1].timestamp,
                )
            )

        return signals

    def _ema(self, data: List[float], period: int) -> List[float]:
        """Calculate exponential moving average."""
        if len(data) < period:
            return [0.0] * len(data)

        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        result = [ema]

        for i in range(period, len(data)):
            ema = (data[i] - ema) * multiplier + ema
            result.append(ema)

        return result

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
