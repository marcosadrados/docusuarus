"""Signal engine that orchestrates strategy generation."""

from typing import List

from .strategy import MarketTrendStrategy, Candle, Signal


class SignalEngine:
    """Orchestrates signal generation using configured strategies."""

    def __init__(self):
        self.strategies = [MarketTrendStrategy()]

    def add_strategy(self, strategy: MarketTrendStrategy):
        """Add a strategy to the engine."""
        self.strategies.append(strategy)

    def generate_signals(self, candles: List[Candle]) -> List[Signal]:
        """Generate signals from all configured strategies."""
        all_signals = []
        for strategy in self.strategies:
            signals = strategy.generate(candles)
            all_signals.extend(signals)
        return all_signals
