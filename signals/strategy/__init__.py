"""Signal strategies for the scalping trading bot.

``signals.strategy`` is a lightweight package (rather than a single module) so
that the equity strategy can live alongside the futures strategy without a
name clash on the package root.

Exports
-------
Candle
Signal
MarketTrendStrategy
EquityStrategy
EquitySignal
"""

from .core import Candle, Signal
from .equity import EquityStrategy, EquitySignal  # noqa: F401
from .strategy import MarketTrendStrategy  # noqa: F401

__all__ = [
    "Candle",
    "Signal",
    "MarketTrendStrategy",
    "EquityStrategy",
    "EquitySignal",
]
