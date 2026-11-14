"""Shared primitives for the signals package."""

from dataclasses import dataclass
from typing import List


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    direction: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    strength: float  # 0.0 to 1.0
    reason: str
    timestamp: int
    symbol: str = ""  # signal service / scanner only
    connector: str = ""  # data-source connector, e.g. 'cme', 'binance'


__all__ = ["Candle", "Signal"]
