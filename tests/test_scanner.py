"""Tests for the on-demand signal scanner.

Run with:: pytest tests/test_scanner.py
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.scanner import ScannerConfig, ScannerError, SignalScanner  # noqa: E402
from signals.strategy import Candle, Signal  # noqa: E402


class DummyStrategy:
    """Strategy that always fires BUY with a fixed confidence."""

    def __init__(self, conf=0.8):
        self.conf = conf

    def generate(self, candles):
        return [
            Signal(
                direction="BUY",
                confidence=self.conf,
                strength=0.6,
                reason="dummy",
                timestamp=candles[-1].timestamp,
            )
        ]


class FailingStrategy:
    def generate(self, candles):
        raise RuntimeError("boom")


class FakeFetcher:
    def __init__(self, candles_by_symbol):
        self.candles_by_symbol = {k: list(v) for k, v in candles_by_symbol.items()}

    def fetch_candles(self, symbol, timeframe="5m", since=None, limit=None):
        return [c for c in self.candles_by_symbol.get(symbol, [])]


def make_candles(n=50, base=100.0):
    return [
        Candle(
            timestamp=1700000000 + i * 300,
            open=base,
            high=base + 1.0,
            low=base - 1.0,
            close=base + 0.5,
            volume=1000000 + i,
        )
        for i in range(n)
    ]


class FakeRegistry:
    """Registry whose ``__call__`` returns a strategy class by name."""

    def __init__(self, mapping):
        self._mapping = mapping

    def __call__(self, name):
        return self._mapping.get(name)

    def register(self, name, cls):
        self._mapping[name] = cls


@pytest.fixture
def config():
    return ScannerConfig(timeframe="5m", limit=20, min_confidence=0.5, min_strength=0.4)


def test_scanner_returns_ranked_signals(config):
    symbols = ["CL=F", "GC=F"]
    fetcher = FakeFetcher({"CL=F": make_candles(40), "GC=F": make_candles(40)})
    registry = FakeRegistry({"market_trend": DummyStrategy})
    scanner = SignalScanner(
        registry=registry,
        fetcher=fetcher,
        symbols=symbols,
        strategies=["market_trend"],
        config=config,
    )
    signals = scanner.scan()
    assert len(signals) == 2
    # Higher-confidence strategy ranks first (both same symbol here, so
    # rank by confidence).
    assert signals[0].confidence == 0.8
    # Every signal carries its symbol and connector.
    assert signals[0].symbol == "CL=F"
    assert signals[0].connector == "cme"


def test_scanner_filters_by_confidence_strength(config):
    symbols = ["CL=F"]
    fetcher = FakeFetcher({"CL=F": make_candles(40)})
    registry = FakeRegistry({"market_trend": DummyStrategy})
    scanner = SignalScanner(
        registry=registry,
        fetcher=fetcher,
        symbols=symbols,
        strategies=["market_trend"],
        config=config,
    )
    signals = scanner.scan()
    assert len(signals) == 1
    assert signals[0].confidence == 0.8


def test_scanner_skips_empty_symbol():
    fetcher = FakeFetcher({"CL=F": make_candles(40)})
    registry = FakeRegistry({"market_trend": DummyStrategy})
    scanner = SignalScanner(
        registry=registry,
        fetcher=fetcher,
        symbols=["CL=F", "NOPE=F"],
        strategies=["market_trend"],
    )
    signals = scanner.scan()
    # 'NOPE=F' has no candles -> skipped; only CL=F produces a signal.
    assert len(signals) == 1


def test_scanner_returns_empty_when_no_fetcher():
    registry = FakeRegistry({"market_trend": DummyStrategy})
    scanner = SignalScanner(
        registry=registry, fetcher=None, symbols=["CL=F"], strategies=["market_trend"]
    )
    assert scanner.scan() == []


def test_scanner_applies_config_floor():
    config = ScannerConfig(min_confidence=0.9)
    symbols = ["CL=F"]
    fetcher = FakeFetcher({"CL=F": make_candles(40)})
    registry = FakeRegistry({"market_trend": DummyStrategy})
    scanner = SignalScanner(
        registry=registry,
        fetcher=fetcher,
        symbols=symbols,
        strategies=["market_trend"],
        config=config,
    )
    assert scanner.scan() == []


def test_scanner_resolves_callable_strategy():
    symbols = ["CL=F"]
    fetcher = FakeFetcher({"CL=F": make_candles(40)})
    scanner = SignalScanner(
        registry=FakeRegistry({}),
        fetcher=fetcher,
        symbols=symbols,
        strategies=[DummyStrategy],
    )
    assert len(scanner.strategies) == 1
    signals = scanner.scan()
    assert len(signals) == 1


def test_scanner_raises_on_unknown_registry_name():
    registry = FakeRegistry({})
    fetcher = FakeFetcher({"CL=F": make_candles(40)})
    with pytest.raises(ScannerError):
        SignalScanner(
            registry=registry, fetcher=fetcher, symbols=["CL=F"], strategies=["nope"]
        )


def test_scanner_ignores_failing_strategy():
    symbols = ["CL=F"]
    fetcher = FakeFetcher({"CL=F": make_candles(40)})
    registry = FakeRegistry({"failer": FailingStrategy})
    scanner = SignalScanner(
        registry=registry, fetcher=fetcher, symbols=symbols, strategies=["failer"]
    )
    signals = scanner.scan()
    assert len(signals) == 1
    assert signals[0].direction == "HOLD"


def test_scan_once_returns_summary(config):
    symbols = ["CL=F", "GC=F"]
    fetcher = FakeFetcher({"CL=F": make_candles(40), "GC=F": make_candles(40)})
    registry = FakeRegistry({"market_trend": DummyStrategy})
    scanner = SignalScanner(
        registry=registry,
        fetcher=fetcher,
        symbols=symbols,
        strategies=["market_trend"],
        config=config,
    )
    summary = scanner.scan_once()
    assert summary["scanned"] == 2
    assert summary["signals"] == 2
    assert summary["top"]["symbol"] == "CL=F"


def test_scanner_inherits_default_strategies():
    symbols = ["CL=F"]
    fetcher = FakeFetcher({"CL=F": make_candles(40)})
    registry = FakeRegistry({"market_trend": DummyStrategy, "rule": DummyStrategy})
    scanner = SignalScanner(
        registry=registry, fetcher=fetcher, symbols=symbols, strategies=None
    )
    assert set(s.__class__.__name__ for s in scanner.strategies) == {
        "DummyStrategy",
        "DummyStrategy",
    }
