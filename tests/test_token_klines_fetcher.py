"""Tests for the token kline fetcher."""

from backtest.token_klines_fetcher import (
    SyntheticTokenKlinesFetcher,
    TokenKlinesProvider,
)


def test_synthetic_fetcher_returns_ohlcv():
    fetcher = SyntheticTokenKlinesFetcher(symbol="SOL/USDT", limit=100)
    df = fetcher.fetch_ohlcv()
    assert len(df) == 100
    assert {"timestamp", "open", "high", "low", "close", "volume"} <= set(df.columns)
    assert df.timestamp.dt.tz is not None
    assert df.close.iloc[0] > 78.9


def test_provider_registry_registration():
    TokenKlinesProvider.register("test_provider", SyntheticTokenKlinesFetcher())
    assert "test_provider" in TokenKlinesProvider.list()
    TokenKlinesProvider._instances.pop("test_provider")
