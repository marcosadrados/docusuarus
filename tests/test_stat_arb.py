"""Tests for the pairs / statistical-arbitrage mean-reversion entrypoint."""

from signals.stat_arb import StatArbStrategy
from signals.strategy import Candle

# (lead, relate, beta, window)
PAIRS = [("TEST/A", "TEST/B", 1.0, 6)]


def _candle(close: float) -> Candle:
    return Candle(
        timestamp=1700000000,
        open=100.0,
        high=100.1,
        low=99.9,
        close=close,
        volume=1000000,
    )


def _bars(close: float, n: int = 6) -> list:
    return [_candle(close) for _ in range(n)]


def _bars_with_tail(tail_close: float) -> list:
    # 5 bars at 100 then one at tail_close; window=6 makes the tail bar the
    # last element of the window and cross the +/-2 Z entry threshold.
    return [_candle(100.0) for _ in range(5)] + [_candle(tail_close)]


def _make_strategy() -> StatArbStrategy:
    return StatArbStrategy(PAIRS, relate_prices={"TEST/B": 95.0})


def test_cheap_spread_fires_buy():
    strat = _make_strategy()
    sigs = strat.generate(_bars_with_tail(50.0))  # far below 100 -> cheap
    assert sigs, "a cheap lead spread must fire a BUY"
    assert sigs[0].direction == "BUY"
    assert sigs[0].reason.startswith("stat-arb pairs TEST/A/TEST/B")


def test_rich_spread_fires_sell():
    strat = _make_strategy()
    sigs = strat.generate(_bars_with_tail(150.0))  # far above 100 -> rich
    assert sigs, "a rich lead spread must fire a SELL"
    assert sigs[0].direction == "SELL"


def test_mid_spread_fires_nothing():
    strat = _make_strategy()
    # flat lead: spread Z ~ 0 -> nothing
    assert strat.generate(_bars_with_tail(100.0)) == []


def test_insufficient_history_returns_nothing():
    strat = _make_strategy()
    assert strat.generate(_bars_with_tail(90.0)[:3]) == []


def test_update_caches_relate_close():
    strat = _make_strategy()
    candles = _bars_with_tail(100.0)[:3]
    strat.update("TEST/A", candles)
    assert strat._relate_closes["TEST/B"] == 95.0
    assert strat._relate_closes["TEST/A"] == 100.0
