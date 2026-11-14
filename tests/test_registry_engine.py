"""Self-contained tests for the orchestrator Engine and the strategy registry.

No live broker, no network: synthetic candles, an in-memory fake broker and a
real but tiny RiskManager are used throughout.  Run with::

    pytest tests/test_registry_engine.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

# Project root so ``orchestrator`` / ``signals`` are importable from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.engine import Engine  # noqa: E402
from signals.registry import (  # noqa: E402
    StrategyRegistry,
    get,
    load_strategy_preset,
    register,
)
from signals.strategy import Candle, Signal  # noqa: E402

# ----------------------------------------------------------------------
# Helpers / synthetic fixtures
# ----------------------------------------------------------------------


def make_candles(n: int = 100, base_price: float = 80.0, volume_scale: int = 1) -> list:
    """Synthetic rising-candle bars with a bullish breakout on the last bar."""
    candles = []
    for i in range(n):
        open_price = base_price + i * 0.01
        close = open_price + 0.05
        candles.append(
            Candle(
                timestamp=1700000000 + i * 60,
                open=open_price,
                high=open_price + 0.1,
                low=open_price - 0.1,
                close=close,
                volume=1000000 + i * volume_scale,
            )
        )
    return candles


class AlwaysBuyStrategy:
    """Synthetic strategy that fires one BUY per tick (deterministic)."""

    def generate(self, candles):
        return [
            Signal(
                direction="BUY",
                confidence=0.8,
                strength=0.7,
                reason="synthetic momentum",
                timestamp=candles[-1].timestamp,
            )
        ]


class SyntheticFetcher:
    """Fetcher over synthetic candles stored per symbol."""

    def __init__(self, candles_by_symbol):
        self.candles_by_symbol = {k: list(v) for k, v in candles_by_symbol.items()}
        self.calls = []

    def fetch_candles(self, symbol, timeframe="1m", since=None, limit=None):
        self.calls.append((symbol, timeframe, since, limit))
        candles = self.candles_by_symbol.get(symbol, [])
        return [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]


class FakeBroker:
    """In-memory broker: records orders, reports filled orders."""

    def __init__(self):
        self.orders = []
        self.positions = {}
        self._seq = 0

    def execute_buy(self, symbol, amount=1.0, price=None, order_type="market"):
        self._seq += 1
        order = {
            "symbol": symbol,
            "side": "BUY",
            "amount": amount,
            "order_id": f"buy-{self._seq}",
        }
        self.orders.append(order)
        self.positions[symbol] = self.positions.get(symbol, 0.0) + amount
        return {"id": order["order_id"], "status": "filled", "execution_price": 82.0}

    def execute_sell(self, symbol, amount=1.0, price=None, order_type="market"):
        self._seq += 1
        order = {
            "symbol": symbol,
            "side": "SELL",
            "amount": amount,
            "order_id": f"sell-{self._seq}",
        }
        self.orders.append(order)
        self.positions[symbol] = self.positions.get(symbol, 0.0) - amount
        return {"id": order["order_id"], "status": "filled", "execution_price": 80.0}


@pytest.fixture
def risk():
    """Real RiskManager with a small account (no live broker needed)."""
    from risk.mgr import RiskManager

    return RiskManager(
        account_balance=10000.0,
        max_per_trade=0.0025,
        daily_loss_limit=0.015,
        max_drawdown=0.03,
    )


def run_engine(engine, seconds: float):
    """Start the engine, wait ``seconds`` (both symbol tasks fire in
    parallel), then stop it.  Returns the coroutine so it can be
    awaited inside a real asyncio loop."""

    async def _run():
        engine.start()
        await asyncio.sleep(seconds)
        engine.stop()

    return _run()


# ----------------------------------------------------------------------
# Strategy registry tests
# ----------------------------------------------------------------------


def test_registry_registers_and_retrieves():
    class DummyStrategy:
        pass

    registry = StrategyRegistry()
    registry.register("dummy", DummyStrategy)

    assert registry.get("dummy") is DummyStrategy
    assert registry.has("dummy")
    assert "dummy" in registry
    assert registry.get("missing") is None
    assert registry.names() == ("dummy",)

    # module-level convenience API operated on the default registry
    register("dummy2", DummyStrategy)
    assert get("dummy2") is DummyStrategy
    assert get("missing-registry-name") is None


def test_registry_loads_preset_by_name():
    class DummyStrategy:
        pass

    # register a brand-new strategy by name, then load it by name
    register("dummy", DummyStrategy)
    assert load_strategy_preset("dummy") is not None
    assert load_strategy_preset("dummy").__class__.__name__ == "DummyStrategy"
    with pytest.raises(ValueError):
        load_strategy_preset("missing")

    # builtin presets ship with the registry out of the box
    assert get("market_trend") is not None
    assert load_strategy_preset("market_trend") is not None


# ----------------------------------------------------------------------
# Engine tests
# ----------------------------------------------------------------------


def test_engine_processes_two_symbols_concurrently(risk):
    """Two symbols each get their own asyncio task; both are scanned and
    both open positions (uncorrelated symbols, custom correlation map)."""
    fetcher = SyntheticFetcher(
        {
            "NG=F": make_candles(100),
            "GC=F": make_candles(100),
        }
    )
    broker = FakeBroker()
    engine = Engine(
        symbols=["NG=F", "GC=F"],
        risk_manager=risk,
        fetcher=fetcher,
        strategy=AlwaysBuyStrategy(),
        broker=broker,
        correlation_map={"NG=F": "gas", "GC=F": "metals"},
        config={"tick_interval": 0.005},
    )

    asyncio.run(run_engine(engine, 0.05))

    # both symbols were scanned (parallel tasks, not a serial loop)
    codes = sorted({call[0] for call in fetcher.calls})
    assert codes == ["GC=F", "NG=F"]
    assert len(engine.tasks) == 2

    # both positions opened
    assert set(engine.open_positions) == {"NG=F", "GC=F"}
    assert len(broker.orders) == 2
    assert len(engine.trades) == 2  # one filled order per symbol


def test_correlation_guard_blocks_second_position(risk):
    """CL=F + BZ=F share one correlation family: at most one position may
    exist between them.  The second symbol's attempts are refused."""
    fetcher = SyntheticFetcher(
        {
            "CL=F": make_candles(100),
            "BZ=F": make_candles(100),
        }
    )
    broker = FakeBroker()
    engine = Engine(
        symbols=["CL=F", "BZ=F"],
        risk_manager=risk,
        fetcher=fetcher,
        strategy=AlwaysBuyStrategy(),
        broker=broker,
        correlation_map={"CL=F": "oil", "BZ=F": "oil"},
        config={"tick_interval": 0.005},
    )
    opened, blocked = [], []
    engine.subscribe(Engine.EVENT_POSITION_OPENED, opened.append)
    engine.subscribe(Engine.EVENT_POSITION_BLOCKED, blocked.append)

    asyncio.run(run_engine(engine, 0.05))

    # both symbols were scanned, but only one position was allowed
    codes = sorted({call[0] for call in fetcher.calls})
    assert codes == ["BZ=F", "CL=F"]
    assert len(engine.tasks) == 2
    assert (
        len(engine.open_positions) == 1
    ), "correlation guard: at most 1 open position per family (CL=F+BZ=F)"
    assert len(opened) == 1
    assert opened[0]["symbol"] in ("CL=F", "BZ=F")
    assert len(blocked) >= 1
    assert all(b["symbol"] == "BZ=F" or b["symbol"] == "CL=F" for b in blocked)


def test_engine_executes_and_tracks_real_time_pnl(risk):
    """A filled buy order updates the portfolio P&L in real time."""
    fetcher = SyntheticFetcher({"CL=F": make_candles(100)})
    broker = FakeBroker()
    engine = Engine(
        symbols=["CL=F"],
        risk_manager=risk,
        fetcher=fetcher,
        strategy=AlwaysBuyStrategy(),
        broker=broker,
        config={"tick_interval": 0.005},
    )

    asyncio.run(run_engine(engine, 0.05))

    assert len(engine.trades) == 1
    trade = engine.trades[0]
    assert trade["symbol"] == "CL=F"
    assert trade["side"] == "BUY"
    # last candle close of make_candles(100): 80.0 + 99*0.01 + 0.05 (open+0.05)
    expected_entry = 80.0 + 99 * 0.01 + 0.05
    assert trade["entry_price"] == pytest.approx(expected_entry)
    # entry 81.04, fill 82.0, ~0.123 lots, lot size 1000 -> +~118.5
    assert engine.portfolio_pnl == pytest.approx(118.5, rel=0.05)
    assert engine.portfolio_pnl > 0


def test_stop_cancels_all_symbol_tasks(risk):
    """Ctrl+C / stop() shuts the engine down gracefully: every symbol task
    is cancelled and awaited to completion."""
    fetcher = SyntheticFetcher({"CL=F": make_candles(100)})
    broker = FakeBroker()
    engine = Engine(
        symbols=["CL=F"],
        risk_manager=risk,
        fetcher=fetcher,
        strategy=AlwaysBuyStrategy(),
        broker=broker,
        config={"tick_interval": 0.005},
    )

    asyncio.run(run_engine(engine, 0.05))

    assert all(task.done() for task in engine.tasks)
    assert not engine._running
