"""Multi-asset, event-driven orchestrator engine.

Replaces the single-symbol hard-coded :class:`orchestrator.main.Orchestrator`
with an :class:`Engine` that:

* accepts a list of :class:`Symbol` entries (``'CL=F'`` plus its market type
  and base price),
* runs one :class:`asyncio.Task` per symbol so every market is scanned and
  executed in parallel (event-driven, Nautilus-style),
* enforces a *per-symbol* risk limit through the supplied
  :class:`risk.mgr.RiskManager` and a *portfolio-level correlation guard*:
  a symbol may open a position only if no symbol in its correlation family
  already has an open position (the default map treats ``CL=F`` + ``BZ=F``
  as one family, so they may hold at most one position between them),
* broadcasts an event whenever a strategy fires; the event subscriber asks
  the :class:`RiskManager` for a size, executes through the broker and keeps
  the portfolio P&L current, and
* shuts down gracefully on ``Ctrl+C`` or :meth:`Engine.stop` (all symbol
  tasks are cancelled and awaited).

The engine is broker- and data-source-agnostic: it needs any object
exposing ``fetch_candles`` and any broker exposing ``execute_buy``/
``execute_sell``.  Supply no broker to run *dry-run*: orders are recorded
instead of sent, so the engine can be fully tested with synthetic fixtures
and no live broker.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from signals.strategy import Candle, Signal

logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Base exception for the orchestrator engine."""


class DuplicateSymbolError(EngineError):
    """Raised when the same symbol code is listed twice."""


@dataclass(frozen=True)
class Symbol:
    """A tradable instrument.

    Parameters
    ----------
    code:
        cTrader CFD code or crypto pair quote (e.g. ``'CL=F'`` or ``'BTC/USDT'``).
    market_type:
        Market family used by the correlation guard, e.g. ``'futures'`` or
        ``'crypto'``.
    base_price:
        Last-known price used as the fallback entry when no candle has been
        fetched for the current tick yet.
    connector:
        Data-source connector id (e.g. ``'cme'``, ``'binance'``).
    exchange:
        Alias for :attr:`connector` (kept for backwards compatibility).
    lot_size:
        Base units per lot (e.g. 1000 for WTI CL=F, 1.0 for per-contract BTC).
    """

    code: str
    market_type: str = "futures"
    base_price: float = 0.0
    connector: str = "cme"
    exchange: Optional[str] = None
    lot_size: float = 1.0


class EventBus:
    """Tiny publish/subscribe bus for engine events."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List] = {}

    def subscribe(self, event_type: str, handler: Any) -> Any:
        """Register ``handler(payload)`` for ``event_type``."""
        self._handlers.setdefault(event_type, []).append(handler)
        return handler

    def unsubscribe(self, event_type: str, handler: Any) -> None:
        """Unregister a previously subscribed handler."""
        self._handlers.get(event_type, []).remove(handler)

    def publish(self, event_type: str, payload: Any = None) -> None:
        """Call every handler for ``event_type`` with ``payload``."""
        for handler in list(self._handlers.get(event_type, [])):
            try:
                handler(payload)
            except Exception:
                logger.exception("Event handler failed for %s", event_type)


class Engine:
    """Multi-asset event-driven orchestrator.

    Parameters
    ----------
    symbols:
        One or more :class:`Symbol` entries.  Bare strings (``'CL=F'``) are
        accepted and normalised to :class:`Symbol`.
    risk_manager:
        Configured :class:`risk.mgr.RiskManager` instance; its
        ``calculate_position_size`` drives every order size.
    fetcher:
        Data source with ``fetch_candles(symbol, timeframe, limit)``.
    strategy:
        Strategy object with ``generate(candles) -> list[Signal]``.
    broker:
        Optional broker with ``execute_buy(symbol, amount, price, order_type)``
        and ``execute_sell``.  When ``None`` the engine runs in *dry-run*
        mode and records the orders it would have placed.
    config:
        Optional dict; recognised keys: ``tick_interval`` (seconds between
        scan cycles), everything else is ignored.
    correlation_map:
        Mapping symbol code -> correlation family name.  Symbols that share
        a family are treated as correlated and may never hold more than
        ``max_open_per_family`` positions between them.  Defaults come from
        :attr:`Engine.DEFAULT_CORRELATION_MAP`.
    lot_sizes:
        Mapping symbol code -> base units per lot (1000 for WTI CL=F, ...).
        Defaults come from :attr:`Engine.DEFAULT_LOT_SIZES`.
    max_open_per_family:
        Portfolio-level cap on open positions per correlation family
        (default 1).
    logger_:
        Optional logger (defaults to the ``engine`` logger).

    Events published on the engine event bus::

        engine_started, symbol_started, symbol_stopped,
        signal_fire, position_requested, position_opened,
        position_blocked, trade_executed, portfolio_pnl, engine_stopped
    """

    #: Fallback symbol lot sizes (base units per lot), mirrors the legacy
    #: Orchestrator.SYMBOL_LOT_SIZES.
    DEFAULT_LOT_SIZES = {
        "CL=F": 1000.0,
        "BZ=F": 1000.0,
        "NG=F": 10000.0,
        "GC=F": 100.0,
        "HG=F": 250.0,
        "SI=F": 50.0,
        "XAU=F": 100.0,
        "XAG=F": 5000.0,
    }

    #: Default correlation map: symbols sharing a family are treated as
    #: correlated; the portfolio cap applies per family.  The crude complex
    #: (WTI CL=F, Brent BZ=F) is the canonical correlated pair.
    DEFAULT_CORRELATION_MAP = {
        "CL=F": "oil",
        "BZ=F": "oil",
        "NG=F": "gas",
        "GC=F": "metals",
        "HG=F": "metals",
        "SI=F": "metals",
        "XAU=F": "metals",
        "XAG=F": "metals",
    }

    #: Default connector map: symbol code -> data-source connector id
    #: (``'cme'`` = futures, ``'binance'`` = crypto).  Only the markets below
    #: are pre-registered; additional connectors are added by the
    #: Orchestrator/Gateway from config.
    DEFAULT_CONNECTOR_MAP = {
        "CL=F": "cme",
        "BZ=F": "cme",
        "NG=F": "cme",
        "GC=F": "cme",
        "HG=F": "cme",
        "SI=F": "cme",
        "XAU=F": "cme",
        "XAG=F": "cme",
        "BTC/USDT": "binance",
        "ETH/USDT": "binance",
        "SOL/USDT": "binance",
        "XRP/USDT": "binance",
        "BNB/USDT": "binance",
        "ADA/USDT": "binance",
        "DOGE/USDT": "binance",
        "AVAX/USDT": "binance",
    }

    #: Engine event types (see :class:`EventBus` for the bus itself).
    EVENT_ENGINE_STARTED = "engine_started"
    EVENT_ENGINE_STOPPED = "engine_stopped"
    EVENT_SYMBOL_STARTED = "symbol_started"
    EVENT_SYMBOL_STOPPED = "symbol_stopped"
    EVENT_SIGNAL_FIRE = "signal_fire"
    EVENT_POSITION_REQUESTED = "position_requested"
    EVENT_POSITION_OPENED = "position_opened"
    EVENT_POSITION_BLOCKED = "position_blocked"
    EVENT_TRADE_EXECUTED = "trade_executed"
    EVENT_PORTFOLIO_PNL = "portfolio_pnl"

    def __init__(
        self,
        symbols: Sequence[Any],
        risk_manager: Any,
        fetcher: Any,
        strategy: Any,
        *,
        broker: Any = None,
        connector_map: Optional[Dict[str, str]] = None,
        fetcher_map: Optional[Dict[str, Any]] = None,
        broker_map: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        correlation_map: Optional[Dict[str, str]] = None,
        lot_sizes: Optional[Dict[str, float]] = None,
        max_open_per_family: int = 1,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.symbols: List[Symbol] = [self._coerce_symbol(s) for s in symbols]
        self._check_duplicates()
        self.risk_manager = risk_manager
        self.fetcher = fetcher
        self.strategy = strategy
        self.broker = broker
        self.config = dict(config or {})
        self.correlation_map: Dict[str, str] = dict(self.DEFAULT_CORRELATION_MAP)
        if correlation_map:
            self.correlation_map.update(correlation_map)
        self.lot_sizes: Dict[str, float] = dict(self.DEFAULT_LOT_SIZES)
        if lot_sizes:
            self.lot_sizes.update(lot_sizes)
        self.connector_map: Dict[str, str] = dict(self.DEFAULT_CONNECTOR_MAP)
        if connector_map:
            self.connector_map.update(connector_map)
        self.fetcher_map: Dict[str, Any] = {}
        self.broker_map: Dict[str, Any] = {}
        if fetcher_map:
            self.fetcher_map.update(fetcher_map)
        if broker_map:
            self.broker_map.update(broker_map)
        self.max_open_per_family = max_open_per_family
        self.tick_interval: float = float(self.config.get("tick_interval", 0.01))
        self.logger = logger_ or logging.getLogger(f"{__name__}.Engine")

        self._lock = threading.RLock()
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._running = False

        # Live portfolio state, updated as trades are opened (and, in a full
        # implementation, closed), plus the audit trail.
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_pnl: float = 0.0
        self._event_bus = EventBus()

    # ------------------------------------------------------------------ #
    # Source selection (per-symbol data source + broker)
    # ------------------------------------------------------------------ #

    def _fetcher_for(self, symbol: Symbol) -> Any:
        """Return the data source for ``symbol`` (connector id -> fetcher)."""
        return self.fetcher_map.get(symbol.connector, self.fetcher)

    def _broker_for(self, symbol: Symbol) -> Any:
        """Return the broker for ``symbol`` (connector id -> broker)."""
        return self.broker_map.get(symbol.connector, self.broker)

    # ------------------------------------------------------------------
    # Symbol normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_symbol(symbol: Any) -> Symbol:
        """Accept ``Symbol``, a symbol string, or any object with ``code``."""
        if isinstance(symbol, Symbol):
            return symbol
        if isinstance(symbol, str):
            return Symbol(code=symbol.strip().upper())
        if hasattr(symbol, "code"):
            s = symbol
            return Symbol(
                code=str(getattr(s, "code")).upper(),
                market_type=getattr(s, "market_type", "futures"),
                base_price=float(getattr(s, "base_price", 0.0)),
            )
        raise TypeError(f"cannot coerce {symbol!r} to Symbol")

    def _check_duplicates(self) -> None:
        seen: Dict[str, List[Symbol]] = {}
        for s in self.symbols:
            seen.setdefault(self._code(s), []).append(s)
        duplicates = {c: ids for c, ids in seen.items() if len(ids) > 1}
        if duplicates:
            raise DuplicateSymbolError(f"duplicate symbol codes: {sorted(duplicates)}")

    def _code(self, symbol: Any) -> str:
        return self._coerce_symbol(symbol).code

    def _family(self, symbol: Any) -> str:
        return self.correlation_map.get(self._code(symbol), "uncorrelated")

    def _lot_size(self, symbol: Any) -> float:
        return self.lot_sizes.get(self._code(symbol), 1.0)

    def _entry_price(self, symbol: Any, candles: Sequence[Candle]) -> float:
        if candles:
            return self._close(candles[-1])
        return float(getattr(symbol, "base_price", 0.0) or 0.0)

    @staticmethod
    def _close(candle: Any) -> float:
        if isinstance(candle, Candle):
            return float(candle.close)
        if isinstance(candle, dict):
            return float(candle["close"])
        if isinstance(candle, (list, tuple)):
            return float(candle[4])
        raise TypeError(f"cannot read close from {candle!r}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tasks(self) -> tuple:
        """All per-symbol asyncio tasks currently managed by the engine."""
        return tuple(self._tasks)

    @property
    def symbols_codes(self) -> tuple:
        return tuple(s.code for s in self.symbols)

    def subscribe(self, event_type: str, handler: Any) -> Any:
        """Subscribe ``handler(payload)`` to ``event_type``."""
        return self._event_bus.subscribe(event_type, handler)

    def unsubscribe(self, event_type: str, handler: Any) -> None:
        """Unsubscribe a handler."""
        self._event_bus.unsubscribe(event_type, handler)

    def broadcast(self, event_type: str, payload: Any = None) -> None:
        """Publish an event on the engine event bus."""
        self._event_bus.publish(event_type, payload)

    def can_open_position(self, symbol: Any) -> tuple[bool, str]:
        """Portfolio-level correlation guard (and per-symbol cap).

        Returns ``(allowed, reason)``.  A position is allowed only if the
        symbol is not already open and no open position in the same
        correlation family exists.  The per-symbol contract is one open
        position per symbol at a time.
        """
        code = self._code(symbol)
        family = self._family(code)
        with self._lock:
            if self.open_positions.get(code):
                return False, f"position already open for {code}"
            for other in self.open_positions:
                if self._family(other) == family:
                    return False, (
                        f"correlation guard: family {family!r} already has an "
                        f"open position ({other})"
                    )
            return True, "ok"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start one scan/trade task per symbol (asyncio Tasks).

        Announces :attr:`EVENT_ENGINE_STARTED` and returns once every
        symbol task has been scheduled.  Ctrl+C (KeyboardInterrupt) or
        :meth:`stop` shut the engine down gracefully.
        """
        if self._running:
            raise EngineError("Engine is already running")
        self._stop_event.clear()
        loop = asyncio.get_running_loop()
        self._tasks = [
            loop.create_task(self._scan_symbol(symbol)) for symbol in self.symbols
        ]
        self._running = True
        self.broadcast(self.EVENT_ENGINE_STARTED, {"symbols": self.symbols_codes})
        self.logger.info(
            "Engine started for %d symbols: %s",
            len(self.symbols),
            ", ".join(self.symbols_codes),
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Graceful shutdown: stop the scan loop, cancel every symbol task,
        await their completion and (optionally) close the fetcher."""
        if not self._running:
            return
        self._stop_event.set()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            try:
                asyncio.get_running_loop().run_until_complete(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                )
            except RuntimeError:
                # Called outside a running loop (e.g. unit-test teardown).
                pass
        try:
            if hasattr(self.fetcher, "close"):
                self.fetcher.close()
        except Exception:
            self.logger.exception("fetcher.close() failed")
        self._running = False
        self.broadcast(self.EVENT_ENGINE_STOPPED, {"symbols": self.symbols_codes})
        self.logger.info("Engine stopped")

    # ------------------------------------------------------------------
    # Per-symbol scan loop
    # ------------------------------------------------------------------

    async def _scan_symbol(self, symbol: Symbol) -> None:
        """One symbol's event loop: fetch -> normalise -> scan -> trade."""
        self.broadcast(self.EVENT_SYMBOL_STARTED, {"symbol": symbol.code})
        self.logger.info("Starting symbol %s", symbol.code)
        try:
            while not self._stop_event.is_set():
                try:
                    self._scan_once(symbol)
                except Exception:
                    self.logger.exception("Scan failed for symbol %s", symbol.code)
                try:
                    await asyncio.sleep(self.tick_interval)
                except asyncio.CancelledError:
                    raise
        finally:
            self.broadcast(self.EVENT_SYMBOL_STOPPED, {"symbol": symbol.code})
            self.logger.info("Stopped symbol %s", symbol.code)

    def _scan_once(self, symbol: Symbol) -> None:
        """Fetch, normalise, scan for signals and act on them (one cycle)."""
        fetcher = self._fetcher_for(symbol)
        candles = self._fetch_and_normalize(fetcher, symbol)
        if not candles:
            return
        self.risk_manager.add_candles(candles)
        entry = self._entry_price(symbol, candles)
        atr = self.risk_manager.current_atr
        for signal in self.strategy.generate(candles):
            self.broadcast(
                self.EVENT_SIGNAL_FIRE,
                {"symbol": symbol.code, "signal": signal},
            )
            if signal.direction == "BUY":
                self._on_buy(symbol, signal, entry, atr)
            elif signal.direction == "SELL":
                self._on_sell(symbol, signal, entry, atr)

    def _on_buy(
        self, symbol: Symbol, signal: Signal, entry: float, atr: Optional[float]
    ) -> None:
        self._request_position(symbol, signal, entry, atr, "BUY")

    def _on_sell(
        self, symbol: Symbol, signal: Signal, entry: float, atr: Optional[float]
    ) -> None:
        self._request_position(symbol, signal, entry, atr, "SELL")

    def _request_position(
        self,
        symbol: Symbol,
        signal: Signal,
        entry: float,
        atr: Optional[float],
        side: str,
    ) -> None:
        allowed, reason = self.can_open_position(symbol)
        if not allowed:
            self.broadcast(
                self.EVENT_POSITION_BLOCKED,
                {"symbol": symbol.code, "side": side, "reason": reason},
            )
            self.logger.warning(
                "Position for %s refused (%s): %s",
                symbol.code,
                side,
                reason,
            )
            return
        size = self._trade_size(symbol, signal, entry, atr)
        order = self._execute(symbol, signal, size, side)
        with self._lock:
            self.open_positions[symbol.code] = {
                "symbol": symbol.code,
                "side": side,
                "size": size,
                "entry_price": entry,
                "order": order,
            }
        self._record_trade(symbol, side, size, order, entry)
        self.broadcast(
            self.EVENT_POSITION_OPENED,
            {"symbol": symbol.code, "side": side, "size": size, "order": order},
        )

    # ------------------------------------------------------------------
    # Trading helpers
    # ------------------------------------------------------------------

    def _trade_size(
        self, symbol: Symbol, signal: Signal, entry: float, atr: Optional[float]
    ) -> float:
        """Risk-budgeted size in lots, paced by ATR volatility."""
        stop = entry * 0.9975  # 0.25% protective stop (legacy convention)
        return self.risk_manager.calculate_position_size(
            signal=signal.direction,
            entry_price=entry,
            stop_price=stop,
            atr=atr,
            lot_size=self._lot_size(symbol),
        )

    def _execute(
        self, symbol: Symbol, signal: Signal, size: float, side: str
    ) -> Dict[str, Any]:
        broker = self._broker_for(symbol)
        if broker is None:
            return self._dry_run(symbol, side, size)
        if side == "BUY":
            return broker.execute_buy(
                symbol=str(symbol.code),
                amount=size,
                price=None,
                order_type="market",
            )
        return broker.execute_sell(
            symbol=str(symbol.code),
            amount=size,
            price=None,
            order_type="market",
        )

    def _dry_run(self, symbol: Symbol, side: str, size: float) -> Dict[str, Any]:
        """Record an order that would have been placed (no live broker)."""
        order = {
            "symbol": symbol.code,
            "side": side,
            "amount": size,
            "order_id": f"dryrun-{len(self.trades)}",
            "status": "dry-run",
        }
        self.trades.append(order)
        return order

    def _record_trade(
        self,
        symbol: Symbol,
        side: str,
        size: float,
        order: Dict[str, Any],
        entry: float,
    ) -> Dict[str, Any]:
        """Persist the trade and update the portfolio P&L in real time."""
        trade = {
            "symbol": symbol.code,
            "side": side,
            "size": size,
            "entry_price": entry,
            "order_id": order.get("order_id"),
            "timestamp": order.get("timestamp", time.time()),
            "status": order.get("status", "filled"),
        }
        self.trades.append(trade)
        fill = order.get("execution_price")
        if fill is not None and size:
            pnl = self._pnl_for(symbol, side, size, entry, float(fill))
            self.portfolio_pnl += pnl
            self.broadcast(
                self.EVENT_PORTFOLIO_PNL,
                {"portfolio_pnl": self.portfolio_pnl},
            )
        return trade

    def _pnl_for(
        self, symbol: Symbol, side: str, size: float, entry: float, fill: float
    ) -> float:
        """Mark-to-market P&L in account currency.

        size is expressed in broker lots, so the P&L is sized * the symbol's
        lot size (base units per lot) * the price move since entry.
        """
        direction = 1.0 if side == "BUY" else -1.0
        return direction * size * self._lot_size(symbol) * (fill - entry)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _fetch_and_normalize(self, fetcher: Any, symbol: Symbol) -> List[Candle]:
        rows = fetcher.fetch_candles(
            self._code(symbol),
            "1m",
            limit=100,
        )
        candles: List[Candle] = []
        for row in rows:
            if isinstance(row, Candle):
                candles.append(row)
            elif isinstance(row, dict):
                candles.append(
                    Candle(
                        timestamp=row["timestamp"],
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row.get("volume", 0.0),
                    )
                )
            elif isinstance(row, (list, tuple)):
                candles.append(Candle(row[0], row[1], row[2], row[3], row[4], row[5]))
            else:
                raise TypeError(f"unsupported candle row: {row!r}")
        return candles
