"""Signal scanner: on-demand market scan across all registered instruments.

The scanner is a fire-and-forget consumer of market data.  It fetches a
rolling window of recent candles for every symbol in the market set, runs
one or more signal strategies over each window, and returns a ranked list of
the strongest signals.  It does **not** open positions or place orders
itself; it only discovers signal pressure across the market.

The scanner exists to (a) give the strategy registry a market-wide view
before execution, and (b) feed the signal-service channel (see
:mclass:`monitoring.notifier` > ``signal_fire`` handler) so status tools and
webhooks see a unified signal surface.

Usage::

    from signals.scanner import SignalScanner
    from signals.registry import get

    scanner = SignalScanner(
        registry=get,          # strategy registry callable
        fetcher=my_fetcher,    # data source with fetch_candles()
        symbols=my_symbols,    # list of Symbol
        strategies=['market_trend', 'rule'],
        timeframe='5m',
        limit=100,
    )
    signals = scanner.scan()          # ranked list of Signal
    for s in signals:
        print(s.symbol, s.direction, s.confidence)

Scan results are consumable by the control surface, the marketplace
presets, or a downstream signal consumer (e.g. TradingView alert).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from signals.registry import get as _default_get
from signals.strategy import Candle, Signal

logger = logging.getLogger(__name__)

#: How a symbol object may be described.
SymbolLike = Union[str, Any]  # code or Symbol(dataclass)


class ScannerError(Exception):
    """Base exception for the signal scanner."""


class ScannerConfig:
    """Plain configuration for a scan run."""

    def __init__(
        self,
        timeframe: str = "5m",
        limit: int = 100,
        min_confidence: float = 0.0,
        min_strength: float = 0.0,
    ) -> None:
        self.timeframe = timeframe
        self.limit = limit
        self.min_confidence = min_confidence
        self.min_strength = min_strength


class SignalScanner:
    """Run one or more strategies over every symbol and rank the results."""

    #: Default strategy name order when none is supplied.
    DEFAULT_STRATEGIES: List[str] = ["market_trend", "rule"]

    def __init__(
        self,
        registry: Any = None,
        fetcher: Any = None,
        symbols: Sequence[SymbolLike] = None,
        strategies: Optional[Sequence[Union[str, Any]]] = None,
        connector_map: Optional[Dict[str, str]] = None,
        config: Optional[ScannerConfig] = None,
    ) -> None:
        self.registry = registry if registry is not None else _default_get
        self.fetcher = fetcher
        self.symbols = list(symbols or [])
        self.timeframe = config.timeframe if config else "5m"
        self.limit = config.limit if config else 100
        self.min_confidence = config.min_confidence if config else 0.0
        self.min_strength = config.min_strength if config else 0.0
        self.connector_map: Dict[str, str] = dict(connector_map or {})

        # Resolve strategy names to callables (instances).
        self.strategies: List[Any] = []
        for name in self.DEFAULT_STRATEGIES if strategies is None else strategies:
            call = self._resolve_strategy(name)
            if call is not None:
                self.strategies.append(call)

        self.config = config or ScannerConfig()

    # ------------------------------------------------------------------ #
    # Strategy resolution
    # ------------------------------------------------------------------ #

    def _resolve_strategy(self, name: Union[str, Any]) -> Optional[Any]:
        """Resolve a strategy name (from the registry) or a callable."""
        if isinstance(name, str):
            cls = self.registry(name)
            if cls is None:
                raise ScannerError(f"unknown strategy {name!r}")
            return cls()
        if callable(name):
            return name()
        raise ScannerError(f"cannot resolve strategy {name!r}")

    def _connector_for(self, symbol: SymbolLike) -> str:
        """Return the data-source connector for ``symbol``."""
        code = self._code(symbol)
        return self.connector_map.get(code, "cme")

    def _code(self, symbol: SymbolLike) -> str:
        if isinstance(symbol, str):
            return symbol.strip().upper()
        if hasattr(symbol, "code"):
            return str(getattr(symbol, "code")).upper()
        raise ScannerError(f"cannot resolve symbol {symbol!r}")

    # ------------------------------------------------------------------ #
    # Data acquisition
    # ------------------------------------------------------------------ #

    def _fetch(self, symbol: SymbolLike) -> List[Candle]:
        code = self._code(symbol)
        connector = self._connector_for(symbol)
        if self.fetcher is None:
            return []
        try:
            return self.fetcher.fetch_candles(code, self.timeframe, limit=self.limit)
        except Exception:
            logger.exception("Failed to fetch candles for %s (%s)", code, connector)
            return []

    # ------------------------------------------------------------------ #
    # Scanning
    # ------------------------------------------------------------------ #

    def _generate(
        self, symbol: SymbolLike, strategy: Any, candles: List[Candle]
    ) -> List[Signal]:
        """Run a single strategy over ``candles`` for ``symbol``."""
        base = Signal(
            direction="HOLD",
            confidence=0.0,
            strength=0.0,
            reason="scan",
            timestamp=candles[-1].timestamp if candles else int(self._now()),
        )
        base.symbol = self._code(symbol)
        base.connector = self._connector_for(symbol)
        try:
            raw = strategy.generate(candles)
        except Exception:
            logger.exception("Strategy %r failed for %s", strategy, self._code(symbol))
            return [base]
        signals = []
        for s in raw:
            s = Signal(
                direction=str(s.direction or "HOLD"),
                confidence=float(getattr(s, "confidence", 0.0) or 0.0),
                strength=float(getattr(s, "strength", 0.0) or 0.0),
                reason=str(getattr(s, "reason", "")),
                timestamp=int(getattr(s, "timestamp", self._now())),
            )
            s.symbol = self._code(symbol)
            s.connector = self._connector_for(symbol)
            signals.append(s)
        return signals

    @staticmethod
    def _now() -> float:
        import time

        return time.time()

    def scan(self) -> List[Signal]:
        """Scan every symbol and return a ranked list of Signals."""
        collected: List[Signal] = []
        for symbol in self.symbols:
            candles = self._fetch(symbol)
            if not candles:
                logger.debug("No candles for %s", self._code(symbol))
                continue
            for strategy in self.strategies:
                collected.extend(self._generate(symbol, strategy, candles))

        if not collected:
            return []

        ranked = sorted(
            collected,
            key=lambda s: (s.confidence * 100.0 + s.strength, s.direction == "BUY"),
            reverse=True,
        )
        # Apply the configured floor so only meaningful signals are surfaced.
        return [
            s
            for s in ranked
            if s.confidence >= self.min_confidence and s.strength >= self.min_strength
        ]

    def scan_once(self) -> Dict[str, Any]:
        """One-shot scan returning a summary dict (engine / control surface)."""
        signals = self.scan()
        if not signals:
            return {
                "scanned": 0,
                "strategies": len(self.strategies),
                "signals": [],
                "top": None,
            }
        top = signals[0]
        by_symbol: Dict[str, List[Signal]] = {}
        for s in signals:
            by_symbol.setdefault(s.symbol, []).append(s)
        return {
            "scanned": len(self.symbols),
            "strategies": len(self.strategies),
            "signals": len(signals),
            "top": {
                "symbol": top.symbol,
                "direction": top.direction,
                "confidence": top.confidence,
                "strength": top.strength,
                "reason": top.reason,
            },
            "by_symbol": {k: len(v) for k, v in by_symbol.items()},
        }
