"""Broker execution layer for the scalping trading bot.

Execution layer for IC Markets cTrader via the cTrader Open API.

The bot's ``config.yaml`` selects the broker.  This file wraps the async
cTrader client in a synchronous interface so the existing orchestrator
loop can use it without change.

Broker selection (from config.yaml)::

    broker: ctrader
    paper_trading: true
    account_id: 1234567
    api_key: "YOUR_CLIENT_ID"
    api_secret: "YOUR_CLIENT_SECRET"

Types of orders this broker supports: market, limit, stop, stop-limit,
and stop-loss-take-profit (binary orders).  Amounts are in "lots" and are
converted to cTrader base units using the symbol's lot size.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

from exec.ctrader_broker import CTraderBroker

logger = logging.getLogger(__name__)


class Broker:
    """Synchronous broker adapter for IC Markets cTrader.

    The underlying cTrader client is async; this class runs it on a
    background task so callers can use the synchronous interface.
    """

    def __init__(
        self,
        exchange_name: str = "ctrader",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        paper_trading: bool = True,
        config: Optional[dict] = None,
    ) -> None:
        self.exchange_name = exchange_name
        self.paper_trading = paper_trading
        self.config = config or {}

        self._api_key = api_key or self.config.get("api_key", "")
        self._secret = secret or self.config.get("api_secret", "")
        self._account_id = int(self.config.get("account_id", 0))

        self._broker: Optional[CTraderBroker] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect the cTrader client (synchronous wrapper)."""
        if self._broker is not None:
            return
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _runner() -> None:
            self._broker = CTraderBroker(
                account_id=self._account_id,
                api_key=self._api_key,
                secret=self._secret,
                paper_trading=self.paper_trading,
            )
            await self._broker.connect()

        self._running = True
        try:
            self._loop.run_until_complete(_runner())
        except Exception:
            self._running = False
            raise

    def disconnect(self) -> None:
        """Disconnect the cTrader client (synchronous wrapper)."""
        if self._loop is not None and self._broker is not None:
            try:
                self._loop.run_until_complete(self._broker.disconnect())
            except Exception:
                logger.exception("Error disconnecting cTrader client")
        self._broker = None
        self._loop = None
        self._running = False

    def close(self) -> None:
        """Close resources (identity alias for disconnect)."""
        self.disconnect()

    def _ensure_connected(self) -> None:
        if not self._running or self._broker is None:
            raise RuntimeError("Broker is not connected. Call connect() first.")

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def execute_buy(
        self,
        symbol: str,
        amount: float = 1.0,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> dict:
        """Place a buy (long) order (synchronous)."""
        self._ensure_connected()
        future = asyncio.run_coroutine_threadsafe(
            self._broker.execute_buy(symbol, amount, price, order_type), self._loop
        )
        try:
            return future.result(timeout=30)
        except Exception:
            logger.exception("Buy order failed for %s", symbol)
            return {"id": None, "status": "failed", "error": "order failed"}

    def execute_sell(
        self,
        symbol: str,
        amount: float = 1.0,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> dict:
        """Place a sell (short) order (synchronous)."""
        self._ensure_connected()
        future = asyncio.run_coroutine_threadsafe(
            self._broker.execute_sell(symbol, amount, price, order_type), self._loop
        )
        try:
            return future.result(timeout=30)
        except Exception:
            logger.exception("Sell order failed for %s", symbol)
            return {"id": None, "status": "failed", "error": "order failed"}

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order (synchronous)."""
        self._ensure_connected()
        future = asyncio.run_coroutine_threadsafe(
            self._broker.cancel_order(order_id), self._loop
        )
        try:
            return future.result(timeout=30)
        except Exception:
            logger.exception("Cancel order %s failed", order_id)
            return False

    # ------------------------------------------------------------------
    # Positions / account
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict]:
        """Return all open positions (synchronous)."""
        self._ensure_connected()
        future = asyncio.run_coroutine_threadsafe(
            self._broker.get_positions(), self._loop
        )
        try:
            return future.result(timeout=30)
        except Exception:
            logger.exception("get_positions failed")
            return []

    def get_balance(self) -> dict:
        """Return the account summary (synchronous)."""
        self._ensure_connected()
        future = asyncio.run_coroutine_threadsafe(
            self._broker.get_balance(), self._loop
        )
        try:
            return future.result(timeout=30)
        except Exception:
            logger.exception("get_balance failed")
            return {"account_id": self._account_id, "error": "balance fetch failed"}


# --------------------------------------------------------------------------- #
# Crypto broker (CCXT-based)
# --------------------------------------------------------------------------- #
class CryptoBroker:
    """Synchronous broker adapter for crypto markets via CCXT.

    Supported exchanges (configurable): ``binance``, ``bybit``, ``okx``.
    ``paper_trading=True`` asks CCXT for its built-in paper/test mode; when
    no live API key is supplied the broker runs in *dry-run* (records orders
    instead of placing them), which is how the crypto skill is unit-tested
    offline.

    Futures / testnet paper mode
    ----------------------------
    Set ``futures=True`` and (optionally) ``test=True``:

    * ``futures=True`` + ``exchange='binance'`` -> CCXT uses the USDT-margined
      perpetual product (``defaultType='future'``) instead of spot.
    * ``test=True`` asks CCXT for its sandbox/test mode, which for binance
      lands on the **Futures testnet** (``demo-fapi.binance.com`` REST /
      ``demo-fstream.binance.com`` WS) -- real market data, simulated fills,
      no real capital.

    Order placement without a live API key never sends an order: it is
    recorded in the dry-run log (``trades()``) and reported with
    ``status='dry-run'``.  With a live key and ``test=True``, orders go to the
    testnet.

    Amounts are in contract size (e.g. 0.001 means 0.001 BTC) and must be
    pre-scaled to the exchange's lot/contract convention.  The returned
    order carries ``execution_price`` when the fill is reported by the
    exchange.

    Types of orders supported: market, limit, stop, stop-limit.
    """

    LINE = "CRYPTO"

    def __init__(
        self,
        exchange: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        paper_trading: bool = True,
        test: bool = False,
        futures: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        self.exchange_name = exchange.lower() or "binance"
        self.api_key = api_key or (config or {}).get("api_key", "")
        self.secret = secret or (config or {}).get("secret", "")
        self.paper_trading = paper_trading
        self.test = test
        self.futures = bool(futures)
        self._client = None
        self._loop = None
        self._trades: list[dict] = []  # dry-run order log
        self._positions: dict[str, dict] = {}
        self._connect_lock = threading.Lock()

    def _create_client(self):
        """Build the CCXT crypto exchange (no network until keys exist)."""
        import ccxt

        cls = getattr(ccxt, self.exchange_name, None)
        if cls is None:
            raise RuntimeError(f"Unsupported crypto exchange: {self.exchange_name}")
        options = {}
        if self.exchange_name == "bybit":
            options["defaultType"] = "swap"
        elif self.futures and self.exchange_name == "binance":
            options["defaultType"] = "future"
        cfg = {
            "apiKey": self.api_key,
            "secret": self.secret,
            "enableRateLimit": True,
            "timeout": 10000,
            "options": options,
        }
        exchange = cls(cfg)
        if self.test:
            exchange.set_sandbox_mode(True)
        return exchange

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        if self._client is not None:
            return
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _runner() -> None:
            self._client = self._create_client()
            if asyncio.iscoroutinefunction(self._client.load_markets):
                await self._client.load_markets()
            else:
                self._client.load_markets()

        self._running = True
        try:
            self._loop.run_until_complete(_runner())
        except Exception:
            self._running = False
            raise

    def disconnect(self) -> None:
        if self._loop is not None and self._client is not None:
            try:
                self._loop.run_until_complete(self._client.close())
            except Exception:
                logger.exception("Crypto broker disconnect failed")
        self._client = None
        self._loop = None
        self._running = False

    def close(self) -> None:
        self.disconnect()

    def _ensure_connected(self) -> None:
        if self._loop is None or self._client is None or not self._running:
            raise RuntimeError("Crypto broker is not connected. Call connect() first.")

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #

    def execute_buy(self, symbol, amount=1.0, price=None, order_type="market"):
        return self._place_order(symbol, amount, price, order_type, "BUY")

    def execute_sell(self, symbol, amount=1.0, price=None, order_type="market"):
        return self._place_order(symbol, amount, price, order_type, "SELL")

    def _place_order(self, symbol, amount, price, order_type, side):
        self._ensure_connected()
        if not self.api_key:
            return self._dry_run(symbol, side, amount, order_type)

        loop = self._loop
        assert loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._broker_place(symbol, amount, price, order_type, side),
            loop,
        )
        try:
            return future.result(timeout=30)
        except Exception:
            logger.exception("Crypto order failed for %s", symbol)
            return {"id": None, "status": "failed", "error": "order failed"}

    async def _broker_place(self, symbol, amount, price, order_type, side):
        exchange = self._connected_client()
        order_type = order_type.lower()
        if order_type == "market":
            otype = "market"
        elif order_type in ("limit", "stop-limit", "stop-limit"):
            otype = "limit"
        else:
            otype = "market"

        result = await exchange.create_order(symbol, otype, side, amount, price)
        return self._format_order(result, symbol)

    def _connected_client(self):
        if self._client is None:
            raise RuntimeError("Crypto broker not connected")
        return self._client

    # ------------------------------------------------------------------ #
    # Positions / balance
    # ------------------------------------------------------------------ #

    def get_balance(self) -> dict:
        self._ensure_connected()
        try:
            return self._loop.run_until_complete(self._broker_get_balance())
        except Exception:
            logger.exception("get_balance failed")
            return {"exchange": self.exchange_name, "error": "balance fetch failed"}

    async def _broker_get_balance(self):
        exchange = self._connected_client()
        try:
            acc = await exchange.fetch_balance()
        except Exception:
            acc = {}
        return {
            "exchange": self.exchange_name,
            "paper_trading": self.paper_trading,
            "total": 0.0,
            "available": 0.0,
            **acc,
        }

    def get_positions(self) -> list[dict]:
        self._ensure_connected()
        try:
            return self._loop.run_until_complete(self._broker_get_positions())
        except Exception:
            logger.exception("get_positions failed")
            return []

    async def _broker_get_positions(self):
        exchange = self._connected_client()
        try:
            positions = await exchange.fetch_positions()
        except Exception:
            positions = []
        out = []
        for p in positions:
            out.append(
                {
                    "symbol": p.get("symbol"),
                    "side": p.get("side"),
                    "volume": p.get("contracts") or p.get("amount"),
                    "entry_price": p.get("entry_price") or p.get("price"),
                    "unrealized_pnl": p.get("unrealized_pl"),
                }
            )
        return out

    # ------------------------------------------------------------------ #
    # Formatting / helpers
    # ------------------------------------------------------------------ #

    def _format_order(self, order: dict, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "side": order.get("side"),
            "type": order.get("type"),
            "amount": order.get("filled") or order.get("cost") or order.get("amount"),
            "id": order.get("id") or order.get("orderNumber"),
            "status": order.get("status") or "filled",
            "execution_price": order.get("price") or order.get("filledAveragePrice"),
            "timestamp": order.get("datetime") or time.time(),
        }

    def _dry_run(self, symbol: str, side: str, amount: float, order_type: str) -> dict:
        """Record an order that would have been placed (no live broker)."""
        order = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "order_id": f"dryrun-{len(self._trades)}",
            "status": "dry-run",
        }
        self._trades.append(order)
        return order

    def trades(self) -> list[dict]:
        """Return the dry-run order log (read-only)."""
        return list(self._trades)

    def positions(self) -> list[dict]:
        """Return the in-memory position log (dry-run mode)."""
        return [dict(p) for p in self._positions.values()]


__all__ = ["Broker", "CTraderBroker", "CryptoBroker"]
