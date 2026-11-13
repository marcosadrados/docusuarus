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
        self, symbol: str, amount: float = 1.0, price: Optional[float] = None,
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
        self, symbol: str, amount: float = 1.0, price: Optional[float] = None,
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
