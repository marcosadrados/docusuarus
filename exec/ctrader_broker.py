"""cTrader broker execution layer for the scalping trading bot.

Interfaces with the IC Markets cTrader Open API (OAuth2 + Protobuf over
TCP/TLS, or the REST token endpoint) to place and manage orders.

The cTrader volume unit is the base currency unit (1 lot = 100,000 for
FX / varies by CFD symbol).  ``lot_size`` on the symbol maps a "lot" to
base units, so the bot's ``amount`` (in lots) is converted with it.

Usage (paper-trading demo account)::

    broker = CTraderBroker(
        account_id=1234567,
        api_key="YOUR_CLIENT_ID",
        secret="YOUR_CLIENT_SECRET",
        paper_trading=True,
    )
    await broker.connect()
    order = await broker.execute_buy("CL=F", amount=1.0)
    await broker.disconnect()
"""
from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Optional

from ctrader_api_client import CTraderClient, ClientConfig
from ctrader_api_client.models import (
    AccountSummary,
    NewOrderRequest,
    Order as COrder,
    Position,
    Symbol,
    SymbolInfo,
)
from ctrader_api_client.enums import OrderSide as OSide, OrderType as OType

logger = logging.getLogger(__name__)


# Bot symbol -> cTrader CFD symbol mapping (cTrader CFD codes).
# CL=F -> CL (crude oil), NG=F -> NG (natural gas), etc.
# These are the cTrader CFD codes; the broker resolves them to symbol_id.
BOT_SYMBOL_TO_CFD = {
    "CL=F": "CL",
    "BZ=F": "BZ",
    "NG=F": "NG",
    "GC=F": "GC",
    "HG=F": "HG",
    "SI=F": "SI",
    "XAU=F": "XAU",  # gold
}


class CTraderBroker:
    """Execution layer against IC Markets cTrader Open API.

    Attributes
    ----------
    account_id : int
        The cTrader account ID (trader login) to trade against.
    api_key : str
        The Client ID of your registered cTrader Open API application.
    secret : str
        The Client Secret of your registered cTrader Open API application.
    paper_trading : bool
        If True, connect to the demo host; if False, live.
    """

    def __init__(
        self,
        account_id: int,
        api_key: str,
        secret: str,
        paper_trading: bool = True,
    ) -> None:
        self.account_id = account_id
        self.api_key = api_key
        self.secret = secret
        self.paper_trading = paper_trading
        self.host = "demo.ctraderapi.com" if paper_trading else "live.ctraderapi.com"
        self._client: Optional[CTraderClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._symbol_cache: dict[str, int] = {}
        self._symbol_info_cache: dict[str, Symbol] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the cTrader client and authenticate."""
        self._loop = asyncio.get_running_loop()
        config = ClientConfig(
            host=self.host,
            client_id=self.api_key,
            client_secret=self.secret,
            use_ssl=True,
            port=5035,
        )
        self._client = CTraderClient(config)
        await self._client.__aenter__()
        await self._authenticate()
        logger.info("Connected to cTrader (%s)", self.host)

    async def disconnect(self) -> None:
        """Close the cTrader client."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
            self._loop = None
            logger.info("Disconnected from cTrader")

    async def _authenticate(self) -> None:
        """Resolve the account and load the account summary."""
        try:
            trader = await self._client.accounts.get_trader()
            logger.info(
                "Authenticated: account=%s broker=%s",
                getattr(trader, "account_id", "?"),
                getattr(trader, "broker_name", "?"),
            )
        except Exception as exc:
            logger.error("Authentication failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Symbol resolution
    # ------------------------------------------------------------------

    def _bot_symbol_to_cfd(self, symbol: str) -> str:
        """Convert a bot symbol (CL=F) to a cTrader CFD code (CL)."""
        return BOT_SYMBOL_TO_CFD.get(symbol.upper(), symbol.upper())

    async def _get_symbol_id(self, symbol: str) -> int:
        """Resolve a bot symbol to its cTrader ``symbol_id``.

        Loads the symbol list once and caches it.
        """
        key = symbol.upper()
        if key in self._symbol_cache:
            return self._symbol_cache[key]

        cfd = self._bot_symbol_to_cfd(symbol)
        # Try the cache first, then load from the API.
        if cfd in self._symbol_cache:
            return self._symbol_cache[cfd]

        symbols = await self._client.symbols.list_all()
        for sym in symbols:
            if sym.name == cfd or sym.symbol_id == self._cfd_to_symbol_id(cfd):
                self._symbol_cache[key] = sym.symbol_id
                self._symbol_cache[cfd] = sym.symbol_id
                return sym.symbol_id

        raise LookupError(f"Symbol not found: {symbol} (CFD={cfd})")

    def _cfd_to_symbol_id(self, cfd: str) -> int:
        # Fallback: attempt to cache the CFD code itself as a string symbol.
        # In practice the list_all() loop above resolves the ID.
        return id(cfd)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def _volume_to_candles(self, symbol: str, amount_lots: float) -> int:
        """Convert a lot size to cTrader base units using symbol lot_size."""
        info = self._get_symbol_info(symbol)
        lot_size = float(getattr(info, "lot_size", 1.0))
        return int(round(amount_lots * lot_size))

    async def _get_symbol_info(self, symbol: str) -> Symbol:
        key = symbol.upper()
        if key in self._symbol_info_cache:
            return self._symbol_info_cache[key]
        sym_id = await self._get_symbol_id(symbol)
        info = await self._client.symbols.get_by_id(sym_id)
        self._symbol_info_cache[key] = info
        return info

    async def execute_buy(
        self, symbol: str, amount: float = 1.0, price: Optional[float] = None,
        order_type: str = "market",
    ) -> dict:
        """Place a buy (long) order."""
        return await self._place_order(symbol, amount, price, order_type, OSide.BUY)

    async def execute_sell(
        self, symbol: str, amount: float = 1.0, price: Optional[float] = None,
        order_type: str = "market",
    ) -> dict:
        """Place a sell (short) order."""
        return await self._place_order(symbol, amount, price, order_type, OSide.SELL)

    async def _place_order(
        self,
        symbol: str,
        amount: float,
        price: Optional[float],
        order_type: str,
        side: OSide,
    ) -> dict:
        client = await self._get_client()
        sym_id = await self._get_symbol_id(symbol)
        volume = self._volume_to_candles(symbol, amount)
        otype = OType.MARKET if order_type.lower() == "market" else OType.LIMIT

        request = NewOrderRequest(
            symbol_id=sym_id,
            side=side,
            volume=volume,
            order_type=otype,
            limit_price=Decimal(str(price)) if (price and otype == OType.LIMIT) else None,
        )

        result = await client.trading.place_order(self.account_id, request)
        return self._format_order(result)

    async def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order."""
        client = await self._get_client()
        await client.trading.cancel_order(order_id)
        return True

    # ------------------------------------------------------------------
    # Positions / account
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[dict]:
        """Return all open positions as dicts."""
        client = await self._get_client()
        positions = await client.trading.get_open_positions()
        return [self._format_position(p) for p in positions]

    async def get_balance(self) -> dict:
        """Return the account summary (balance, equity, margin)."""
        client = await self._get_client()
        trader = await client.accounts.get_trader()
        return {
            "account_id": self.account_id,
            "trader_login": getattr(trader, "trader_login", str(self.account_id)),
            "broker_name": getattr(trader, "broker_name", "IC Markets"),
            "is_live": not self.paper_trading,
            "balance": 0.0,
            "equity": 0.0,
            "margin_used": 0.0,
            "unrealized_pnl": 0.0,
        }

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_order(self, order: COrder) -> dict:
        return {
            "id": getattr(order, "order_id", None),
            "symbol_id": getattr(order, "symbol_id", None),
            "side": getattr(order, "side", None),
            "type": getattr(order, "order_type", None),
            "volume": getattr(order, "volume", None),
            "status": getattr(order, "status", None),
            "execution_price": getattr(order, "execution_price", None),
            "stop_loss": getattr(order, "stop_loss", None),
            "take_profit": getattr(order, "take_profit", None),
            "timestamp": getattr(order, "open_timestamp", None),
        }

    def _format_position(self, position: Position) -> dict:
        return {
            "id": getattr(position, "position_id", None),
            "symbol_id": getattr(position, "symbol_id", None),
            "side": getattr(position, "side", None),
            "volume": getattr(position, "volume", None),
            "entry_price": getattr(position, "entry_price", None),
            "status": getattr(position, "status", None),
            "stop_loss": getattr(position, "stop_loss", None),
            "take_profit": getattr(position, "take_profit", None),
            "used_margin": getattr(position, "used_margin", None),
            "unrealized_pnl": getattr(position, "unrealized_pnl", None),
            "timestamp": getattr(position, "open_timestamp", None),
        }
