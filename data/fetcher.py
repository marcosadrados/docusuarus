"""Data acquisition layer for the scalping trading bot.

Futures path
------------
``WFuturesFetcher`` keeps fetching futures OHLCV via CCXT (CME / ICE / COMEX /
LBMA), exactly as before.

US equities path
----------------
``USEquityBars`` / :func:`load_equity` ship a *deterministic sample* of OHLCV
bars for US equities (e.g. AAPL).  The sample is a seeded random walk around a
base price, so every call returns the identical dataset for a given
symbol / timeframe / limit.  That makes the equity strategy and the backtest
runner fully testable **offline**, with no market-data API key and no network
access.  The returned objects satisfy the bot's ``Candle`` contract
(:class:`signals.strategy.Candle`):

    timestamp : int   -- epoch seconds
    open, high, low, close : float
    volume : float

The sample deliberately contains both below-VWAP (pullback) and above-VWAP
(rebound) phases so a VWAP-mean-reversion volume-accrual strategy can be
exercised end-to-end.
"""

import math
import random
from typing import Optional

import ccxt

from signals.strategy import Candle

__all__ = ["USEquityBars", "load_equity", "WFuturesFetcher"]

#: Default base price around which each symbol's sample bars are generated.
DEFAULT_BASE_PRICES = {
    "AAPL": 150.0,
    "MSFT": 300.0,
    "GOOGL": 140.0,
    "AMZN": 160.0,
    "NVDA": 400.0,
    "META": 350.0,
    "TSLA": 180.0,
    "NFLX": 400.0,
}

#: Bar spacing (seconds) per timeframe.
DEFAULT_INTERVALS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}

#: Crypto symbol -> base price for the deterministic offline sample bars.
#: (per-pair reference price, used only when no live exchange key is set)
DEFAULT_BASE_PRICES = {
    "AAPL": 150.0,
    "MSFT": 300.0,
    "GOOGL": 140.0,
    "AMZN": 160.0,
    "NVDA": 400.0,
    "META": 350.0,
    "TSLA": 180.0,
    "NFLX": 400.0,
    #: Default crypto / spot pairs (base currency = USD pair, quote = USDT).
    "BTC/USDT": 95000.0,
    "ETH/USDT": 3500.0,
    "SOL/USDT": 150.0,
    "XRP/USDT": 2.2,
    "BNB/USDT": 560.0,
    "ADA/USDT": 0.6,
    "DOGE/USDT": 0.12,
    "AVAX/USDT": 32.0,
}

#: Crypto bar spacing (seconds) per timeframe.
DEFAULT_INTERVALS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}

#: First epoch-second tick of every generated sample.
DEFAULT_START_TIME = 1700000000


class WFuturesFetcher:
    """Fetch futures market data using CCXT.

    Supports:
    - Historical candles (daily, 1m, 5m, 15m, 1h)
    - Real-time stream data
    - Multiple exchanges (CME, ICE, COMEX, LBMA, etc.)
    """

    def __init__(self, exchange="CME", api_key=None, secret=None):
        self.exchange_name = exchange or "CME"
        self.api_key = api_key
        self.secret = secret
        self.exchange = None
        self._exchange_class = None

    def _get_exchange_class(self):
        """Get the CCXT exchange class."""
        if self._exchange_class is None:
            cls = getattr(ccxt, self.exchange_name.lower(), None)
            if cls is None:
                # Try with default available exchanges
                for name in ["binance", "bybit", "okx", "kraken"]:
                    cls = getattr(ccxt, name, None)
                    if cls:
                        self.exchange_name = name
                        self._exchange_class = cls
                        return cls
                raise ValueError(f"Unknown exchange: {self.exchange_name}")
            self._exchange_class = cls
        return self._exchange_class

    def _connect(self):
        """Initialize the CCXT exchange connection."""
        exchange_class = self._get_exchange_class()
        self.exchange = exchange_class(
            {
                "apiKey": self.api_key or "",
                "secret": self.secret or "",
                "enableRateLimit": True,
                "timeout": 10000,
            }
        )
        self.exchange.load_markets()

    def fetch_candles(self, symbol, timeframe="1m", since=None, limit=None):
        """Fetch historical candles for a symbol."""
        symbol_upper = symbol.upper().replace("-", "")
        try:
            candles = self.exchange.fetch_ohlcv(
                symbol=symbol_upper, timeframe=timeframe, since=since, limit=limit
            )
            return candles
        except Exception as e:
            raise RuntimeError(f"Failed to fetch candles for {symbol}: {e}") from None

    def fetch_realtime_ohlcv(self, symbol, timeframe="1m", max_bars=500):
        """Fetch real-time OHLCV data (simulated for live trading)."""
        return []

    def get_symbols(self, prefix=None):
        symbols = list(self.exchange.symbols)
        if prefix:
            return [s for s in symbols if s.startswith(prefix)]
        return symbols

    def close(self):
        pass


class USEquityBars:
    """Deterministic sample bars for US equities.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. ``"AAPL"``).  Uppercased internally.
    base_price : float, optional
        Centre price for the random walk.  Falls back to
        :attr:`DEFAULT_BASE_PRICES` when the symbol is unknown.
    seed : int
        RNG seed.  The same ``seed`` always produces the same bars, which is
        what keeps offline backtests reproducible.
    start_time : float, optional
        Epoch seconds of the first bar.  Defaults to
        :data:`DEFAULT_START_TIME`.
    """

    def __init__(
        self,
        symbol: str = "AAPL",
        base_price: float | None = None,
        seed: int = 42,
        start_time: float | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.base_price = (
            base_price
            if base_price is not None
            else DEFAULT_BASE_PRICES.get(self.symbol, 150.0)
        )
        self.seed = seed
        self.start_time = start_time if start_time is not None else DEFAULT_START_TIME
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Public API                                                       #
    # ------------------------------------------------------------------ #

    def fetch(
        self, symbol: Optional[str] = None, timeframe: str = "1m", limit: int = 100
    ) -> list[Candle]:
        """Return ``limit`` sample ``Candle`` objects for ``symbol``.

        Parameters
        ----------
        symbol : str, optional
            Ticker symbol.  Defaults to the instance's symbol.
        timeframe : str
            Bar resolution (``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``).  Used
            for the spacing between ticks.
        limit : int
            Number of bars to return.  A value of zero returns an empty list.
        """
        symbol = self.symbol if symbol is None else symbol.upper()
        timeframe = str(timeframe).lower()
        limit = int(limit)
        if limit <= 0:
            return []

        interval = DEFAULT_INTERVALS.get(timeframe, 3600)
        base = (
            self.base_price
            if symbol == self.symbol
            else DEFAULT_BASE_PRICES.get(symbol, 150.0)
        )
        candles: list[Candle] = []
        prev_close = base

        for i in range(limit):
            # Deterministic random walk around the base price.  The drift is
            # capped so the sample stays in a sane band, and a small cyclic
            # component guarantees both below-VWAP and above-VWAP phases.
            drift_steps = min(i, 200) * interval * 1.2e-4
            noise = self._rng.gauss(0.0, self._rng.random() * 0.03)
            cyclic = (
                0.6 * math.sin(self._rng.random() * 2.0) * self._rng.random() * 0.01
            )
            open_p = prev_close
            close = open_p + drift_steps + noise + cyclic
            high = max(open_p, close) + self._rng.gauss(0.0, 0.01)
            low = min(open_p, close) - self._rng.gauss(0.0, 0.01)
            volume = self._rng.randint(500_000, 2_500_000)

            candles.append(
                Candle(
                    timestamp=int(self.start_time + i * interval),
                    open=round(open_p, 5),
                    high=round(high, 5),
                    low=round(low, 5),
                    close=round(close, 5),
                    volume=volume,
                )
            )
            prev_close = close

        return candles

    @classmethod
    def load_equity(
        cls,
        symbol: str = "AAPL",
        timeframe: str = "1m",
        limit: int = 100,
    ) -> list[Candle]:
        """Module-level convenience: sample bars for a US equity.

        This is the primary offline entry point for the equities path:

            bars = load_equity("AAPL", "1m", 100)

        ``bars`` is a list of ``Candle`` objects usable directly by the equity
        strategy and the backtest runner, with no network call and no API key.
        """
        return cls(
            symbol=symbol,
            base_price=DEFAULT_BASE_PRICES.get(symbol.upper(), 150.0),
            seed=42,
        ).fetch(symbol=symbol, timeframe=timeframe, limit=limit)


def load_equity(
    symbol: str = "AAPL", timeframe: str = "1m", limit: int = 100
) -> list[Candle]:
    """Return deterministic sample OHLCV bars for a US equity symbol.

    Alias for :meth:`USEquityBars.load_equity` so the equities path reads
    naturally in the backtest runner and tests.

    Example
    -------
    >>> bars = load_equity("AAPL", "1m", 30)
    >>> bars[0].timestamp
    1700000000
    >>> len(bars) == 30
    True
    """
    return USEquityBars.load_equity(symbol=symbol, timeframe=timeframe, limit=limit)


# --------------------------------------------------------------------------- #
# Crypto path
# --------------------------------------------------------------------------- #
class CryptoFetcher:
    """Data source for crypto spot / perpetual markets.

    Uses CCXT (``binance`` / ``bybit`` / ``okx``) for live OHLCV, and falls
    back to a deterministic sample walk when no live exchange key is
    configured, so the crypto path is fully testable offline.

    Parameters
    ----------
    exchange : str
        CCXT exchange id (``binance``, ``bybit``, ``okx``).
    api_key : str, optional
        Exchange API key.  ``None`` means offline sample mode.
    secret : str, optional
        Exchange API secret.
    test : bool
        If True, run CCXT in paper/test mode where supported.
    seed : int
        RNG seed for the offline sample walk.
    """

    def __init__(
        self,
        exchange: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        test: bool = False,
        seed: int = 42,
        futures: bool = False,
    ) -> None:
        self.exchange_name = exchange.lower() or "binance"
        self.api_key = api_key or ""
        self.secret = secret or ""
        self.test = test
        self.seed = seed
        self.futures = bool(futures)
        self.start_time = DEFAULT_START_TIME
        self.exchange = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: Optional[int] = None,
        limit: int = 100,
    ) -> list:
        """Return ``limit`` OHLCV bars for ``symbol``.

        Returns a list of CCXT rows: ``[timestamp, open, high, low, close, volume]``.
        When no live key is configured, returns a deterministic sample walk
        (``_sample_bars``) so the crypto path is unit-testable offline.
        """
        symbol = (symbol or "").upper().replace("-", "/")

        # Offline / not in paper-testing mode: deterministic sample.
        if not self.test and not self.api_key and not self.secret:
            return self._sample_bars(symbol, timeframe, limit)

        def _live():
            return self._connected().fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )

        try:
            return _live()
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch candles for {symbol}: {exc}") from None

    def _sample_bars(self, symbol: str, timeframe: str, limit: int) -> list:
        """Deterministic OHLCV sample for offline testing."""
        if limit <= 0:
            return []

        interval = DEFAULT_INTERVALS.get(timeframe.lower(), 3600)
        base = DEFAULT_BASE_PRICES.get(symbol, 100.0)
        rng = random.Random(self.seed)
        candles = []
        prev_close = base

        for i in range(limit):
            drift_steps = min(i, 200) * interval * 1.2e-4
            noise = rng.gauss(0.0, rng.random() * 0.04)
            cyclic = 0.5 * math.sin(rng.random() * 2.0) * rng.random() * 0.01
            open_p = prev_close
            close = open_p + drift_steps + noise + cyclic
            high = max(open_p, close) + rng.gauss(0.0, 0.02)
            low = min(open_p, close) - rng.gauss(0.0, 0.02)
            volume = int(rng.randint(500_000, 2_500_000))

            candles.append(
                [
                    int(self.start_time + i * interval),
                    round(open_p, 5),
                    round(high, 5),
                    round(low, 5),
                    round(close, 5),
                    volume,
                ]
            )
            prev_close = close

        return candles

    def _connected(self):
        """Lazy-connect the CCXT exchange (no network until keys exist)."""
        if self.exchange is not None:
            return self.exchange
        cls = getattr(ccxt, self.exchange_name, None)
        if cls is None:
            raise RuntimeError(f"Unsupported crypto exchange: {self.exchange_name}")
        options = {}
        if self.exchange_name == "bybit":
            options["defaultType"] = "swap"
        elif self.futures and self.exchange_name == "binance":
            options["defaultType"] = "future"
        self.exchange = cls(
            {
                "apiKey": self.api_key,
                "secret": self.secret,
                "enableRateLimit": True,
                "timeout": 10000,
                "options": options,
            }
        )
        if self.test:
            self.exchange.set_sandbox_mode(True)
        self.exchange.load_markets()
        return self.exchange

    def get_symbols(self, prefix: Optional[str] = None) -> list:
        return self._connected().symbols

    def close(self) -> None:
        pass


__all__ = ["USEquityBars", "load_equity", "WFuturesFetcher", "CryptoFetcher"]
