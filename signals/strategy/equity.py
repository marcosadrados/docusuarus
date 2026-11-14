"""Equity scalping strategy: VWAP-pullback + volume-accrual.

This is deliberately distinct from :class:`signals.strategy.MarketTrendStrategy`
(oil/gas/metals, EMA + 50-period MA).  It is an equity-specific,
VWAP-mean-reversion engine:

- It computes a rolling VWAP (typical price * volume).
- A **pullback** (price sitting below VWAP) with a volume accrual is a BUY.
- A **rebound** (price sitting above VWAP) with a volume accrual is a SELL.

Both conditions require the latest bar's volume to exceed a multiple of the
window average volume, which confirms participation.  The strategy returns at
most one ``Signal`` per call so it is safe against the backtest runner's
``signals[0]`` access pattern.
"""

from dataclasses import dataclass
from typing import List, Optional

from .core import Candle, Signal

__all__ = ["EquityStrategy", "EquitySignal"]

#: Minimum window (bars) required before indicators are meaningful.
DEFAULT_LOOKBACK = 20
#: Pullback / rebound depth as a fraction of VWAP (0.3% = 0.003).
DEFAULT_PULLBACK = 0.003
#: Volume must exceed this multiple of the window average.
DEFAULT_VOLUME_MULTIPLE = 1.5
#: VWAP denominator smoothing (typical price weights).
VWAP_WEIGHT = 1.0


@dataclass
class EquitySignal(Signal):
    """Equity strategy signal; a ``Signal`` with equity-specific fields."""

    equity_symbol: Optional[str] = None
    vwap: Optional[float] = None
    volume_ratio: Optional[float] = None


class EquityStrategy:
    """VWAP-pullback + volume-accrual scalping strategy.

    Rules:
    - BUY  (pullback + accumulation): price < VWAP * (1 - pullback) AND
      current volume > volume_multiple * average volume.
    - SELL (rebound + accumulation):  price > VWAP * (1 + pullback) AND
      current volume > volume_multiple * average volume.

    The strategy requires a warm-up window (``lookback`` bars).  When the
    provided window is too short it returns no signal (``[]``), matching the
    ``signals.strategy.MarketTrendStrategy`` convention.
    """

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        pullback: float = DEFAULT_PULLBACK,
        volume_multiple: float = DEFAULT_VOLUME_MULTIPLE,
        symbol: str = "AAPL",
    ) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if not 0.0 < pullback < 1.0:
            raise ValueError("pullback must be in (0, 1)")
        if volume_multiple <= 1.0:
            raise ValueError("volume_multiple must be > 1.0")

        self.lookback = lookback
        self.pullback = pullback
        self.volume_multiple = volume_multiple
        self.symbol = symbol
        self._vwap = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                       #
    # ------------------------------------------------------------------ #

    def generate(self, candles: List[Candle]) -> List[Signal]:
        """Generate signals from a trailing window of candles.

        The backtest runner hands the strategy a trailing window (0..49 bars,
        per ``backtest/runner.py``).  This method returns a single signal
        derived from the *last* bar of that window when the pullback/rebound
        and volume-accrual conditions are met, otherwise a HOLD signal.

        Parameters
        ----------
        candles : List[Candle]
            Trailing window of bars, most-recent-first order of indices
            (the last element is the current tick).

        Returns
        -------
        List[Signal]
            ``[Signal(...)]`` when a condition fires, ``[HOLD]`` when the
            market is price-neutral, or ``[]`` when the window is too small.
        """
        if len(candles) < self.lookback:
            return []

        closes = [c.close for c in candles]
        vols = [c.volume for c in candles]

        # ---- Rolling VWAP over the lookback window ---------------------- #
        window = candles[-self.lookback :]
        typical = [(c.high + c.low + c.close) / 3.0 for c in window]
        window_vol = [float(c.volume) for c in window]
        vwap = sum(t * v for t, v in zip(typical, window_vol, strict=True)) / sum(window_vol)
        self._vwap = vwap

        # ---- Volume accrual check --------------------------------------- #
        avg_vol = sum(vols) / len(vols)
        current_vol = float(vols[-1])
        volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        current_close = closes[-1]
        vol_confirmed = volume_ratio >= self.volume_multiple

        # ---- Pullback (BUY) --------------------------------------------- #
        if current_close < vwap * (1.0 - self.pullback) and vol_confirmed:
            depth = (vwap - current_close) / vwap
            confidence = min(0.9, 0.5 + depth * 40.0)
            return [
                EquitySignal(
                    direction="BUY",
                    confidence=confidence,
                    strength=0.6,
                    reason=f"VWAP pullback ({depth*100:.2f}%) + volume accrual x{volume_ratio:.1f}",
                    timestamp=candles[-1].timestamp,
                    equity_symbol=self.symbol,
                    vwap=vwap,
                    volume_ratio=volume_ratio,
                )
            ]

        # ---- Rebound (SELL) --------------------------------------------- #
        if current_close > vwap * (1.0 + self.pullback) and vol_confirmed:
            depth = (current_close - vwap) / vwap
            confidence = min(0.9, 0.5 + depth * 40.0)
            return [
                EquitySignal(
                    direction="SELL",
                    confidence=confidence,
                    strength=0.5,
                    reason=f"VWAP rebound ({depth*100:.2f}%) + volume accrual x{volume_ratio:.1f}",
                    timestamp=candles[-1].timestamp,
                    equity_symbol=self.symbol,
                    vwap=vwap,
                    volume_ratio=volume_ratio,
                )
            ]

        # ---- Range-bound ------------------------------------------------- #
        return [
            EquitySignal(
                direction="HOLD",
                confidence=0.0,
                strength=0.0,
                reason="No VWAP signal",
                timestamp=candles[-1].timestamp,
                equity_symbol=self.symbol,
                vwap=vwap,
                volume_ratio=volume_ratio,
            )
        ]

    @property
    def vwap(self) -> float:
        """Latest computed VWAP (0.0 before the first warm-up window)."""
        return self._vwap
