"""ATR (Average True Range) computation and volatility-based position scaling.

The bot's fixed 0.25%-per-trade risk ceiling assumes a "typical" stop width.  On
a choppy (high ATR) bar a fixed-percentage stop is widened by noise and a
constant-size position is quietly over-exposing; on a calm bar the same
position is under-exposing.  This module lets the size "breathe with the
market": it shrinks size when ATR is high and grows it when ATR is low, so the
round-trip dollar risk stays pinned near the per-trade limit.

Pure functions; no network, no broker.  Works with Candle dataclasses
(unit tests) and raw CCXT OHLCV rows (timestamp, open, high, low, close,
volume), which is what the live data pipeline returns.

Example
-------
>>> from risk.atr import average_true_range, volatility_factor
>>> atr = average_true_range(candles, period=14)[-1]
>>> volatility_factor(atr, reference_atr=atr)   # neutral
1.0
>>> volatility_factor(atr * 3, reference_atr=atr)   # highly volatile -> shrink
0.5
"""

from __future__ import annotations

from typing import List

#: How many candles are used to smooth the Average True Range (EMA basis).
DEFAULT_ATR_PERIOD = 14

#: Hard lower/upper bounds on the volatility adjustment factor.
MIN_VOL_FACTOR = 0.5
MAX_VOL_FACTOR = 2.0

__all__ = ["average_true_range", "volatility_factor"]


def _hlcv(row):
    """Open/high/low/close of one bar, from a Candle object, dict, or OHLCV row."""
    if isinstance(row, dict):
        # dict: {open, high, low, close, ...} (or timestamp-keyed)
        return row["open"], row["high"], row["low"], row["close"]
    if isinstance(row, (list, tuple)):
        # CCXT OHLCV: [timestamp, open, high, low, close, volume]
        return row[1], row[2], row[3], row[4]
    return row.open, row.high, row.low, row.close


def average_true_range(candles: List, period: int = DEFAULT_ATR_PERIOD) -> List[float]:
    """Return ATR(period) for each candle, length ``len(candles)``.

    The first ``period`` samples are warm-up (EMA needs a full lookback); the
    list is still aligned so ``atr[i]`` is the average true range ending at
    ``candles[i]`` (for candles with enough history).

    Accepts Candle dataclasses or raw CCXT OHLCV rows so it works both in unit
    tests and against live-fetched data.
    """
    if not candles:
        return []
    n = len(candles)
    if n < period + 1:
        # Not enough bars for a real EMA; fall back to a small flat ATR so the
        # min_factor cap on the position scaler keeps this from over-shrinking.
        return [0.1] * n

    # True range per bar (TTR[0] is the first bar's full body).
    tr: List[float] = [_hlcv(candles[0])[1] - _hlcv(candles[0])[2]]
    for i in range(1, n):
        h, low = _hlcv(candles[i])[1], _hlcv(candles[i])[2]
        prev_close = _hlcv(candles[i - 1])[3]
        tr.append(max(h - low, abs(h - prev_close), abs(low - prev_close)))

    multiplier = 2.0 / (period + 1)
    # Warm-up: the first ``period`` TR values feed the initial EMA.  We map
    # the resulting EMA back to candle indices, so a value exists only once
    # enough history has elapsed.
    atr = sum(tr[:period]) / period
    result: List[float] = [0.0] * (period - 1)  # not enough history yet
    result.append(atr)  # ATR ending at candle period-1
    for v in tr[period:]:
        atr = (v - atr) * multiplier + atr
        result.append(atr)
    return result


def volatility_factor(
    atr: float,
    reference_atr: float,
    min_factor: float = MIN_VOL_FACTOR,
    max_factor: float = MAX_VOL_FACTOR,
) -> float:
    """Scale position size relative to a reference ATR.

    - ``atr == reference_atr``               -> 1.0 (no change)
    - ``atr`` is low (calm)                 -> > 1.0 (grow the position, up to ``max_factor``)
    - ``atr`` is high (volatile)             -> < 1.0 (shrink the position, down to ``min_factor``)

    Raises ``ValueError`` if either value is non-positive.
    """
    if atr <= 0 or reference_atr <= 0:
        raise ValueError("ATR and reference ATR must be positive")
    factor = reference_atr / atr
    return min(max_factor, max(min_factor, factor))
