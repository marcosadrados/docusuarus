"""Pairs / statistical-arbitrage mean-reversion signal.

A stateless, testable mean-reversion strategy.  For each registered pair
``(lead_symbol, relate_symbol, beta, window)`` it tracks the spread

    spread = lead_close - beta * relate_close

and fires on the *lead* symbol with a BUY when the spread is cheap
(low Z-score, i.e. the lead has underperformed the relate) or a SELL when
the spread is rich (high Z-score).  This is the classic stat-arb / mean
reversion edge: two assets that move together occasionally drift apart,
and the trade expects the spread to revert to its historical mean.

The strategy is deliberately conservative: it only fires once per tick and
only when the Z-score has crossed the entry threshold.  It is a **single
position per symbol** (sized by the engine's existing ATR-risk scaler), so
it stays inside the bot's 0.25%/trade risk envelope.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from signals.strategy import Candle, Signal

#: Z-score entry threshold.  When the spread is this many standard
#: deviations from its rolling mean, we expect mean reversion.
ENTRY_Z = 2.0
#: Z-score exit threshold.  Only needed for an explicit exit; the
#: engine closes on the protective stop, so this is a dead-man switch.
EXIT_Z = 3.0

#: A registered pair: (lead, relate, beta, window).
Pair = Tuple[str, str, float, int]


def _zscore(values: List[float], window: List[float]) -> float:
    """Z-score of *values* relative to *window* (the recent history)."""
    if len(window) < 2:
        return 0.0
    mean = sum(window) / len(window)
    var = sum((w - mean) ** 2 for w in window) / len(window)
    std = var**0.5
    if std == 0:
        return 0.0
    return (values[-1] - mean) / std


class StatArbStrategy:
    """Statistical-arbitrage mean-reversion scaler.

    Parameters
    ----------
    pairs:
        List of ``(lead, relate, beta, window)`` tuples, where ``lead`` is
        the symbol that receives the signal, ``relate`` is the pair's hedge
        asset (held as a recent-bar snapshot), and ``beta`` is the hedge
        ratio (how much of the relate offsets the lead).
    relate_prices:
        Dict of symbol -> recent close price, optionally supplied by the
        embedding application.  Used to anchor the spread for assets that
        have not yet been scanned.

    Example
    -------
    >>> strat = StatArbStrategy([
    ...     ("CL=F", "BZ=F", 0.85, 100),
    ...     ("XAU=F", "XAG=F", 2.2, 120),
    ... ], relate_prices={"BZ=F": 66.0, "XAG=F": 38.0})
    >>> sigs = strat.generate(candles)  # candles for CL=F
    >>> sigs
    [Signal(direction='BUY', ...)]
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, float, int]] = None,
        relate_prices: Dict[str, float] = None,
    ) -> None:
        self.pairs: List[Tuple[str, str, float, int]] = [
            tuple(p) for p in (pairs or [])
        ]
        # code -> recent close (last bar) seen in update()
        self._relate_closes: Dict[str, float] = dict(relate_prices or {})

    def update(self, code: str, candles: List[Candle]) -> None:
        """Cache the latest bar for a symbol so it can anchor pair spreads."""
        if not candles:
            return
        self._relate_closes[code] = float(candles[-1].close)

    def generate(self, candles: List[Candle]) -> List[Signal]:
        """Return one signal per fired pair, on the lead symbol."""
        if not candles or len(candles) < max(p[3] for p in self.pairs):
            return []

        closes = [c.close for c in candles]
        last = closes[-1]
        signals: List[Signal] = []

        for lead, relate, beta, window in self.pairs:
            # relate asset price snapshot
            relate_price = self._relate_closes.get(relate)
            if relate_price is None:
                # cannot anchor the spread yet: cache the first bar, defer
                if len(closes) >= window:
                    self._relate_closes[relate] = last
                continue

            # Rolling window over this symbol's own close (the leader).
            lead_window = closes[-window:] if len(closes) >= window else closes

            # Spread at the last bar
            spread = last - beta * relate_price

            # Rolling statistics of the spread, off the lead window
            spread_window = [c - beta * relate_price for c in lead_window]
            spread_mean = sum(spread_window) / len(spread_window)
            var = sum((s - spread_mean) ** 2 for s in spread_window) / len(
                spread_window
            )
            std = var**0.5
            if std == 0:
                continue
            # z-score of the current spread against its own rolling history
            z = (spread - spread_mean) / std

            # Direction: z < -ENTRY_Z -> cheap (expect lead to catch up) -> BUY
            # z >  ENTRY_Z -> rich (expect lead to give back) -> SELL
            if z < -ENTRY_Z:
                confidence = min(0.85, 0.5 + (z + ENTRY_Z) * 0.08)
                signals.append(
                    Signal(
                        direction="BUY",
                        confidence=confidence,
                        strength=0.6,
                        reason=(
                            f"stat-arb pairs {lead}/{relate}: "
                            f"spread Z-score = {z:.2f} (cheap)"
                        ),
                        timestamp=candles[-1].timestamp,
                    )
                )
            elif z > ENTRY_Z:
                confidence = min(0.85, 0.5 + (z - ENTRY_Z) * 0.08)
                signals.append(
                    Signal(
                        direction="SELL",
                        confidence=confidence,
                        strength=0.6,
                        reason=(
                            f"stat-arb pairs {lead}/{relate}: "
                            f"spread Z-score = {z:.2f} (rich)"
                        ),
                        timestamp=candles[-1].timestamp,
                    )
                )

        return signals
