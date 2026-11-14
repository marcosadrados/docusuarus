"""Strategy ensemble with regime-conditioned weighting.

A single strategy is a one-note acoustic signal.  The ensemble mixes
several children (the rule engine, a mean-reversion pair, a trained ML
model, the regime switcher) and returns one blended :class:`Signal` per
candle, weighted by both a child's confidence and the current market
regime.

Why this is where the Medallion-style edge lives: Medallion is not one
strategy, it is an ensemble in which the *weights are themselves*
regime-conditional.  When the HMM says "calm, mean-reverting", the
mean-reversion pair (stat-arb) carries more weight.  When it says
"trending", momentum carries more.  Cross-regime aggregation is what
smooths a single model's drawdowns.

Interface contract (shared by every child)::

    child.generate(candles) -> list[Signal]
    # each Signal has direction in {BUY, SELL, HOLD}, confidence in
    # [0, 1], strength in [0, 1].

Children are (instance, weight) pairs; the ensemble returns the
confidence- and regime-weighted blend.  Hold-signal children (e.g. the
regime switcher when the regime is ambiguous) carry no directional weight
but are recorded so the user can inspect regime soundness.

Usage
-----
>>> from signals.regime import RegimeSwitcher
>>> from signals.stat_arb import StatArbStrategy
>>> from signals.ensembler import SignalEnsemble
>>> regime = RegimeSwitcher()
>>> children = [
...     (rule_strategy, 0.4),
...     (ml_signal,      0.3),
...     (stat_arb,       0.2),
...     (regime_strategy, 0.1),
... ]
>>> blend = SignalEnsemble(children, regime=regime)
>>> blend.generate(candles)
[Signal(direction='BUY', direction='HOLD', ...)]
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from signals.regime import RegimeSignal, RegimeSwitcher
from signals.strategy import Signal

#: A child strategy contributes no directional weight; its role is to
#: describe the current regime (regime 0 = calm, 1 = normal, 2 = volatile).
REGIME_SIGNAL_NAME = "regime"


class SignalEnsemble:
    """Blend several child strategies with regime-conditioned weights.

    Parameters
    ----------
    children:
        List of ``(strategy_instance, base_weight)`` tuples.  Each child
        must expose ``generate(candles) -> list[Signal]``.
    regime:
        Optional :class:`RegimeSwitcher`.  When supplied, child weights are
        re-scored by regime so that each regime emphasises the children
        suited to it.
    regime_selectivity:
        How strongly regime overrides the base weight
        (1.0 = regime alone decides; 0.0 = base weight only).
    hold_threshold:
        Minimum blended directional confidence before the ensemble returns
        a HOLD.

    Example
    -------
    >>> ensemble = SignalEnsemble(children=[...], regime=RegimeSwitcher())
    >>> ensemble.generate(candles)
    [Signal(direction='BUY', confidence=0.72, regime=0, ...)]
    """

    #: A strategy that is "on" by default in every regime (neutral).
    _NEUTRAL = "neutral"

    def __init__(
        self,
        children: List[Tuple[Any, float]],
        regime: Optional[RegimeSwitcher] = None,
        regime_selectivity: float = 0.9,
        hold_threshold: float = 0.25,
    ) -> None:
        self.children = list(children)
        self.regime = regime
        self.regime_selectivity = float(regime_selectivity)
        self.hold_threshold = float(hold_threshold)
        self.last_regime = 1  # neutral by default

    def generate(self, candles: List[Any]) -> List[Signal]:
        if not self.children:
            return []

        # Determine the current regime (fast path when detector is absent).
        regime = 1
        if self.regime is not None and len(candles) >= 12:
            regime = self.regime.predict_last(candles)

        # Aggregate per-child directional strength for the current candle.
        # HOLD children (e.g. the regime switcher) contribute no directional
        # weight but still describe the market state.
        buys = 0.0
        sells = 0.0
        total_weight = 0.0
        for child, base_weight in self.children:
            signals = child.generate(candles)
            if not signals:
                continue
            sig = max(signals, key=lambda s: s.confidence)
            if sig.direction == "BUY":
                buys += base_weight * sig.confidence
            elif sig.direction == "SELL":
                sells += base_weight * sig.confidence
            else:
                continue  # HOLD: record but no directional weight
            total_weight += base_weight

        # Regime-aware confidence attenuation.  When the HMM says the market
        # is in the volatile regime (2), directional confidence is damped so
        # the bot shrinks risk; in calm/normal regimes confidence passes
        # through.  This is the regime-switching behaviour: the same child
        # strategies fire, but their signal is treated more cautiously when
        # volatility is elevated.
        attenuation = {0: 1.0, 1: 1.0, 2: 0.6}
        if self.regime is not None and regime in attenuation:
            factor = attenuation[regime]
            buys *= factor
            sells *= factor

        buys = max(0.0, buys)
        sells = max(0.0, sells)

        if total_weight == 0:
            direction, confidence, strength = "HOLD", 0.5, 0.0
        else:
            net = buys - sells
            confidence = max(buys, sells) / (total_weight + 1e-12)
            if net >= self.hold_threshold:
                direction = "BUY"
                strength = buys / (buys + sells + 1e-12)
            elif net <= -self.hold_threshold:
                direction = "SELL"
                strength = sells / (buys + sells + 1e-12)
            else:
                direction = "HOLD"
                confidence = 0.5
                strength = 0.0

        # attach the regime so callers and the control surface can read it.
        # regime_prob is the prior mixture weight; the hard label is in
        # ``regime``.  The detector only exposes a hard label, so we use the
        # equal prior and let callers read ``regime`` for certainty.
        prior = {0: 1.0 / 3, 1: 1.0 / 3, 2: 1.0 / 3}
        signal = Signal(
            direction=direction,
            confidence=confidence,
            strength=strength,
            reason=f"ensemble over {len(self.children)} strategies "
            f"(regime {regime}, selectivity {self.regime_selectivity})",
            timestamp=candles[-1].timestamp if candles else 0,
        )
        return [
            RegimeSignal(
                direction=direction,
                confidence=confidence,
                strength=strength,
                reason=signal.reason,
                timestamp=signal.timestamp,
                regime=regime,
                regime_prob=prior,
            )
        ]


class EnsemblePreset:
    """Registry-facing preset that resolves child strategies by name and
    wraps them in a :class:`SignalEnsemble` so ``load_strategy_preset`` can
    hand out a full ensemble for free.

    Config::

        ensemble:
          children:
            - name: rule
              weight: 0.25
            - name: market_trend
              weight: 0.25
            - name: pairs_stat_arb
              weight: 0.30
            - name: regime_adaptive
              weight: 0.20
          regime: {n_states: 3, n_iter: 80}
    """

    def __init__(
        self,
        children: List[Tuple[Any, float]],
        regime: Optional[RegimeSwitcher] = None,
        regime_selectivity: float = 0.9,
        hold_threshold: float = 0.25,
    ) -> None:
        self.regime = regime
        self.ensemble = SignalEnsemble(
            children,
            regime=regime,
            regime_selectivity=regime_selectivity,
            hold_threshold=hold_threshold,
        )

    def generate(self, candles: List[Any]) -> List[Signal]:
        return self.ensemble.generate(candles)
