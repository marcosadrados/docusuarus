"""Tests for the signal ensemble and its regime-conditioned weighting."""

from signals.ensembler import SignalEnsemble
from signals.regime import RegimeSignal, RegimeStrategy, RegimeSwitcher
from signals.rules import RuleStrategy, parse_rules
from signals.strategy import Candle, Signal

RULE_DSL = """
if close > ema(5): BUY
if close < ema(20): SELL
"""
RULE_STRATEGY = RuleStrategy(parse_rules(RULE_DSL))


def _sample(n=10):
    return [
        Candle(
            timestamp=1700000000 + i,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000,
        )
        for i in range(n)
    ]


def test_ensemble_returns_a_signal():
    ensemble = SignalEnsemble(
        children=[(RULE_STRATEGY, 0.5), (StatArbMock(), 0.5)],
        regime=None,
    )
    sigs = ensemble.generate(_sample())
    assert sigs
    s = sigs[0]
    assert s.direction in ("BUY", "SELL", "HOLD")
    assert 0.0 <= s.confidence <= 1.0
    assert s.strength >= 0.0


def test_ensemble_attaches_regime():
    ensemble = SignalEnsemble(
        children=[(RULE_STRATEGY, 0.5)],
        regime=RegimeSwitcher(),
    )
    sigs = ensemble.generate(_sample())
    assert sigs
    s = sigs[0]
    assert isinstance(s, RegimeSignal)
    assert s.regime in (0, 1, 2)


def test_ensemble_with_rule_weight():
    ensemble = SignalEnsemble(
        children=[(RULE_STRATEGY, 0.5), (StatArbMock(), 0.5)],
        regime=None,
    )
    sigs = ensemble.generate(_sample())
    assert sigs[0].direction == "BUY"


def test_ensemble_empty_children_returns_empty():
    ensemble = SignalEnsemble(children=[], regime=None)
    assert ensemble.generate(_sample()) == []


def test_ensemble_with_regime_strategy_regime_field():
    switcher = RegimeSwitcher()
    ensemble = SignalEnsemble(
        children=[
            (RULE_STRATEGY, 0.5),
            (RegimeStrategy(switcher), 0.5),
        ],
        regime=None,
    )
    sigs = ensemble.generate(_sample())
    assert sigs
    s = sigs[0]
    assert isinstance(s, RegimeSignal)
    assert s.regime in (0, 1, 2)


class StatArbMock:
    """Minimal mock child: returns a BUY signal on every generate call."""

    def generate(self, candles):
        return [
            Signal(
                "BUY",
                confidence=0.6,
                strength=0.5,
                reason="stat-arb mock",
                timestamp=candles[-1].timestamp,
            )
        ]
