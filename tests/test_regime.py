"""Tests for the Gaussian-mixture market-regime switcher."""

from signals.regime import RegimeStrategy, RegimeSwitcher
from signals.strategy import Candle


# A strong uptrend then a violent reversion -> the HMM should move from a
# low-vol regime toward a high-vol mean-reversion regime.
def _candles(n, close_base=100.0, trend=0.1):
    return [
        Candle(
            timestamp=1700000000 + i,
            open=close_base + trend * i,
            high=close_base + trend * i + 0.15,
            low=close_base + trend * i - 0.15,
            close=close_base + trend * i,
            volume=1000000 + i * 100,
        )
        for i in range(n)
    ]


def test_fit_predict_returns_valid_regime_labels():
    rst = RegimeSwitcher(n_states=3, n_iter=30)
    candles = _candles(30)
    labels = rst.fit_predict(candles)
    assert len(labels) == 30
    assert set(labels.tolist()).issubset({0, 1, 2})
    assert rst.predict_last(candles) in (0, 1, 2)


def test_predict_last_returns_valid_regime():
    rst = RegimeSwitcher(n_iter=30)
    rng = list(range(60))
    # slow trend then a crash
    c = []
    for i in rng:
        close = 100.0 + i * 0.05 if i < 40 else 100.0 + 40 * 0.05 - 0.4 * (i - 40)
        c.append(
            Candle(
                timestamp=1700000000 + i,
                open=close,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=1000000,
            )
        )
    assert rst.predict_last(c) in (0, 1, 2)


def test_regime_strategy_emits_regime_signal():
    rst = RegimeSwitcher()
    strat = RegimeStrategy(rst)
    candles = _candles(20)
    sigs = strat.generate(candles)
    assert sigs and sigs[0].regime in (0, 1, 2)
    assert sigs[0].direction == "HOLD"  # state descriptor, no directional bet
