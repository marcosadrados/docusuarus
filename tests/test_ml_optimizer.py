"""Tests for the ML signal model and the walk-forward parameter optimizer.

Coverage:
    1. FeatureBuilder -> fixed-dimension feature matrix, deterministic.
    2. MLSignal -> label synthesis, predict with confidence in [0, 1].
    3. train_model -> fitted signal ready for online prediction.
    4. ParameterOptimizer / optimize_parameters -> grid search over a
       strategy's parameters with rolling walk-forward splits, best params
       carrying a positive walk-forward score on synthetic data.
"""

import os
import sys

import numpy as np

# Project root so ``signals`` and ``backtest`` are importable from anywhere.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtest.optimizer import (  # noqa: E402
    ParamGrid,
    optimize_parameters,
)
from signals.ml import FeatureBuilder, MLSignal, train_model  # noqa: E402
from signals.strategy import Candle, Signal  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic candle factory (same convention as tests/test_signals.py)
# --------------------------------------------------------------------------- #
def make_candles(n=60, base_price=80.0, step=0.01, seed=0):
    """Upward-trending OHLCV candles with rising volume."""
    rng = np.random.default_rng(seed)
    candles = []
    for i in range(n):
        open_price = base_price + i * step + rng.normal(0, 0.01)
        close_price = open_price + 0.05 + rng.normal(0, 0.01)
        candles.append(
            Candle(
                timestamp=1700000000 + i * 60,
                open=open_price,
                high=open_price + 0.1 + abs(rng.normal(0, 0.01)),
                low=open_price - 0.1 - abs(rng.normal(0, 0.01)),
                close=close_price,
                volume=1000000 + i * 10 + rng.normal(0, 100),
            )
        )
    return candles


def make_optimizer_candles(
    n=900, seed=7, base=80.0, drift=0.002, amp=0.4, period=100, noise=0.0
):
    """Synthetic uptrend + cycle (deterministic, no noise).

    An EMA-cross strategy earns a profit here: it buys at cycle troughs and
    sells at cycle peaks, so the walk-forward score is reliably positive.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    level = base + drift * t + amp * np.sin(2 * np.pi * t / period)
    price = np.clip(level, base - 2.0, base + 3.0)

    opens = np.empty(n)
    opens[0] = price[0]
    opens[1:] = price[:-1]
    highs = np.maximum(opens, price) + rng.uniform(0.03, 0.12, n)
    lows = np.minimum(opens, price) - rng.uniform(0.03, 0.12, n)
    volumes = rng.uniform(100.0, 220.0, n)

    return [
        Candle(
            timestamp=1700000000 + i,
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(price[i]),
            volume=float(volumes[i]),
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# Test strategy: parameterised EMA cross (readable by the optimizer)
# --------------------------------------------------------------------------- #
class SimpleEMACrossStrategy:
    """BUY on an up-cross of EMA(ema_period), SELL on a down-cross.

    Only the signal of the last candle in the window is consumed by the
    backtest loop, which keeps the fold evaluation deterministic and cheap.
    Constructor parameters double as optimizer grid keys.
    """

    def __init__(self, ema_period=20, rsi_period=14, lookback=50, **kwargs):
        self.ema_period = int(ema_period)
        self.rsi_period = int(rsi_period)
        self.lookback = int(lookback)

    def _ema(self, data, period):
        if len(data) < period:
            return np.zeros_like(data)
        alpha = 2.0 / (period + 1)
        ema = float(np.mean(data[:period]))
        out = np.zeros_like(data)
        out[period - 1] = ema
        for i in range(period, len(data)):
            ema = (data[i] - ema) * alpha + ema
            out[i] = ema
        return out

    def generate(self, candles):
        signals = []
        if len(candles) < self.ema_period + 2:
            return signals
        closes = np.array([c.close for c in candles], dtype=float)
        ema = self._ema(closes, self.ema_period)
        i = len(candles) - 1
        if closes[i] > ema[i] and closes[i - 1] <= ema[i - 1]:
            signals.append(
                Signal("BUY", 0.6, 0.5, "ema up-cross", candles[i].timestamp)
            )
        elif closes[i] < ema[i] and closes[i - 1] >= ema[i - 1]:
            signals.append(
                Signal("SELL", 0.6, 0.5, "ema down-cross", candles[i].timestamp)
            )
        return signals


# --------------------------------------------------------------------------- #
# FeatureBuilder
# --------------------------------------------------------------------------- #
def test_featurebuilder_returns_consistent_dims():
    candles = make_candles(80)
    builder = FeatureBuilder()

    features, names = builder.transform(candles)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] == len(candles)
    assert features.shape[1] == len(names)
    assert len(names) == 8, names
    assert list(names) == [
        "ema5",
        "ema20",
        "ema50",
        "rsi14",
        "atr14",
        "volume_z",
        "spread",
        "volatility_z",
    ]

    # Byte-identical output on the same input (determinism).
    features2, names2 = builder.transform(candles)
    np.testing.assert_array_equal(features, features2)
    assert names == names2

    # Fixed width regardless of series length; empty input -> (0, 8).
    features_short, names_short = builder.transform(make_candles(5))
    assert features_short.shape == (5, 8)
    assert builder.transform([])[0].shape == (0, 8)


# --------------------------------------------------------------------------- #
# MLSignal
# --------------------------------------------------------------------------- #
def test_mlsignal_predicts_with_confidence_in_range():
    model = train_model(make_candles(80), test_size=0.25, seed=42)

    assert model.model is not None
    assert model.classes_ is not None
    assert set(model.classes_) <= {1, -1}

    candle = make_candles(80)[-1]
    direction, score = model.predict(candle)

    assert direction in ("BUY", "SELL", "HOLD")
    assert 0.0 <= score <= 1.0
    assert isinstance(score, float)


def test_mlsignal_fit_uses_synthetic_and_explicit_labels():
    candles = make_candles(60)

    # None labels -> synthetic close-direction labels.
    model = MLSignal(strategy_name="test")
    model.fit([(c, None) for c in candles])
    assert model.model is not None
    assert model.classes_ is not None
    candle = candles[-1]
    direction, score = model.predict(candle)
    assert direction in ("BUY", "SELL", "HOLD")
    assert 0.0 <= score <= 1.0

    # Explicit labels.
    labels = [1] * 58 + [-1]  # 59 labels for the 59 train rows
    model2 = MLSignal(strategy_name="test2")
    model2.fit(list(zip(candles[:-1], labels)))
    assert model2.model is not None
    assert len(model2.classes_) == 2


def test_train_model_returns_fitted_signal():
    model = train_model(make_candles(100), test_size=0.25, seed=42)

    assert model.model is not None
    assert model.classes_ is not None
    assert len(model._candles) == 100

    # Online prediction on fresh candles: confidence stays in [0, 1].
    for candle in make_candles(10):
        direction, score = model.predict(candle)
        assert direction in ("BUY", "SELL", "HOLD")
        assert 0.0 <= score <= 1.0

    # Same candle twice -> identical (direction, score).
    candle = make_candles(10)[-1]
    assert model.predict(candle) == model.predict(candle)

    # Deep-dip candle (outside the whole training range) still yields a
    # valid direction and a confidence in [0, 1].
    deep = Candle(
        timestamp=1700000000, open=50.0, high=50.1, low=49.9, close=50.0, volume=1.0
    )
    direction, score = model.predict(deep)
    assert direction in ("BUY", "SELL", "HOLD")
    assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# ParameterOptimizer / grid search
# --------------------------------------------------------------------------- #
def test_paramgrid_yields_all_combinations():
    grid = ParamGrid(
        {"ema_period": [5, 10, 20], "rsi_period": [14, 20], "lookback": [20, 50]}
    )
    assert len(grid) == 12
    combos = list(grid)
    assert len(combos) == 12
    assert combos[0] == {"ema_period": 5, "rsi_period": 14, "lookback": 20}
    # Deduplicate: the same combination must not appear twice.
    assert len({tuple(sorted(c.items())) for c in combos}) == 12


def test_optimizer_returns_best_params_with_positive_score(capsys):
    candles = make_optimizer_candles(900, seed=7)

    best_params, best_score, results = optimize_parameters(
        SimpleEMACrossStrategy, candles, n_splits=3
    )

    captured = capsys.readouterr()
    assert "BEST_PARAMS" in captured.out
    assert "walk-forward avg net_pnl" in captured.out

    assert best_params, "optimizer should select parameters"
    assert best_score > 0.0, (
        "walk-forward score should be positive on synthetic data: " f"{best_score!r}"
    )
    assert results, "optimizer should return per-combination results"

    # The winning combination beats the rest on the walk-forward score.
    scores = [r["score"] for r in results]
    assert best_score == max(scores)

    # Every fold of the best combo recorded real test metrics.
    best_fold = results[0]["folds"][0]
    assert best_fold["test_rows"] > 0
    assert best_fold["total_trades"] >= 0


def test_optimizer_prints_summary_on_smoke_run(capsys):
    """Even with a degenerate grid the optimizer prints a best-param summary."""
    candles = make_optimizer_candles(300)
    best_params, best_score, _ = optimize_parameters(
        SimpleEMACrossStrategy,
        candles,
        n_splits=3,
        param_grid={"ema_period": [5, 10], "rsi_period": [14], "lookback": [20]},
    )

    captured = capsys.readouterr()
    assert "BEST_PARAMS" in captured.out

    # A real score must be computed on the synthetic data (positive when the
    # strategy profits, else zero when no trades fired in the test window).
    assert np.isfinite(best_score)
    assert abs(best_score) <= 100.0
    assert best_params
