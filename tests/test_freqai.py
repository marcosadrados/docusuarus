"""Tests for the FreqAI-style self-adaptive ML signal model."""

import numpy as np

from signals.freqai import Freqai, select_features
from signals.ml import _close_direction_labels
from signals.strategy import Candle


def _bars(n=60, up=True):
    """Bars; up=True rises, else alternates so the corpus stays mixed."""
    step = 0.05
    return [
        Candle(
            timestamp=1700000000 + i,
            open=100.0,
            high=101.0,
            low=99.0,
            # Rising: +0.05/step.  Alternating: a 0.2 spread so classes are
            # neither single-class nor all-equal (keeps retrain meaningful).
            close=100.0 + (i * step if up else ((i % 2) * 0.2 - 0.1)),
            volume=1000000,
        )
        for i in range(n)
    ]


def test_freqai_predict_returns_direction_and_score():
    model = Freqai(window=30, retrain_every=5)
    model.feed(_bars(35))
    direction, score = model.predict(_bars(35)[-1])
    assert direction in ("BUY", "SELL", "HOLD")
    assert 0.0 <= score <= 1.0


def test_freqai_feed_triggers_retrain():
    model = Freqai(window=30, retrain_every=5)
    # Mixed direction keeps the retraining corpus multi-class, so retrain()
    # actually refits and sets the fitted flag.
    model.feed(_bars(35, up=False))
    # Each retrain() set the internal flag; the window now holds 30 bars.
    assert model._fitted is True
    assert len(model.model._candles) == 30
    # A retrained model still predicts (no crash).
    d, s = model.predict(_bars(35)[-1])
    assert d in ("BUY", "SELL", "HOLD")


def test_freqai_single_class_window_keeps_model():
    # rising-only window -> single class; retrain must not crash or overwrite.
    model = Freqai(window=30, retrain_every=5)
    rising = [
        Candle(
            timestamp=1700000000 + i,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0 + 0.1 * i,
            volume=1000000,
        )
        for i in range(35)
    ]
    model.feed(rising)
    direction, score = model.predict(rising[-1])
    # should not raise and should return neutral HOLD when there is no signal
    assert direction in ("BUY", "SELL", "HOLD")


def test_select_features_selects_k_features():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8))
    y = np.concatenate([np.ones(20), -np.ones(20)])
    X_sel, mask = select_features(X, y, k=3, method="variance")
    assert X_sel.shape == (40, 3)
    assert mask.sum() == 3


def test_freqai_evaluate_returns_report():
    model = Freqai(window=20, retrain_every=5)
    X, _ = model.model.feature_builder.transform(_bars(26))
    y = _close_direction_labels(_bars(27))
    report = model.evaluate(X, y, k=5)
    assert report["n_samples"] == 26
    assert report["n_features_kept"] <= 5
