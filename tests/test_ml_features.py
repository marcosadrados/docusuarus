"""Tests for the wider / feature-selected ML signal model."""

import numpy as np

from signals.freqai import select_features
from signals.ml import FeatureBuilder, _close_direction_labels
from signals.strategy import Candle

# --------------------------------------------------------------------------- #
# Feature builder
# --------------------------------------------------------------------------- #


def _bars(n=15):
    return [
        Candle(
            timestamp=1700000000 + i,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0 + i * 0.1,
            volume=1000000 + i * 100,
        )
        for i in range(n)
    ]


def test_extended_returns_wider_fixed_vector():
    fb = FeatureBuilder(extended=True)
    features, names = fb.transform(_bars(12))
    assert len(names) == 14
    assert features.shape == (12, 14)
    # core columns present in the same order as before
    assert names[:8] == [
        "ema5",
        "ema20",
        "ema50",
        "rsi14",
        "atr14",
        "volume_z",
        "spread",
        "volatility_z",
    ]


def test_extended_default_still_core_only():
    fb = FeatureBuilder(extended=False)
    features, names = fb.transform(_bars(8))
    assert len(names) == 8
    assert features.shape == (8, 8)


def test_close_direction_labels():
    candles = _bars(5)
    labels = _close_direction_labels(candles)
    assert len(labels) == 4
    # strictly rising closes -> all +1
    rising = [
        Candle(
            timestamp=1700000000 + i,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0 + 0.5 * i,
            volume=1000000,
        )
        for i in range(5)
    ]
    assert np.all(_close_direction_labels(rising) == 1.0)


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #


def test_select_features_returns_mask_and_reduced_columns():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    y = np.concatenate([np.ones(20), -np.ones(20)])
    X_sel, mask = select_features(X, y, k=3, method="variance")
    assert X_sel.shape == (40, 3)
    assert mask.sum() == 3


def test_select_features_variance_fallback():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 5))
    y = np.ones(20)
    X_sel, mask = select_features(X, y, k=2, method="variance")
    assert X_sel.shape == (20, 2)
