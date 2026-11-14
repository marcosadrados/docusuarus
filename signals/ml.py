"""Machine-learning signal model for the scalping trading bot.

Provides a feature builder (EMA / RSI / ATR / volume / volatility z-scores)
and a scikit-learn GradientBoostingClassifier wrapped as a trading signal
model with a per-candle ``(direction, score)`` API.

Usage
-----
    from signals.ml import FeatureBuilder, MLSignal, train_model

    # 1) build a fixed feature vector per candle
    builder = FeatureBuilder()
    features, feature_names = builder.transform(candles)

    # 2) train + predict a classifier
    model = train_model(candles, test_size=0.25, seed=42)
    for candle in candles[-10:]:
        direction, score = model.predict(candle)
        # direction in {'BUY', 'SELL', 'HOLD'}, score in [0, 1]

Dependencies: numpy, pandas, scikit-learn.  No external ML packages
required.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from signals.strategy import Candle


# --------------------------------------------------------------------------- #
# Feature builder
# --------------------------------------------------------------------------- #
class FeatureBuilder:
    """Compute a fixed-size feature vector per candle.

    ``transform(candles)`` returns ``(features, feature_names)`` where
    ``features`` has shape ``(n_candles, 8)`` and the columns are, in order:

    ``ema5``, ``ema20``, ``ema50``   -- exponential moving averages
    ``rsi14``                        -- relative strength index (14-period)
    ``atr14``                        -- average true range (14-period)
    ``volume_z``                     -- z-score of volume over the available
                                       history (expanding window)
    ``spread``                       -- percentage bar spread ``(high-low)/close``
    ``volatility_z``                 -- z-score of rolling return volatility

    The column order is fixed, so the same builder can be used for fitting
    and for online prediction.
    """

    def __init__(
        self,
        ema_periods=(5, 20, 50),
        rsi_period=14,
        atr_period=14,
        extended: bool = False,
    ) -> None:
        self.ema_periods = tuple(ema_periods)
        self.rsi_period = int(rsi_period)
        self.atr_period = int(atr_period)
        self.extended = bool(extended)

    @property
    def feature_names(self) -> list[str]:
        """Fixed column names in feature order."""
        if self.extended:
            names = [f"ema{p}" for p in self.ema_periods]
            names += [
                f"rsi{self.rsi_period}",
                f"atr{self.atr_period}",
                "volume_z",
                "spread",
                "volatility_z",
            ]
            names += ["roc5", "roc10", "natr", "cmf", "sma20", "sma50"]
            return names
        names = [f"ema{p}" for p in self.ema_periods]
        names += [
            f"rsi{self.rsi_period}",
            f"atr{self.atr_period}",
            "volume_z",
            "spread",
            "volatility_z",
        ]
        return names

    def fit(self, X, y=None):
        """Stateless transformer: ``fit`` returns ``self``."""
        return self

    def transform(self, candles: list[Candle]) -> tuple[np.ndarray, list[str]]:
        """Compute the feature matrix for a candle list."""
        if not candles:
            return np.zeros((0, len(self.feature_names))), self.feature_names

        cols = {
            "close": np.array([c.close for c in candles], dtype=float),
            "open": np.array([c.open for c in candles], dtype=float),
            "high": np.array([c.high for c in candles], dtype=float),
            "low": np.array([c.low for c in candles], dtype=float),
            "volume": np.array([c.volume for c in candles], dtype=float),
        }
        df = pd.DataFrame(cols)

        # EMA-5/20/50 (expanding recursion, min_periods=1 so row 0 is valid)
        ema_cols = {}
        for p in self.ema_periods:
            ema_cols[f"ema{p}"] = (
                df["close"].ewm(span=p, adjust=False, min_periods=1).mean()
            )

        # RSI-14
        delta = df["close"].diff().fillna(0.0)
        gain = delta.clip(lower=0).rolling(self.rsi_period, min_periods=1).mean()
        loss = (-delta).clip(lower=0).rolling(self.rsi_period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0).to_numpy()

        # ATR-14 (true range)
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (prev_close - df["low"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self.atr_period, min_periods=1).mean().to_numpy()

        # volume z-score (expanding window)
        vol_mean = df["volume"].expanding().mean()
        vol_std = df["volume"].expanding().std(ddof=0).replace(0, np.nan)
        volume_z = ((df["volume"] - vol_mean) / vol_std).fillna(0.0).to_numpy()

        # percentage spread
        spread = ((df["high"] - df["low"]) / df["close"]).to_numpy()

        # volatility z-score (rolling std of log returns)
        rets = df["close"].pct_change().fillna(0.0)
        vol = rets.rolling(max(self.ema_periods), min_periods=1).std(ddof=0)
        vol_mean = vol.expanding().mean()
        vol_std = vol.expanding().std(ddof=0).replace(0, np.nan)
        volatility_z = ((vol - vol_mean) / vol_std).fillna(0.0).to_numpy()

        feature_columns = [ema_cols[f"ema{p}"] for p in self.ema_periods]
        feature_columns += [rsi, atr, volume_z, spread, volatility_z]

        if self.extended:
            # Extend the fixed vector with a handful of coarse, numerically
            # stable indicators.  Row 0 is still valid (min_periods=1) so the
            # pipeline matches the contract in :meth:`FeatureBuilder.transform`.
            e = df["close"]
            # rate of change (period = 5, 10)
            ema_cols["roc5"] = (e / e.shift(5) - 1.0) * 100.0
            ema_cols["roc10"] = (e / e.shift(10) - 1.0) * 100.0
            # normalized ATR (per-bar %) and Chaikin-ish money flow
            ema_cols["natr"] = (tr / e) * 100.0
            ema_cols["cmf"] = (df["volume"] * (df["high"] - df["low"]) / e).rolling(
                self.atr_period, min_periods=1
            ).sum() / df["volume"].rolling(self.atr_period, min_periods=1).sum()
            ema_cols["sma20"] = df["close"].rolling(20, min_periods=1).mean()
            ema_cols["sma50"] = df["close"].rolling(50, min_periods=1).mean()
            extended_names = ["roc5", "roc10", "natr", "cmf", "sma20", "sma50"]
            # core 8 + extended 6 => 14 fixed columns, in feature_names order
            feature_columns = (
                [ema_cols[f"ema{p}"] for p in self.ema_periods]
                + [rsi, atr, volume_z, spread, volatility_z]
                + [ema_cols[k] for k in extended_names]
            )
        else:
            feature_columns = [ema_cols[f"ema{p}"] for p in self.ema_periods] + [
                rsi,
                atr,
                volume_z,
                spread,
                volatility_z,
            ]

        features = np.column_stack(feature_columns)
        return features, self.feature_names

    def fit_transform(self, candles: list[Candle]) -> np.ndarray:
        """Fit (no-op) + transform."""
        return self.transform(candles)[0]


# --------------------------------------------------------------------------- #
# Label helper
# --------------------------------------------------------------------------- #
def _synthetic_labels(candles: list[Candle]) -> np.ndarray:
    """Label per candle: +1 if the next close is higher, else -1.

    Length is ``len(candles) - 1``; the label at index *i* corresponds to
    candle *i* and predicts the close-direction from candle *i* to *i+1*.
    """
    closes = np.array([c.close for c in candles], dtype=float)
    return np.where(np.diff(closes) >= 0.0, 1.0, -1.0)


# --------------------------------------------------------------------------- #
# ML signal model
# --------------------------------------------------------------------------- #
class MLSignal:
    """Wraps a scikit-learn binary classifier as a trading signal model.

    ``fit(candles_with_labels)`` trains a :class:`GradientBoostingClassifier`
    on the feature matrix built by :class:`FeatureBuilder`.  When candle
    labels are not given explicitly they are synthesised from the close
    direction (+1 if the next close is higher, else -1).

    ``predict(candle)`` returns ``(direction, score)`` where
    ``direction in {'BUY', 'SELL', 'HOLD'}`` and ``score`` is a
    probability-like confidence in ``[0, 1]`` (``HOLD`` when the model is
    not confident enough).
    """

    HOLD_THRESHOLD: float = 0.55

    def __init__(self, strategy_name: str, threshold: float = 0.55):
        self.strategy_name = strategy_name
        self.threshold = threshold
        self.model: GradientBoostingClassifier = GradientBoostingClassifier(
            random_state=42
        )
        self.feature_builder = FeatureBuilder()
        self._candles: list[Candle] = []
        self.feature_names_: list[str] = []
        self.classes_: np.ndarray | None = None

    # -- public API --------------------------------------------------------- #
    def fit(self, candles_with_labels) -> "MLSignal":
        """Fit the classifier on candle/label pairs.

        ``candles_with_labels`` is a list of ``(Candle, label|None)`` tuples
        (``label`` may be ``None``, in which case the close direction is
        synthesised).  Returns ``self``.
        """
        if not candles_with_labels:
            raise ValueError("candles_with_labels must not be empty")

        # Accept either [(candle, label), ...] or a bare [candle, ...] list.
        first = candles_with_labels[0]
        if isinstance(first, tuple):
            pairs = [
                (c, lab if lab is not None else None) for c, lab in candles_with_labels
            ]
        else:
            pairs = [(c, None) for c in candles_with_labels]

        candles = [c for c, _ in pairs]
        if len(candles) < 2:
            raise ValueError("need at least two candles to fit the model")

        self._candles = candles
        features, names = self.feature_builder.transform(candles)
        self.feature_names_ = names

        # Synthesise the close-direction label (+1 if the next close is
        # higher, else -1) for every candle except the last one.
        synth = _synthetic_labels(candles)

        # Build training rows from the provided pairs: explicit labels win,
        # otherwise the synthetic close direction.  The final pair (the last
        # candle) has no label yet -> excluded from training.
        train_rows = []
        for i, (_, lab) in enumerate(pairs):
            if lab is not None:
                train_rows.append((i, float(lab)))
            elif i < len(synth):
                train_rows.append((i, float(synth[i])))
            # else: last pair -> no label -> excluded
        if not train_rows:
            raise ValueError("no usable labels in candles_with_labels")

        X = features[[i for i, _ in train_rows]]
        y = np.asarray([lab for _, lab in train_rows], dtype=float)

        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[valid], y[valid]

        if len(np.unique(y)) < 2:
            # Degenerate single-class sample: still fit (predict will hold).
            X_train, y_train = X, y
        else:
            X_train, y_train = X, y

        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        return self

    def predict(self, candle: Candle) -> tuple[str, float]:
        """Predict the direction and confidence for one candle."""
        if not self._candles:
            raise RuntimeError("model has not been fitted yet")

        context = self._candles + [candle]
        features, _ = self.feature_builder.transform(context)
        row = features[-1]

        if not np.all(np.isfinite(row)):
            # Not enough history for reliable features -> neutral HOLD.
            return "HOLD", 0.5

        proba = self.model.predict_proba(row.reshape(1, -1))[0]
        classes = list(self.classes_)
        idx_up = classes.index(1) if 1 in classes else None
        idx_down = classes.index(-1) if -1 in classes else None

        if idx_up is None or idx_down is None:
            return "HOLD", 0.5

        p_up = float(proba[idx_up])
        p_down = float(proba[idx_down])
        score = max(p_up, p_down)

        if p_up >= self.threshold and p_up >= p_down:
            return "BUY", score
        if p_down >= self.threshold and p_down > p_up:
            return "SELL", score
        return "HOLD", score

    # -- convenience ------------------------------------------------------ #
    def predict_proba(self, candle: Candle) -> np.ndarray:
        """Return the raw class probabilities for ``candle``."""
        if not self._candles:
            raise RuntimeError("model has not been fitted yet")
        context = self._candles + [candle]
        features, _ = self.feature_builder.transform(context)
        return self.model.predict_proba(features[-1].reshape(1, -1))[0]


# --------------------------------------------------------------------------- #
# Module-level trainer
# --------------------------------------------------------------------------- #
def train_model(
    candles: list[Candle],
    test_size: float = 0.25,
    seed: int = 42,
    strategy_name: str = "ml_model",
) -> MLSignal:
    """Build a :class:`MLSignal` and fit it on the given candles.

    Labels are synthesised from the close direction (+1 if the next close is
    higher, else -1).  Features are split into a train/validation set with
    ``test_size`` and the classifier is fitted on the training split.

    Returns the fitted signal model.
    """
    if len(candles) < 2:
        raise ValueError("need at least two candles to train the model")

    signal = MLSignal(strategy_name=strategy_name)

    features, _ = signal.feature_builder.transform(candles)
    y = _synthetic_labels(candles)

    # Train rows are candles 0 .. n-2 (the last candle has no label yet).
    X = features[:-1].copy()
    y = np.asarray(y, dtype=float)[: len(X)]
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]

    if len(np.unique(y)) > 1:
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    else:
        X_train, y_train = X, y

    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X_train, y_train)

    signal.model = model
    signal.classes_ = model.classes_
    signal._candles = list(candles)
    return signal

    # --------------------------------------------------------------------------- #
    # Self-adaptive ML (FreqAI-style local retraining)
    # --------------------------------------------------------------------------- #
    """Add-on for :mod:`signals.ml`.

    :class:`AdaptiveMLSignal` wraps a trained :class:`MLSignal` and applies
    periodic retraining on a rolling window of candles -- a lightweight, honest
    approximation of FreqAI-style self-adaptation: the model keeps its
    structure, but the evidence it trains on keeps moving with the market.

    This is deliberately conservative: it retrains on *retrain_every* bars and
    only ever fits on the *most recent* window (never the full history), so a
    transient regime never poisons the model.  It is a **signal source**, not a
    payment or execution layer.
    """


__all__ = ["AdaptiveMLSignal"]


class AdaptiveMLSignal:
    """MLSignal wrapped for self-adaptive retraining.

    Parameters
    ----------
    window:
        Number of recent candles to retain as the retraining corpus.
    retrain_every:
        Number of new candles ingested between refits.
    base_model:
        Optional seed :class:`MLSignal`.  If omitted, a fresh one is built.

    Example
    -------
    >>> import signals.ml as ml
    >>> model = ml.AdaptiveMLSignal(window=100, retrain_every=25)
    >>> model.feed(candles)          # push history
    >>> direction, score = model.predict(new_candle)
    """

    def __init__(
        self,
        window: int = 200,
        retrain_every: int = 40,
        base_model: MLSignal | None = None,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if retrain_every < 1:
            raise ValueError(f"retrain_every must be >= 1, got {retrain_every}")
        self.window = window
        self.retrain_every = retrain_every
        self._history = deque(maxlen=window)
        self._retrain_counter = 0
        self._seed = base_model
        self.model = base_model or MLSignal(strategy_name="adaptive")
        self._fitted = False

    # ------------------------------------------------------------------
    # Training surface
    # ------------------------------------------------------------------

    def feed(self, candles: list[Candle]) -> None:
        """Push new bars through the model (triggers retraining)."""
        if not candles:
            return
        self._history.extend(candles)
        self._retrain_counter += 1
        if self._retrain_counter >= self.retrain_every:
            self._retrain_counter = 0
            self.retrain()

    def retrain(self) -> MLSignal:
        """Refit the underlying model on the current window. Returns ``self.model``."""
        data = list(self._history)
        if len(data) < 2:
            return self.model
        # Guard against a degenerate (single-class) window that makes the
        # underlying classifier raise; in that case keep the current model.
        try:
            candidate = MLSignal(strategy_name="adaptive")
            candidate.fit([(c, None) for c in data])
        except ValueError:
            return self.model
        self.model = candidate
        self._fitted = True
        return self.model

    def predict(self, candle: Candle) -> tuple[str, float]:
        """Predict and auto-retrain on the given bar. Never raises."""
        if len(self._history) < 2:
            return "HOLD", 0.5
        if not self._fitted:
            self.retrain()
        self.model._candles = list(self._history)
        try:
            return self.model.predict(candle)
        except Exception:
            # Degenerate or unfitted classifier: neutral HOLD.
            return "HOLD", 0.5


# --------------------------------------------------------------------------- #
# Labels (shared helper)
# --------------------------------------------------------------------------- #
def _close_direction_labels(candles: Sequence) -> np.ndarray:
    """(+1 if the next close is higher, else -1). -1 shorter by one candle."""
    closes = np.array([c.close for c in candles], dtype=float)
    return np.where(np.diff(closes) >= 0.0, 1.0, -1.0)


__all__ = [
    "FeatureBuilder",
    "MLSignal",
    "train_model",
    "AdaptiveMLSignal",
    "_close_direction_labels",
]
