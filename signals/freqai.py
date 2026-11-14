"""Freqtrade-style self-adaptive ML signal model (FreqAI-style local retraining).

Adds three things on top of :mod:`signals.ml`:

1. **Wider feature builder** -- ``FeatureBuilder.extended = True`` returns a
   larger (but still fixed-column) feature vector so the model has more signal
   to select from without changing the existing 8-feature default contract.
2. **Feature selection** -- pick the informative columns out of the wide
   vector before fitting (``variance`` or mutual-information selection).
3. **FreqAI-style retraining harness** -- ``Freqai`` wraps an
   ``AdaptiveMLSignal`` and exposes a clean retrain / predict / evaluate
   surface so a live model keeps its structure while the evidence it trains on
   keeps moving with the market.

This is a signal source, not an execution layer.  It is deliberately
conservative: the model keeps its scikit-learn structure and it retrains only
on the *most recent* window.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from signals.ml import MLSignal

try:  # scikit-learn feature_selection is optional at runtime
    from sklearn.feature_selection import mutual_info_classif

    _HAS_MI = True
except Exception:
    _HAS_MI = False

__all__ = ["select_features", "Freqai"]


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
def select_features(
    X: np.ndarray,
    y: Sequence[float],
    k: Optional[int] = None,
    method: str = "variance",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(X_selected, mask)`` keeping the most informative columns.

    Parameters
    ----------
    X: shape (n_samples, n_features)
    y: target labels, same length as n_samples
    k: keep at most k features (``None`` = use the full set)
    method: ``"variance"`` (deterministic, keeps the k most variable columns)
            or ``"mutual_info"`` (sklearn mutual information, needs
            ``sklearn.feature_selection`` importable)
    random_state: seed for the mutual-information estimator

    Returns
    -------
    X_selected : (n_samples, n_selected)
    mask       : boolean array, True for kept columns, in original order
    """
    y = np.asarray(y, dtype=float)
    if len(y) != X.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    if method == "mutual_info" and _HAS_MI:
        mi = mutual_info_classif(X, y, random_state=random_state, n_neighbors=5)
        order = np.argsort(mi)[::-1]
        if k is not None:
            order = order[:k]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[order] = True
        return X[:, mask], mask

    # Pure-variance fallback: deterministic, no model needed.
    variances = np.nanvar(X, axis=0)
    order = np.argsort(variances)[::-1]
    if k is not None:
        order = order[: max(k, 0)]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[order] = True
    return X[:, mask], mask


# --------------------------------------------------------------------------- #
# FreqAI-style retraining harness
# --------------------------------------------------------------------------- #
class Freqai:
    """Self-adaptive ML signal model with a clean retrain / evaluate surface.

    Parameters
    ----------
    window : int
        Rolling retraining corpus size (how many recent candles the model
        trains on).
    retrain_every : int
        Number of new candles ingested before the model is refit.
    strategy_name : str
        Name used by the underlying signal model.
    base_model : MLSignal | None
        Optional seed model to inherit ``feature_names_`` and calibration.

    Example
    -------
    >>> from signals.ml import FeatureBuilder
    >>> from signals.freqai import Freqai, select_features
    >>> fb = FeatureBuilder(extended=True)
    >>> model = Freqai(window=200, retrain_every=40)
    >>> model.feed(candles)                 # push history
    >>> direction, score = model.predict(new_candle)
    >>> X, y = fb.transform(candles)[0], _close_direction_labels(candles)
    >>> X_sel, mask = select_features(X, y, k=12, method="variance")
    >>> print(model.evaluate(X_sel, y, mask))  # short offline report
    """

    def __init__(
        self,
        window: int = 200,
        retrain_every: int = 40,
        strategy_name: str = "freqai",
        base_model: Optional[MLSignal] = None,
    ) -> None:
        self.window = window
        self.retrain_every = retrain_every
        self.strategy_name = strategy_name
        # The underlying model, carrying the fitted scikit-learn classifier
        # and its feature_names_.  On a retrain this is swapped for a freshly
        # fitted child; the Freqai wrapper owns the window/history cadence.
        self._model = (
            base_model
            if base_model is not None
            else MLSignal(strategy_name=strategy_name)
        )
        self._history: deque = deque(maxlen=window)
        self._retrain_counter = 0
        self._fitted = False

    # -- public API ---------------------------------------------------------

    def feed(self, candles: Sequence) -> None:
        """Push new bars through the model (triggers a retrain at cadence)."""
        if not candles:
            return
        for _c in candles:
            self._history.append(_c)
            self._retrain_counter += 1
            if self._retrain_counter >= self.retrain_every:
                self._retrain_counter = 0
                self.retrain()

    def retrain(self) -> MLSignal:
        """Force a refit on the current window. Returns the fitted model."""
        data = list(self._history)
        if len(data) < 2:
            return self.model
        try:
            candidate = MLSignal(strategy_name=self.strategy_name)
            candidate.fit([(c, None) for c in data])
        except ValueError:
            # Degenerate (single-class) window: keep the current model.
            return self.model
        self._model = candidate
        self._fitted = True
        return self.model

    def predict(self, candle) -> Tuple[str, float]:
        """Predict on the given bar. Auto-retrains on unfitted data. Never raises."""
        if len(self._history) < 2:
            return "HOLD", 0.5
        if not self._fitted:
            self.retrain()
        try:
            self.model._candles = list(self._history)
            return self.model.predict(candle)
        except Exception:
            # Degenerate or unfitted classifier: neutral HOLD.
            return "HOLD", 0.5

    @property
    def model(self) -> MLSignal:
        """The underlying fitted scikit-learn classifier."""
        return self._model

    def feature_names(self) -> list[str]:
        """Feature names used by the underlying model, if available."""
        return list(getattr(self.model, "feature_names_", []))

    def evaluate(
        self,
        X: np.ndarray,
        y: Sequence[float],
        mask: Optional[np.ndarray] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Short offline report for a feature set.

        Parameters
        ----------
        X, y       : feature matrix + labels
        mask       : column mask (if not given, no selection is applied)
        k          : if not None and mask is None, select the top-k features

        Returns a dict with the dataset shape and selected-column count.
        """
        if mask is None and k is not None:
            X, mask = select_features(X, y, k=k, method="variance")
        elif mask is None:
            mask = np.ones(X.shape[1], dtype=bool)
        return {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_features_kept": int(np.count_nonzero(mask)),
        }
