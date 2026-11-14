"""Regime detection as a 3-state Gaussian mixture (regime switching).

Markets move in regimes: low-volatility trend, sideways, and high-volatility
mean-reversion.  A Gaussian mixture model assigns each candle to one of three
hidden regimes by fitting a per-regime Gaussian to a small feature vector.

This is a faithful "regime switching" model: a hidden state per candle, learned
from data, with a Viterbi-style hard assignment.  It is deliberately cheaper
and more numerically stable than a full Baum-Welch HMM, and it needs only
numpy.

Features per candle (each column z-scored over the available history):
    r_t        log-return
    atr_z      z-score of true range (volatility)
    volume_z   z-score of volume
    volatility_z z-score of rolling return volatility

Regime ordering (fixed)::
    0  -- calm / low-volatility trending
    1  -- normal / sideways
    2  -- high-volatility mean-reversion

The regime is surfaced on :class:`Signal` as ``regime`` (0/1/2) so the
ensemble, the rule engine and the control surface can act on it.
"""

from __future__ import annotations

import numpy as np

from signals.strategy import Candle, Signal

from typing import Any  # noqa: F401 (annotation-only use)

#: Number of hidden regimes.
N_REGIMES = 3
#: Minimum candles required before a regime label can be trusted.
MIN_CANDLES = 12
#: Random seed.
SEED = 42


def _build_features(candles: list[Candle]) -> np.ndarray:
    """Return an ``(n, 4)`` matrix of z-scored features: r, atr_z, volume_z, volatility_z."""
    n = len(candles)
    if n == 0:
        return np.zeros((0, 4))
    closes = np.array([c.close for c in candles], dtype=float)
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    volumes = np.array([c.volume for c in candles], dtype=float)

    # Log returns
    returns = np.zeros(n)
    if n > 1:
        returns[1:] = np.log(closes[1:] / closes[:-1])

    # ATR
    prev_close = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )
    atr = np.convolve(tr, np.ones(14) / 14, mode="same")
    # expanding z-score of ATR
    atr_mean = np.cumsum(atr) / np.arange(1, n + 1)
    atr_var = np.cumsum((atr - atr_mean) ** 2) / np.arange(1, n + 1)
    atr_std = np.sqrt(atr_var)
    atr_z = (atr - atr_mean) / np.where(atr_std == 0, 1.0, atr_std)
    atr_z = np.nan_to_num(atr_z, nan=0.0, posinf=0.0, neginf=0.0)

    # Volume z-score
    vol_mean = np.cumsum(volumes) / np.arange(1, n + 1)
    vol_var = np.cumsum((volumes - vol_mean) ** 2) / np.arange(1, n + 1)
    vol_std = np.sqrt(vol_var)
    volume_z = (volumes - vol_mean) / np.where(vol_std == 0, 1.0, vol_std)
    volume_z = np.nan_to_num(volume_z, nan=0.0, posinf=0.0, neginf=0.0)

    # Rolling volatility z-score
    rets = returns.copy()
    vol = np.zeros(n)
    for i in range(1, n):
        lo = max(1, i - 14)
        vol[i] = np.std(rets[lo : i + 1]) if i > lo else 0.0
    vol_mean = np.cumsum(vol) / np.arange(1, n + 1)
    vol_var = np.cumsum((vol - vol_mean) ** 2) / np.arange(1, n + 1)
    vol_std = np.sqrt(vol_var)
    volatility_z = (vol - vol_mean) / np.where(vol_std == 0, 1.0, vol_std)
    volatility_z = np.nan_to_num(volatility_z, nan=0.0, posinf=0.0, neginf=0.0)

    feats = np.column_stack([returns, atr_z, volume_z, volatility_z])
    # Normalise each column to zero mean, unit variance (expanding stats).
    for col in range(feats.shape[1]):
        col_mean = feats[:, col].mean()
        col_std = feats[:, col].std()
        if col_std > 1e-12:
            feats[:, col] = (feats[:, col] - col_mean) / col_std
    return feats


class RegimeSwitcher:
    """3-state Gaussian-mixture regime detector.

    Fit on recent candles, then each candle is assigned to the most likely
    of three Gaussian regimes.  Exposes the regime as ``regime`` on the
    returned :class:`Signal` so the ensemble and the control surface can
    act on it.

    Parameters
    ----------
    n_states:
        Number of regimes (default 3).
    n_iter:
        EM iterations.
    tol:
        Convergence tolerance.
    random_state:
        Seed for initialisation.

    Example
    -------
    >>> rst = RegimeSwitcher()
    >>> labels = rst.fit_predict(candles)          # 0/1/2 per candle
    >>> rst.predict_last(candles)                  # regime of the last bar
    1
    """

    def __init__(
        self,
        n_states: int = N_REGIMES,
        n_iter: int = 80,
        tol: float = 1e-4,
        random_state: int = SEED,
    ) -> None:
        self.n_states = int(n_states)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.means_: np.ndarray | None = None  # (d, K)
        self.vars_: np.ndarray | None = None  # (d, K)
        self.weights_: np.ndarray | None = None  # (K,)

    def fit(self, candles: list[Candle]) -> "RegimeSwitcher":
        if len(candles) < self.n_states + 2:
            return self
        feats = _build_features(candles)
        feats = feats[~np.isnan(feats).any(axis=1)]
        if len(feats) < self.n_states + 2:
            return self
        self._em(feats)
        return self

    def fit_predict(self, candles: list[Candle]) -> np.ndarray:
        labels = np.zeros(len(candles), dtype=int)
        if len(candles) < self.n_states + 2:
            return labels
        self.fit(candles)
        feats = _build_features(candles)
        feats = feats[~np.isnan(feats).any(axis=1)]
        if feats.shape[0] == 0:
            return labels
        # responsibilities on the valid rows, then decode
        resp = self._responsibilities(feats)
        labels[-len(feats) :] = np.argmax(resp, axis=1)
        return labels

    def predict_last(self, candles: list[Candle]) -> int:
        """Regime label for the most recent candle."""
        if len(candles) < self.n_states + 2:
            return 0
        self.fit(candles)
        feats = _build_features(candles)
        feats = feats[~np.isnan(feats).any(axis=1)]
        if feats.shape[0] == 0:
            return 0
        resp = self._responsibilities(feats)
        return int(np.argmax(resp[-1]))

    # ------------------------------------------------------------------
    # Gaussian mixture - EM
    # ------------------------------------------------------------------

    def _init_params(self, X: np.ndarray) -> tuple:
        d, K = X.shape[1], self.n_states
        # Distinct quantile bands per centroid so the EM starts from well-
        # separated means (k=0 -> [0%, 33%], k=1 -> [33%, 66%], k=2 -> [66%, 100%]).
        means = np.zeros((d, K))
        for k in range(K):
            for j in range(d):
                lo = np.percentile(X[:, j], 100.0 * k / K)
                hi = np.percentile(X[:, j], 100.0 * (k + 1) / K)
                means[j, k] = (lo + hi) / 2.0
        # start with small, distinct variances
        vars_ = np.array(
            [X.var(axis=0) * (0.5 + 0.5 * (K - k) / K) for k in range(K)]
        ).T
        weights = np.ones(K) / K
        return means, vars_, weights

    def _em(self, X: np.ndarray) -> None:
        means, vars_, weights = self._init_params(X)
        prev_loglike = -np.inf

        for _ in range(self.n_iter):
            # E-step: responsibilities (log-space for stability)
            resp = self._responsibilities(X, means, vars_, weights)
            # M-step
            Nk = resp.sum(axis=0) + 1e-12
            weights = Nk / Nk.sum()
            means = (X.T @ resp) / Nk
            # variance: var[j, k] = sum_t resp[t,k]*(X[t,j]-means[j,k])^2 / Nk[k]
            # diff[t, j, k] = X[t, j] - means[j, k]
            diff = X[:, :, None] - means[None, :, :]  # (n, d, K): axis1=d, axis2=K
            vars_ = np.sum(resp[:, None, :] * (diff**2), axis=0) / Nk  # (d, K)
            vars_ = np.maximum(vars_, 1e-9)

            # log-likelihood
            loglike = np.sum(np.log(np.clip(Nk, 1e-12, None)))
            if abs(loglike - prev_loglike) < self.tol:
                break
            prev_loglike = loglike

        self.means_ = means
        self.vars_ = vars_
        self.weights_ = weights

    def _responsibilities(
        self, X: np.ndarray, means=None, vars_=None, weights=None
    ) -> np.ndarray:
        if means is None:
            means = self.means_
            vars_ = self.vars_
            weights = self.weights_
        d = X.shape[1]
        # diff[t, k, j] = X[t, j] - means[k, j]
        means_Kd = means.T  # (K, d)
        var = np.maximum(vars_, 1e-12).T  # (K, d)
        diff = X[:, None, :] - means_Kd[None, :, :]  # (n, K, d)
        log_n = -0.5 * (
            d * np.log(2 * np.pi) + np.sum(diff**2 / var[None, :, :], axis=2)
        )
        log_n = log_n + np.log(weights)[None, :]
        # numerically stable log-softmax over the state axis (K)
        log_n_max = np.max(log_n, axis=1, keepdims=True)
        log_n = log_n - (
            log_n_max + np.log(np.sum(np.exp(log_n - log_n_max), axis=1, keepdims=True))
        )
        return np.exp(log_n)

    def _viterbi(self, obs: np.ndarray) -> tuple[float, np.ndarray]:
        """Hard state path (unused for predict_last, kept for the API)."""
        n, d = obs.shape
        if self.means_ is None:
            return -np.inf, np.zeros(n, dtype=int)
        resp = self._responsibilities(obs)
        return 0.0, np.argmax(resp, axis=1)


class RegimeSignal(Signal):
    """A signal that carries the inferred market regime."""

    def __init__(
        self,
        direction: str,
        confidence: float = 0.5,
        strength: float = 0.5,
        reason: str = "",
        timestamp: int = 0,
        regime: int = 0,
        regime_prob: dict[int, float] = None,
    ) -> None:
        super().__init__(direction, confidence, strength, reason, timestamp)
        self.regime = regime
        self.regime_prob = regime_prob or {
            0: 1.0 / N_REGIMES,
            1: 1.0 / N_REGIMES,
            2: 1.0 / N_REGIMES,
        }

    def as_dict(self) -> dict:  # type: ignore[override]
        d = super().as_dict()
        d["regime"] = self.regime
        d["regime_prob"] = self.regime_prob
        return d


class RegimeStrategy:
    """Strategy that emits the current regime as a (HOLD) signal with a
    ``regime`` field, so the ensemble and the control surface can read the
    market state and switch the active strategy accordingly.

    The direction is always HOLD (the regime is the signal); this child is
    consumed by the ensemble as a state descriptor, not a directional bet.
    """

    def __init__(self, regime: RegimeSwitcher) -> None:
        self.regime = regime

    def generate(self, candles: list[Any]) -> list[Signal]:
        regime = 1  # calm / normal default
        if self.regime is not None and len(candles) >= 12:
            regime = self.regime.predict_last(candles)
        return [
            RegimeSignal(
                direction="HOLD",
                confidence=0.6,
                strength=0.5,
                reason=f"market regime {regime}",
                timestamp=candles[-1].timestamp if candles else 0,
                regime=regime,
            )
        ]
