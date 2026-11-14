"""Freqtrade-style strategy hyperparameter optimizer with rolling walk-forward
splits.

Dependency-light: pure numpy/pandas plus a lightweight evaluator that mirrors
the existing :class:`backtest.runner.BacktestRunner` trade accounting.  No
Optuna / Ray Tune, no database — a plain grid search over a parameter dict.

Usage
-----
    from backtest.optimizer import optimize_parameters
    from my.strategies import TrendStrategy

    best_params, best_score, results = optimize_parameters(
        TrendStrategy, candles, n_splits=3, param_grid={
            "ema_period": [5, 10, 20, 50],
            "rsi_period": [14, 20],
            "lookback": [20, 50, 100],
        },
    )
    print(best_params, best_score)

Walk-forward scoring
--------------------
The sample is split into rolling train/test windows of ``train_ratio`` /
``step`` fractions.  Each parameter combination is evaluated on every test
window and the fold score is the average net PnL per trade in that window.
The combination with the highest mean fold score wins.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Local import (never a cycle: backtest/runner imports signals, never the
# reverse).  Only used to materialise a DataFrame argument.
from backtest.runner import BacktestRunner
from signals.strategy import Candle

#: Default parameter grid.  Keys must match the constructor parameters of
#: ``strategy_class`` (the optimizer passes them through as keyword args).
DEFAULT_GRID: Dict[str, Sequence[int]] = {
    "ema_period": [5, 10, 20, 50],
    "rsi_period": [14, 20],
    "lookback": [20, 50, 100],
}

#: Slippage / commission mirrored from the backtest runner (fraction -> %)
_BUY_SLIPPAGE_PT = BacktestRunner.SLIPPAGE_RATE * 100
_SELL_SLIPPAGE_PT = -BacktestRunner.SLIPPAGE_RATE * 100
_COMISSION_PT = BacktestRunner.COMMISSION_RATE * 2 * 100


# --------------------------------------------------------------------------- #
# Parameter grid
# --------------------------------------------------------------------------- #
class ParamGrid:
    """Cartesian product of parameter lists.

    Example::

        grid = ParamGrid({"ema_period": [5, 10, 20], "rsi_period": [14, 20]})
        for params in grid:
            print(params)   # {'ema_period': 5, 'rsi_period': 14}, ...
    """

    def __init__(self, params: Dict[str, Iterable[Any]]):
        self.params = dict(params)
        self.keys: List[str] = list(self.params)
        self._values: List[List[Any]] = [list(v) for v in self.params.values()]

    def combinations(self) -> Iterable[Dict[str, Any]]:
        """Yield every parameter combination as a dict."""
        for values in itertools.product(*self._values):
            yield dict(zip(self.keys, values, strict=True))

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return self.combinations()

    def __len__(self) -> int:
        n = 1
        for v in self._values:
            n *= len(v)
        return n


# --------------------------------------------------------------------------- #
# Walk-forward splits
# --------------------------------------------------------------------------- #
def walk_forward_splits(
    n: int,
    n_splits: int = 3,
    train_ratio: float = 0.7,
    step: float = 0.2,
    min_train: int = 100,
    min_test: int = 50,
) -> List[Tuple[slice, slice]]:
    """Return rolling (train, test) slices of a sample of length ``n``.

    The first fold anchors the train window at the start of the sample and
    the test window immediately after it; later folds slide that pair
    forward by ``step``.  Returns ``[]`` when the sample is too small.
    """
    train_size = int(np.ceil(train_ratio * n))
    test_size = int(np.ceil((1.0 - train_ratio) * n))
    step_size = max(1, int(np.ceil(step * n)))

    train_size = max(1, min(train_size, n))
    test_size = max(1, min(test_size, n))
    step_size = max(1, min(step_size, n))

    if train_size + test_size > n:
        test_size = n - train_size
    if train_size < min_train or test_size < min_test:
        return []

    splits: List[Tuple[slice, slice]] = []
    start = 0
    while start + train_size <= n:
        test_start = start + train_size
        test_end = min(test_start + test_size, n)
        test_rows = test_end - test_start
        if test_rows < min_test:
            break
        splits.append((slice(start, start + train_size), slice(test_start, test_end)))
        start += step_size
    return splits


# --------------------------------------------------------------------------- #
# Lightweight backtest evaluator (mirrors BacktestRunner trade accounting)
# --------------------------------------------------------------------------- #
def _evaluate_strategy(strategy, candles: Sequence[Candle]) -> Dict[str, float]:
    """Fast deterministic backtest over ``candles``.

    Mirrors :class:`backtest.runner.BacktestRunner` semantics (trailing
    49-candle window, 0.02% slippage per side, 0.04% commission per trade):
    BUY opens, SELL closes.  Returns per-trade statistics.
    """
    position = 0
    entry_price = 0.0
    take_profit = 0.0
    stop_loss = 0.0

    total_pnl = 0.0
    trade_count = 0
    wins = 0

    for i, candle in enumerate(candles):
        lookback = min(i, 49)
        window = candles[max(0, i - lookback) : i + 1]
        signals = strategy.generate(window)
        signal = signals[0] if signals else None

        if signal is not None and signal.direction == "BUY":
            position = 1
            entry_price = candle.close
            stop_loss = entry_price * 0.9975
            take_profit = entry_price * 1.0025

        elif signal is not None and signal.direction == "SELL":
            if position > 0:
                exit_price = max(min(candle.close, take_profit), stop_loss)
                gross_pnl = (exit_price - entry_price) / entry_price * 100.0
                slippage = (
                    _BUY_SLIPPAGE_PT if signal.direction == "BUY" else _SELL_SLIPPAGE_PT
                )
                commission = _COMISSION_PT
                net_pnl = gross_pnl + slippage - commission
                total_pnl += net_pnl
                trade_count += 1
                if net_pnl > 0:
                    wins += 1
                position = 0
                entry_price = 0.0
                stop_loss = 0.0
                take_profit = 0.0

    return {
        "total_pnl": total_pnl,
        "total_trades": trade_count,
        "wins": wins,
        "win_rate": wins / trade_count if trade_count > 0 else 0.0,
        "avg_pnl": total_pnl / trade_count if trade_count > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
class ParameterOptimizer:
    """Grid search over strategy parameters with rolling walk-forward splits.

    Parameters
    ----------
    strategy_class:
        Strategy class whose ``__init__`` accepts the grid parameter names
        as keyword arguments and whose ``generate(candles)`` returns
        ``Signal`` objects (``BUY`` opens, ``SELL`` closes).
    candles:
        List of ``Candle`` records (or a DataFrame with the standard OHLCV
        columns, which is converted automatically).
    param_grid:
        Parameter grid; defaults to :data:`DEFAULT_GRID` when ``None``.
    n_splits:
        Number of rolling walk-forward folds to use (``min`` is enforced:
        folds are generated on demand and the search keeps as many as the
        sample allows).
    """

    def __init__(
        self,
        strategy_class,
        candles: Sequence[Any],
        param_grid: Optional[Dict[str, Sequence[Any]]] = None,
        n_splits: int = 3,
        train_ratio: float = 0.7,
        step: float = 0.2,
        min_train: int = 100,
        min_test: int = 50,
    ):
        self.strategy_class = strategy_class
        self.candles = self._materialise(candles)
        self.param_grid = ParamGrid(
            param_grid if param_grid is not None else DEFAULT_GRID
        )
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.step = step
        self.min_train = min_train
        self.min_test = min_test

    # -- internals ---------------------------------------------------------- #
    @staticmethod
    def _materialise(candles: Sequence[Any]) -> List[Candle]:
        """Convert a DataFrame (or anything else) into a Candle list."""
        if isinstance(candles, pd.DataFrame):
            return BacktestRunner().to_candles(candles)
        return list(candles)

    def walk_forward_splits(self) -> List[Tuple[slice, slice]]:
        return walk_forward_splits(
            len(self.candles),
            n_splits=self.n_splits,
            train_ratio=self.train_ratio,
            step=self.step,
            min_train=self.min_train,
            min_test=self.min_test,
        )

    def evaluate_fold(
        self, params: Dict[str, Any], train_slice: slice, test_slice: slice
    ) -> Dict[str, Any]:
        """Run the strategy on one walk-forward fold and score it."""
        strategy = self.strategy_class(**params)
        test_candles = self.candles[test_slice]
        metrics = _evaluate_strategy(strategy, test_candles)
        return {
            **params,
            "train_rows": train_slice.stop - train_slice.start,
            "test_rows": test_slice.stop - test_slice.start,
            **metrics,
        }

    def optimize(
        self, verbose: bool = True
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Search the parameter grid with walk-forward validation.

        Returns ``(best_params, best_score, results)`` where
        ``best_score`` is the mean fold score (average net PnL per trade)
        of the winning combination, and ``results`` is a sorted list of
        ``{params, score, folds, ...}` dicts.
        """
        splits = self.walk_forward_splits()
        if not splits:
            msg = (
                f"sample too small for walk-forward validation "
                f"(need >= {self.min_train + self.min_test} candles, got {len(self.candles)})"
            )
            if verbose:
                print(f"WARN: {msg}")
            return {}, 0.0, []

        all_results: List[Dict[str, Any]] = []
        n_combos = len(self.param_grid)

        if verbose:
            print(
                f"Optimizing {n_combos} parameter combinations x "
                f"{len(splits)} walk-forward splits (train {self.train_ratio:.0%}, "
                f"step {self.step:.0%}) ..."
            )

        for combo_idx, params in enumerate(self.param_grid, 1):
            fold_scores: List[float] = []
            fold_metrics: List[Dict[str, Any]] = []

            for fold_idx, (train_slice, test_slice) in enumerate(splits, 1):
                metrics = self.evaluate_fold(params, train_slice, test_slice)
                fold_metrics.append(metrics)
                if metrics["total_trades"] > 0:
                    fold_scores.append(metrics["avg_pnl"])

            score = float(np.mean(fold_scores)) if fold_scores else 0.0
            all_results.append(
                {
                    "params": params,
                    "score": score,
                    "folds": fold_metrics,
                    "n_folds": len(splits),
                }
            )

            if verbose:
                print(
                    f"  [{combo_idx}/{n_combos}] {params} -> "
                    f"avg net pnl {score:+.4f}"
                )

        # Stable sort: highest walk-forward score first.
        all_results.sort(key=lambda r: r["score"], reverse=True)
        best = all_results[0]

        if verbose:
            self._print_summary(best, len(splits))

        return best["params"], best["score"], all_results

    @staticmethod
    def _print_summary(best: Dict[str, Any], n_splits: int) -> None:
        param_str = ",".join(f"{k}={v}" for k, v in best["params"].items())
        print(
            f"BEST_PARAMS {param_str} | "
            f"walk-forward avg net_pnl = {best['score']:+.4f} "
            f"({n_splits} folds, "
            f"{best['folds'][0]['test_rows'] if best['folds'] else 0} test rows/fold)"
        )


# --------------------------------------------------------------------------- #
# Module-level entry point
# --------------------------------------------------------------------------- #
def optimize_parameters(
    strategy_class,
    candles: Sequence[Any],
    n_splits: int = 3,
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    """Freqtrade-style grid search with rolling walk-forward splits.

    Args:
        strategy_class: Strategy class accepting ``**params`` and exposing
            ``generate(candles)``.
        candles: Candle list (or OHLCV DataFrame).
        n_splits: Rolling walk-forward split count.
        param_grid: Parameter grid; defaults over EMA/RSI/lookback periods.
        verbose: Print a per-combination progress line plus the summary.

    Returns:
        ``(best_params, best_score, results)``.  ``best_score`` is the
        walk-forward average net PnL per trade of the winning combination.
    """
    optimizer = ParameterOptimizer(
        strategy_class=strategy_class,
        candles=candles,
        param_grid=param_grid,
        n_splits=n_splits,
    )
    return optimizer.optimize(verbose=verbose)
