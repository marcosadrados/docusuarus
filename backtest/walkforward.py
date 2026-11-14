"""Walk-forward / out-of-sample validation for the scalping bot.

Rolling the evaluation window forward produces per-window out-of-sample metrics
that a single-pass backtest cannot reveal.  That is precisely the gap the
big-win vendors have to defend against: a strategy can look excellent on one
in-sample pass and collapse on the next, so an out-of-sample profile is the
credibility anchor.

Design
------
Only the existing bot classes (:class:`backtest.runner.BacktestRunner`,
:class:`signals.strategy.MarketTrendStrategy`) and numpy/pandas are used.

``walkforward_report(strategy, candles, ...)`` is the primary entry point:
operates on Candle records (or an OHLCV DataFrame) and returns per-fold OOS
metrics plus aggregate stats.  ``evaluate_walkforward`` is retained as a thin
CSV driver for backwards compatibility.

The strategy is deterministic in this codebase (no fitted coefficients), so
"train" only selects the context slice; the strategy is re-scored on every
contiguous "test" window, each time extended slightly further out of sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.runner import BacktestRunner
from signals.strategy import Candle, MarketTrendStrategy

# Slippage / commission in percentage points, mirrored from the runner.
_BUY_SLIPPAGE_PT = BacktestRunner.SLIPPAGE_RATE * 100
_SELL_SLIPPAGE_PT = -BacktestRunner.SLIPPAGE_RATE * 100
_COMISSION_PT = BacktestRunner.COMMISSION_RATE * 2 * 100


def _to_candles(candles) -> list[Candle]:
    """Convert a DataFrame (or Candle list) into a Candle list.

    ``load_data`` (above) stores timestamps as pandas datetime64, and
    ``to_candles`` reads ``row["timestamp"].timestamp()`` on each row, so the
    timestamp column must be datetime-aware before that conversion.
    """
    if isinstance(candles, pd.DataFrame):
        df = candles.copy()
        if pd.api.types.is_integer_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return BacktestRunner().to_candles(df)
    return list(candles)


def _evaluate_window(strategy, candles: list[Candle], account_balance: float = 10000.0) -> dict:
    """Backtest a window of candles through the strategy over a short horizon.

    Returns per-trade and risk metrics the same way ``BacktestRunner.run`` does:
    win rate, total pnl, max drawdown, sharpe, avg trade, trade count.
    """
    position = 0
    entry_price = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    total_pnl = 0.0
    trade_count = 0
    wins = 0
    trade_pnls: list[float] = []

    # Equity bookkeeping, mirrored from BacktestRunner.run.
    trade_pnl = 0.0
    portfolio_value = account_balance
    peak_value = portfolio_value
    max_drawdown = 0.0

    for i, candle in enumerate(candles):
        lookback = min(i, 49)
        window = candles[max(0, i - lookback):i + 1]
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
                slippage = _BUY_SLIPPAGE_PT if signal.direction == "BUY" else _SELL_SLIPPAGE_PT
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
                trade_pnl = net_pnl
                trade_pnls.append(net_pnl)

        # Equity steps on every closed trade (plus the initial state).
        current_value = account_balance * (1 + trade_pnl / 100.0)
        portfolio_value = current_value
        if portfolio_value > peak_value:
            peak_value = portfolio_value
        drawdown = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0.0
        max_drawdown = min(max_drawdown, drawdown)

    return {
        "total_pnl": total_pnl,
        "win_rate": (wins / trade_count) if trade_count else 0.0,
        "total_trades": trade_count,
        "wins": wins,
        "max_drawdown": max_drawdown,
        "sharpe": _sharpe(trade_pnls),
        "avg_trade": (total_pnl / trade_count) if trade_count else 0.0,
    }


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns))
    return float(mean / std) if std > 0 else 0.0


def _rolling_splits(n: int, train_ratio: float = 0.7, step: float = 0.2,
                    min_train: int = 100, min_test: int = 50):
    """Yield (train_start, train_end, test_start, test_end) index tuples."""
    train_size = int(np.ceil(train_ratio * n))
    test_size = int(np.ceil((1.0 - train_ratio) * n))
    step_size = max(1, int(np.ceil(step * n)))

    train_size = max(1, min(train_size, n))
    test_size = max(1, min(test_size, n))
    step_size = max(1, min(step_size, n))

    if train_size + test_size > n:
        test_size = n - train_size
    if train_size < min_train or test_size < min_test:
        return

    start = 0
    while start + train_size <= n:
        test_start = start + train_size
        test_end = min(test_start + test_size, n)
        test_rows = test_end - test_start
        if test_rows < min_test:
            break
        yield (start, start + train_size, test_start, test_end)
        start += step_size


def walkforward_report(
    strategy,
    candles,
    train_ratio: float = 0.7,
    step: float = 0.2,
    min_train: int = 100,
    min_test: int = 50,
    account_balance: float = 10000.0,
) -> dict:
    """Run a rolling train/test walk-forward over a strategy + data.

    ``candles`` may be a list of :class:`signals.strategy.Candle` or an OHLCV
    DataFrame.  ``train`` only selects the context slice (the strategy has no
    fitted parameters); each contiguous test window is scored independently.

    Returns ``'folds'`` (per-fold OOS metrics) plus aggregate stats:
    ``'mean_sharpe'``, ``'std_sharpe'``, ``'mean_win_rate'``, ``'std_win_rate'``,
    ``'mean_max_drawdown'``, ``'mean_pnl'``, ``'train_ratio'``, ``'step'``,
    ``'n_folds'``.
    """
    strategy = strategy or MarketTrendStrategy()
    data = _to_candles(candles)
    n = len(data)

    splits = list(_rolling_splits(n, train_ratio, step, min_train, min_test))
    if not splits:
        return {
            "folds": [],
            "mean_sharpe": 0.0, "std_sharpe": 0.0,
            "mean_win_rate": 0.0, "std_win_rate": 0.0,
            "mean_max_drawdown": 0.0, "mean_pnl": 0.0,
            "train_ratio": train_ratio, "step": step, "n_folds": 0,
        }

    folds = []
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
        test_candles = data[test_start:test_end]
        metrics = _evaluate_window(strategy, test_candles, account_balance)
        folds.append(
            {
                "fold": fold_idx + 1,
                "train_rows": train_end - train_start,
                "test_start": test_start,
                "test_end": test_end,
                "test_rows": test_end - test_start,
                **metrics,
            }
        )

    sharpe = np.array([f["sharpe"] for f in folds], dtype=float)
    win_rate = np.array([f["win_rate"] for f in folds], dtype=float)
    max_drawdown = np.array([f["max_drawdown"] for f in folds], dtype=float)
    pnl = np.array([f["total_pnl"] for f in folds], dtype=float)

    return {
        "folds": folds,
        "mean_sharpe": float(np.mean(sharpe)),
        "std_sharpe": float(np.std(sharpe)),
        "mean_win_rate": float(np.mean(win_rate)),
        "std_win_rate": float(np.std(win_rate)),
        "mean_max_drawdown": float(np.mean(max_drawdown)),
        "mean_pnl": float(np.mean(pnl)),
        "train_ratio": train_ratio,
        "step": step,
        "n_folds": len(folds),
    }


# --------------------------------------------------------------------------- #
# Backward-compatible CSV driver (kept for external callers).
# --------------------------------------------------------------------------- #
def evaluate_walkforward(
    csv_path: str,
    strategy=None,
    train_ratio: float = 0.7,
    step: float = 0.2,
    min_train: int = 100,
    min_test: int = 50,
) -> dict:
    """Run a rolling train/test walk-forward over a raw backtest CSV.

    Retained for compatibility; prefers :func:`walkforward_report` on Candle
    data.
    """
    strategy = strategy or MarketTrendStrategy()
    df = BacktestRunner().load_data(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return walkforward_report(
        strategy, df,
        train_ratio=train_ratio, step=step,
        min_train=min_train, min_test=min_test,
    )
