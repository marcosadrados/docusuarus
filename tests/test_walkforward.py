"""Tests for the walk-forward / out-of-sample validation module."""

from pathlib import Path

import pandas as pd
import types

from backtest.runner import BacktestRunner
from backtest.walkforward import evaluate_walkforward, walkforward_report
from signals.strategy import Candle, MarketTrendStrategy

BASE = Path(__file__).resolve().parent
SAMPLE_CSV = BASE / "fixtures" / "sample_candles.csv"


class _Alternating:
    """Deterministic strategy: BUY on even ticks, SELL on odd ticks."""

    def generate(self, window):
        idx = len(window) - 1
        signal = types.SimpleNamespace(direction="BUY" if idx % 2 == 0 else "SELL")
        return [signal]


def _candles_from_csv(path: Path) -> list[Candle]:
    df = BacktestRunner().load_data(path)
    return BacktestRunner().to_candles(df)


def test_walkforward_report_structural():
    rows = [
        Candle(timestamp=t * 60 + 1700000000, open=80.0, high=80.1, low=79.9,
               close=80.0, volume=100.0) for t in range(40)
    ]
    report = walkforward_report(_Alternating(), rows, train_ratio=0.5, step=0.3,
                                min_train=10, min_test=5)
    assert report["n_folds"] >= 1
    assert report["train_ratio"] == 0.5
    assert report["step"] == 0.3
    for f in report["folds"]:
        assert f["fold"] >= 1
        assert f["test_start"] >= 0
        assert f["test_start"] < f["test_end"]
        assert f["test_rows"] == f["test_end"] - f["test_start"]
        assert 0.0 <= f["win_rate"] <= 1.0
        assert isinstance(f["sharpe"], float)
        assert isinstance(f["max_drawdown"], float)


def test_walkforward_uses_markettrend_strategy():
    report = walkforward_report(MarketTrendStrategy(), _candles_from_csv(SAMPLE_CSV))
    assert report["n_folds"] >= 0
    assert isinstance(report["mean_sharpe"], float)
    assert isinstance(report["mean_pnl"], float)
    assert len(report["folds"]) == report["n_folds"]


def test_walkforward_tiny_sample_returns_empty():
    rows = [Candle(timestamp=t, open=80.0, high=80.1, low=79.9, close=80.0, volume=100.0)
            for t in range(10)]
    report = walkforward_report(_Alternating(), rows, min_train=20, min_test=10)
    assert report["n_folds"] == 0
    assert report["folds"] == []


def test_evaluate_walkforward_csv_backcompat():
    report = evaluate_walkforward(str(SAMPLE_CSV), min_train=5, min_test=3)
    assert isinstance(report["n_folds"], int)
    assert report["n_folds"] >= 0
