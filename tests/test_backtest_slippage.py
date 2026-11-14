"""Smoke tests for the backtest slippage + commission model.

Verifies that backtest/runner.py returns the pre-existing keys plus the new
gross_pnl / slippage_pct / commission_pct keys, and that net pnl reconciles
to gross plus slippage minus commission.
"""

import csv
import math
import os

import pytest

from backtest.runner import BacktestRunner

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURE_DIR, "sample_candles.csv")


def _ensure_sample_csv(path=SAMPLE_CSV):
    """Write a deterministic ~30-row candle csv (price trending ~80 with noise)."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = 80.0
    rows = []
    for i in range(30):
        close = base + 0.25 * i + 0.6 * math.sin(i * 1.7) + 0.12 * math.sin(i * 5.3)
        open_p = close + 0.08 * (1 if i % 2 == 0 else -1)
        rows.append(
            {
                "timestamp": 1700000000 + i * 60,
                "open": round(open_p, 5),
                "high": round(max(open_p, close) + 0.08, 5),
                "low": round(min(open_p, close) - 0.08, 5),
                "close": round(close, 5),
                "volume": round(100.0 + i, 2),
            }
        )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(rows)


class _Signal:
    """Minimal signal object exposing .direction, matching the runner's access
    pattern (``signal.direction``)."""

    def __init__(self, direction):
        self.direction = direction


class _AlternatingStrategy:
    """Alternating BUY/SELL signals so each BUY is closed by a SELL -> 15 trades.

    The runner passes the strategy a trailing window of Candle dataclasses
    (lookback 0..49); the last candle in the window is the current one.
    """

    def generate(self, candles):
        # The runner passes a trailing window of Candle dataclasses; the last
        # candle in that window is the current tick.
        ts = candles[-1].timestamp
        # Timestamps are 1700000000 + i*60, so (ts - base) // 60 is the tick i.
        i = (ts - 1700000000) // 60
        if i % 2 == 0:
            return [_Signal("BUY")]
        return [_Signal("SELL")]


def test_backtest_runner_runs_with_slippage_and_commission():
    _ensure_sample_csv()
    runner = BacktestRunner(account_balance=10000.0)
    runner.set_strategy(_AlternatingStrategy())
    results = runner.run(SAMPLE_CSV)

    # Pre-existing keys still present.
    for key in [
        "total_trades",
        "wins",
        "win_rate",
        "total_pnl",
        "max_drawdown",
        "sharpe",
        "avg_trade",
    ]:
        assert key in results, f"missing pre-existing key: {key}"

    # New keys added.
    for key in ["gross_pnl", "slippage_pct", "commission_pct"]:
        assert key in results, f"missing new key: {key}"

    # Net pnl = gross + slippage - commission (slippage is signed: negative for
    # long exits, positive for short exits).
    assert results["total_pnl"] == pytest.approx(
        results["gross_pnl"] + results["slippage_pct"] - results["commission_pct"],
        abs=1e-9,
    ), "total_pnl does not reconcile to gross + slippage - commission"

    # Every closed trade records per-trade slippage/commission.
    assert len(runner.trades) == results["total_trades"]
    for trade in runner.trades:
        assert "slippage" in trade
        assert "commission" in trade
        # Slippage on exit fill: long (SELL close) reduced (-0.02%), short (BUY
        # close) increased (+0.02%).
        assert trade["slippage"] == pytest.approx(
            -0.02 if trade["signal"] == "SELL" else 0.02, abs=1e-12
        )
        assert trade["commission"] == pytest.approx(0.04, abs=1e-12)

    # Sanity: the alternating strategy closes exactly 15 trades.
    assert results["total_trades"] == 15
    assert results["win_rate"] == pytest.approx(
        results["wins"] / results["total_trades"], abs=1e-12
    )
    assert results["gross_pnl"] > 0  # sample trends upward overall
