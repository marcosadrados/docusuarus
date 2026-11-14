"""End-to-end demo: run the backtest on the bundled sample candles, write an
auditable per-trade ledger, and print the comparative metrics the biggest-win
bots publish (SQN, expectancy, profit factor, monthly breakdown).

Usage:
    python3 scripts/demo_ledger.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.analyzer import Analyzer  # noqa: E402
from backtest.ledger import summarize, write_ledger  # noqa: E402
from backtest.runner import BacktestRunner  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
CANDLES = BASE / "tests" / "fixtures" / "sample_candles.csv"
LEDGER = BASE / "logs" / "demo_trades.jsonl"


def _reset_ledger(path: Path) -> None:
    """Start the demo from a clean ledger so re-runs are reproducible.

    Note: production backtests keep this file APPEND-ONLY (see :func:`write_ledger`
    and :class:`backtest.ledger`). This clean step applies only to the
    demonstration script, not to the ledger itself.
    """
    if path.exists():
        path.unlink()


class _Signal:
    def __init__(self, direction):
        self.direction = direction


class _AlternatingStrategy:
    """Deterministic pseudo-scalper: BUY on even ticks, SELL on odd ticks."""

    def generate(self, window):
        idx = len(window) - 1
        return [_Signal("BUY" if idx % 2 == 0 else "SELL")]


def main() -> int:
    runner = BacktestRunner(account_balance=10000.0)
    runner.set_strategy(_AlternatingStrategy())
    results = runner.run(str(CANDLES))

    _reset_ledger(LEDGER)

    trades = results.get("trades", []) or []
    timestamps = [int(t["timestamp"]) for t in trades]
    write_ledger(trades, str(LEDGER), symbol="CL=F", timestamps=timestamps)

    print("Backtest results (runner.return):")
    for k in ("total_trades", "wins", "win_rate", "total_pnl",
              "max_drawdown", "sharpe", "avg_trade"):
        print("  %-14s: %s" % (k, results.get(k)))

    print("\nAnalyzer comparative metrics:")
    print("-" * 52)
    print(Analyzer.analyze_comparative(results))

    print("\nAuditable ledger written to:")
    print("  %s" % LEDGER)
    print("\nLedger summary (comparative metrics):")
    print("-" * 52)
    print(summarize(str(LEDGER)))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
