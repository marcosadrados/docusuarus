"""Smoke-runner driver: exercise the equity strategy on synthetic OHLCV data.

Usage
-----
    python backtest/equity_backtest.py
    # or, from the repository root:
    python -m backtest.equity_backtest

This loads the synthetic equity sample (tests/fixtures/sample_equity_candles.csv),
converts it to ``Candle`` records with :meth:`backtest.runner.BacktestRunner.to_candles`,
runs it through :class:`backtest.runner.BacktestRunner` with :class:`signals.strategy.equity.EquityStrategy`,
and prints the raw results dict plus the audit log written by the runner.

Designed to exit 0 and print a ``total_pnl`` that is positive (and a trade count
greater than zero), proving the equity path is wired end-to-end through the
existing runner without touching the futures data pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow `python backtest/equity_backtest.py` from any cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.runner import BacktestRunner  # noqa: E402
from signals.strategy import Candle  # noqa: E402
from signals.strategy.equity import EquityStrategy  # noqa: E402

SAMPLE_CSV = ROOT / "tests" / "fixtures" / "sample_equity_candles.csv"


def load_candles(path: Path) -> list[Candle]:
    """Parse the sample CSV into ``Candle`` records (same contract as the runner)."""
    df = pd.read_csv(path)
    return [
        Candle(
            timestamp=int(row["timestamp"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for _, row in df.iterrows()
    ]


def run_equity_smoke(
    csv_path: Path,
    lookback: int = 20,
    pullback: float = 0.003,
    volume_multiple: float = 1.5,
    account_balance: float = 10000.0,
) -> dict:
    """Run the equity strategy over the sample CSV through the existing runner."""
    strategy = EquityStrategy(
        lookback=lookback,
        pullback=pullback,
        volume_multiple=volume_multiple,
        symbol="AAPL",
    )
    runner = BacktestRunner(account_balance=account_balance)
    runner.set_strategy(strategy)
    return runner.run(str(csv_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Equity backtest smoke run")
    parser.add_argument(
        "--csv", type=Path, default=SAMPLE_CSV, help="synthetic equity sample CSV"
    )
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--pullback", type=float, default=0.003)
    parser.add_argument("--volume-multiple", type=float, default=1.5)
    parser.add_argument("--account-balance", type=float, default=10000.0)
    args = parser.parse_args(argv)

    results = run_equity_smoke(
        args.csv,
        lookback=args.lookback,
        pullback=args.pullback,
        volume_multiple=args.volume_multiple,
        account_balance=args.account_balance,
    )

    print(json.dumps(results, indent=2))
    print("\n--- raw stdout summary ---")
    print(f"sample rows : {len(load_candles(args.csv))}")
    print(f"equity trades: {results['total_trades']}")
    print(f"wins         : {results['wins']}")
    print(f"win rate     : {results['win_rate']:.3f}")
    print(f"total pnl    : {results['total_pnl']:.4f}%")
    print(f"max drawdown : {results['max_drawdown']:.4f}")

    # The equity path is wired through the existing runner only if it produced
    # at least one trade and a positive total P&L (or at least one trade).
    assert results["total_trades"] > 0, "no trades were executed"
    assert results["total_pnl"] > 0.0, "total_pnl is not positive"
    return results


if __name__ == "__main__":
    main()
