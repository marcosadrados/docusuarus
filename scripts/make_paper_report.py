"""Tiny-capital paper report generator.

Runs the market trend strategy on the same deterministic sample at a tiny
account balance (default $10K) and writes a publishable per-trade audit ledger
plus a paper-performance report.  This is the "no real-money proof exists yet"
figure every vendor is transparent about, and the same shape competitors use
for their paper-trading claims (Galileo's $3,200, Alpha Arena's $10,000).

Usage:
    python scripts/make_paper_report.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.ledger import summarize, write_ledger  # noqa: E402
from backtest.runner import BacktestRunner  # noqa: E402
from signals.strategy.strategy import MarketTrendStrategy  # noqa: E402

# Reuse the same deterministic sample as the walk-forward report, so both
# figures can be compared on identical data.
from scripts.make_walkforward_report import make_data  # noqa: E402

OUT = ROOT / "docs" / "paper_report.md"
N_ROWS = 5000
SEED = 42
BALANCE = 10000.0  # tiny paper account


def build_paper_report(rows: pd.DataFrame, out_path: Path) -> None:
    strategy = MarketTrendStrategy()
    runner = BacktestRunner(account_balance=BALANCE)
    runner.set_strategy(strategy)

    import tempfile

    tmp = rows.to_csv(index=False)
    fd, tmpname = tempfile.mkstemp(suffix=".csv", dir=ROOT)
    try:
        with os.fdopen(fd, "w") as _fh:
            _fh.write(tmp)
        results = runner.run(tmpname)
    finally:
        os.unlink(tmpname)

    trades = results.get("trades", []) or []
    ledger_path = ROOT / "logs" / "paper_trades.jsonl"
    write_ledger(
        trades, str(ledger_path), symbol="CL=F",
        timestamps=[int(t["timestamp"]) for t in trades],
        balance=BALANCE,
    )
    metrics = summarize(str(ledger_path))

    net = metrics["net_pnl"]
    lines = []
    lines.append("=" * 66)
    lines.append("PAPER-TRADING PERFORMANCE REPORT (TINY CAPITAL)")
    lines.append("Scalping Trading Bot  |  Oil, Gas & Metals futures")
    lines.append("=" * 66)
    lines.append("")
    lines.append("## 1. Setup")
    lines.append("")
    lines.append("- sample        : deterministic synthetic OHLCV (seed=42, 5,000 rows)")
    lines.append(f"- account balance : ${BALANCE:,.0f}  (paper only, no real money)")
    lines.append("- strategy      : MarketTrendStrategy (EMA 5/20 + volume)")
    lines.append("- runner        : BacktestRunner (0.02% slippage/side, 0.04% commission/trade)")
    lines.append("- period        : ~3.5 days of 1-min bars")
    lines.append("")
    lines.append("## 2. Results (tiny paper account)")
    lines.append("")
    lines.append("| metric | result |")
    lines.append("|---|---|")
    lines.append(f"| total trades | {metrics['total_trades']} |")
    lines.append(f"| win rate | {metrics['win_rate']*100:.1f}% |")
    lines.append(f"| net PnL | {net:+.2f}% |")
    lines.append(f"| mean trade | {metrics['avg_trade']:+.3f}% |")
    lines.append(f"| Sharpe (per-day) | {metrics['sharpe']:+.2f} |")
    lines.append(f"| Sortino | {metrics['sortino']:+.2f} |")
    lines.append(f"| max drawdown | {metrics['max_drawdown']*100:.2f}% |")
    lines.append(f"| SQN (Kaiser) | {metrics['sqn']:+.2f} |")
    lines.append(f"| expectancy | {metrics['expectancy']:+.3f}%/trade |")
    lines.append(f"| profit factor | {metrics['profit_factor']:.2f} |")
    lines.append(f"| avg win / loss | {metrics['avg_win']:+.3f} / {metrics['avg_loss']:+.3f} |")
    lines.append(f"| ledger | {ledger_path} |")
    lines.append("")
    lines.append("## 3. Transparency note")
    lines.append("")
    lines.append(
        "This is paper-only at a tiny account. Paper results are not a substitute for")
    lines.append(
        "live, funded trading, and sample-specific numbers move with market regime.")
    lines.append("Re-run `make_paper_report.py` to reproduce this ledger and these stats.")
    lines.append("")
    lines.append("=" * 66)
    lines.append("End of paper report")
    lines.append("=" * 66)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    data = make_data(N_ROWS)
    build_paper_report(data, OUT)
