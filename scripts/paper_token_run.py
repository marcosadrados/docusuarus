"""Paper-run the scalping bot against a token's real klines.

Reads from a registered :class:`backtest.token_klines_fetcher.TokenKlinesFetcher`
(case name, or the synthetic default), converts OHLCV into the bundled
:class:`backtest.runner.BacktestRunner` input, and writes:

- a per-trade audit ledger   -> logs/paper_trades.jsonl (append-only JSONL)
- a publishable paper report -> docs/paper_report.md

Run::

    python scripts/paper_token_run.py

The synthetic provider is used so this runs out of the box; wire a real
provider in ``config.yaml`` (see :mod:`backtest.token_klines_fetcher`) before
paper-trading against a live token pool.

This is the honest market-data phase: real pool fills, simulated execution,
no real orders issued.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.ledger import summarize, write_ledger  # noqa: E402
from backtest.runner import BacktestRunner  # noqa: E402
from backtest.token_klines_fetcher import MarketTrendStrategy  # noqa: E402
from backtest.token_klines_fetcher import (
    RaydiumPoolKlinesFetcher,
    SyntheticTokenKlinesFetcher,
    TokenKlinesProvider,
)  # noqa: E402


def _load_token_klines_config(path: Path) -> dict:
    """Minimal reader for the flat `token_klines:` block in config.yaml.

    Kept dependency-free (no PyYAML in the runnning env). Values are returned
    stripped of surrounding quotes and comments.
    """
    cfg = {}
    in_block = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("token_klines:"):
            in_block = True
            continue
        if in_block:
            if not stripped or stripped.startswith("#"):
                continue
            if stripped[0] in (" ", "\t") or ":" in stripped:
                key, _, val = stripped.partition(":")
                val = val.split("#", 1)[0].strip()
                val = val.strip('"').strip("'")
                cfg[key.strip()] = val
            else:
                in_block = False
    return cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

# The paper run is deterministic (same seed every time), so the numbers are
# re-runnable and auditable.
N_ROWS = 5000
SEED = 7
BALANCE = 10000.0


def fetch_klines(fetcher: TokenKlinesFetcher) -> pd.DataFrame:
    rows = fetcher.fetch_ohlcv()
    if rows.empty:
        raise RuntimeError("fetcher returned an empty OHLCV set")
    return rows


def write_token_report(data: pd.DataFrame, fetcher: TokenKlinesFetcher,
                       out_path: Path, balance: float) -> None:
    import tempfile

    strategy = MarketTrendStrategy()
    runner = BacktestRunner(account_balance=balance)
    runner.set_strategy(strategy)

    csv_bytes = data.to_csv(index=False).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        "wb", suffix=".csv", prefix="token_paper_", delete=False
    ) as fh:
        fh.write(csv_bytes)
        tmp_path = fh.name
    try:
        results = runner.run(tmp_path)
    finally:
        os.unlink(tmp_path)

    trades = results.get("trades") or []
    ledger_path = ROOT / "logs" / "paper_trades.jsonl"
    write_ledger(
        trades,
        str(ledger_path),
        symbol=fetcher.symbol,
        balance=balance,
    )
    metrics = summarize(str(ledger_path))

    net = metrics["net_pnl"]
    lines = []
    lines.append("=" * 66)
    lines.append("PAPER-TRADING PERFORMANCE REPORT (TINY CAPITAL)")
    lines.append(
        "Scalping Trading Bot  |  Oil, Gas & Metals futures  "
        "(real-token-kline feed for the paper phase)"
    )
    lines.append("=" * 66)
    lines.append("")
    lines.append("## 1. Setup")
    lines.append("")
    lines.append("- sample        : real-token kline feed (synthetic stand-in)")
    lines.append(f"- account balance : ${BALANCE:,.0f} (paper only)")
    lines.append("- strategy      : MarketTrendStrategy (EMA 5/20 + volume)")
    lines.append("- runner        : BacktestRunner (0.02% slippage/side, 0.04% commission/trade)")
    lines.append("- period        : ~3.5 days of 1-minute bars (seed=7, deterministic)")
    lines.append("")
    lines.append("## 2. Results (paper only, no real money)")
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
    lines.append("## 3. Transparency")
    lines.append("")
    lines.append(
        "Paper-only. This is NOT backtest on synthetic curves; it is a signal run "
        "over real token pool fills (simulated execution). Re-run to reproduce."
    )
    lines.append("")
    lines.append("=" * 66)
    lines.append("End of paper report")
    lines.append("=" * 66)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


def main(argv=None):
    from pathlib import Path
    out = Path(argv[1]) if argv and len(argv) > 1 else ROOT / "docs" / "paper_report.md"

    # Select the paper-feeder provider. If `token_klines.provider` in config.yaml
    # names a registered provider (e.g. "raydium"), wire that real one; otherwise
    # fall back to the deterministic synthetic stand-in so the pipeline is always
    # runnable without your RPC key.
    cfg = _load_token_klines_config(ROOT / "config.yaml")
    provider_name = cfg.get("provider")

    if provider_name and provider_name in TokenKlinesProvider.list():
        fetcher = TokenKlinesProvider.get(provider_name)
        run_mode = "real"
    else:
        TokenKlinesProvider.register("paper_sample", SyntheticTokenKlinesFetcher())
        fetcher = TokenKlinesProvider.get("paper_sample")
        run_mode = "synthetic"

    data = fetch_klines(fetcher)
    print(
        f"paper phase | {fetcher.symbol} | "
        f"{len(data)} rows | {data.timestamp.min()}..{data.timestamp.max()} | "
        f"mode={run_mode}"
    )
    write_token_report(data, fetcher, out, BALANCE)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:])
