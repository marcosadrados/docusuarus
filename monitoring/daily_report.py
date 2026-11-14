"""CLI: per-day performance report for the scalping trading bot.

Reads the trade audit JSON-lines log and prints a per-day performance report
(net PnL, win rate, trade counts, drawdown) so you can audit the bot without
opening code.

Usage
-----
    python -m monitoring.daily_report                 # last 7 days
    python -m monitoring.daily_report logs/trades.jsonl
    python -m monitoring.daily_report --days 14
"""

from __future__ import annotations

import argparse
import sys

from monitoring.performance_report import generate_daily_report, report_to_text

DEFAULT_LOG = "logs/trades.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a per-day performance report for the scalping bot."
    )
    parser.add_argument(
        "log",
        nargs="?",
        default=DEFAULT_LOG,
        help=f"trade audit log path (default: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="how many days of history to show (default: 7)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        result = generate_daily_report(args.log)
    except FileNotFoundError:
        print(f"error: log not found: {args.log}")
        return 1

    print(report_to_text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
