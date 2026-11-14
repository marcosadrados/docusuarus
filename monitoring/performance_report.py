"""Per-day performance report for the scalping trading bot.

Reads the trade audit log (JSON-lines) or a list of trade records and produces a
per-day breakdown: net PnL, win/loss counts, win rate, cumulative equity and
drawdown, plus an overall summary.  Used by the daily report generator and the
audit trail.

The report is deliberately offline and deterministic - no broker, no clock
dependency beyond the timestamps already embedded in each trade record.

Example
-------
>>> trades = [
...     {"timestamp": 1700000000, "pnl": 5.0, "side": "BUY"},
...     {"timestamp": 1700000060, "pnl": -2.0, "side": "SELL"},
... ]
>>> report = build_daily_report(trades)
>>> report[0]["pnl"]
3.0
>>> report[-1]["win_rate"]
0.5
"""

from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

# A day in epoch seconds (UTC).
_DAY = 86400

# Default starting account balance for cumulative-equity math.
DEFAULT_ACCOUNT_BALANCE = 10000.0

TradeLike = Dict[str, Any]


def _to_epoch(value: Any) -> float:
    """Coerce a timestamp to epoch seconds (None counts as 0)."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return value.timestamp()
    except AttributeError:
        return float(value)


def _date_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")


def _trade_pnl(trade: TradeLike) -> float:
    if isinstance(trade, dict):
        return float(trade.get("pnl", 0.0))
    # dataclass-style object
    return float(trade.pnl or 0.0)


def build_daily_report(trades: Iterable[TradeLike]) -> List[Dict[str, Any]]:
    """Build the per-day performance breakdown from a list of trades.

    Returns a list of dicts, one per trading day (UTC), each with the daily net
    PnL, number of trades, win/loss counts, win rate and cumulative equity.
    Empty or None input yields an empty list.

    Example
    -------
    >>> trades = [
    ...     {"timestamp": 1700000000, "pnl": 5.0},
    ...     {"timestamp": 1700000060, "pnl": -2.0},
    ... ]
    >>> [d["pnl"] for d in build_daily_report(trades)]
    [3.0]
    """
    days: Dict[str, Dict[str, Any]] = OrderedDict()
    order: List[str] = []

    for trade in trades:
        pnl = _trade_pnl(trade)
        epoch = _to_epoch(
            trade.get("timestamp")
            if isinstance(trade, dict)
            else getattr(trade, "timestamp", 0.0)
        )
        date = _date_from_epoch(epoch)

        if date not in days:
            days[date] = {
                "date": date,
                "pnl": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "max_drawdown": 0.0,
                "peak_equity": DEFAULT_ACCOUNT_BALANCE,
                "equity_curve": [],
            }
            order.append(date)

        d = days[date]
        d["pnl"] += pnl
        d["trades"] += 1
        if pnl >= 0:
            d["wins"] += 1
        else:
            d["losses"] += 1
        d["win_rate"] = (d["wins"] / d["trades"]) if d["trades"] else 0.0

    # Walk each day's trades in order; build the equity curve and record the
    # worst draw-down within each day.
    for date in order:
        d = days[date]
        running = DEFAULT_ACCOUNT_BALANCE
        peak = running
        d["max_drawdown"] = 0.0
        for trade in trades:
            t = (
                trade.get("timestamp")
                if isinstance(trade, dict)
                else getattr(trade, "timestamp", 0.0)
            )
            tdate = _date_from_epoch(t)
            if tdate == date:
                running += _trade_pnl(trade)
                d["equity_curve"].append(running)
                if running > peak:
                    peak = running
                dd = (peak - running) / peak if peak > 0 else 0.0
                d["max_drawdown"] = max(d["max_drawdown"], dd)
    return [days[date] for date in order]


def generate_daily_report(
    log_file: str, account_balance: float = DEFAULT_ACCOUNT_BALANCE
) -> Dict[str, Any]:
    """Read the trade JSONL log and return a daily performance report.

    Parameters
    ----------
    log_file : str
        Path to the JSON-lines log written by :class:`monitoring.logger.TradeLogger`
        (default ``logs/trades.jsonl``).
    account_balance : float
        Starting account balance, used for cumulative-equity math.

    Returns
    -------
    dict
        ``"daily"``: list of per-day dicts (same shape as ``build_daily_report``).
        ``"summary"``: overall net PnL, total trades, win rate, average trade
        size and maximum drawdown.
    """
    try:
        with open(log_file, "r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        records = []

    trades = [r for r in records if r.get("kind") == "trade"]
    daily = build_daily_report(trades)

    total_pnl = sum(d["pnl"] for d in daily)
    total_trades = sum(d["trades"] for d in daily)
    total_wins = sum(d["wins"] for d in daily)
    total_losses = sum(d["losses"] for d in daily)
    avg_trade = (total_pnl / total_trades) if total_trades else 0.0
    win_rate = (total_wins / total_trades) if total_trades else 0.0

    max_dd = max((d["max_drawdown"] for d in daily), default=0.0)

    return {
        "daily": daily,
        "summary": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "account_balance": account_balance,
            "net_pnl": total_pnl,
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "win_rate": win_rate,
            "avg_trade": avg_trade,
            "max_drawdown": max_dd,
            "days": len(daily),
        },
    }


def report_to_text(result: Dict[str, Any], lines: int = 8) -> str:
    """Render a human-readable report from a ``generate_daily_report`` result."""
    s = result["summary"]
    lines_out = [
        "=" * 64,
        "PERFORMANCE REPORT",
        "=" * 64,
        f"period          : {s['days']} day(s)",
        f"net PnL         : {s['net_pnl']:+.2f}",
        f"trades          : {s['total_trades']}  (wins {s['total_wins']}, losses {s['total_losses']})",
        f"win rate        : {s['win_rate'] * 100:.1f}%",
        f"avg trade       : {s['avg_trade']:+.2f}",
        f"max drawdown    : {s['max_drawdown'] * 100:.2f}%",
        "-" * 64,
        f"{'date':<12}{'pnl':>9}{'trades':>8}{'wins':>7}{'loss':>7}{'wr%':>8}",
        "-" * 64,
    ]
    for d in result["daily"][-lines:]:
        wr = (d["wins"] / d["trades"] * 100) if d["trades"] else 0.0
        lines_out.append(
            f"{d['date']:<12}{d['pnl']:+9.2f}{d['trades']:>8}"
            f"{d['wins']:>7}{d['losses']:>7}{wr:>8.1f}"
        )
    lines_out.append("-" * 64)
    return "\n".join(lines_out)
