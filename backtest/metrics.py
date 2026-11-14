"""Comparative performance metrics.

The axes on which the "biggest-win" vendors are compared (latest: Trendhoo
live trail, Galileo FX unverified spikes, Alpha Arena LLM results, prop-firm
payout trails). This module computes them from a per-trade ledger so the same
numbers can be produced for backtests, paper runs, and live trades alike.

Every figure here is derived from trade records, never hand-edited.

Reference axes (what competitors publish):
- Win rate, profit factor, Sharpe, Sortino, max drawdown, SQN, expectancy,
  avg win / avg loss, monthly breakdown, R-multiple distribution.
- SQN (Kaiser): sqrt(N) * mean / std, measured on trade returns.
- Profit factor = gross_profit / abs(gross_loss).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

#: Dollar-denominated scaling so "R" (risk) is meaningful. See
#: `profit_factor_per_r` below.
RISK_PER_R = 1.0


def trades_from_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract per-trade dicts from a backtest ``results`` dict.

    Backwards compatible: if ``results`` carries no ``trades`` key (old
    runners), fall back to empty so the caller can still render.
    """
    trades = results.get("trades") if isinstance(results, dict) else None
    if trades:
        return [t if isinstance(t, dict) else {"pnl": t.pnl or 0.0} for t in trades]
    return []


def _net_pnl(trade: Dict[str, Any]) -> float:
    return float(trade.get("pnl", 0.0))


def _gross_pnl(trade: Dict[str, Any]) -> float:
    return float(trade.get("gross_pnl", 0.0))


def _side(trade: Dict[str, Any]) -> str:
    return str(trade.get("side", trade.get("signal", "")))


def norm_timestamp(value: Any) -> Optional[str]:
    """Coerce a trade timestamp to an ISO ``YYYY-MM`` month key."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime("%Y-%m")
    try:
        return value.timestamp()
    except AttributeError:
        pass
    if isinstance(value, str):
        # Handle epoch-string or already-formatted: try ISO first.
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
        return dt.strftime("%Y-%m")
    return None


def compute(
    trades: Iterable[Dict[str, Any]],
    timestamps: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Compute comparative performance metrics from a list of trade records.

    Parameters
    ----------
    trades : iterable of dicts
        Each trade needs a ``pnl`` (net, e.g. percent), ``gross_pnl`` and a
        ``side``/``signal``. Dataclass-style objects with ``.pnl`` also work.
    timestamps : iterable, optional
        One timestamp per trade (epoch seconds or datetime). When supplied,
        ``monthly_breakdown`` is populated.

    Returns
    -------
    dict
        ``total_trades``, ``win_rate``, ``net_pnl``, ``avg_trade``,
        ``sharpe`` (per-trade, annualized over 1 trading day for comparability),
        ``sortino``, ``max_drawdown``, ``sqn``, ``expectancy``, ``profit_factor``,
        ``profit_factor_per_r``, ``avg_win``, ``avg_loss``, ``monthly_breakdown``.
    """
    items = [t for t in trades]
    if timestamps:
        items = [dict(t, _timestamp=t) for t, t_ in zip(items, timestamps)]

    n = len(items)
    if n == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "avg_trade": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "sqn": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "profit_factor_per_r": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "monthly_breakdown": [],
        }

    pnl = [_net_pnl(t) for t in items]
    gross = [_gross_pnl(t) for t in items]
    side = [_side(t) for t in items]
    wins = [p for p, s in zip(pnl, side) if p >= 0]
    losses = [p for p, s in zip(pnl, side) if p < 0]

    mean = sum(pnl) / n
    variance = sum((x - mean) ** 2 for x in pnl) / n
    std = math.sqrt(variance)

    sharpe = (mean / std) if std > 0 else 0.0
    # Diminished downside: sortino = mean / (downside std) -- annualized factor
    # so shares units with per-trade percent. Here we keep it as a per-day
    # ratio: downside = sqrt(mean of min(pnl,0)^2).
    downside = math.sqrt(sum(min(x, 0.0) ** 2 for x in pnl) / n)
    sortino = (mean / downside) if downside > 0 else 0.0

    running = 100.0
    peak = running
    max_dd = 0.0
    for x in pnl:
        running += x
        if running > peak:
            peak = running
        dd = (peak - running) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    sqn = (math.sqrt(n) * mean / std) if (n > 0 and std > 0) else 0.0
    expectancy = mean
    gross_profit = sum(w for w in wins)
    gross_loss = abs(sum(l for l in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    profit_factor_per_r = (gross_profit / max(abs(gross_loss), RISK_PER_R)) if n else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (abs(sum(losses)) / len(losses)) if losses else 0.0

    monthly: Dict[str, Dict[str, Any]] = OrderedDict()
    if timestamps is not None:
        ts_list = list(timestamps)
    else:
        ts_list = [getattr(t, "_timestamp", t.get("timestamp")) for t in items]
    for t, p, tm in zip(items, pnl, ts_list):
        m = norm_timestamp(tm)
        if not m:
            continue
        if m not in monthly:
            monthly[m] = {"month": m, "pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
        monthly[m]["pnl"] += p
        monthly[m]["trades"] += 1
        if p >= 0:
            monthly[m]["wins"] += 1
        else:
            monthly[m]["losses"] += 1
    monthly_breakdown = [
        {"month": m["month"], "pnl": m["pnl"], "trades": m["trades"],
         "win_rate": m["wins"] / m["trades"] if m["trades"] else 0.0}
        for m in monthly.values()
    ]

    return {
        "total_trades": n,
        "win_rate": (len(wins) / n),
        "net_pnl": sum(pnl),
        "avg_trade": (sum(pnl) / n),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "sqn": sqn,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "profit_factor_per_r": profit_factor_per_r,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "monthly_breakdown": monthly_breakdown,
    }


def monthly_summary(results: Dict[str, Any], timestamps: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """Convenience wrapper that computes comparative metrics from a backtest
    results dict (a.k.a. `results` from ``BacktestRunner.run``)."""
    trades = trades_from_results(results)
    return compute(trades, timestamps=timestamps)
