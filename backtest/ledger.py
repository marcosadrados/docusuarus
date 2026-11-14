"""Auditable per-trade ledger.

Writes a JSON-lines file that records every trade with a unique append-only
rowid, an ISO timestamp, and the full economics (entry/exit, gross pnl,
slippage, commission, net pnl, win flag). Once written the file is never
rewritten: each run appends. This is the same shape competitors use for
public performance trails (third-party trackers, prop-firm payout logs,
Alpha Arena broadcasts) and the only shape auditors will accept.

Shape:
    {"rowid": 1, "ts": "2026-11-14T10:32:00", "timestamp_epoch": 173157...,
     "symbol": "CL=F", "side": "BUY", "entry": 80.1, "exit": 80.75,
     "gross_pnl": 0.81, "slippage": -0.02, "commission": 0.04,
     "net_pnl": 0.75, "pnl": 0.75, "win": true, "timestamp": 1731...}
"""

from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .metrics import compute, monthly_summary

DEFAULT_ACCOUNT_BALANCE = 10000.0

#: Files are read/write. Each "ledger" is actually one JSONL file.
LedgerPath = Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")


def _epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value.timestamp())
    except AttributeError:
        return float(value)


def _side_value(trade: Dict[str, Any]) -> str:
    return str(trade.get("side", trade.get("signal", ""))).upper()


def _make_record(
    trade: Dict[str, Any],
    symbol: str,
    ts: Optional[float],
    balance: float,
) -> Dict[str, Any]:
    pnl = float(trade.get("pnl", 0.0))
    gross = float(trade.get("gross_pnl", 0.0))
    return {
        "rowid": uuid.uuid4().hex,
        "ts": _now_iso(),
        "timestamp_epoch": _epoch(ts) if ts is not None else None,
        "symbol": symbol,
        "account_balance": balance,
        "side": _side_value(trade),
        "entry": trade.get("entry", None),
        "exit": trade.get("exit", None),
        "gross_pnl": gross,
        "slippage": trade.get("slippage", 0.0),
        "commission": trade.get("commission", 0.0),
        "net_pnl": pnl,
        "pnl": pnl,
        "timestamp": _epoch(ts) if ts is not None else None,
        "win": pnl >= 0,
    }


def write_ledger(
    trades: Iterable[Dict[str, Any]],
    path: Optional[str] = None,
    symbol: str = "UNKNOWN",
    balance: float = DEFAULT_ACCOUNT_BALANCE,
    timestamps: Optional[Iterable[float]] = None,
) -> Path:
    """Append trade records to an auditable JSONL ledger.

    ``path`` may be a directory (a per-session file is created under
    ``<path>/trades-<YYYY-MM-DD>.jsonl``) or a file path. Returns the
    resolved path. Idempotent append: files are opened in ``a`` mode, so
    re-running a backtest does not silently overwrite history.
    """
    tracepath = Path(path) if path is not None else None
    if tracepath is None or tracepath.is_dir():
        if tracepath is None:
            tracepath = Path("logs")
        tracepath.mkdir(parents=True, exist_ok=True)
        tracepath = tracepath / f"trades-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    elif tracepath.parent != Path("."):
        tracepath.parent.mkdir(parents=True, exist_ok=True)

    if timestamps is not None:
        timestamps = list(timestamps)
    else:
        timestamps = None

    written = 0
    with tracepath.open("a", encoding="utf-8") as fh:
        for i, trade in enumerate(trades):
            ts = None
            if timestamps is not None and i < len(timestamps):
                ts = timestamps[i]
            record = _make_record(trade, symbol, ts, balance)
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1
    return tracepath


def load_ledger(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL ledger back into memory."""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize(
    path: str,
    balance: float = DEFAULT_ACCOUNT_BALANCE,
    timestamps: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Load a ledger and compute the comparative metrics plus run counts."""
    trades = load_ledger(path)
    result = compute(trades, timestamps=timestamps)
    result.update(
        {
            "ledger_path": path,
            "ledger_rows": len(trades),
            "ledger_first_ts": (
                trades[0].get("timestamp_epoch") if trades else None
            ),
            "ledger_last_ts": (
                trades[-1].get("timestamp_epoch") if trades else None
            ),
        }
    )
    return result
