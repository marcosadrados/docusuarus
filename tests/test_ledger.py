"""Tests for the auditable per-trade ledger."""

import json
from pathlib import Path

import pytest

from backtest.ledger import load_ledger, summarize, write_ledger

BASE = Path(__file__).resolve().parent


def test_write_and_load(tmp_path):
    trades = [
        {"entry": 100.0, "exit": 101.0, "gross_pnl": 1.0, "slippage": -0.02,
         "commission": 0.04, "pnl": 0.94, "side": "BUY", "timestamp": 1700000000},
        {"entry": 101.0, "exit": 99.0, "gross_pnl": -1.5, "slippage": 0.02,
         "commission": 0.04, "pnl": -1.56, "side": "SELL", "timestamp": 1700000060},
    ]
    path = write_ledger(
        trades, tmp_path / "trades.jsonl", symbol="CL=F",
        balance=10000.0, timestamps=[1700000000, 1700000060],
    )
    records = load_ledger(str(path))
    assert len(records) == 2
    assert records[0]["symbol"] == "CL=F"
    assert records[0]["side"] == "BUY"
    assert records[0]["rowid"]
    assert records[0]["timestamp_epoch"] == 1700000000
    assert records[1]["side"] == "SELL"
    assert records[1]["net_pnl"] == pytest.approx(-1.56)


def test_write_appends_without_overwriting(tmp_path):
    p = tmp_path / "trades.jsonl"
    t1 = [{"entry": 1.0, "exit": 1.1, "gross_pnl": 0.1, "pnl": 0.08}]
    t2 = [{"entry": 1.1, "exit": 1.0, "gross_pnl": -0.1, "pnl": -0.12}]
    write_ledger(t1, str(p), symbol="CL=F")
    write_ledger(t2, str(p), symbol="CL=F")
    assert len(load_ledger(str(p))) == 2


def test_summarize_derives_metrics(tmp_path):
    trades = [
        {"entry": 1.0, "exit": 1.1, "gross_pnl": 0.2, "pnl": 0.17},
        {"entry": 1.1, "exit": 1.2, "gross_pnl": 0.1, "pnl": 0.08},
        {"entry": 1.2, "exit": 1.0, "gross_pnl": -0.3, "pnl": -0.34},
    ]
    path = write_ledger(trades, tmp_path / "t.jsonl", symbol="CL=F",
                        timestamps=[1, 2, 3])
    m = summarize(str(path))
    assert m["ledger_rows"] == 3
    assert m["total_trades"] == 3
    assert m["win_rate"] == 2 / 3
    assert m["profit_factor"] == 0.25 / 0.34
    assert len(m["monthly_breakdown"]) == 1
