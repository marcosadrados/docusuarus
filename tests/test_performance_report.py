"""Smoke tests for the per-day performance report generator.

Verifies that the performance report reads the trade audit log, builds a
per-day breakdown with win/loss counts and draw-down, and that the overall
summary reconciles to the trade records.
"""

import json

import pytest

from monitoring.performance_report import (
    build_daily_report,
    generate_daily_report,
    report_to_text,
)

DEFAULT_ACCOUNT_BALANCE = 10000.0


def _sample_log(path):
    """Write a small deterministic trade audit log and return its path.

    Trades at 2023-11-14 + 2023-11-15 so the report spans two days.
    """
    records = [
        {
            "kind": "trade",
            "timestamp": 1700000000,
            "symbol": "CL=F",
            "side": "BUY",
            "size": 1.0,
            "entry_price": 80.0,
            "exit_price": 80.5,
            "pnl": 3.0,
            "order_id": "o1",
        },
        {
            "kind": "trade",
            "timestamp": 1700000060,
            "symbol": "CL=F",
            "side": "SELL",
            "size": 1.0,
            "entry_price": 80.5,
            "exit_price": 80.2,
            "pnl": -1.0,
            "order_id": "o2",
        },
        {
            "kind": "trade",
            "timestamp": 1700086400,
            "symbol": "CL=F",
            "side": "BUY",
            "size": 1.0,
            "entry_price": 80.0,
            "exit_price": 79.8,
            "pnl": -2.0,
            "order_id": "o3",
        },
        {
            "kind": "trade",
            "timestamp": 1700086460,
            "symbol": "CL=F",
            "side": "SELL",
            "size": 1.0,
            "entry_price": 79.8,
            "exit_price": 80.6,
            "pnl": 1.0,
            "order_id": "o4",
        },
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def test_build_daily_report_groups_by_day():
    trades = [
        {"timestamp": 1700000000, "pnl": 4.0},
        {"timestamp": 1700000060, "pnl": -2.0},
        {"timestamp": 1700086400, "pnl": -2.0},
        {"timestamp": 1700086460, "pnl": 1.0},
    ]
    report = build_daily_report(trades)
    assert [d["date"] for d in report] == ["2023-11-14", "2023-11-15"]
    assert len(report) == 2
    assert report[0]["pnl"] == pytest.approx(2.0)
    assert report[0]["win_rate"] == pytest.approx(0.5)


def test_build_daily_report_is_empty_for_empty_input():
    assert build_daily_report([]) == []


def test_generate_daily_report_reconciles(tmp_path):
    log_path = str(tmp_path / "trades.jsonl")
    _sample_log(log_path)

    report = generate_daily_report(log_path)

    s = report["summary"]
    assert s["net_pnl"] == pytest.approx(1.0)
    assert s["total_trades"] == 4
    assert s["total_wins"] == 2
    assert s["total_losses"] == 2
    assert s["win_rate"] == pytest.approx(0.5)
    assert len(report["daily"]) == 2


def test_report_to_text(tmp_path):
    log_path = str(tmp_path / "trades.jsonl")
    _sample_log(log_path)
    report = generate_daily_report(log_path)
    text = report_to_text(report)
    assert "PERFORMANCE REPORT" in text
    assert "2023-11-14" in text
    assert "2023-11-15" in text


def test_generate_daily_report_missing_log_returns_empty(tmp_path):
    report = generate_daily_report(str(tmp_path / "nope.jsonl"))
    assert report["summary"]["total_trades"] == 0
    assert report["summary"]["net_pnl"] == 0.0
