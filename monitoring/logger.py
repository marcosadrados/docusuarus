"""Audit-log subsystem for the scalping trading bot.

TradeLogger records every trade, signal event, and risk check to a JSON-lines
log file (append-only, human-readable) and keeps an in-memory tail of the last
events so tests and the orchestrator can inspect what was recorded without
re-reading the file.

Every logged record is a JSON object on its own line with a ``kind`` field:
``trade``, ``signal``, or ``risk``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A filled trade with full audit metadata."""

    timestamp: float
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float | None
    pnl: float | None
    order_id: str | None = None
    slippage: float = 0.0
    commission: float = 0.0


@dataclass
class SignalEvent:
    """A signal fired by the signal engine."""

    timestamp: float
    symbol: str
    signal: str
    confidence: float
    direction: str | None = None


@dataclass
class RiskCheck:
    """Result of a single risk check."""

    timestamp: float
    check: str
    passed: bool
    details: dict = field(default_factory=dict)


class TradeLogger:
    """Persist trades, signal events, and risk checks; keep an in-memory tail."""

    #: Default JSON-lines location (relative to process cwd).
    DEFAULT_LOG_FILE = "logs/trades.jsonl"

    def __init__(
        self,
        log_file: str = DEFAULT_LOG_FILE,
        account_balance: float = 10000.0,
        max_per_trade: float = 0.0025,
        daily_loss_limit: float = 0.015,
        max_drawdown: float = 0.03,
    ) -> None:
        self.log_file = log_file
        self.account_balance = account_balance
        self.max_per_trade = max_per_trade
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown

        self._trades: list[Trade] = []
        self._signals: list[SignalEvent] = []
        self._risk_checks: list[RiskCheck] = []

        # Rolling account state the logger tracks for the audit trail.
        self._daily_pnl = 0.0
        self._peak_equity = account_balance

        self._closed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_dir(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.log_file))
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _write(self, record: dict) -> None:
        self._ensure_dir()
        with open(self.log_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=lambda obj: asdict(obj)) + "\n")

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------
    def log_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        order_id: str | None = None,
        slippage: float = 0.0,
        commission: float = 0.0,
    ) -> Trade:
        """Record a trade and append it to the audit log."""
        trade = Trade(
            timestamp=time_time(),
            symbol=symbol.upper(),
            side=side.upper(),
            size=size,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            order_id=order_id,
            slippage=slippage,
            commission=commission,
        )
        self._trades.append(trade)
        self._write({"kind": "trade", **asdict(trade)})
        return trade

    def log_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        direction: str | None = None,
    ) -> SignalEvent:
        """Record a signal event and append it to the audit log."""
        event = SignalEvent(
            timestamp=time_time(),
            symbol=symbol.upper(),
            signal=str(signal).upper(),
            confidence=confidence,
            direction=direction,
        )
        self._signals.append(event)
        self._write({"kind": "signal", **asdict(event)})
        return event

    def log_risk_check(
        self,
        check: str,
        passed: bool,
        details: dict | None = None,
    ) -> RiskCheck:
        """Record a risk check and append it to the audit log."""
        record = RiskCheck(
            timestamp=time_time(),
            check=check,
            passed=passed,
            details=dict(details or {}),
        )
        self._risk_checks.append(record)
        self._write({"kind": "risk", **asdict(record)})
        return record

    # Convenience risk-check helpers (compute + log in one step).
    def check_risk_per_trade(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        stop_price: float,
    ) -> RiskCheck:
        """Round-trip risk of one trade vs the 0.25% per-trade limit."""
        risk_per_share = abs(entry_price - stop_price)
        risk = risk_per_share * size
        limit = self.account_balance * self.max_per_trade
        passed = risk <= limit
        return self.log_risk_check(
            "position_size",
            passed,
            {
                "symbol": symbol,
                "size": size,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "risk_per_share": risk_per_share,
                "risk": risk,
                "limit": limit,
            },
        )

    def check_daily_loss(self, pnl: float) -> RiskCheck:
        """Daily loss vs the 1.5% daily-loss limit."""
        self._daily_pnl += pnl
        loss = -self._daily_pnl
        limit = self.account_balance * self.daily_loss_limit
        passed = loss <= limit
        return self.log_risk_check(
            "daily_loss",
            passed,
            {
                "daily_pnl": self._daily_pnl,
                "loss": loss,
                "limit": limit,
            },
        )

    def check_max_drawdown(self, current_equity: float) -> RiskCheck:
        """Max drawdown vs the 3% drawdown limit."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        passed = drawdown <= self.max_drawdown
        return self.log_risk_check(
            "max_drawdown",
            passed,
            {
                "peak_equity": self._peak_equity,
                "current_equity": current_equity,
                "drawdown": drawdown,
                "limit": self.max_drawdown,
            },
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    @property
    def signals(self) -> list[SignalEvent]:
        return list(self._signals)

    @property
    def risk_checks(self) -> list[RiskCheck]:
        return list(self._risk_checks)

    @property
    def recent_events(self) -> list:
        """All logged events in chronological order (trades, signals, risks)."""
        return self._trades + self._signals + self._risk_checks

    def flush(self) -> None:
        """No-op for append-only JSONL; present for an explicit flush API."""
        pass

    def close(self) -> None:
        """Explicitly close the logger (no-op for file appender)."""
        self._closed = True


def time_time() -> float:
    """Epoch-seconds helper so the module is easy to monkeypatch in tests."""
    return datetime.now(timezone.utc).timestamp()
