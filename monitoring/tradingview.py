"""TradingView alert integration for the scalping trading bot.

Trader-facing way to push bot signals into TradingView alert/strategy
management.  You configure a TradingView alert with **Webhook URL** as the
action; this module builds the JSON TradingView expects and POSTs it whenever
the bot emits a DONE signal.

The bot already POSTs to arbitrary URLs, so the only piece you add is a
TradingViewChannel configured in :class:`monitoring.notifier.Notifier` plus
the webhook URL.  No external library required.

TradingView alert JSON shape
-----------------------------
{
  "symbol": "CL=F",
  "strategy": "rule_momentum",
  "time": "2026-11-14T01:34:00.123Z",
  "close": 82.5,
  "histogramColor": "green",
  "orderType": "buy",
  "stakeAmount": "0.100",
  "sl": "82.14",
  "tp": "",
}

With the local Webhook CIDR rule enabled, TradingView may also POST alerts
into the bot (see :class:`TradingViewReceiver`), but the outbound path above
is the primary use case for this module.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict

__all__ = ["TradingViewAlert", "TradingViewChannel", "TradingViewReceiver"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class TradingViewAlert:
    """Build a TradingView-compatible JSON payload for a DONE signal."""

    def __init__(self, strategy_name: str = "rule_momentum") -> None:
        self.strategy_name = strategy_name

    @staticmethod
    def _color_for(side: str | None) -> str:
        return "green" if side == "BUY" else "red"

    @staticmethod
    def _order_for(side: str | None) -> str:
        return "buy" if side == "BUY" else "sell"

    def build(
        self,
        payload: Dict[str, Any],
        *,
        stop_distance: float = 0.0025,
        risk: float = 0.0025,
    ) -> Dict[str, Any]:
        """Build the TradingView alert JSON from a *normalized* message.

        Parameters
        ----------
        payload:
            Message a notifier built (see :func:`TradingViewChannel.send`).
        stop_distance:
            Fractional stop distance from entry (default 0.25%, the bot's
            risk ceiling).
        risk:
            Fractional risk per trade (default 0.25%).
        """
        symbol = payload.get("symbol")
        side = payload.get("side")
        price = payload.get("price") or payload.get("close")
        size = payload.get("size")
        reason = payload.get("summary") or payload.get("reason")
        ts = payload.get("ts") or _utcnow()

        # ``ts`` may be a float epoch (planner) or an ISO string.
        if isinstance(ts, (int, float)):
            from datetime import datetime, timezone

            ts = (
                datetime.fromtimestamp(ts, timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                )[:-3]
                + "Z"
            )

        entry = float(price) if price else 0.0
        stop = (
            entry * (1 - stop_distance)
            if side == "BUY"
            else entry * (1 + stop_distance)
        )
        sl = f"{stop:.4f}" if stop else ""
        tp = f"{entry * (1 + 2 * risk):.4f}" if entry else ""
        size_fmt = f"{size:.4f}" if size else ""

        return {
            "symbol": symbol,
            "strategy": self.strategy_name,
            "time": ts,
            "close": entry,
            "histogramColor": self._color_for(side),
            "orderType": self._order_for(side),
            "stakeAmount": size_fmt,
            "sl": sl,
            "tp": tp,
            "reason": reason,
            "risk": risk,
        }

    def post(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: int = 10,
    ) -> None:
        """POST the built payload to a TradingView webhook URL."""
        if not url:
            return
        body = json.dumps(self.build(payload)).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status >= 400:
                    print(f"[tradingview] webhook returned {resp.status}")
        except Exception:
            print("[tradingview] webhook POST failed")


class TradingViewChannel:
    """Standalone channel that pushes DONE signals to TradingView via webhook.

    Used by :class:`monitoring.notifier.Notifier` via the ``tradingview``
    channel type.  (Defined here rather than in notifier.py so this module
    stays free of the circular import that would result from either side
    importing the other at module load.)
    """

    def __init__(
        self,
        url: str = "",
        strategy_name: str = "rule_momentum",
        enabled: bool = True,
    ) -> None:
        self.url = url.strip()
        self.strategy_name = strategy_name
        self.enabled = enabled
        self.alert = TradingViewAlert(strategy_name=strategy_name)

    def send(self, message: Dict[str, Any]) -> None:
        if not self.enabled or not self.url:
            return
        try:
            self.alert.post(self.url, message)
        except Exception:
            print("[tradingview] channel send failed")


class TradingViewReceiver:
    """Inbound TradingView alert handler (draft: TradingView POSTs alerts here).

    Documented as the inbound side of the pair.  The current implementation
    is a zero-operation stub; wiring the bot to actually ingest TradingView
    alerts is a follow-up so the outbound (push) direction above is the
    primary path today.
    """

    def __init__(self) -> None:
        self.name = "tradingview_in"
        self.incoming_event = "tradingview_alert"

    def handle(self, payload: Dict[str, Any]) -> None:
        # Placeholder: TradingView -> bot ingest not wired yet.
        return
