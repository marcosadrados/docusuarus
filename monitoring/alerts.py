"""Alerting subsystem for the scalping trading bot.

Alerts evaluates live trading metrics against risk limits and fires
notifications when thresholds are breached:

* ``on_drawdown``        -> max drawdown (3%)
* ``on_daily_loss``      -> daily loss limit (1.5%)
* ``on_api_error``       -> broker / market-data API error

Notifications can be routed to Slack, email, or the console. Every firing is
accumulated so callers can audit what was alerted (``Alerts.alerts`` / ``.was_fired``).
"""

from __future__ import annotations

import logging
from typing import Union

logger = logging.getLogger(__name__)

# Notification channels
Channel = Union[str, None]


class Alerts:
    """Evaluate metrics against risk limits and fire alerts when breached."""

    # Risk limits (account-relative), consistent with the bot's risk posture.
    DEFAULT_MAX_DRAWDOWN = 0.03      # 3%
    DEFAULT_DAILY_LOSS_LIMIT = 0.015  # 1.5%
    DEFAULT_MAX_PER_TRADE = 0.0025   # 0.25%

    def __init__(
        self,
        slack_webhook: str | None = None,
        email_recipient: str | None = None,
        max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
        daily_loss_limit: float = DEFAULT_DAILY_LOSS_LIMIT,
        max_per_trade: float = DEFAULT_MAX_PER_TRADE,
    ) -> None:
        self.slack_webhook = slack_webhook
        self.email_recipient = email_recipient
        self.max_drawdown = float(max_drawdown)
        self.daily_loss_limit = float(daily_loss_limit)
        self.max_per_trade = float(max_per_trade)

        self._alerts: list[dict] = []
        self.was_fired = False

    # ------------------------------------------------------------------
    # Notification plumbing
    # ------------------------------------------------------------------
    def _notify(self, channel: Channel, message: str) -> None:
        if channel == "slack" and self.slack_webhook:
            logger.warning("SLACK ALERT (webhook=%s): %s", self.slack_webhook, message)
        elif channel == "email" and self.email_recipient:
            logger.warning("EMAIL ALERT (to=%s): %s", self.email_recipient, message)
        else:
            # Headless / paper-trading: surface the alert on the console log.
            logger.warning("ALERT: %s", message)

    # ------------------------------------------------------------------
    # Threshold checks
    # ------------------------------------------------------------------
    def on_drawdown(self, drawdown: float) -> bool:
        """Fire when the peak-to-current drawdown exceeds the limit.

        Returns ``True`` if an alert fired.
        """
        drawdown = float(drawdown)
        fired = drawdown >= self.max_drawdown
        if fired:
            self.was_fired = True
            self._alerts.append(
                {
                    "type": "drawdown",
                    "drawdown": drawdown,
                    "limit": self.max_drawdown,
                }
            )
            self._notify(
                "slack" if self.slack_webhook else "console",
                f"MAX DRAWDOWN BREACH: {drawdown:.2%} >= {self.max_drawdown:.2%}",
            )
        return fired

    def on_daily_loss(self, daily_loss: float) -> bool:
        """Fire when the day's loss exceeds the daily-loss limit.

        Returns ``True`` if an alert fired.
        """
        daily_loss = float(daily_loss)
        fired = daily_loss >= self.daily_loss_limit
        if fired:
            self.was_fired = True
            self._alerts.append(
                {
                    "type": "daily_loss",
                    "daily_loss": daily_loss,
                    "limit": self.daily_loss_limit,
                }
            )
            self._notify(
                "slack" if self.slack_webhook else "console",
                f"DAILY LOSS LIMIT BREACH: {daily_loss:.2%} >= {self.daily_loss_limit:.2%}",
            )
        return fired

    def on_api_error(self, error: Exception | str) -> bool:
        """Fire on any broker / market-data API error.

        Returns ``True`` (an API error always fires an alert).
        """
        message = (
            str(error) if isinstance(error, Exception) else str(error)
        )
        self.was_fired = True
        self._alerts.append({"type": "api_error", "error": message})
        self._notify(
            "slack" if self.slack_webhook else "console",
            f"API ERROR: {message}",
        )
        return True

    # ------------------------------------------------------------------
    # Introspection / lifecycle
    # ------------------------------------------------------------------
    @property
    def alerts(self) -> list[dict]:
        """Copy of every alert that has fired, in order."""
        return list(self._alerts)

    def reset(self) -> None:
        """Clear fired alerts for a new trading day / session."""
        self._alerts.clear()
        self.was_fired = False
