"""Notification / signal-publishing layer for the scalping trading bot.

Push DONE signals (trades opened, blocked, P&L) to any channel: console,
webhook, Discord, or Telegram.  The notifier is a thin publish/subscribe
wrapper around the orchestrator event bus, so trades are broadcast
automatically as they happen -- no polling, no extra threads.

Channels
--------
- ConsoleChannel     -- print to stdout (default; no credentials needed)
- WebhookChannel     -- generic HTTP POST (JSON) to any URL
- DiscordChannel     -- Discord incoming webhook (embeds)
- TelegramChannel    -- Telegram bot API (bot_token + chat_id)
- SlackChannel       -- Slack incoming webhook (plain text)

Usage (programmatic)::

    from notifier import Notifier, ConsoleChannel, DiscordChannel
    notifier = Notifier([ConsoleChannel(), DiscordChannel(url)])
    notifier.subscribe(engine)              # engine must be running
    ...
    notifier.unsubscribe(engine)

Usage (config-driven)::

    notifier:
      enabled: true
      channels:
        - type: console
        - type: discord
          url: "[REDACTED]"
        - type: telegram
          bot_token: "[REDACTED]"
          chat_id: "123456789"

The notifier is intentionally cheap: with no channels or ``enabled: false`` it is a
no-op, so tests and dry-run mode are unaffected.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

#: Events we listen to on the engine event bus (DONE signals only).
LISTEN_EVENTS = (
    "signal_fire",  # strategy fired (pre-trade alert)
    "position_opened",  # trade opened (DONE signal)
    "position_blocked",  # trade blocked (risk guard)
    "portfolio_pnl",  # P&L update
)


class Channel:
    """Base class for every notification channel."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def send(self, message: Dict[str, Any]) -> None:
        """Push a *normalized* message dict to this channel."""
        raise NotImplementedError


class ConsoleChannel(Channel):
    """Print each message to stdout as formatted JSON."""

    def send(self, message: Dict[str, Any]) -> None:
        print(json.dumps(message, indent=2, default=str))


class WebhookChannel(Channel):
    """Generic HTTP POST (JSON) to any REST endpoint."""

    def __init__(self, url: str, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self.url = url.strip() if url else None

    def send(self, message: Dict[str, Any]) -> None:
        if not self.enabled or not self.url:
            return
        try:
            resp = requests.post(
                self.url,
                data=json.dumps(message).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Webhook POST to %s returned %s", self.url, resp.status_code
                )
            else:
                logger.info("Webhook POST to %s -> %s", self.url, resp.status_code)
        except Exception:
            logger.exception("Webhook POST to %s failed", self.url)


class DiscordChannel(WebhookChannel):
    """Discord incoming webhook: renders a rich embed."""

    def send(self, message: Dict[str, Any]) -> None:
        if not self.enabled or not self.url:
            return
        embed = self._build_embed(message)
        super().send({"embeds": [embed]})

    def _color_for(self, message: Dict[str, Any]) -> int:
        side = message.get("side")
        if side == "BUY":
            return 0x2ECC71  # green
        if side == "SELL":
            return 0xE74C3C  # red
        return 0x95A5A6  # grey

    def _build_embed(self, message: Dict[str, Any]) -> Dict[str, Any]:
        fields = [
            {"name": "Symbol", "value": message.get("symbol", "-"), "inline": True},
            {"name": "Side", "value": message.get("side", "-"), "inline": True},
            {
                "name": "Size",
                "value": f"{message.get('size', 0):.4f} lots",
                "inline": True,
            },
            {
                "name": "Price",
                "value": f"{message.get('price', 0):.4f}",
                "inline": True,
            },
            {
                "name": "Conf",
                "value": f"{message.get('confidence', 0):.2f}",
                "inline": True,
            },
            {"name": "Status", "value": message.get("status", "-"), "inline": True},
        ]
        if message.get("reason"):
            fields.append(
                {"name": "Reason", "value": message["reason"], "inline": False}
            )
        return {
            "title": f"{message.get('side', '').upper()} {message.get('symbol', '-')}",
            "description": message.get("summary", "-"),
            "color": self._color_for(message),
            "fields": fields,
            "timestamp": message.get("ts", ""),
            "footer": {"text": "scalping-bot"},
        }


class TelegramChannel(Channel):
    """Telegram bot API: bot_token + chat_id."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.bot_token = bot_token.strip()
        self.chat_id = str(chat_id).strip()

    def send(self, message: Dict[str, Any]) -> None:
        if not self.enabled or not self.bot_token or not self.chat_id:
            return
        try:
            text = self._build_text(message)
            resp = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                data={"chat_id": self.chat_id, "text": text},
                timeout=10,
            )
            if resp.status_code >= 400:
                logger.warning("Telegram POST -> %s", resp.status_code)
            else:
                logger.info("Telegram message sent")
        except Exception:
            logger.exception("Telegram POST failed")

    def _build_text(self, message: Dict[str, Any]) -> str:
        parts = [
            f"{message.get('ts', '')}  [{message.get('symbol', '-')}]",
            f"{message.get('side', '')} {message.get('summary', '')}",
        ]
        if message.get("confidence"):
            parts.append(f"conf {message['confidence']:.2f}")
        return "\n".join(parts)


class SlackChannel(WebhookChannel):
    """Slack incoming webhook: renders a plain-text message."""

    def send(self, message: Dict[str, Any]) -> None:
        if not self.enabled or not self.url:
            return
        text = self._build_text(message)
        super().send({"text": text})

    def _build_text(self, message: Dict[str, Any]) -> str:
        return (
            message.get("summary", "")
            or f"{message.get('side','')} {message.get('symbol','-')}"
        )


class Notifier:
    """Subscribe to engine events and push DONE signals to channels."""

    def __init__(self, channels: Optional[List[Channel]] = None) -> None:
        self.channels = [c for c in channels] if channels else []
        self._subscriptions: List[tuple] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def subscribe(self, engine: Any) -> None:
        """Subscribe to the engine event bus (idempotent)."""
        for event_type in LISTEN_EVENTS:
            handler = getattr(self, f"_on_{event_type}")
            wrapped = engine.subscribe(event_type, handler)
            self._subscriptions.append((event_type, wrapped))

    def unsubscribe(self, engine: Any) -> None:
        for event_type, handler in self._subscriptions:
            try:
                engine.unsubscribe(event_type, handler)
            except Exception:
                logger.debug("Failed to unsubscribe %s", event_type)
        self._subscriptions = []

    # ------------------------------------------------------------------
    # Engine event handlers
    # ------------------------------------------------------------------

    def _on_signal_fire(self, payload: Dict[str, Any]) -> None:
        self._dispatch(self._with_event_type(payload, "signal_fire"))

    def _on_position_opened(self, payload: Dict[str, Any]) -> None:
        self._dispatch(self._with_event_type(payload, "position_opened"))

    def _on_position_blocked(self, payload: Dict[str, Any]) -> None:
        self._dispatch(self._with_event_type(payload, "position_blocked"))

    def _on_portfolio_pnl(self, payload: Dict[str, Any]) -> None:
        self._dispatch(self._with_event_type(payload, "portfolio_pnl"))

    @staticmethod
    def _with_event_type(payload: Dict[str, Any], event_type: str) -> Dict[str, Any]:
        """Attach the engine event type so _build_message can classify it."""
        return {**payload, "__event_type__": event_type}

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, payload: Dict[str, Any]) -> None:
        message = self._build_message(payload)
        for channel in self.channels:
            if channel.enabled:
                try:
                    channel.send(message)
                except Exception:
                    logger.exception(
                        "Channel.send() failed for %s", type(channel).__name__
                    )

    def _build_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        event_type = payload.get("__event_type__", "unknown")
        symbol = self._get(payload, "symbol")
        side = self._get(payload, "side")
        confidence = self._get(payload, "confidence")
        price = self._get(payload, "price")
        size = self._get(payload, "size")
        status = self._get(payload, "status")
        reason = self._get(payload, "reason")
        pnl = self._get(payload, "pnl")

        if event_type == "signal_fire":
            signal = self._get(payload, "signal")
            direction = getattr(signal, "direction", None)
            summary = f"signal fired: {getattr(signal, 'reason', '-')}"
        elif event_type == "position_opened":
            direction = side
            summary = f"{side} {symbol} opened @ {price or 0:.4f} "
            summary += f"({size or 0:.4f} lots)"
        elif event_type == "position_blocked":
            direction = side
            summary = f"{side} {symbol} blocked: {reason or '-'}"
        elif event_type == "portfolio_pnl":
            direction = None
            summary = f"portfolio pnl: {pnl or 0:+.4f}"
        else:
            direction = side
            summary = f"{event_type}: {symbol}"

        return {
            "ts": time.time(),
            "event": event_type,
            "symbol": symbol,
            "side": side,
            "direction": direction,
            "price": price,
            "size": size,
            "status": status,
            "reason": reason,
            "confidence": confidence,
            "pnl": pnl,
            "summary": summary,
            "payload": payload,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get(payload: Dict[str, Any], key: str) -> Any:
        return payload.get(key)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "Notifier":
        """Build a Notifier from a config dict (gateway wiring)."""
        cfg = dict(config or {})
        section = cfg.get("notifier") or {}
        channels: List[Channel] = []
        if not section.get("enabled", True):
            return cls(channels)
        for ch in section.get("channels") or []:
            ctype = ch.get("type", "console").lower()
            if ctype in ("console",):
                channels.append(ConsoleChannel())
            elif ctype in ("webhook",):
                channels.append(
                    WebhookChannel(ch.get("url"), enabled=ch.get("enabled", True))
                )
            elif ctype in ("discord",):
                channels.append(
                    DiscordChannel(ch.get("url"), enabled=ch.get("enabled", True))
                )
            elif ctype in ("telegram",):
                channels.append(
                    TelegramChannel(
                        ch.get("bot_token"),
                        ch.get("chat_id"),
                        enabled=ch.get("enabled", True),
                    )
                )
            elif ctype in ("slack",):
                channels.append(
                    SlackChannel(ch.get("url"), enabled=ch.get("enabled", True))
                )
            elif ctype in ("tradingview",):
                from monitoring.tradingview import TradingViewChannel

                channels.append(
                    TradingViewChannel(
                        ch.get("url"),
                        strategy_name=ch.get("strategy_name", "rule_momentum"),
                        enabled=ch.get("enabled", True),
                    )
                )
            else:
                logger.warning("Unknown notifier channel type: %s", ctype)
        return cls(channels)
