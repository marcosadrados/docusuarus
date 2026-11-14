"""Tests for the notification / signal-publishing layer."""


from monitoring.notifier import (
    Channel,
    ConsoleChannel,
    DiscordChannel,
    Notifier,
    SlackChannel,
    TelegramChannel,
    WebhookChannel,
)
from orchestrator.engine import Engine, Symbol


class RecordingChannel(Channel):
    """Captures every message it receives (used for tests only)."""

    def __init__(self) -> None:
        super().__init__(enabled=True)
        self.messages: list[dict] = []

    def send(self, message: dict) -> None:
        self.messages.append(message)


def _engine() -> Engine:
    from data.fetcher import WFuturesFetcher
    from risk.mgr import RiskManager
    from signals.strategy import MarketTrendStrategy

    return Engine(
        symbols=[Symbol("CL=F")],
        risk_manager=RiskManager(account_balance=10000, max_per_trade=0.0025),
        fetcher=WFuturesFetcher(),
        strategy=MarketTrendStrategy(),
    )


def test_notifier_no_channels_is_noop():
    n = Notifier()
    assert n.channels == []
    assert n._subscriptions == []


def test_notifier_subscribes_and_dispatch():
    ch = RecordingChannel()
    n = Notifier([ch])
    engine = _engine()
    n.subscribe(engine)

    engine.broadcast(
        "position_opened",
        {
            "symbol": "CL=F",
            "side": "BUY",
            "price": 82.5,
            "size": 0.1,
            "summary": "opened",
        },
    )
    engine.broadcast(
        "position_blocked",
        {
            "symbol": "CL=F",
            "side": "SELL",
            "reason": "correlation",
        },
    )
    engine.broadcast(
        "signal_fire",
        {
            "symbol": "CL=F",
            "signal": type(
                "S", (), {"direction": "BUY", "confidence": 0.8, "reason": "mom"}
            )(),
        },
    )

    assert len(ch.messages) == 3
    assert ch.messages[0]["event"] == "position_opened"
    assert ch.messages[0]["side"] == "BUY"
    assert ch.messages[0]["summary"] == "BUY CL=F opened @ 82.5000 (0.1000 lots)"
    assert ch.messages[1]["event"] == "position_blocked"
    assert "correlation" in ch.messages[1]["summary"]
    assert ch.messages[2]["event"] == "signal_fire"
    assert ch.messages[2]["direction"] == "BUY"


def test_notifier_message_builds_workflow_fields():
    n = Notifier([])
    msg = n._build_message(
        {
            "__event_type__": "portfolio_pnl",
            "symbol": "BTC/USDT",
            "side": "SELL",
            "price": 3500.0,
            "size": 1.0,
            "confidence": 0.75,
            "pnl": -12.5,
            "summary": "closed",
        }
    )
    assert msg["event"] == "portfolio_pnl"
    assert msg["pnl"] == -12.5
    assert msg["direction"] is None


def test_notifier_from_config_console_and_discord():
    n = Notifier.from_config(
        {
            "notifier": {
                "enabled": True,
                "channels": [
                    {"type": "console"},
                    {"type": "discord", "url": "http://x.test"},
                ],
            }
        }
    )
    assert len(n.channels) == 2
    assert isinstance(n.channels[0], ConsoleChannel)
    assert isinstance(n.channels[1], DiscordChannel)


def test_notifier_disabled_config_returns_empty():
    n = Notifier.from_config({"notifier": {"enabled": False}})
    assert n.channels == []


def test_webhook_channel_post_succeeds(monkeypatch):
    received = {}

    def fake_post(url, data, headers, timeout):
        received["url"] = url
        received["data"] = data
        return type("Resp", (), {"status_code": 200})()

    monkeypatch.setattr("monitoring.notifier.requests.post", fake_post)
    ch = WebhookChannel("https://example.com/hook")
    ch.send({"ts": 1, "event": "position_opened", "symbol": "CL=F"})
    assert received["url"] == "https://example.com/hook"
    assert b'"symbol"' in received["data"]


def test_discord_channel_embed_contains_fields():
    ch = DiscordChannel("http://x.test")
    msg = ch._build_embed(
        {
            "side": "SELL",
            "symbol": "BTC/USDT",
            "price": 3500.0,
            "size": 1.0,
            "confidence": 0.75,
            "summary": "closed",
            "ts": 1700000000,
        }
    )
    assert msg["title"] == "SELL BTC/USDT"
    assert msg["color"] == 0xE74C3C
    values = [f["value"] for f in msg["fields"]]
    assert "3500.0000" in values


def test_telegram_channel_builds_text():
    ch = TelegramChannel("bot-token", "123456789")
    text = ch._build_text(
        {
            "ts": 1,
            "symbol": "BTC/USDT",
            "side": "BUY",
            "confidence": 0.8,
            "summary": "opened",
        }
    )
    assert "BTC/USDT" in text
    assert "BUY" in text
    assert "0.80" in text


def test_slack_channel_builds_text():
    ch = SlackChannel("http://x.test")
    text = ch._build_text({"summary": "SELL BTC/USDT opened"})
    assert text == "SELL BTC/USDT opened"
