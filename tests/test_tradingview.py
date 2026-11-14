"""Tests for the TradingView alert integration."""

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


from monitoring.tradingview import (
    TradingViewAlert,
    TradingViewChannel,
)


def _sample_message():
    return {
        "symbol": "CL=F",
        "side": "BUY",
        "price": 82.5,
        "size": 0.1,
        "ts": 1700000000,
        "confidence": 0.8,
        "summary": "opened",
    }


def test_tradingview_alert_builds_json():
    alert = TradingViewAlert(strategy_name="rule_momentum")
    msg = alert.build(_sample_message())
    assert msg["symbol"] == "CL=F"
    assert msg["orderType"] == "buy"
    assert msg["histogramColor"] == "green"
    assert msg["stakeAmount"] == "0.1000"
    assert "sl" in msg
    assert "tp" in msg
    assert msg["risk"] == 0.0025


def test_tradingview_alert_sell_color():
    alert = TradingViewAlert()
    msg = alert.build({**_sample_message(), "side": "SELL"})
    assert msg["histogramColor"] == "red"
    assert msg["orderType"] == "sell"


def test_tradingview_alert_adapts_iso_ts():
    alert = TradingViewAlert()
    msg = alert.build({**_sample_message(), "ts": "2026-11-14T01:34:00.000Z"})
    assert "time" in msg


def test_tradingview_alert_epoch_ts_converted():
    alert = TradingViewAlert()
    msg = alert.build({**_sample_message(), "ts": 1700000000})
    assert msg["time"].startswith("2023-11-14T")


class _CaptureHandler(BaseHTTPRequestHandler):
    """Minimal HTTP server that records the body of incoming POSTs."""

    received: bytes = b""
    lock = threading.Lock()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        _CaptureHandler.received = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args: object) -> None:
        pass


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_tradingview_channel_sends_to_webhook():
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/hook"
        ch = TradingViewChannel(url=url)
        ch.send(_sample_message())
        assert _CaptureHandler.received != b""
        body = _CaptureHandler.received.decode("utf-8")
        assert '"symbol": "CL=F"' in body
        assert '"strategy": "rule_momentum"' in body
        assert '"orderType": "buy"' in body
        assert '"histogramColor": "green"' in body
        assert '"stakeAmount": "0.1000"' in body
        assert '"sl"' in body
        assert '"tp"' in body
        assert '"risk": 0.0025' in body
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ------------------------------------------------------------------
# TradingViewReceiver (inbound alerts) -- documented, not wired by default
# ------------------------------------------------------------------


def test_receiver_shape_is_draft_only():
    # Shape of an inbound TradingView alert (not an executable route in this
    # module yet); this test locks the documented contract.
    from monitoring.tradingview import TradingViewReceiver

    r = TradingViewReceiver()
    assert r.name == "tradingview_in"
    assert r.incoming_event == "tradingview_alert"
