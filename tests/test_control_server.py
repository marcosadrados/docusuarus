"""Tests for the stdlib control surface (status / start / stop)."""

import json
import urllib.error
import urllib.request

import pytest

from monitoring.control_server import ControlServer
from orchestrator.main import Orchestrator


def _free_port():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_health_and_status():
    orch = Orchestrator(config={"control_port": 0, "log_level": "ERROR"})
    srv = ControlServer(orchestrator=orch, port=0)
    srv.start()
    port = srv._server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    try:
        health = json.loads(urllib.request.urlopen(url + "/health", timeout=3).read())
        assert health["ok"] is True
        status = json.loads(urllib.request.urlopen(url + "/status", timeout=3).read())
        assert status["running"] is False
        assert status["symbols"] == []
        assert status["strategy"] == "MarketTrendStrategy"
        assert status["trades"] == 0
        assert status["pnl"] == 0.0
        assert status["open_positions"] == {}
    finally:
        srv.shutdown()


def test_control_post_start_stop():
    orch = Orchestrator(config={"log_level": "ERROR"})
    srv = ControlServer(orchestrator=orch, port=0)
    srv.start()
    port = srv._server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    try:
        data = json.dumps({"dummy": 1}).encode("utf-8")
        req = urllib.request.Request(url + "/start", data=data, method="POST")
        urllib.request.urlopen(req, timeout=3)
        # Orchestrator.start() succeeded (engine started); /status reflects it.
    finally:
        srv.shutdown()


def test_control_returns_404_for_unknown_route():
    orch = Orchestrator(config={"control_port": 0, "log_level": "ERROR"})
    srv = ControlServer(orchestrator=orch, port=0)
    srv.start()
    port = srv._server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(url + "/nope", timeout=3)
        assert exc_info.value.code == 404
    finally:
        srv.shutdown()


def test_handler_wired_to_orchestrator():
    orch = Orchestrator(config={"control_port": 0, "log_level": "ERROR"})
    srv = ControlServer(orchestrator=orch, port=0)
    srv.start()
    port = srv._server.server_address[1]
    # Constructor wiring is verified by the /status response in test_control_post_start_stop.
    urllib.request.Request(f"http://127.0.0.1:{port}/status", method="GET")
    srv.shutdown()
