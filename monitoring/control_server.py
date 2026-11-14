"""Lightweight control surface for the trading bot (no external web framework).

Exposes a stdlib HTTP server with a tiny REST API so the bot can be
controlled and inspected without a dashboard:

    GET  /health        -> {'ok': true}
    GET  /status        -> {running, symbols, strategy, open_positions, pnl, trades}
    GET  /signals       -> latest trades (open-position snapshot)
    POST /start         -> start the trading loop
    POST /stop          -> stop the trading loop
    POST /restart       -> stop + start

Runs entirely on the Python standard library (ThreadingHTTPServer), so it
follows the same zero-dependency, self-hosted philosophy as the rest of the
bot.  It is a control surface, not a trading or execution layer.

Architecture note
-----------------
The engine is async: :meth:`orchestrator.engine.Engine.start` schedules one
asyncio task per symbol and relies on a *running* event loop.  The control
surface is a synchronous stdlib HTTP server, so it cannot depend on a running
loop from the embedding application.  To keep ``/start`` and ``/stop`` truly
functional, this server runs the engine's scan loop on a dedicated event loop
in a daemon background thread.  In production, where the bot is embedded in
an async runtime, the engine is started directly by the application and the
control surface simply mirrors that state through ``/status``.

Usage
-----
    from monitoring.control_server import ControlServer
    from orchestrator.main import Gateway

    gw = Gateway(config={'control_port': 8999})
    # Gateway wires the server automatically when config['control_port'] is set.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

__all__ = ["ControlHandler", "ControlServer"]

_OK = {"ok": True, "ts": 0.0}


def _now() -> float:
    return time.time()


class ControlHandler(BaseHTTPRequestHandler):
    """REST handler wired to a :class:`ControlServer`."""

    orchestrator: Any = None  # set by ControlServer.start()
    # Engine-loop state is kept on the handler so both the HTTP handler and
    # the loop thread read and write the same variables.
    _engine_thread: Optional[threading.Thread] = None
    _engine_loop: Optional[asyncio.AbstractEventLoop] = None
    _engine_started = False

    # -- helpers ---------------------------------------------------------

    def _send(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}

    # -- lifecycle --------------------------------------------------------

    def do_GET(self) -> None:
        if self.path in ("/health", "/"):
            return self._send(_OK)
        if self.path == "/status":
            return self._status()
        if self.path == "/signals":
            return self._signals()
        self._send({"error": "not found", "path": self.path}, 404)

    def do_POST(self) -> None:
        if self.path in ("/start", "/stop", "/restart"):
            return self._control(self.path)
        self._send({"error": "not found", "path": self.path}, 404)

    # -- endpoints ---------------------------------------------------------

    def _status(self) -> None:
        orch = self.orchestrator
        if orch is None or orch._engine is None:
            return self._send(
                {
                    "running": False,
                    "symbols": [],
                    "strategy": None,
                    "open_positions": {},
                    "pnl": 0.0,
                    "trades": 0,
                }
            )
        engine = orch._engine
        return self._send(
            {
                "running": orch.running,
                "symbols": engine.symbols_codes,
                "strategy": type(orch.strategy).__name__,
                "open_positions": engine.open_positions,
                "pnl": engine.portfolio_pnl,
                "trades": len(engine.trades),
            }
        )

    def _signals(self) -> None:
        orch = self.orchestrator
        if orch is None or orch._engine is None:
            return self._send([])
        engine = orch._engine
        return self._send({"signals": list(engine.trades[-20:])})

    def _control(self, action: str) -> None:
        orch = self.orchestrator
        if orch is None:
            return self._send({"error": "no orchestrator"}, 500)
        if action == "/start" and not orch.running:
            if self._engine_thread is not None and self._engine_thread.is_alive():
                return self._send({"status": "already running", "running": True})
            self._engine_thread = threading.Thread(
                target=self._run_loop, daemon=True, name="bot-engine-loop"
            )
            self._engine_thread.start()
            return self._send({"status": "started", "running": True})
        if action == "/stop":
            self._stop_loop()
            orch.stop()
            return self._send({"status": "stopped", "running": False})
        if action == "/restart" and orch.running:
            orch.stop()
            try:
                orch.start()
            except Exception as exc:
                return self._send({"error": str(exc)}, 500)
            return self._send({"status": "restarted"})
        return self._send({"error": "not found", "path": self.path}, 404)

    # -- engine loop (runs on a background event loop) ---------------------

    def _run_loop(self) -> None:
        """Create a fresh event loop, start the orchestrator, keep it alive."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            self.orchestrator.start()
            self._engine_started = True
            loop.run_forever()
        except Exception:
            self._engine_started = False
            raise
        finally:
            self._engine_started = False
            loop.close()

    def _stop_loop(self) -> None:
        """Stop the background engine event loop and release resources."""
        if self._engine_thread is None:
            return
        loop = self._engine_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        try:
            self._engine_thread.join(timeout=5.0)
        except Exception:
            pass
        self._engine_thread = None
        self._engine_loop = None
        self._engine_started = False

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet: control surface logs to stdout/stderr via the logger only if needed.
        pass


class ControlServer:
    """Owns the stdlib control surface; start/stop in a background thread.

    One :class:`ControlHandler` subclass (shared across all requests) holds
    the engine-loop state; :class:`ControlServer` only wires it to an
    :class:`orchestrator` and serves the HTTP socket.
    """

    def __init__(
        self,
        orchestrator: Any = None,
        host: str = "127.0.0.1",
        port: int = 8999,
    ) -> None:
        self.orchestrator = orchestrator
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = ThreadingHTTPServer((self.host, self.port), ControlHandler)
        # Wire the orchestrator instance into the shared handler class.
        ControlHandler.orchestrator = self.orchestrator
        ControlHandler._engine_thread = None
        ControlHandler._engine_loop = None
        ControlHandler._engine_started = False
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="bot-control-server"
        )
        self._thread.start()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    @property
    def url(self) -> str:
        if self._server is None:
            return f"http://{self.host}:{self.port}"
        return f"http://localhost:{self.port}"

    def __enter__(self) -> "ControlServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
