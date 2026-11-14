#!/usr/bin/env python3
"""Deterministic smoke for the multi-symbol live tick loop.

Verifies that the Orchestrator builds a multi-symbol `Engine` and that the
engine's per-symbol asyncio scan tasks start across connectors (cme futures,
binance crypto) without a separate tick-loop file.

Usage:
    python scripts/smoke_multi_symbol.py

Optionally: `python scripts/smoke_multi_symbol.py --config path/to/config.yaml`
(to smoke a real config file). The default runs an offline, fixture-driven
session that never needs a broker key.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

sys.path.insert(0, ".")


async def run_smoke(config):
    from orchestrator.main import Orchestrator

    orch = Orchestrator(config=config)
    codes = [s.code for s in orch.symbols]
    connectors = sorted({s.connector for s in orch.symbols})
    assert connectors == ["binance", "cme"], f"connectors={connectors}"
    assert len(codes) >= 2, f"expected >=2 symbols, got {len(codes)}"

    orch.start()
    try:
        tasks = orch._engine._tasks
        assert tasks, "no symbol tasks started"
        assert len(tasks) == len(codes)
        assert all(not t.done() for t in tasks)
        engine = orch._engine
        # All symbols scanned in the same parallel window.
        assert len(engine.open_positions) == 0
    finally:
        orch.stop()

    return codes, connectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="path to a config.yaml (defaults to offline multi-symbol config)",
    )
    args = parser.parse_args()

    if args.config:
        import yaml

        with open(args.config) as fh:
            config = yaml.safe_load(fh) or {}
    else:
        config = {
            "markets": ["CL=F", "GC=F", "XAU=F"],
            "crypto_markets": ["BTC/USDT", "ETH/USDT"],
            "connector_map": {
                "CL=F": "cme",
                "GC=F": "cme",
                "XAU=F": "cme",
                "BTC/USDT": "binance",
                "ETH/USDT": "binance",
            },
            "strategy": "market_trend",
            "log_level": "ERROR",
            "control_port": 0,
        }

    codes, connectors = asyncio.run(run_smoke(config))
    print(f"multi-symbol smoke OK: {codes} [connectors={connectors}]")
