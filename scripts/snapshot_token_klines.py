"""Token-kline snapshot job: appends fresh pool fills to the data/fetcher CSV
shape the BacktestRunner reads.

Run from cron (see README) or an incoming webhook. Reads the token_klines block
in config.yaml, pulls recent swaps for the configured Raydium pool, and appends
new 1m observations to data/fetcher/<pool>_klines.csv.

Always append-only; never rewrites existing rows.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.token_klines_fetcher import RaydiumPoolKlinesFetcher  # noqa: E402

FETCHER = Path("data/fetcher")
os.makedirs(FETCHER, exist_ok=True)


def load_cfg(path=ROOT / "config.yaml"):
    cfg = {}
    in_block = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s.startswith("token_klines:"):
            in_block = True
            continue
        if in_block:
            if not s or s.startswith("#"):
                continue
            if s[0] in (" ", "\t") or ":" in s:
                k, _, v = s.partition(":")
                v = v.split("#", 1)[0].strip().strip('"').strip("'")
                cfg[k.strip()] = v
            else:
                in_block = False
    return cfg


def snapshot_once(cfg):
    fetcher = RaydiumPoolKlinesFetcher(
        pool=cfg["pool"],
        rpc_url=cfg["rpc_url"],
        program_id=cfg.get("program_id"),
        limit=int(cfg.get("limit", 500)),
    )
    df = fetcher.fetch_ohlcv()
    if df.empty:
        return 0

    path = FETCHER / f"{cfg['pool']}_klines.csv"
    new = 0
    if path.exists():
        with path.open() as fh:
            reader = csv.DictReader(fh)
            last_ts = max(int(r["timestamp"]) for r in reader) if reader else 0
        new = int((df.timestamp.min() - last_ts).total_seconds() // 60)
        if new <= 0:
            return 0

    with path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if not path.exists() or path.stat().st_size == 0:
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for _, row in df.iterrows():
            writer.writerow([
                int(row.timestamp.timestamp()),
                f"{row.open:.4f}", f"{row.high:.4f}",
                f"{row.low:.4f}", f"{row.close:.4f}",
                f"{row.volume:.4f}",
            ])
    return new


if __name__ == "__main__":
    cfg = load_cfg()
    if not cfg.get("provider"):
        print("no token_klines.provider in config.yaml; run skipped (synthetic fallback active in runner)")
        sys.exit(0)
    n = snapshot_once(cfg)
    print(f"snapshot | {cfg.get('pool')} | +{n} rows -> data/fetcher/{cfg['pool']}_klines.csv")
