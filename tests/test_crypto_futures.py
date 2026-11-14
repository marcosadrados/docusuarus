"""Tests: Binance Futures / crypto paper-trading (testnet + dry-run)."""


from data.fetcher import CryptoFetcher
from exec.broker import CryptoBroker


def test_cryptofetcher_futures_uses_testnet():
    """Paper mode must hit the live testnet (real bars), not the sample."""
    f = CryptoFetcher(exchange="binance", test=True, futures=True)
    bars = f.fetch_candles("BTC/USDT", "1m", limit=2)
    assert len(bars) == 2
    # Testnet rows carry a future-epoch timestamp, not the offline sample seed.
    assert bars[0][0] > 1600000000
    # Fresh bars (coordinates are not the deterministic sample values).
    assert bars[0][4] != 95000.0
    f.close()


def test_cryptofetcher_futures_offline_uses_sample():
    """Without test=True and with no key, must fall back to the sample."""
    f = CryptoFetcher(exchange="binance", futures=True)
    bars = f.fetch_candles("BTC/USDT", "5m", limit=1)
    assert len(bars) == 1
    assert abs(bars[0][4] - 95000.0) < 0.1
    f.close()


def test_cryptobroker_futures_dry_run_without_key():
    """No key + futures=True -> orders recorded, not sent."""
    b = CryptoBroker(exchange="binance", test=True, futures=True)
    b.connect()
    res = b.execute_buy("BTC/USDT", amount=0.001, price=None, order_type="market")
    assert res["status"] == "dry-run"
    assert res["symbol"] == "BTC/USDT"
    b.disconnect()
    assert b.trades() == [res]


def test_cryptobroker_futures_limit_order_mapping():
    b = CryptoBroker(exchange="binance", test=True, futures=True)
    b.connect()
    for otype in ("market", "limit", "stop-limit"):
        res = b.execute_buy("BTC/USDT", amount=0.001, price=60000.0, order_type=otype)
        assert res["status"] == "dry-run"
        assert res["type"] == otype
    b.disconnect()
