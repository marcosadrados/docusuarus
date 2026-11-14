"""Live trading orchestrator for the scalping trading bot."""

import logging
from typing import Dict, cast

from monitoring.control_server import ControlServer
from monitoring.logger import TradeLogger
from monitoring.notifier import Notifier
from orchestrator.engine import Engine, Symbol
from risk.mgr import RiskManager
from signals.ensembler import EnsemblePreset
from signals.regime import RegimeStrategy, RegimeSwitcher
from signals.registry import get, load_strategy_preset
from signals.rules import RuleStrategy


class Orchestrator:
    SYMBOL_LOT_SIZES = {
        "CL=F": 1000,  # WTI crude
        "BZ=F": 1000,  # Brent
        "NG=F": 10000,  # natural gas
        "GC=F": 100,  # gold
        "HG=F": 250,  # copper
        "SI=F": 50,  # silver
        "XAU=F": 100,  # gold spot
        "XAG=F": 5000,  # silver
    }

    #: Crypto per-contract lot size (1.0 = 1 contract of the quoted asset).
    CRYPTO_LOT_SIZES = {
        "BTC/USDT": 1.0,
        "ETH/USDT": 1.0,
        "SOL/USDT": 1.0,
        "XRP/USDT": 1.0,
        "BNB/USDT": 1.0,
        "ADA/USDT": 1.0,
        "DOGE/USDT": 1.0,
        "AVAX/USDT": 1.0,
    }

    #: Connector id -> data-source class.
    CONNECTOR_FETCHERS = {
        "cme": "data.fetcher.WFuturesFetcher",
        "binance": "data.fetcher.CryptoFetcher",
        "bybit": "data.fetcher.CryptoFetcher",
        "okx": "data.fetcher.CryptoFetcher",
    }

    #: Connector id -> broker class.
    CONNECTOR_BROKERS = {
        "cme": "exec.broker.Broker",
        "binance": "exec.broker.CryptoBroker",
        "bybit": "exec.broker.CryptoBroker",
        "okx": "exec.broker.CryptoBroker",
    }

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = TradeLogger()
        self.signals = []
        self.trades = []
        self.running = False

        self._build_sources()
        strat_config = self.config.get("strategy", "market_trend")

        # Resolve the stat-arb pairs (lead / relate / beta / window) and the
        # relate-asset price anchor from config so the spread can be
        # constructed even before a candle arrives for the relate asset.
        pairs_cfg = self.config.get("stat_arb", {}) or {}
        pairs = pairs_cfg.get("pairs", [])
        relate_prices_cfg = pairs_cfg.get("relate_prices") or {}
        relate_prices: Dict[str, float] = {
            str(k): float(v) for k, v in relate_prices_cfg.items()
        }

        # -------------------------------------------------
        # Strategy selection: single preset OR ensemble
        # -------------------------------------------------
        ensemble_cfg = self.config.get("ensemble") or {}
        regime = None
        if ensemble_cfg.get("enabled", False):
            # Construct the regime model first so it can be handed to the
            # regime-aware child (and the ensemble).
            regime_cfg = self.config.get("regime") or {}
            regime_kwargs = {
                k: regime_cfg[k]
                for k in ("n_states", "n_iter", "tol", "random_state")
                if k in regime_cfg
            }
            regime = RegimeSwitcher(**regime_kwargs) if regime_cfg else None

            children = []
            for child_cfg in ensemble_cfg.get("children", []):
                name = child_cfg["name"]
                weight = float(child_cfg.get("weight", 0.25))
                if name == "regime_adaptive":
                    # regime is guaranteed non-None in the ensemble branch
                    children.append(
                        (RegimeStrategy(cast(RegimeSwitcher, regime)), weight)
                    )
                else:
                    children.append(
                        (
                            load_strategy_preset(
                                name, pairs=pairs, relate_prices=relate_prices
                            ),
                            weight,
                        ),
                    )
            self.strategy = EnsemblePreset(
                children=children,
                regime=regime,
                regime_selectivity=ensemble_cfg.get("regime_selectivity", 0.9),
                hold_threshold=ensemble_cfg.get("hold_threshold", 0.25),
            )
        else:
            # Single preset (not an ensemble): pass stat-arb arguments
            # only to the stat-arb strategy class itself.
            strat_cls = get(strat_config)
            if strat_cls is None:
                # Unknown strategy name: fall back to RuleStrategy so the
                # bot keeps running, and never silently returns None.
                strat_cls = RuleStrategy
            if strat_cls.__name__ == "StatArbStrategy":
                self.strategy = strat_cls(pairs=pairs, relate_prices=relate_prices)
            else:
                # Contract: Orchestrator.strategy is always an *instance* so
                # the control surface can read it uniformly via
                # strategy.__class__.__name__.
                self.strategy = strat_cls()
        self.risk = RiskManager(
            account_balance=10000,
            max_per_trade=0.0025,
            daily_loss_limit=0.015,
            max_drawdown=0.03,
        )
        self.setup_logging()
        self._engine = Engine(
            self.symbols,
            self.risk,
            self.fetcher_map["cme"],
            self.strategy,
            connector_map=self.connector_map,
            fetcher_map=self.fetcher_map,
            broker_map=self.broker_map,
            lot_sizes=self.lot_sizes,
            config=self.config,
        )

    # ------------------------------------------------------------------ #
    # Source construction
    # ------------------------------------------------------------------ #

    def _build_sources(self):
        config = self.config

        connector_map = dict(config.get("connector_map") or {})
        crypto_markets = config.get("crypto_markets") or []
        markets = config.get("markets") or []

        for m in markets:
            connector_map.setdefault(str(m).upper(), "cme")
        for m in crypto_markets:
            connector_map.setdefault(str(m).upper(), "binance")

        if not connector_map:
            # Config lists no markets: default to a minimal futures basket so
            # the orchestrator still constructs and the engine stays finite.
            connector_map = {"CL=F": "cme", "GC=F": "cme"}

        self.connector_map = {
            str(k).upper(): self._canonical_connector(v)
            for k, v in connector_map.items()
        }

        self.connector_map = connector_map

        self.fetcher_map = {}
        self.broker_map = {}
        self.lot_sizes = dict(self.SYMBOL_LOT_SIZES)
        self.lot_sizes.update(self.CRYPTO_LOT_SIZES)

        for code, connector in self.connector_map.items():
            fetcher_cls_name = self.CONNECTOR_FETCHERS.get(
                connector, "data.fetcher.WFuturesFetcher"
            )
            broker_cls_name = self.CONNECTOR_BROKERS.get(
                connector, "exec.broker.Broker"
            )
            self.fetcher_map[connector] = self._import(fetcher_cls_name)()
            self.broker_map[connector] = self._import(broker_cls_name)(config=config)

    @staticmethod
    def _canonical_connector(value):
        value = str(value).lower()
        if value in ("cme", "ctrader", "cTrader"):
            return "cme"
        if value in ("binance", "bybit", "okx"):
            return value
        return "cme"

    @staticmethod
    def _import(dotted):
        from importlib import import_module

        mod, _, cls = dotted.rpartition(".")
        return getattr(import_module(mod), cls)

    @property
    def symbols(self):
        codes = []
        for m in self.config.get("markets") or []:
            codes.append(str(m).upper())
        for m in self.config.get("crypto_markets") or []:
            codes.append(str(m).upper())
        return [Symbol(code=c, connector=self.connector_map[c]) for c in codes]

    def _lot_size(self, symbol_code):
        return self.lot_sizes.get(str(symbol_code).upper(), 1.0)

    def setup_logging(self):
        log_level = self.config.get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("logs/trading.log"), logging.StreamHandler()],
        )
        self.logger_obj = logging.getLogger(__name__)

    def start(self):
        self.running = True
        self.logger_obj.info("Trading loop started")
        try:
            self._engine.start()
        except Exception as exc:
            self.logger_obj.error(f"Failed to start engine: {exc}")
            raise

    def stop(self):
        self.running = False
        self._engine.stop()
        self.logger_obj.info("Trading loop stopped")


class Gateway:
    def __init__(self, config=None):
        self.config = config or {}
        self.orchestrator = None
        self.notifier = None
        self.control_server = None

    def start(self):
        self.orchestrator = Orchestrator(self.config)
        self.notifier = Notifier.from_config(self.config)
        if self.orchestrator and self.notifier:
            self.notifier.subscribe(self.orchestrator._engine)
        if self.config.get("control_port"):
            self.control_server = ControlServer(
                orchestrator=self,
                port=int(self.config.get("control_port") or 0),
            )
            self.control_server.start()
        self.orchestrator.start()

    def stop(self):
        if self.control_server:
            self.control_server.shutdown()
            self.control_server = None
        if self.notifier:
            self.notifier.unsubscribe(self.orchestrator._engine)
            self.notifier = None
        if self.orchestrator:
            self.orchestrator.stop()
