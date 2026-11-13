"""Live trading orchestrator for the scalping trading bot."""

import time
import logging
from signals.strategy import MarketTrendStrategy
from risk.mgr import RiskManager
from data.fetcher import WFuturesFetcher
from exec.broker import Broker
from monitoring.logger import TradeLogger


class Orchestrator:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = TradeLogger()
        self.signals = []
        self.trades = []
        self.running = False
        self.fetcher = WFuturesFetcher()
        self.broker = Broker(config=self.config)
        self.strategy = MarketTrendStrategy()
        self.risk = RiskManager(account_balance=10000,
                max_per_trade=0.0025,
                daily_loss_limit=0.015,
                max_drawdown=0.03)
        self.setup_logging()

    def setup_logging(self):
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger_obj = logging.getLogger(__name__)

    def start(self):
        self.running = True
        self.logger_obj.info("Trading loop started")
        try:
            self.broker.connect()
        except Exception as exc:
            self.logger_obj.error(f"Failed to connect cTrader broker: {exc}")
            raise
        while self.running:
            try:
                self._tick()
                time.sleep(self.config.get('tick_interval', 5))
            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                self.logger_obj.error(f"Tick error: {e}")
                time.sleep(5)

    def _tick(self):
        self.logger_obj.info("Fetching data...")
        candles = self.fetcher.fetch_candles('CL=F', '1m', limit=100)
        self.logger_obj.info("Generating signals...")
        signals = self.strategy.generate(candles)
        self.signals.extend(signals)
        self.logger_obj.info("Checking risk...")
        risk_check = self.risk.check_daily_loss()
        if risk_check['daily_loss_limit'] and risk_check['daily_loss'] > 0.015:
            self.logger_obj.warning("Daily loss limit reached")
            return
        for signal in signals:
            self.logger.log_signal(
                symbol='CL=F',
                signal=signal['direction'],
                confidence=0.7,
                direction=signal.get('direction', 'BUY')
            )
            if signal['direction'] == 'BUY':
                self._execute_buy(signal, candles[0] if candles else None)
            elif signal['direction'] == 'SELL':
                self._execute_sell(signal, candles[0] if candles else None)
        self.logger.log_signals(signals)
        self.logger.log_trades(self.trades)

    def _execute_buy(self, signal, candle):
        try:
            order = self.broker.execute_buy(symbol='CL=F', amount=1.0, order_type='market')
            self.trades.append({
                'type': 'buy', 'symbol': 'CL=F', 'status': 'filled',
                'order_id': order.get('id'), 'timestamp': order.get('timestamp')
            })
            self.logger_obj.info(f"Buy order executed: {order.get('id')}")
        except Exception as e:
            self.logger_obj.error(f"Buy order failed: {e}")

    def _execute_sell(self, signal, candle):
        try:
            order = self.broker.execute_sell(symbol='CL=F', amount=1.0, order_type='market')
            self.trades.append({
                'type': 'sell', 'symbol': 'CL=F', 'status': 'filled',
                'order_id': order.get('id'), 'timestamp': order.get('timestamp')
            })
            self.logger_obj.info(f"Sell order executed: {order.get('id')}")
        except Exception as e:
            self.logger_obj.error(f"Sell order failed: {e}")

    def stop(self):
        self.running = False
        self.fetcher.close()
        self.broker.close()
        self.logger_obj.info("Trading loop stopped")


class Gateway:
    def __init__(self, config=None):
        self.config = config or {}
        self.orchestrator = None

    def start(self):
        self.orchestrator = Orchestrator(self.config)
        self.orchestrator.start()

    def stop(self):
        if self.orchestrator:
            self.orchestrator.stop()
