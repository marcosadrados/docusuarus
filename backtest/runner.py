"""Backtest runner for the scalping trading bot."""

import pandas as pd
import numpy as np


class BacktestRunner:
    def __init__(self, account_balance=10000.0, start_date=None, end_date=None):
        self.account_balance = account_balance
        self.start_date = start_date
        self.end_date = end_date
        self.strategy = None
        self.risk = None
        self.logger = None
        self.trades = []
        self.signals = []
        self.portfolio_value = account_balance
        self.daily_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_value = account_balance

    def set_strategy(self, strategy):
        self.strategy = strategy

    def set_risk(self, risk):
        self.risk = risk

    def set_logger(self, logger):
        self.logger = logger

    def load_data(self, csv_path, timestamp_col='timestamp'):
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df[timestamp_col], unit='s')
        df = df.sort_values('timestamp')
        return df

    def to_candles(self, df):
        candles = []
        for _, row in df.iterrows():
            candles.append({
                'timestamp': int(row['timestamp'].timestamp()),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        return candles

    def run(self, csv_path):
        df = self.load_data(csv_path)
        if self.start_date:
            df = df[df['timestamp'] >= pd.to_datetime(self.start_date)]
        if self.end_date:
            df = df[df['timestamp'] <= pd.to_datetime(self.end_date)]

        candles = self.to_candles(df)
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trade_pnl = 0
        trade_count = 0
        wins = 0

        for i, candle in enumerate(candles):
            if self.strategy:
                signal = self.strategy.generate([candle])[0] if i < len(candles) else None
            else:
                signal = None

            if signal and signal.get('direction') == 'BUY':
                position = 1
                entry_price = candle['close']
                stop_loss = entry_price * 0.9975
                take_profit = entry_price * 1.0025

            elif signal and signal.get('direction') == 'SELL':
                if position > 0:
                    exit_price = max(min(candle['close'], take_profit), stop_loss)
                    pnl = (exit_price - entry_price) / entry_price * 100
                    trade_pnl = pnl
                    trade_count += 1
                    if pnl > 0:
                        wins += 1
                    position = 0
                    self.trades.append({
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                        'signal': signal['direction']
                    })

            current_value = self.account_balance * (1 + trade_pnl / 100)
            self.portfolio_value = current_value

            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value
            drawdown = (self.portfolio_value - self.peak_value) / self.peak_value
            self.max_drawdown = min(self.max_drawdown, drawdown)

        return {
            'total_trades': trade_count,
            'wins': wins,
            'win_rate': wins / trade_count if trade_count > 0 else 0,
            'total_pnl': sum(t['pnl'] for t in self.trades),
            'max_drawdown': self.max_drawdown,
            'sharpe': self._calculate_sharpe(),
            'avg_trade': self.total_pnl / trade_count if trade_count > 0 else 0,
        }

    def _calculate_sharpe(self):
        returns = [t['pnl'] for t in self.trades]
        if len(returns) < 2:
            return 0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return mean_return / std_return
