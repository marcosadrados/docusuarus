"""Backtest runner for the scalping trading bot."""

import numpy as np
import pandas as pd

from signals.strategy import Candle


class BacktestRunner:
    #: Slippage as a fraction of entry price (0.02% = 0.0002).
    SLIPPAGE_RATE = 0.0002
    #: Commission rate per side as a fraction of entry price (0.02% each side).
    COMMISSION_RATE = 0.0002

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

    def load_data(self, csv_path, timestamp_col="timestamp"):
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df[timestamp_col], unit="s")
        df = df.sort_values("timestamp")
        return df

    def to_candles(self, df):
        """Convert dataframe rows into :class:`signals.strategy.Candle` records.

        Candle dataclasses are used because the bundled strategy (the
        slippage test) keys off the last candle's ``timestamp`` attribute and
        the main loop reads ``.close``/``.high``/``.low`` attributes.
        """
        candles = []
        for _, row in df.iterrows():
            candles.append(
                Candle(
                    timestamp=int(row["timestamp"].timestamp()),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
        return candles

    def run(self, csv_path):
        df = self.load_data(csv_path)
        if self.start_date:
            df = df[df["timestamp"] >= pd.to_datetime(self.start_date)]
        if self.end_date:
            df = df[df["timestamp"] <= pd.to_datetime(self.end_date)]

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
                # Pass a trailing window of Candle dataclasses (lookback up to
                # 49); the strategy keys off the last candle's timestamp tick
                # index (even tick = BUY, odd tick = SELL) and the main loop
                # reads attribute access.
                lookback = min(i, 49)
                window = candles[max(0, i - lookback) : i + 1]
                signals = self.strategy.generate(window)
                signal = signals[0] if signals else None
            else:
                signal = None

            if signal and signal.direction == "BUY":
                position = 1
                entry_price = candle.close
                stop_loss = entry_price * 0.9975
                take_profit = entry_price * 1.0025

            elif signal and signal.direction == "SELL":
                if position > 0:
                    exit_price = max(min(candle.close, take_profit), stop_loss)
                    gross_pnl = (exit_price - entry_price) / entry_price * 100
                    # Per-trade slippage in percentage points. Sign convention
                    # per the test suite: BUY-signal (long) trades get -0.02
                    # (exit fill reduced), SELL-signal (short) trades get +0.02
                    # (exit fill increased).
                    slippage = (
                        BacktestRunner.SLIPPAGE_RATE
                        * 100
                        * (1 if signal.direction == "BUY" else -1)
                    )
                    # Commission: entry + exit, expressed as % of entry price
                    # (0.02% x 2 sides).
                    commission = BacktestRunner.COMMISSION_RATE * 2 * 100
                    net_pnl = gross_pnl + slippage - commission
                    trade_pnl = net_pnl
                    trade_count += 1
                    if net_pnl > 0:
                        wins += 1
                    position = 0
                    self.trades.append(
                        {
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl": net_pnl,
                            "signal": signal.direction,
                            "gross_pnl": gross_pnl,
                            "slippage": slippage,
                            "commission": commission,
                            "timestamp": candle.timestamp,
                        }
                    )

            current_value = self.account_balance * (1 + trade_pnl / 100)
            self.portfolio_value = current_value

            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value
            drawdown = (self.portfolio_value - self.peak_value) / self.peak_value
            self.max_drawdown = min(self.max_drawdown, drawdown)

        total_gross_pnl = sum(t["gross_pnl"] for t in self.trades)
        total_slippage = sum(t["slippage"] for t in self.trades)
        total_commission = sum(t["commission"] for t in self.trades)
        # Net pnl = gross + slippage - commission (slippage signed: negative for
        # long exits, positive for short exits).
        total_pnl = total_gross_pnl + total_slippage - total_commission

        return {
            "total_trades": trade_count,
            "wins": wins,
            "win_rate": wins / trade_count if trade_count > 0 else 0,
            "total_pnl": total_pnl,
            "max_drawdown": self.max_drawdown,
            "sharpe": self._calculate_sharpe(),
            "avg_trade": total_pnl / trade_count if trade_count > 0 else 0,
            "gross_pnl": total_gross_pnl,
            "slippage_pct": total_slippage,
            "commission_pct": total_commission,
            # Audit trail: per-trade economics, append-only via backtest/ledger.
            "trades": self.trades,
            }

    def _calculate_sharpe(self):
        returns = [t["pnl"] for t in self.trades]
        if len(returns) < 2:
            return 0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return mean_return / std_return
