"""
Risk management engine for the scalping trading bot.

Enforces strict risk limits:

- Max per trade: 0.25% of account
- Daily loss limit: 1.5% of account
- Max drawdown: 3% of account

The ``RiskManager`` calculates position sizes that keep the round-trip risk
per trade within the configured limit, tracks cumulative daily P&L to raise
a daily-loss alert, and monitors drawdown against the peak equity.
"""

from .limits import MAX_PER_TRADE, DAILY_LOSS_LIMIT, MAX_DRAWDOWN


class RiskManager:
    """Manages risk limits and position sizing for the trading bot.

    Parameters
    ----------
    account_balance : float
        Current account balance (equity) in account currency.
    max_per_trade : float, optional
        Max risk per trade as a fraction of account. Defaults to
        :data:`risk.limits.MAX_PER_TRADE` (0.25%).
    daily_loss_limit : float, optional
        Daily loss limit as a fraction of account. Defaults to
        :data:`risk.limits.DAILY_LOSS_LIMIT` (1.5%).
    max_drawdown : float, optional
        Max drawdown as a fraction of account. Defaults to
        :data:`risk.limits.MAX_DRAWDOWN` (3%).
    """

    def __init__(self, account_balance, max_per_trade=None,
                 daily_loss_limit=None, max_drawdown=None):
        self.account_balance = float(account_balance)
        self.max_per_trade = (
            float(max_per_trade) if max_per_trade is not None else MAX_PER_TRADE
        )
        self.daily_loss_limit = (
            float(daily_loss_limit) if daily_loss_limit is not None
            else DAILY_LOSS_LIMIT
        )
        self.max_drawdown = (
            float(max_drawdown) if max_drawdown is not None else MAX_DRAWDOWN
        )

        # --- state tracked by the engine --------------------------- #
        self.daily_pnl = 0.0
        self.daily_loss_threshold = self.account_balance * self.daily_loss_limit
        self.peak_equity = self.account_balance
        self.current_equity = self.account_balance

    # ------------------------------------------------------------------ #
    # Position sizing
    # ------------------------------------------------------------------ #

    def calculate_position_size(self, signal, entry_price, stop_price):
        """Calculate the number of units to trade so the round-trip risk is
        capped at ``max_per_trade`` of the account balance.

        Parameters
        ----------
        signal : str
            Direction of the trade (``BUY`` or ``SELL``). The direction is
            not used in the size calculation; only the distance between the
            entry and stop determines the risk per unit.
        entry_price : float
            Entry price of the trade.
        stop_price : float
            Stop-out price of the trade.

        Returns
        -------
        float
            Position size in units (e.g. contracts).

        Raises
        ------
        ValueError
            If ``entry_price`` and ``stop_price`` are equal (zero risk
            per unit would divide by zero).
        """
        risk_per_unit = abs(float(entry_price) - float(stop_price))
        if risk_per_unit <= 0:
            raise ValueError(
                "entry_price and stop_price must differ; "
                "risk per unit is zero"
            )

        risk_budget = self.account_balance * self.max_per_trade
        return risk_budget / risk_per_unit

    # ------------------------------------------------------------------ #
    # Daily loss tracking
    # ------------------------------------------------------------------ #

    def record_trade(self, pnl):
        """Record the P&L of a closed trade and update daily loss tracking.

        Parameters
        ----------
        pnl : float
            Profit or loss of the closed trade (positive = win).
        """
        self.daily_pnl += float(pnl)

    def check_daily_loss(self):
        """Return True if the cumulative daily P&L has hit or exceeded the
        daily loss limit.

        A daily loss limit of 1.5% means the bot must stop trading for the
        day once its net daily P&L is at or below -1.5% of the account.

        Returns
        -------
        bool
            True if the daily loss limit has been hit.
        """
        return self.daily_pnl <= -self.daily_loss_threshold

    # ------------------------------------------------------------------ #
    # Max drawdown tracking
    # ------------------------------------------------------------------ #

    def update_equity(self, current_equity):
        """Update the current account equity and record the running peak.

        Parameters
        ----------
        current_equity : float
            Current account equity.

        Returns
        -------
        float
            Current drawdown as a fraction of the running peak equity.
        """
        current_equity = float(current_equity)
        self.current_equity = current_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        return (self.peak_equity - current_equity) / self.peak_equity

    def check_max_drawdown(self):
        """Return True if the current drawdown has hit or exceeded the
        maximum drawdown limit.

        A max drawdown limit of 3% means the bot must stop trading once
        equity falls 3% or more below its running peak.

        Returns
        -------
        bool
            True if the max drawdown limit has been hit.
        """
        drawdown = (
            self.peak_equity - self.current_equity
        ) / self.peak_equity
        return drawdown >= self.max_drawdown
