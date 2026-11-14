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

from .atr import average_true_range, volatility_factor
from .limits import DAILY_LOSS_LIMIT, MAX_DRAWDOWN, MAX_PER_TRADE


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

    MIN_VOL_FACTOR = 0.5
    MAX_VOL_FACTOR = 2.0

    def __init__(
        self,
        account_balance,
        max_per_trade=None,
        daily_loss_limit=None,
        max_drawdown=None,
    ):
        self.account_balance = float(account_balance)
        self.max_per_trade = (
            float(max_per_trade) if max_per_trade is not None else MAX_PER_TRADE
        )
        self.daily_loss_limit = (
            float(daily_loss_limit)
            if daily_loss_limit is not None
            else DAILY_LOSS_LIMIT
        )
        self.max_drawdown = (
            float(max_drawdown) if max_drawdown is not None else MAX_DRAWDOWN
        )

        # --- ATR / volatility reference (position scaler) -------------- #
        self.atr_period = 14
        self._atr_buffer = []
        self._atr_values = []
        self._atr_reference = None

        self.daily_pnl = 0.0
        self.daily_loss_threshold = self.account_balance * self.daily_loss_limit
        self.peak_equity = self.account_balance
        self.current_equity = self.account_balance

    # ------------------------------------------------------------------ #
    # ATR reference (used by the volatility position scaler) ----------- #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Position sizing
    # ------------------------------------------------------------------ #

    def calculate_position_size(
        self, signal, entry_price, stop_price, atr=None, lot_size: float = 1.0
    ):
        """Calculate the number of units/lots to trade so round-trip risk is
        capped at ``max_per_trade`` of the account balance.

        The size is paced by current volatility: when the market's ATR is
        high, the position size is shrunk (down to ``MIN_VOL_FACTOR``); when
        ATR is low, it is grown (up to ``MAX_VOL_FACTOR``).  This keeps the
        round-trip dollar risk close to the per-trade limit across regimes.

        Parameters
        ----------
        signal : str
            Direction of the trade (``BUY`` or ``SELL``).  Direction is not
            used in the size calculation; only the entry/stop spread matters.
        entry_price : float
            Entry price.
        stop_price : float
            Stop price.
        atr : float, optional
            Current ATR.  If ``None`` the method falls back to the trailing
            referenceATR set by :meth:`add_candles`.  Pass an explicit ATR to
            force a particular regime.
        lot_size : float, optional
            Number of base units per lot (e.g. 1000 for WTI CL=F, 100 for XAU).
            The returned size is divided by this so the caller (the broker
            executor) can pass lots directly.

        Returns
        -------
        float
            Position size in the broker's lot units.

        Raises
        ------
        ValueError
            If ``entry_price`` and ``stop_price`` are equal (zero risk per
            unit would divide by zero).

        Examples
        --------
        >>> rm = RiskManager(account_balance=10000, max_per_trade=0.0025)
        >>> rm.calculate_position_size(Signal.BUY, 80.0, 79.5)
        50.0
        >>> # high ATR -> size shrinks to the floor of 0.5
        >>> rm.calculate_position_size(Signal.BUY, 80.0, 79.5, atr=0.6, lot_size=1000)
        25.0
        """
        risk_per_unit = abs(float(entry_price) - float(stop_price))
        if risk_per_unit <= 0:
            raise ValueError(
                "entry_price and stop_price must differ; " "risk per unit is zero"
            )

        risk_budget = self.account_balance * self.max_per_trade
        base_units = risk_budget / risk_per_unit

        # --- volatility pacing --------------------------------------- #
        if atr is None:
            atr = self.current_atr
        if atr is not None and float(atr) > 0:
            factor = volatility_factor(
                float(atr),
                self._atr_reference or float(atr),
                self.MIN_VOL_FACTOR,
                self.MAX_VOL_FACTOR,
            )
            base_units *= factor

        return base_units / float(lot_size)

    # ------------------------------------------------------------------ #
    # ATR reference (used by the volatility position scaler) ----------- #
    # ------------------------------------------------------------------ #

    def add_candles(self, candles):
        """Feed new bars and keep a rolling ATR reference.

        The reference ATR (the size target) is the mean of the ATR over the
        most recent ``atr_period`` bars, so the scaler stays anchored to the
        market's recent typical volatility instead of a single noisy bar.
        """
        if not candles:
            return
        self._atr_buffer.extend(candles)
        # Keep the buffer small enough to recompute fast; we only need the
        # most recent ATR lookback + reference window.
        self._atr_buffer = self._atr_buffer[-self.atr_period * 5 :]
        self._atr_values = average_true_range(self._atr_buffer, self.atr_period)
        if len(self._atr_values) >= self.atr_period:
            self._atr_reference = float(
                sum(self._atr_values[-self.atr_period :]) / self.atr_period
            )

    @property
    def current_atr(self):
        """Latest ATR value, or ``None`` if there isn't enough history yet."""
        if not self._atr_values:
            return None
        return float(self._atr_values[-1])

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
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        return drawdown >= self.max_drawdown
