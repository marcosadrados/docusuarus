"""
Risk limit constants for the scalping trading bot.

Used by :mod:`risk.mgr` to enforce strict position sizing, daily loss,
and drawdown limits.

Markets: oil (WTI CL, Brent BZ), gas (NG), metals (GC, SI, HG) futures.
"""


#: Maximum risk per single trade as a fraction of account balance.
#: 0.25% of a $10,000 account = $25.
MAX_PER_TRADE = 0.0025

#: Daily loss limit as a fraction of account balance.
#: 1.5% of a $10,000 account = $150.
DAILY_LOSS_LIMIT = 0.015

#: Maximum drawdown as a fraction of account balance.
#: 3% of a $10,000 account = $300.
MAX_DRAWDOWN = 0.03
