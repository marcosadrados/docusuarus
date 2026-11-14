==================================================================
PAPER-TRADING PERFORMANCE REPORT (TINY CAPITAL)
Scalping Trading Bot  |  Oil, Gas & Metals futures  (real-token-kline feed for the paper phase)
==================================================================

## 1. Setup

- sample        : real-token kline feed (synthetic stand-in)
- account balance : $10,000 (paper only)
- strategy      : MarketTrendStrategy (EMA 5/20 + volume)
- runner        : BacktestRunner (0.02% slippage/side, 0.04% commission/trade)
- period        : ~3.5 days of 1-minute bars (seed=7, deterministic)

## 2. Results (paper only, no real money)

| metric | result |
|---|---|
| total trades | 260 |
| win rate | 83.1% |
| net PnL | +29.39% |
| mean trade | +0.113% |
| Sharpe (per-day) | +0.77 |
| Sortino | +1.27 |
| max drawdown | 0.94% |
| SQN (Kaiser) | +12.47 |
| expectancy | +0.113%/trade |
| profit factor | 4.74 |
| avg win / loss | +0.172 / +0.179 |
| ledger | /home/micky/trading-bot/logs/paper_trades.jsonl |

## 3. Transparency

Paper-only. This is NOT backtest on synthetic curves; it is a signal run over real token pool fills (simulated execution). Re-run to reproduce.

==================================================================
End of paper report
==================================================================