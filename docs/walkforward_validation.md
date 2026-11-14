==================================================================
WALK-FORWARD / OUT-OF-SAMPLE VALIDATION REPORT
Scalping Trading Bot  |  Oil, Gas & Metals futures
==================================================================

## 1. Data

- synthetic, deterministic OHLCV panel (seed=42, power-law drift)
- rows: 5000  |  time horizon ~3.5 days
- runner: BacktestRunner (0.02% slippage/side, 0.04% commission/trade)
- walk-forward: rolling train/test, train_ratio 50%, step 20%
- metrics: Sharpe, win rate, max drawdown, mean net pnl

## 2. Out-of-sample fold results

| Strategy | folds | mean Sharpe | mean win rate | mean max DD | mean net pnl |
|---|---|---|---|---|---|

### market_trend

- folds: 3
- mean Sharpe: +1.65 (std +1.11)
- mean win rate: 90.1%
- mean max drawdown: -0.39%
- mean net pnl: +4.91%

Per-fold detail:

- Fold 1 (test rows 2500): Sharpe +0.73, win rate 83%, max DD -0.5%, pnl +8.00%
- Fold 2 (test rows 1500): Sharpe +1.01, win rate 87%, max DD -0.5%, pnl +4.87%
- Fold 3 (test rows 500): Sharpe +3.21, win rate 100%, max DD -0.2%, pnl +1.87%

## 3. Single-pass (in-sample curve fit) comparison

- total trades: 130
- Sharpe: +0.77
- win rate: 83.1%
- max drawdown: -0.5%
- net pnl: +14.70%

Single-pass trades on the full sample at once; it cannot reveal
instability across time. Compare its Sharpe / max-Drawdown against
the walk-forward folds above before trusting the curve.

==================================================================
End of walk-forward validation report
==================================================================