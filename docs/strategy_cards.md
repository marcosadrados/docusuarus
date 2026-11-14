====================================================================
STRATEGY CARDS INDEX
Scalping Trading Bot (Oil, Gas & Metals)
====================================================================

| preset | class | fit window | OOS | status |
|---|---|---|---|---|
| market_trend | MarketTrendStrategy | 2026-11-01..2026-11-10 | +1.65 | validated on sample |
| rule_simple | RuleStrategy | 2026-11-01..2026-11-10 | n/a | placeholder (no evidence yet) |
| stat_arb | StatArbStrategy | 2026-11-01..2026-11-07 | 0 trades | runnable; 0 entries on trending sample |
| freqai | Freqai | 2026-11-01..2026-11-07 | self-adaptive retrain harness | runnable (non-directional) |
| regime_adaptive | RegimeStrategy | 2026-11-01..2026-11-07 | regime state descriptor (HOLD) | runnable (non-directional) |

## Cards

### market_trend (MarketTrendStrategy)

| field | value |
|---|---|
| fit window | 2026-11-01..2026-11-10 |
| rule | EMA 5/20 momentum + volume |
| OOS | +1.65 |
| status | validated on sample |
| regime coverage | low_vol, trend, reversal |
| max trades/day | 48 |
| connection | cme (IC Markets cTrader) / binance |
| loadable | True |
| build key | market_trend |
| evidence | Seed=42 sample: OOS Sharpe +1.65, win rate 90.1%, max DD -0.39%, mean pnl +4.91% |

### rule_simple (RuleStrategy)

| field | value |
|---|---|
| fit window | 2026-11-01..2026-11-10 |
| rule | simple DSL rule set (compiled) |
| OOS | n/a |
| status | placeholder (no evidence yet) |
| regime coverage | low_vol, trend, reversal |
| max trades/day | 48 |
| connection | cme (IC Markets cTrader) / binance |
| loadable | False |
| build key | None |
| evidence | Placeholder template; not yet evaluated for OOS directional performance |

### stat_arb (StatArbStrategy)

| field | value |
|---|---|
| fit window | 2026-11-01..2026-11-07 |
| rule | pairs_stat_arb |
| OOS | 0 trades |
| status | runnable; 0 entries on trending sample |
| regime coverage | low_vol, trend, reversal |
| max trades/day | 48 |
| connection | cme (IC Markets cTrader) / binance |
| loadable | True |
| build key | stat_arb |
| evidence | Stat-arb mean-reversion needs a mean-reverting pair spread; trending synthetic sample yields 0 entries (pair fit required) |

### freqai (Freqai)

| field | value |
|---|---|
| fit window | 2026-11-01..2026-11-07 |
| rule | self-adaptive ML retrain harness |
| OOS | self-adaptive retrain harness |
| status | runnable (non-directional) |
| regime coverage | low_vol, trend, reversal |
| max trades/day | 48 |
| connection | cme (IC Markets cTrader) / binance |
| loadable | True |
| build key | freqai |
| evidence | Retrain cadence (window/retrain_every) + offline feature-selection evaluation; no fixed directional OOS stats |

### regime_adaptive (RegimeStrategy)

| field | value |
|---|---|
| fit window | 2026-11-01..2026-11-07 |
| rule | Gaussian-mixture 3-state regime detector |
| OOS | regime state descriptor (HOLD) |
| status | runnable (non-directional) |
| regime coverage | low_vol, trend, reversal |
| max trades/day | 48 |
| connection | cme (IC Markets cTrader) / binance |
| loadable | True |
| build key | regime_adaptive |
| evidence | Regime distribution over sample; emits HOLD state descriptor (regime field), consumed by the ensemble as a state signal |

====================================================================
End of strategy cards
====================================================================