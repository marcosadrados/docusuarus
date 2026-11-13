# Scalping Trading Bot for Oil, Gas & Metals

A scalable trading bot for scalping futures markets (WTI, Brent, Natural Gas, Gold, Silver, Copper).

## Architecture
- data/ - Data acquisition (fetch historical + real-time)
- signals/ - Signal generation (scalping strategies)
- risk/ - Risk management (position sizing, loss limits)
- exec/ - Execution layer (broker API)
- backtest/ - Backtesting framework
- orchestrator/ - Main trading loop
- monitoring/ - Logging, alerts, telemetry

## Risk Parameters
- Max per trade: 0.25% of account
- Daily loss limit: 1.5% of account
- Max drawdown: 3%

## Setup
- Create config.yaml with API keys
- Install dependencies: pip install -r requirements.txt
- Run backtests before live trading

## Setup
# Scalping Trading Bot

A scalable trading bot for scalping futures markets (WTI, Brent, Natural Gas, Gold, Silver, Copper).

## Architecture
- data/ - Data acquisition (fetch historical + real-time)
- signals/ - Signal generation (scalping strategies)
- risk/ - Risk management (position sizing, loss limits)
- exec/ - Execution layer (broker API)
- backtest/ - Backtesting framework
- orchestrator/ - Main trading loop
- monitoring/ - Logging, alerts, telemetry

## Risk Parameters
- Max per trade: 0.25% of account
- Daily loss limit: 1.5% of account
- Max drawdown: 3%

## Setup
- Create config.yaml with API keys
- Install dependencies: pip install -r requirements.txt
- Run backtests before live trading
