# Scalping Trading Bot — Oil, Gas & Metals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a scalping trading bot for oil, gas, and metals futures that scans markets, generates signals, enforces strict risk limits, and executes trades automatically.

**Architecture:** Three-layer architecture — (1) Data Acquisition (real-time market data via futures APIs), (2) Signal Engine (scalping strategy detection with volume/elasticity filters), (3) Risk & Execution (position sizing, stop-loss, take-profit, broker API execution).

**Tech Stack:** Python 3.11+, CCXT or broker-specific API client, Pandas/Ta-Lib for technical analysis, SQLAlchemy for historical data storage, Redis for real-time state, Docker for deployment.

**Spec:** [TradingBot implementation spec] (placeholder) — the plan argues from this spec.

## Global Constraints

- Markets: Oil (WTI CL, Brent BZ), Gas (NG), Metals (GC, SI, HG) futures
- Strategy: Scalping — entries within seconds, holds < 5 minutes
- Risk: Max 0.25% account per trade, 1.5% daily loss limit, 3% max drawdown
- Must validate on historical data before any live trading
- All trades logged with full audit trail
- Use subagent-driven-development for implementation

## Review Focus

- Position sizing calculations (round-trip risk per trade)
- Slippage and commission modeling in backtests
- Overnight position handling (achy futures)
- Rate limiting and API throttling
- Data integrity (missing ticks, stale data)

---

### Task 1: Data Acquisition Layer

**Files:**
- Create: `tradingbot/data/fetcher.py`
- Create: `tradingbot/data/store.py`
- Modify: `tradingbot/config.yaml:1-10`

**Interfaces:**
- Consumes: None (standalone)
- Produces: `RawCandle` objects (OHLCV + volume + timestamp) → stored in SQLite

- [ ] **Step 1: Write the failing test**
```python
def test_fetch_wti_candles_returns_ohlcv():
    fetcher = WFuturesFetcher(api_key="test")
    candles = fetcher.fetch_candles("CL=F", "1m", since="2026-11-10")
    assert len(candles) > 0
    assert all(c["timestamp"] for c in candles)
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_fetcher.py::test_fetch_wti_candles_returns_ohlcv` — Expected: FAIL with "function not defined"
- [ ] **Step 3: Implement WFuturesFetcher in `tradingbot/data/fetcher.py`** — Use CCXT or broker API; implement `fetch_candles(symbol, timeframe, since)`
- [ ] **Step 4: Run test to verify it passes** — Expected: PASS
- [ ] **Step 5: Commit** — `git add tradingbot/data/fetcher.py tests/test_fetcher.py && git commit -m "feat: add WTI futures data fetcher"`
- [ ] **Step 6: Test with real data** — Fetch last 2 hours of CL=F (WTI) 1m candles; verify data integrity

### Task 2: Signal Engine

**Files:**
- Create: `tradingbot/signals/strategy.py`
- Create: `tradingbot/signals/engine.py`

**Interfaces:**
- Consumes: RawCandle objects (from Task 1)
- Produces: Signal objects (BUY, SELL, HOLD) with confidence score

- [ ] **Step 1: Write the failing test**
```python
def test_momentum_signal_fires_on_breakout():
    candles = load_sample_candles()  # from test fixtures
    engine = MomentumStrategy()
    signal = engine.generate(candles)
    assert signal in [Signal.BUY, Signal.SELL, Signal.HOLD]
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL
- [ ] **Step 3: Implement MarketTrendStrategy in `tradingbot/signals/strategy.py`** — Rule: 5-min close > 20-min EMA AND volume > 50th percentile; or mean reversion: price > 2 std dev from 50-period MA → sell
- [ ] **Step 4: Run test to verify it passes** — Expected: PASS
- [ ] **Step 5: Commit** — `git add tradingbot/signals && git commit -m "feat: add scalping signal strategy"`
- [ ] **Step 6: Validate on 1 month of data** — Run on historical CL=F data; verify signals fire at reasonable frequency

### Task 3: Risk Engine

**Files:**
- Create: `tradingbot/risk/mgr.py`
- Create: `tradingbot/risk/limits.py`

**Interfaces:**
- Consumes: Signal objects, account balance
- Produces: Validated trade (or None if risk check fails)

- [ ] **Step 1: Write the failing test**
```python
def test_position_sizing_respect_025_percent_limit():
    risk = RiskManager(account_balance=10000, max_per_trade=0.0025)
    signal = Signal.BUY
    size = risk.calculate_position_size(signal, entry_price=80.0, stop=79.5)
    assert size <= 1000  # 0.25% of 10k = 25, at 80.0/0.25=1000
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL
- [ ] **Step 3: Implement RiskManager with position sizing, daily loss tracking, max drawdown check** — `calculate_position_size()`, `check_daily_loss()`, `check_max_drawdown()`
- [ ] **Step 4: Run test to verify it passes** — Expected: PASS
- [ ] **Step 5: Commit** — `git add tradingbot/risk && git commit -m "feat: add risk management engine"`
- [ ] **Step 6: Test with simulated account** — Verify 1.5% daily loss limit triggers alert

### Task 4: Execution Layer

**Files:**
- Create: `tradingbot/exec/broker.py`
- Create: `tradingbot/exec/client.py`

**Interfaces:**
- Consumes: Validated trade objects (from Task 3)
- Produces: Execution results (filled/partially filled/failed)

- [ ] **Step 1: Write the failing test**
```python
def test_broker_execute_returns_order_id():
    broker = IBKRBroker(api_key="test", secret="test")
    result = broker.execute_sell("CL=F", size=100, limit_price=80.5)
    assert result["order_id"] is not None
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL
- [ ] **Step 3: Implement IBKRBroker or CCXT client for your broker** — Implement `execute_buy()`, `execute_sell()`, `cancel_order()`, `get_positions()`
- [ ] **Step 4: Run test to verify it passes** — Expected: PASS
- [ ] **Step 5: Commit** — `git add tradingbot/exec && git commit -m "feat: add broker execution layer"`
- [ ] **Step 6: Test paper trading** — Execute paper trades on demo account; verify fills

### Task 5: Backtesting Framework

**Files:**
- Create: `tradingbot/backtest/runner.py`
- Create: `tradingbot/backtest/analyzer.py`

**Interfaces:**
- Consumes: Historical data (from Task 1) + signals (from Task 2)
- Produces: Performance report (returns, Sharpe, max drawdown, win rate, total trades)

- [ ] **Step 1: Write the failing test**
```python
def test_backtest_returns_positive_sharpe():
    results = run_backtest(symbol="CL=F", start="2026-10-01", end="2026-11-01")
    assert results["sharpe_ratio"] > 0
    assert results["total_trades"] > 0
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL
- [ ] **Step 3: Implement BacktestRunner** — Walk through historical candles, apply signals, simulate fills with slippage, calculate P&L
- [ ] **Step 4: Run test to verify it passes** — Expected: PASS
- [ ] **Step 5: Commit** — `git add tradingbot/backtest && git commit -m "feat: add backtesting framework"`
- [ ] **Step 6: Backtest on 3 months of CL=F, NG, GC data** — Verify positive Sharpe ratio and acceptable drawdown

### Task 6: Live Trading Orchestrator

**Files:**
- Create: `tradingbot/orchestrator/main.py`
- Create: `tradingbot/orchestrator/gateway.py`

**Interfaces:**
- Consumes: Signals (from Task 2) + Risk validation (from Task 3) + Data (from Task 1)
- Produces: Executed trades, real-time metrics, alerts

- [ ] **Step 1: Write the failing test**
```python
def test_orchestrator_runs_continuously():
    orch = Orchestrator(broker="paper", risk_limits={"max_daily_loss": 0.015})
    orch.start()
    # Give it 5 seconds to run
    time.sleep(5)
    orch.stop()
    assert orch.analyses_since_start >= 0
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL
- [ ] **Step 3: Implement Orchestrator** — Main loop: fetch data → generate signals → risk check → execute → log
- [ ] **Step 4: Run test to verify it passes** — Expected: PASS
- [ ] **Step 5: Commit** — `git add tradingbot/orchestrator && git commit -m "feat: add live trading orchestrator"`
- [ ] **Step 6: Run on paper trading with real data** — Verify it runs continuously without crashing

### Task 7: Monitoring and Alerting

**Files:**
- Create: `tradingbot/monitoring/logger.py`
- Create: `tradingbot/monitoring/alerts.py`

**Interfaces:**
- Consumes: Execution results, performance metrics
- Produces: Log files, alert notifications (email, Slack, webhook)

- [ ] **Step 1: Write the failing test**
```python
def test_alerts_fire_on_drawdown():
    alerts = Alerts(slack_webhook="https://hooks.slack.com/test")
    alerts.on_drawdown(drawdown=0.05)  # 5% drawdown
    assert alerts.was_fired
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL
- [ ] **Step 3: Implement Monitoring** — Log all trades, signal events, risk checks; alert on 3% daily loss, 5% max drawdown, API errors
- [ ] **Step 4: Run test to verify it passes** — Expected: PASS
- [ ] **Step 5: Commit** — `git add tradingbot/monitoring && git commit -m "feat: add monitoring and alerting"`
- [ ] **Step 6: Test with live paper trading** — Verify logs appear and alerts fire correctly

---

**Review Focus:** Position sizing, slippage, overnight positions, API rate limits, data integrity

---

**Execution method:** Subagent-driven (each task dispatched to a fresh subagent with clear specs, reviewed by you before proceeding to the next).
