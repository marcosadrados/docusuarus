"""Tests for the backtesting framework."""


from backtest.runner import BacktestRunner


def test_backtest_runner_exists():
    """Test that BacktestRunner can be instantiated."""
    runner = BacktestRunner(account_balance=10000.0)
    assert runner is not None
