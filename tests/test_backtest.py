"""Tests for the backtesting framework."""

import pytest
from backtest.runner import BacktestRunner
from backtest.analyzer import Analyzer


def test_backtest_runner_exists():
    """Test that BacktestRunner can be instantiated."""
    runner = BacktestRunner(account_balance=10000.0)
    assert runner is not None
