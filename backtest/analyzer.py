"""Performance analyzer for backtest results."""

import pandas as pd
import numpy as np


class Analyzer:
    @staticmethod
    def analyze(results):
        report = []
        report.append("=" * 50)
        report.append("BACKTEST PERFORMANCE ANALYSIS")
        report.append("=" * 50)
        report.append(f"\nTotal trades:        {results['total_trades']}")
        report.append(f"Win rate:            {results['win_rate'] * 100:.1f}%")
        report.append(f"Total PnL:           {results['total_pnl']:.2f}%")
        report.append(f"Maximum drawdown:    {results['max_drawdown'] * 100:.2f}%")
        report.append(f"Sharpe ratio:        {results['sharpe']:.2f}")
        report.append(f"Average trade:       {results['avg_trade']:.2f}%")
        report.append("\n--- Interpretation ---")
        if results['sharpe'] > 1.0:
            report.append("Sharpe ratio > 1.0: Good risk-adjusted returns")
        elif results['sharpe'] > 0.5:
            report.append("Sharpe ratio > 0.5: Acceptable risk-adjusted returns")
        else:
            report.append("Sharpe ratio < 0.5: Poor risk-adjusted returns")
        if results['win_rate'] > 0.5:
            report.append("Win rate > 50%: Consistent winning strategy")
        else:
            report.append("Win rate < 50%: High-risk strategy, needs validation")
        if results['max_drawdown'] < -0.03:
            report.append("Drawdown > 3%: Risk limits may be too loose")
        return "\n".join(report)
