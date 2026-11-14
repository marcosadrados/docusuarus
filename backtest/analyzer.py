"""Performance analyzer for backtest results."""



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
        if results["sharpe"] > 1.0:
            report.append("Sharpe ratio > 1.0: Good risk-adjusted returns")
        elif results["sharpe"] > 0.5:
            report.append("Sharpe ratio > 0.5: Acceptable risk-adjusted returns")
        else:
            report.append("Sharpe ratio < 0.5: Poor risk-adjusted returns")
        if results["win_rate"] > 0.5:
            report.append("Win rate > 50%: Consistent winning strategy")
        else:
            report.append("Win rate < 50%: High-risk strategy, needs validation")
        if results["max_drawdown"] < -0.03:
            report.append("Drawdown > 3%: Risk limits may be too loose")


    @staticmethod
    def analyze_comparative(results):
        """Compare backtest performance on the axes used by the biggest-win bots
        (SQN, expectancy, profit factor, monthly breakdown) rather than just
        Sharpe / drawdown / win rate."""
        from .metrics import compute

        trades = results.get('trades', []) or []
        if not trades:
            return '(no trades to compare)'

        m = compute(trades)
        lines = []
        lines.append('=' * 52)
        lines.append('COMPARATIVE METRICS (big-win bot axes)')
        lines.append('=' * 52)
        lines.append('Total trades:         %d' % m['total_trades'])
        lines.append('Win rate:             %.1f%%' % (m['win_rate'] * 100))
        lines.append('Net PnL:              %+.2f%%' % m['net_pnl'])
        lines.append('Expectancy (mean):    %+.3f%%/trade' % m['expectancy'])
        lines.append('Profit factor:        %.2f' % m['profit_factor'])
        lines.append('Profit factor / R:    %.2f' % m['profit_factor_per_r'])
        lines.append('SQN (Kaiser):         %.2f' % m['sqn'])
        lines.append('Sharpe (per-day):     %.2f' % m['sharpe'])
        lines.append('Sortino:              %.2f' % m['sortino'])
        lines.append('Max drawdown:         %.2f%%' % (m['max_drawdown'] * 100))
        lines.append('Avg win / loss:       %+.3f / %+.3f' % (m['avg_win'], m['avg_loss']))
        if m['monthly_breakdown']:
            seg = []
            for x in m['monthly_breakdown']:
                tag = 'WIN' if x['win_rate'] >= 0.5 else 'LOSS'
                seg.append('%s: %s %s%%' % (x['month'], tag, x['win_rate'] * 100))
            lines.append('Monthly: ' + ', '.join(seg))
        return chr(10).join(lines)

        return "\n".join(report)
