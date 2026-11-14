"""Walk-forward validation demo for the scalping bot's backtest.

Generates a small synthetic OHLCV CSV (trend + noise around 80), runs
:class:`backtest.walkforward.evaluate_walkforward`, and prints the per-fold
and aggregate metrics in a clean, readable table.

The sample is ~200 rows so that every fold keeps its minimum train/test rows
while still producing in-window signals for the strategy's 50-period
lookback horizon.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtest.walkforward import evaluate_walkforward  # noqa: E402

# Walk-forward settings that give a handful of non-trivial folds on a
# ~200-row synthetic sample.  The strategy's 50-period lookback means a
# test window needs at least ~60 rows to produce signals mid-window.
PARAMS = dict(
    train_ratio=0.55,
    step=0.20,
    min_train=70,
    min_test=50,
)


def make_synthetic_csv(path, n=300, seed=7, base=80.0, drift=0.012, noise=0.6):
    """Write a synthetic uptrend + noise OHLCV CSV and return the DataFrame."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Underlying trend with small random steps plus occasional mean-reversion
    # spikes so the strategy's SELL (mean-reversion) rule fires too.
    level = base + drift * t + rng.normal(0.0, noise, n)
    spikes = rng.random(n) < 0.2
    level[spikes] += rng.choice([7.0, 4.0, 2.0], size=int(spikes.sum()))
    price = np.clip(level, base - 7.0, base + 9.0)

    opens = np.empty(n)
    opens[0] = price[0]
    opens[1:] = price[:-1]

    highs = np.maximum(opens, price) + rng.uniform(0.03, 0.14, n)
    lows = np.minimum(opens, price) - rng.uniform(0.03, 0.14, n)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": price,
            "volume": rng.uniform(90.0, 220.0, n),
        }
    )
    df.to_csv(path, index=False)
    return df


def main():
    tmpdir = tempfile.TemporaryDirectory()
    csv_path = os.path.join(tmpdir.name, "synthetic.csv")
    df = make_synthetic_csv(csv_path)

    print("=" * 78)
    print("WALK-FORWARD VALIDATION  (deterministic strategy, no fitting)")
    print("=" * 78)
    print(f"rows        : {len(df)}")
    print(
        f"params      : train_ratio={PARAMS['train_ratio']}, "
        f"step={PARAMS['step']}, min_train={PARAMS['min_train']}, "
        f"min_test={PARAMS['min_test']}"
    )
    print()

    result = evaluate_walkforward(csv_path, **PARAMS)

    header = (
        f"{'fold':>4} | {'train window':>17} | {'test window':>17} | "
        f"{'trades':>6} | {'win%':>7} | {'pnl%':>8} | {'sharpe':>8} | "
        f"{'drawdown%':>11}"
    )
    print(header)
    print("-" * 78)
    for fold in result["folds"]:
        train = f"{fold['train_rows']}r-{fold['train_end']:>3}"
        test = f"{fold['test_rows']}r-{fold['test_end']:>3}"
        print(
            f"{fold['fold']:>4} | {train:>17} | {test:>17} | "
            f"{fold['total_trades']:>6} | {fold['win_rate']*100:>7.1f}% | "
            f"{fold['total_pnl']:+8.2f}% | {fold['sharpe']:+8.2f} | "
            f"{fold['max_drawdown']*100:>11.2f}%"
        )
    print("-" * 78)
    total_trades = sum(f["total_trades"] for f in result["folds"])
    print(
        f"{'total':>4} | {'':>17} | {'':>17} | {total_trades:>6} | "
        f"{result['mean_win_rate']*100:>7.1f}% | {result['mean_pnl']:+8.2f}% | "
        f"{result['mean_sharpe']:+8.2f} | "
        f"{result['mean_max_drawdown']*100:>11.2f}%"
    )
    print()

    print("Aggregate (across folds):")
    print(
        f"  mean_sharpe   = {result['mean_sharpe']:+.3f}  +/- "
        f"{result['std_sharpe']:.3f}"
    )
    print(
        f"  mean_win_rate = {result['mean_win_rate']*100:+.1f}%  +/- "
        f"{result['std_win_rate']*100:.1f}%"
    )
    print(f"  mean_pnl      = {result['mean_pnl']:+.2f}%")
    print(f"  mean_drawdown = {result['mean_max_drawdown']*100:+.2f}%")
    print(f"  n_folds       = {result['n_folds']}")

    tmpdir.cleanup()


if __name__ == "__main__":
    main()
