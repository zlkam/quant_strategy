"""
Walk-Forward Optimization (WFO) for honest out-of-sample parameter tuning.

Replaces the single grid search with anchored rolling windows:
  - Train on 4-year windows, test on 2-year OOS windows
  - Grid search best params in-sample, evaluate on OOS
  - Report aggregate OOS metrics (no in-sample contamination)

Usage:
    uv run python analysis/optimize.py                    # WFO on all tickers
    uv run python analysis/optimize.py --ticker QQQ       # single ticker
    uv run python analysis/optimize.py --metric calmar    # score by Calmar ratio
    uv run python analysis/optimize.py --fast             # reduced grid for quick test
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime

# Find project root (parent of analysis/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import BacktestConfig
from data.loader import load_historical_data
from backtest.engine import BacktestEngine
from backtest.metrics import MetricsCalculator


OUTPUT_FILE = "logs/wfo_results.json"

# WFO window parameters
TRAIN_YEARS = 4
TEST_YEARS = 2
STEP_YEARS = 2
BARS_PER_YEAR = 252

# Grid: parameters to sweep
PARAM_GRID = {
    "signals.long_entry": [35, 40, 45],
    "signals.long_exit": [10, 15, 20],
    "signals.short_entry": [-45, -50, -55],
}

PARAM_GRID_FAST = {
    "signals.long_entry": [35, 40],
    "signals.long_exit": [15],
    "signals.short_entry": [-50],
}


# ---------------------------------------------------------------------------
# WFO runner
# ---------------------------------------------------------------------------

def set_config_value(config: BacktestConfig, key_path: str, value) -> None:
    """Set a nested dataclass attribute from a dotted key path."""
    parts = key_path.split(".")
    obj = config
    for key in parts[:-1]:
        obj = getattr(obj, key)
    setattr(obj, parts[-1], value)


def build_param_combos(grid: dict) -> list[dict]:
    """Build all parameter combinations from a grid dict."""
    import itertools
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for val_tuple in itertools.product(*values):
        combo = {k: v for k, v in zip(keys, val_tuple)}
        combos.append(combo)
    return combos


def run_wfo_ticker(
    ticker: str,
    df: pd.DataFrame,
    base_config: BacktestConfig,
    grid: dict,
    metric: str = "sharpe",
) -> dict:
    """
    Run walk-forward optimization for a single ticker.

    Parameters
    ----------
    ticker : str
    df : pd.DataFrame
        Full historical OHLCV data.
    base_config : BacktestConfig
        Base config with default parameters.
    grid : dict
        Parameter grid {key_path: [values]}.
    metric : str
        Scoring metric: "sharpe", "calmar", or "sortino".

    Returns
    -------
    dict with folds[], oos_metrics{}, best_params_stability{}
    """
    combos = build_param_combos(grid)
    total_bars = len(df)
    train_bars = TRAIN_YEARS * BARS_PER_YEAR
    test_bars = TEST_YEARS * BARS_PER_YEAR
    step_bars = STEP_YEARS * BARS_PER_YEAR

    folds = []
    oos_returns = []
    oos_sharpes = []
    oos_dds = []
    param_choices = []

    fold_num = 0
    start_bar = 0

    while start_bar + train_bars + test_bars <= total_bars:
        fold_num += 1
        train_start = start_bar
        train_end = start_bar + train_bars
        test_start = train_end
        test_end = min(test_start + test_bars, total_bars)

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        # Grid search on training window
        best_score = -float("inf")
        best_params = None
        best_train_metrics = None

        for combo in combos:
            config = deepcopy(base_config)
            for key_path, value in combo.items():
                set_config_value(config, key_path, value)

            try:
                engine = BacktestEngine(config)
                result = engine.run(ticker, train_df)
                calc = MetricsCalculator(
                    result["trades"], result["equity_curve"],
                    config.risk.initial_capital,
                )
                metrics = calc.compute_all()
                score = metrics.get(f"{metric.capitalize()}_Ratio", 0) or 0
                if score > best_score:
                    best_score = score
                    best_params = combo
                    best_train_metrics = metrics
            except Exception:
                continue

        if best_params is None:
            start_bar += step_bars
            continue

        # Evaluate best params on OOS test window
        config = deepcopy(base_config)
        for key_path, value in best_params.items():
            set_config_value(config, key_path, value)

        try:
            engine = BacktestEngine(config)
            result = engine.run(ticker, test_df)
            calc = MetricsCalculator(
                result["trades"], result["equity_curve"],
                config.risk.initial_capital,
            )
            oos_metrics = calc.compute_all()

            fold_result = {
                "fold": fold_num,
                "train_period": f"{train_df.index[0].date()} -> {train_df.index[-1].date()}",
                "test_period": f"{test_df.index[0].date()} -> {test_df.index[-1].date()}",
                "best_params": best_params,
                "train_score": round(best_score, 4),
                "oos_return": round(oos_metrics["Total_Return_Pct"], 2),
                "oos_ann_return": round(oos_metrics["Annualized_Return_Pct"], 2),
                "oos_sharpe": round(oos_metrics["Sharpe_Ratio"], 4),
                "oos_calmar": round(oos_metrics["Calmar_Ratio"], 4),
                "oos_sortino": round(oos_metrics["Sortino_Ratio"], 4),
                "oos_dd": round(oos_metrics["Max_Drawdown_Pct"], 2),
                "oos_hit_rate": round(oos_metrics["Hit_Rate_Pct"], 2),
                "oos_pf": round(oos_metrics["Profit_Factor"], 2),
            }
            folds.append(fold_result)
            oos_returns.append(oos_metrics["Annualized_Return_Pct"])
            oos_sharpes.append(oos_metrics["Sharpe_Ratio"])
            oos_dds.append(oos_metrics["Max_Drawdown_Pct"])
            param_choices.append(best_params)

            print(f"  Fold {fold_num}: train={fold_result['train_period']}, "
                  f"test={fold_result['test_period']}")
            print(f"    Best params: {best_params}")
            print(f"    OOS: return={fold_result['oos_ann_return']:.2f}%, "
                  f"sharpe={fold_result['oos_sharpe']:.4f}, "
                  f"dd={fold_result['oos_dd']:.2f}%, "
                  f"calmar={fold_result['oos_calmar']:.4f}")

        except Exception as e:
            print(f"  Fold {fold_num}: OOS eval failed: {e}")

        start_bar += step_bars

    if not folds:
        return {"ticker": ticker, "error": "No valid folds", "folds": []}

    # Aggregate OOS metrics
    oos_summary = {
        "n_folds": len(folds),
        "mean_ann_return": round(np.mean(oos_returns), 2),
        "std_ann_return": round(np.std(oos_returns), 2),
        "worst_ann_return": round(np.min(oos_returns), 2),
        "mean_sharpe": round(np.mean(oos_sharpes), 4),
        "std_sharpe": round(np.std(oos_sharpes), 4),
        "worst_sharpe": round(np.min(oos_sharpes), 4),
        "mean_dd": round(np.mean(oos_dds), 2),
        "worst_dd": round(np.min(oos_dds), 2),
        "pct_profitable_folds": round(
            sum(1 for r in oos_returns if r > 0) / len(oos_returns) * 100, 1
        ),
    }

    # Parameter stability: how often each value was chosen
    param_stability = {}
    for key in grid:
        choices = [str(p[key]) for p in param_choices if key in p]
        if choices:
            from collections import Counter
            counts = Counter(choices)
            param_stability[key] = {
                "most_common": counts.most_common(1)[0][0],
                "frequency": round(counts.most_common(1)[0][1] / len(choices) * 100, 1),
                "all_choices": dict(counts),
            }

    return {
        "ticker": ticker,
        "metric": metric,
        "n_folds": len(folds),
        "folds": folds,
        "oos_aggregate": oos_summary,
        "param_stability": param_stability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker")
    parser.add_argument("--metric", type=str, default="sharpe",
                        choices=["sharpe", "calmar", "sortino"])
    parser.add_argument("--fast", action="store_true", help="Reduced grid")
    args = parser.parse_args()

    base_config = BacktestConfig()
    tickers = [args.ticker] if args.ticker else sorted(
        [f[:-4] for f in os.listdir(base_config.data_dir) if f.endswith(".csv")]
    )
    grid = PARAM_GRID_FAST if args.fast else PARAM_GRID

    print(f"=== Walk-Forward Optimization ===")
    print(f"Tickers: {tickers}")
    print(f"Metric: {args.metric}")
    print(f"Windows: {TRAIN_YEARS}yr train / {TEST_YEARS}yr test / {STEP_YEARS}yr step")
    print(f"Grid: {len(build_param_combos(grid))} combinations")
    print(f"Start: {datetime.now().isoformat()}\n")

    all_results = {}
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"WFO: {ticker}")
        print(f"{'='*60}")
        filepath = os.path.join(base_config.data_dir, f"{ticker}.csv")
        df = load_historical_data(filepath)
        print(f"  Data: {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")

        result = run_wfo_ticker(ticker, df, base_config, grid, args.metric)
        all_results[ticker] = result

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        agg = result["oos_aggregate"]
        print(f"\n  WFO Aggregate (OOS only):")
        print(f"    Folds: {agg['n_folds']} ({agg['pct_profitable_folds']:.0f}% profitable)")
        print(f"    Mean Ann Return: {agg['mean_ann_return']:.2f}% (std: {agg['std_ann_return']:.2f}%)")
        print(f"    Mean Sharpe:     {agg['mean_sharpe']:.4f} (worst: {agg['worst_sharpe']:.4f})")
        print(f"    Mean DD:         {agg['mean_dd']:.2f}% (worst: {agg['worst_dd']:.2f}%)")
        print(f"    Worst Ann Ret:   {agg['worst_ann_return']:.2f}%")
        print(f"  Param stability: {result['param_stability']}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
