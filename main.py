"""
Entry point for the Continuous-Signal Quant Strategy Backtester.

Loads all historical data, runs the continuous-signal backtest for each
ticker with LONG / FLAT / SHORT positioning, computes performance metrics,
and writes bar-level indicator logs and structured trade logs.

Output (all in logs/):
  <TICKER>_bars.log    — pipe-delimited per-bar indicator + equity log
  <TICKER>_trades.log  — structured trade blocks with per-stock summary
  portfolio_summary.log — combined metrics for all tickers
"""

import os
import sys

import pandas as pd

from config import BacktestConfig
from data.loader import load_all_tickers
from backtest.engine import BacktestEngine
from backtest.metrics import MetricsCalculator
from backtest.logging import BarLogger, TradeLogger


OUTPUT_DIR = "logs"


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def main() -> None:
    config = BacktestConfig()
    ensure_output_dir()

    # Load all tickers
    print("Loading historical data...")
    tickers = load_all_tickers(config.data_dir)
    if not tickers:
        print(f"No CSV files found in '{config.data_dir}/'")
        sys.exit(1)
    print(f"Loaded {len(tickers)} tickers: {', '.join(sorted(tickers.keys()))}")

    engine = BacktestEngine(config)

    all_metrics: list[dict] = []

    for ticker in sorted(tickers.keys()):
        df = tickers[ticker]
        print(f"\n{'='*60}")
        print(f"Backtesting {ticker}  |  {len(df)} bars  |  "
              f"{df.index[0].date()} -> {df.index[-1].date()}")
        print(f"{'='*60}")

        try:
            result = engine.run(ticker, df)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        # --- Write bar log (per-bar indicator values) ---
        bar_log_path = os.path.join(OUTPUT_DIR, f"{ticker}_bars.log")
        bar_logger = BarLogger(bar_log_path)
        bar_logger.write_header(ticker)
        for bar in result["bar_log"]:
            bar_logger.write_bar(bar)
        bar_logger.close()
        print(f"  -> Bar log: {bar_log_path} ({len(result['bar_log'])} bars)")

        # --- Write trade log ---
        trade_log_path = os.path.join(OUTPUT_DIR, f"{ticker}_trades.log")
        trade_logger = TradeLogger(trade_log_path)
        trade_logger.write_header(ticker)
        for trade in result["trades"]:
            if trade["Action"] in ("BUY", "SELL_SHORT"):
                trade_logger.log_entry(trade)
            else:
                trade_logger.log_exit(trade)

        # --- Metrics ---
        metrics_calc = MetricsCalculator(
            result["trades"],
            result["equity_curve"],
            config.risk.initial_capital,
        )
        metrics = metrics_calc.compute_all()
        metrics["Ticker"] = ticker
        metrics["BuyHold_Return_Pct"] = metrics_calc.benchmark_return(df)

        # Per-stock summary in trade log
        trade_logger.log_summary(metrics)
        trade_logger.close()
        print(f"  -> Trade log: {trade_log_path} ({len(result['trades'])} trades)")

        # Print key stats
        print(f"  Total Return    : {metrics['Total_Return_Pct']:>10.2f}%")
        print(f"  Ann. Return     : {metrics['Annualized_Return_Pct']:>10.2f}%")
        print(f"  Buy & Hold      : {metrics['BuyHold_Return_Pct']:>10.2f}%")
        print(f"  Max Drawdown    : {metrics['Max_Drawdown_Pct']:>10.2f}%")
        print(f"  Sharpe Ratio    : {metrics['Sharpe_Ratio']:>10.2f}")
        print(f"  Sortino Ratio   : {metrics['Sortino_Ratio']:>10.2f}")
        print(f"  Calmar Ratio    : {metrics['Calmar_Ratio']:>10.2f}")
        print(f"  Hit Rate        : {metrics['Hit_Rate_Pct']:>10.2f}%")
        print(f"  Profit Factor   : {metrics['Profit_Factor']:>10.2f}")
        print(f"  Total Trades    : {metrics['Total_Trades']:>10}  "
              f"(L: {metrics['Total_Long_Trades']}, S: {metrics['Total_Short_Trades']})")
        print(f"  Exposure Time   : {metrics['Exposure_Time_Pct']:>10.2f}%  "
              f"(L: {metrics['Long_Exposure_Pct']:.1f}%, S: {metrics['Short_Exposure_Pct']:.1f}%)")
        print(f"  Avg Bars Held   : {metrics['Avg_Bars_Held']:>10.1f}")

        # Drawdown limit check
        dd = abs(metrics["Max_Drawdown_Pct"])
        dd_limit = config.risk.get_max_dd(ticker) * 100
        if dd > dd_limit:
            print(f"  *** WARNING: Max DD ({dd:.2f}%) exceeds limit ({dd_limit:.0f}%) for {ticker}")

        all_metrics.append(metrics)

    # ------------------------------------------------------------------
    # Portfolio-level summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PORTFOLIO SUMMARY")
    print(f"{'='*60}")

    metrics_df = pd.DataFrame(all_metrics).set_index("Ticker")
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 300)
    pd.set_option("display.float_format", "{:.4f}".format)

    # Select key columns for display
    display_cols = [
        "Total_Return_Pct", "Annualized_Return_Pct", "BuyHold_Return_Pct",
        "Max_Drawdown_Pct", "Sharpe_Ratio", "Sortino_Ratio", "Calmar_Ratio",
        "Hit_Rate_Pct", "Profit_Factor", "Total_Trades",
        "Long_Hit_Rate_Pct", "Short_Hit_Rate_Pct",
        "Profit_Factor_Long", "Profit_Factor_Short",
        "Exposure_Time_Pct", "Long_Exposure_Pct", "Short_Exposure_Pct",
        "Avg_Bars_Held", "Max_Leverage",
    ]
    available_cols = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[available_cols])

    # Save portfolio summary log
    summary_path = os.path.join(OUTPUT_DIR, "portfolio_summary.log")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Portfolio Summary\n")
        f.write(f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#\n")
        f.write(metrics_df.to_string())
    print(f"\nPortfolio summary saved to {summary_path}")


if __name__ == "__main__":
    main()
