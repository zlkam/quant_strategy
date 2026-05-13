"""
Cross-validation: compares our vectorized engine against backtesting.py
(event-driven) to verify execution correctness.

Runs both engines in identical baseline mode (long-only, no TP, simple
stops, fixed 100% allocation) to isolate the core execution logic.
Reports equity curve correlation and key metric differences.

Usage:
    uv run python analysis/cross_validate.py              # all tickers
    uv run python analysis/cross_validate.py --ticker QQQ # single ticker
"""

import argparse
import os
import sys

# Find project root (parent of analysis/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from config import BacktestConfig
from data.loader import load_historical_data
from backtest.engine import BacktestEngine as OurEngine
from backtest.metrics import MetricsCalculator
from indicators import calculate_ama, calculate_dsmo, calculate_smfi
from strategy.signal import (
    compute_adx, compute_raw_signal, smooth_signal,
    compute_choppiness_index, compute_sigmoid_regime_weight,
    compute_dual_regime_weight, compute_effective_signal,
    compute_hysteresis_state,
)


# ---------------------------------------------------------------------------
# backtesting.py Strategy — matches our engine's core long-only logic
# ---------------------------------------------------------------------------

class BaselineStrategy(Strategy):
    """
    Event-driven strategy mirroring our engine's core execution:
      - Long-only (no shorts)
      - Entry: hysteresis state = 1 (LONG)
      - Exit: hysteresis state = 0 or -1, or trailing stop (2x ATR)
      - Fixed 100% allocation per trade (no vol targeting)
      - No TP, no pyramiding, no regime filters

    Uses pre-computed signals from our pipeline for identical decision inputs.
    """
    signals_df = None

    def init(self):
        self.hyst_state = self.I(lambda: self.signals_df["hysteresis_state"], name="hyst")
        self.atr = self.I(lambda: self.signals_df["ATR"], name="atr")

    def next(self):
        curr_hyst = int(self.hyst_state[-1])
        curr_atr = float(self.atr[-1])
        curr_close = float(self.data.Close[-1])

        if not self.position:
            if curr_hyst == 1:
                self.buy()
                self.highest_close = curr_close
                self.entry_bar = len(self.data) - 1
        elif self.position.is_long:
            self.highest_close = max(getattr(self, "highest_close", 0), curr_close)
            stop_level = self.highest_close - 2.0 * curr_atr
            exit_pos = False
            if curr_close <= stop_level and curr_atr > 0:
                exit_pos = True
            elif curr_hyst in (0, -1):
                exit_pos = True
            if exit_pos:
                self.position.close()


# ---------------------------------------------------------------------------
# Signal computation (reuses our existing pipeline)
# ---------------------------------------------------------------------------

def compute_our_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full signal pipeline with baseline (simple) config."""
    config = BacktestConfig()
    # Baseline: long-only, no TP, no shorts, simple stops, no CI gate
    config.risk.max_exposure = 1.0
    config.risk.min_exposure = 0.0
    config.risk.tp_enabled = False
    config.risk.time_exit_enabled = False
    config.risk.stop_signal_adaptive = False
    config.risk.tp_adx_adaptive = False
    config.signals.pyramid_initial = 1.0
    config.signals.pyramid_add = 0.0
    config.regime.ci_gate_enabled = False
    config.signals.short_entry = -100.0   # effectively disable shorts
    config.signals.short_exit = -100.0

    df = df.copy()
    ic, sc, rc = config.indicators, config.signals, config.regime

    df = calculate_ama(df, bos_p=ic.ama_bos_period, slow_p=ic.ama_slow_period,
                       fast_p=ic.ama_fast_period, push_fac=ic.ama_push_factor,
                       anch_w=ic.ama_anchor_weight, smth_p=ic.ama_smooth_period,
                       filter_th=ic.ama_filter_threshold)
    df = calculate_smfi(df, flow_period=ic.smfi_flow_period, vol_period=ic.smfi_vol_period,
                        inst_threshold=ic.smfi_inst_threshold, smth_period=ic.smfi_smooth_period,
                        div_period=ic.smfi_div_period, div_th=ic.smfi_div_threshold,
                        accum_th=ic.smfi_accum_threshold, dist_th=ic.smfi_dist_threshold)
    df = calculate_dsmo(df, stoch_period=ic.dsmo_stoch_period, pre_smooth=ic.dsmo_pre_smooth,
                        fast_smooth=ic.dsmo_fast_smooth, slow_smooth=ic.dsmo_slow_smooth,
                        bottom_th=ic.dsmo_bottom_threshold, top_th=ic.dsmo_top_threshold)
    df["ADX"] = compute_adx(df, period=ic.adx_period)
    df["choppiness"] = compute_choppiness_index(df, period=ic.ci_period)
    df["raw_signal"] = compute_raw_signal(df, ama_w=sc.ama_weight,
                                          smfi_w=sc.smfi_weight, dsmo_w=sc.dsmo_weight)
    df["smoothed_signal"] = smooth_signal(df["raw_signal"], period=sc.signal_ema_period)
    df["adx_weight"] = compute_sigmoid_regime_weight(df["ADX"], midpoint=rc.sigmoid_midpoint,
                                                      steepness=rc.sigmoid_steepness,
                                                      floor=rc.adx_floor)
    df["regime_weight"] = compute_dual_regime_weight(df["adx_weight"], df["choppiness"],
                                                      ci_threshold=rc.ci_choppy_threshold,
                                                      ci_enabled=rc.ci_gate_enabled)
    df["effective_signal"] = compute_effective_signal(df["smoothed_signal"], df["regime_weight"])
    df["hysteresis_state"] = compute_hysteresis_state(df["effective_signal"],
                                                       long_entry=sc.long_entry,
                                                       long_exit=sc.long_exit,
                                                       short_entry=sc.short_entry,
                                                       short_exit=sc.short_exit)
    return df


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

def validate_ticker(ticker: str) -> dict:
    """Run both engines on a single ticker and compare."""
    filepath = os.path.join("historical_data", f"{ticker}.csv")
    df = load_historical_data(filepath)

    print(f"\n{'='*60}")
    print(f"Cross-validating {ticker} | {len(df)} bars")
    print(f"{'='*60}")

    # Compute shared signals
    print("  [1/3] Computing signals...")
    signals_df = compute_our_signals(df)

    # Our engine
    print("  [2/3] Running our vectorized engine...")
    config = BacktestConfig()
    config.risk.max_exposure = 1.0
    config.risk.min_exposure = 0.0
    config.risk.tp_enabled = False
    config.risk.time_exit_enabled = False
    config.risk.stop_signal_adaptive = False
    config.risk.tp_adx_adaptive = False
    config.signals.pyramid_initial = 1.0
    config.signals.pyramid_add = 0.0
    config.regime.ci_gate_enabled = False
    config.signals.short_entry = -100.0
    config.signals.short_exit = -100.0

    our_engine = OurEngine(config)
    our_result = our_engine.run(ticker, df)
    our_metrics = MetricsCalculator(our_result["trades"], our_result["equity_curve"],
                                     config.risk.initial_capital)
    our_m = our_metrics.compute_all()

    # backtesting.py engine
    print("  [3/3] Running backtesting.py...")
    ohlc_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    signals_for_bt = signals_df.reindex(ohlc_df.index)

    bt = Backtest(ohlc_df, BaselineStrategy, cash=config.risk.initial_capital,
                  commission=0.0)
    bt_result = bt.run(signals_df=signals_for_bt)
    bt_equity = bt_result["_equity_curve"]["Equity"]

    # Compare equity curves
    our_eq = our_result["equity_curve"]["Equity"]
    our_aligned = our_eq.reindex(bt_equity.index, method="ffill").dropna()
    bt_aligned = bt_equity.reindex(our_aligned.index).dropna()
    common_idx = our_aligned.index.intersection(bt_aligned.index)

    if len(common_idx) > 10:
        corr = our_aligned[common_idx].corr(bt_aligned[common_idx])
        dev_pct = ((our_aligned[common_idx] - bt_aligned[common_idx]) / our_aligned[common_idx] * 100)
        max_dev = dev_pct.abs().max()
        mean_dev = dev_pct.abs().mean()
    else:
        corr = np.nan; max_dev = np.nan; mean_dev = np.nan

    our_final = our_eq.iloc[-1]
    bt_final = bt_equity.iloc[-1]
    init = config.risk.initial_capital

    comparison = {
        "ticker": ticker,
        "our_return_pct": round((our_final - init) / init * 100, 2),
        "bt_return_pct": round((bt_final - init) / init * 100, 2),
        "our_sharpe": round(our_m["Sharpe_Ratio"], 2),
        "bt_sharpe": round(bt_result.get("Sharpe Ratio", 0) or 0, 2),
        "our_dd_pct": round(our_m["Max_Drawdown_Pct"], 2),
        "bt_dd_pct": round(bt_result.get("Max. Drawdown [%]", 0) or 0, 2),
        "our_trades": our_m["Total_Trades"],
        "bt_trades": int(bt_result.get("# Trades", 0)),
        "equity_corr": round(corr, 4) if not np.isnan(corr) else None,
        "max_dev_pct": round(max_dev, 2) if not np.isnan(max_dev) else None,
        "mean_dev_pct": round(mean_dev, 2) if not np.isnan(mean_dev) else None,
    }

    # Verdict
    c = comparison
    if c["equity_corr"] and c["equity_corr"] > 0.95 and c["max_dev_pct"] and c["max_dev_pct"] < 15:
        c["verdict"] = "PASS"
    elif c["equity_corr"] and c["equity_corr"] > 0.90:
        c["verdict"] = "WARN"
    else:
        c["verdict"] = "FAIL"

    print(f"\n  Results:")
    print(f"    Our:  Return={c['our_return_pct']:.1f}%  DD={c['our_dd_pct']:.1f}%  "
          f"Sharpe={c['our_sharpe']:.2f}  Trades={c['our_trades']}")
    print(f"    BT:   Return={c['bt_return_pct']:.1f}%  DD={c['bt_dd_pct']:.1f}%  "
          f"Sharpe={c['bt_sharpe']:.2f}  Trades={c['bt_trades']}")
    print(f"    Corr={c['equity_corr']}  MaxDev={c['max_dev_pct']}%  "
          f"MeanDev={c['mean_dev_pct']}%")
    print(f"    VERDICT: {c['verdict']}")

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate against backtesting.py")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else sorted(
        [f[:-4] for f in os.listdir("historical_data") if f.endswith(".csv")]
    )

    all_comparisons = []
    for ticker in tickers:
        try:
            comp = validate_ticker(ticker)
            all_comparisons.append(comp)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_comparisons.append({"ticker": ticker, "verdict": "ERROR", "error": str(e)})

    # Summary table
    print(f"\n{'='*90}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*90}")
    header = (f"{'Ticker':<8} {'Our Ret':>9} {'BT Ret':>9} {'Our DD':>7} {'BT DD':>7} "
              f"{'Our Sh':>6} {'BT Sh':>6} {'Corr':>7} {'MaxDev':>7} {'Trades':>7} {'Verdict'}")
    print(header)
    print("-" * 90)
    for c in all_comparisons:
        if c.get("error"):
            print(f"{c['ticker']:<8} ERROR: {c['error']}")
        else:
            print(f"{c['ticker']:<8} {c['our_return_pct']:>8.1f}% {c['bt_return_pct']:>8.1f}% "
                  f"{c['our_dd_pct']:>6.1f}% {c['bt_dd_pct']:>6.1f}% "
                  f"{c['our_sharpe']:>5.2f} {c['bt_sharpe']:>5.2f} "
                  f"{str(c['equity_corr']):>7} {str(c['max_dev_pct']):>6}% "
                  f"{str(c['our_trades']):>7} {c['verdict']}")

    passes = sum(1 for c in all_comparisons if c.get("verdict") == "PASS")
    warns = sum(1 for c in all_comparisons if c.get("verdict") == "WARN")
    fails = sum(1 for c in all_comparisons if c.get("verdict") == "FAIL")
    print(f"\n  PASS: {passes}  WARN: {warns}  FAIL: {fails}  TOTAL: {len(all_comparisons)}")


if __name__ == "__main__":
    main()
