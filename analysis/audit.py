"""
Comprehensive Strategy Audit: scenario analysis, peer comparison, live-trading readiness.

Tests the strategy across 8 distinct market regimes, compares against baseline
strategies (200-d MA crossover, buy-and-hold, 60/40), and audits for live-trading
readiness (transaction costs, slippage, turnover, recovery times).

Outputs a full audit report to logs/audit_report.log.

Usage:
    uv run python analysis/audit.py                # full audit on all tickers
    uv run python analysis/audit.py --ticker QQQ   # single ticker
"""

import argparse
import os
import sys
from datetime import datetime

# Find project root (parent of analysis/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import BacktestConfig
from data.loader import load_historical_data
from backtest.engine import BacktestEngine
from backtest.metrics import MetricsCalculator


# ---------------------------------------------------------------------------
# Market regime definitions (QQQ as proxy)
# ---------------------------------------------------------------------------

REGIMES = {
    "1_DotCom_Crash": ("2000-01-01", "2002-10-09"),
    "2_Recovery":      ("2002-10-10", "2007-10-09"),
    "3_GFC_Crash":     ("2007-10-10", "2009-03-09"),
    "4_QE_Bull":       ("2009-03-10", "2020-02-19"),
    "5_COVID_Crash":   ("2020-02-20", "2020-03-23"),
    "6_PostCOVID_Bull":("2020-03-24", "2022-01-03"),
    "7_2022_Bear":     ("2022-01-04", "2022-12-28"),
    "8_AI_Bull":       ("2023-01-01", "2025-12-31"),
}

REGIME_TYPES = {
    "1_DotCom_Crash": "Bear",
    "2_Recovery":      "Bull",
    "3_GFC_Crash":     "Crash",
    "4_QE_Bull":       "Bull",
    "5_COVID_Crash":   "Crash",
    "6_PostCOVID_Bull":"Bull",
    "7_2022_Bear":     "Bear",
    "8_AI_Bull":       "Bull",
}


# ---------------------------------------------------------------------------
# 200-day MA crossover baseline strategy
# ---------------------------------------------------------------------------

def run_ma_crossover_baseline(df: pd.DataFrame, initial_capital: float = 1_000_000.0) -> dict:
    """
    Simple 200-day MA crossover: buy when close > MA200, sell when close < MA200.
    Long-only, 100% allocation, no shorting, no stops.
    Industry-standard baseline for trend-following strategies.
    """
    df = df.copy()
    df["MA200"] = df["Close"].rolling(200, min_periods=1).mean()
    df["Signal"] = (df["Close"] > df["MA200"]).astype(int).diff().fillna(0)

    cash = initial_capital
    shares = 0.0
    equity_log = []

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        signal = df["Signal"].iloc[i]

        if signal == 1 and shares == 0:  # Buy
            shares = cash / price
            cash = 0.0
        elif signal == -1 and shares > 0:  # Sell
            cash = shares * price
            shares = 0.0

        equity = cash + shares * price
        equity_log.append(equity)

    equity_series = pd.Series(equity_log, index=df.index)
    peak = equity_series.expanding().max()
    dd = (equity_series - peak) / peak * 100

    final = equity_series.iloc[-1]
    total_ret = (final - initial_capital) / initial_capital * 100
    days = (df.index[-1] - df.index[0]).days
    ann_ret = ((1 + total_ret/100) ** (365/max(days,1)) - 1) * 100
    daily_ret = equity_series.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd.min()) if dd.min() != 0 else 0

    return {
        "total_return": round(total_ret, 2),
        "ann_return": round(ann_ret, 2),
        "max_dd": round(dd.min(), 2),
        "sharpe": round(sharpe, 4),
        "calmar": round(calmar, 4),
    }


# ---------------------------------------------------------------------------
# 60/40 portfolio baseline
# ---------------------------------------------------------------------------

def run_6040_baseline(df: pd.DataFrame, spy_df: pd.DataFrame = None,
                      initial_capital: float = 1_000_000.0) -> dict:
    """
    60% stocks / 40% bonds proxy: 60% in the ticker, 40% in cash (earning 0%).
    Rebalanced monthly. No bond data available, so we use a simplified model
    where the 40% bond portion earns the risk-free rate (~0% for most of 2000-2020).
    """
    df = df.copy()
    monthly_idx = df.resample("ME").last().index

    cash = initial_capital
    stock_value = 0.0
    equity_log = []
    rebalance_dates = set(monthly_idx)

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        date = df.index[i]

        if date in rebalance_dates:
            total = cash + stock_value
            stock_value = total * 0.60
            cash = total * 0.40

        # Update stock value
        if i > 0:
            prev_price = df["Close"].iloc[i-1]
            if prev_price > 0 and stock_value > 0:
                stock_value *= (price / prev_price)

        equity = cash + stock_value
        equity_log.append(equity)

    equity_series = pd.Series(equity_log, index=df.index)
    peak = equity_series.expanding().max()
    dd = (equity_series - peak) / peak * 100

    final = equity_series.iloc[-1]
    total_ret = (final - initial_capital) / initial_capital * 100
    days = (df.index[-1] - df.index[0]).days
    ann_ret = ((1 + total_ret/100) ** (365/max(days,1)) - 1) * 100
    daily_ret = equity_series.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    calmar = ann_ret / abs(dd.min()) if dd.min() != 0 else 0

    return {
        "total_return": round(total_ret, 2),
        "ann_return": round(ann_ret, 2),
        "max_dd": round(dd.min(), 2),
        "sharpe": round(sharpe, 4),
        "calmar": round(calmar, 4),
    }


# ---------------------------------------------------------------------------
# Scenario Analysis
# ---------------------------------------------------------------------------

def run_scenario_analysis(ticker: str, df: pd.DataFrame, config: BacktestConfig) -> dict:
    """Run backtest on each market regime separately."""
    results = {}
    engine = BacktestEngine(config)

    for regime_name, (start, end) in REGIMES.items():
        mask = (df.index >= start) & (df.index <= end)
        regime_df = df[mask]

        if len(regime_df) < 50:
            results[regime_name] = {"error": "Not enough data"}
            continue

        try:
            result = engine.run(ticker, regime_df)
            mc = MetricsCalculator(result["trades"], result["equity_curve"],
                                    config.risk.initial_capital)
            m = mc.compute_all()
            results[regime_name] = {
                "regime_type": REGIME_TYPES.get(regime_name, "Unknown"),
                "bars": len(regime_df),
                "period": f"{regime_df.index[0].date()} -> {regime_df.index[-1].date()}",
                "total_return": m["Total_Return_Pct"],
                "ann_return": m["Annualized_Return_Pct"],
                "max_dd": m["Max_Drawdown_Pct"],
                "sharpe": m["Sharpe_Ratio"],
                "calmar": m["Calmar_Ratio"],
                "sortino": m["Sortino_Ratio"],
                "hit_rate": m["Hit_Rate_Pct"],
                "profit_factor": m["Profit_Factor"],
                "trades": m["Total_Trades"],
                "exposure": m["Exposure_Time_Pct"],
            }
        except Exception as e:
            results[regime_name] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Live-Trading Readiness Audit
# ---------------------------------------------------------------------------

def audit_live_readiness(ticker: str, df: pd.DataFrame, config: BacktestConfig) -> dict:
    """Audit for live-trading readiness."""
    engine = BacktestEngine(config)
    result = engine.run(ticker, df)
    trades = result["trades"]
    eq = result["equity_curve"]

    mc = MetricsCalculator(trades, eq, config.risk.initial_capital)
    m = mc.compute_all()
    bm_bh = mc.benchmark_return(df)

    audit = {}

    # 1. Transaction cost impact (0.1% per trade, 0.05% slippage per side)
    n_entries = m["Total_Trades"]
    n_exits = len([t for t in trades if t["Action"] in ("SELL", "BUY_TO_COVER")])
    total_trades = n_entries + n_exits
    # Avg trade notional
    notionals = [t.get("Notional", 0) for t in trades if t.get("Notional")]
    avg_notional = np.mean(notionals) if notionals else 0
    # Commission: 0.1% per trade → 0.1% × notional × total_trades
    commission_cost_pct = 0.001 * total_trades * 100  # as % of avg trade
    # Slippage: 0.05% per side → 0.05% × notional × total_trades
    slippage_cost_pct = 0.0005 * total_trades * 100
    total_cost_pct = commission_cost_pct + slippage_cost_pct
    audit["total_trades"] = total_trades
    audit["commission_impact_pct"] = round(commission_cost_pct, 2)
    audit["slippage_impact_pct"] = round(slippage_cost_pct, 2)
    audit["total_cost_estimate_pct"] = round(total_cost_pct, 2)
    # Rough estimate: costs reduce annual return by this much
    ann_trades_per_year = total_trades / max((df.index[-1] - df.index[0]).days / 365, 1)
    audit["annual_turnover"] = round(ann_trades_per_year, 1)

    # 2. Signal stability
    # Compute effective signal volatility (lower = more stable)
    if "effective_signal" in result.get("equity_curve", pd.DataFrame()).columns:
        sig_std = float(eq.get("effective_signal", pd.Series([0])).std())
    else:
        sig_std = 0.0
    audit["signal_volatility"] = round(sig_std, 2)

    # 3. Drawdown recovery analysis
    if not eq.empty and "Drawdown" in eq.columns:
        dd_series = eq["Drawdown"]
        # Find drawdown periods
        in_dd = dd_series < -1.0  # in drawdown > 1%
        # Count drawdown episodes
        dd_episodes = 0
        in_episode = False
        for v in in_dd:
            if v and not in_episode:
                dd_episodes += 1
                in_episode = True
            elif not v:
                in_episode = False
        audit["drawdown_episodes"] = dd_episodes
        audit["deepest_dd_pct"] = round(dd_series.min(), 2)
        # Recovery time: bars from trough to breakeven
        trough_idx = dd_series.idxmin()
        trough_pos = eq.index.get_loc(trough_idx)
        trough_val = eq["Equity"].iloc[trough_pos]
        recovered = False
        recovery_bars = 0
        for j in range(trough_pos + 1, len(eq)):
            if eq["Equity"].iloc[j] >= eq["Equity"].iloc[:trough_pos+1].max():
                recovered = True
                recovery_bars = j - trough_pos
                break
        audit["max_recovery_bars"] = recovery_bars if recovered else -1
        audit["max_recovery_days"] = recovery_bars if recovered else -1

    # 4. Max consecutive losing trades
    exits = [t for t in trades if t["Action"] in ("SELL", "BUY_TO_COVER")]
    max_consec_loss = 0
    curr_streak = 0
    for t in exits:
        if t.get("PnL", 0) <= 0:
            curr_streak += 1
            max_consec_loss = max(max_consec_loss, curr_streak)
        else:
            curr_streak = 0
    audit["max_consecutive_losses"] = max_consec_loss

    # 5. Worst single-trade loss
    worst_loss = min((t.get("PnL", 0) for t in exits), default=0)
    worst_loss_pct = min((t.get("PnL_Pct", 0) for t in exits), default=0)
    audit["worst_single_loss"] = round(worst_loss, 2)
    audit["worst_single_loss_pct"] = round(worst_loss_pct, 2)

    # 6. Capacity check: avg daily volume vs position size
    avg_vol = df["Volume"].mean()
    max_position_notional = max((t.get("Notional", 0) for t in trades), default=0)
    if avg_vol > 0 and max_position_notional > 0:
        # Assume we can trade 5% of daily volume without impact
        capacity_ratio = max_position_notional / (avg_vol * 0.05)
        audit["capacity_ratio"] = round(capacity_ratio, 2)
        audit["capacity_ok"] = capacity_ratio < 1.0
    else:
        audit["capacity_ok"] = True

    return audit


# ---------------------------------------------------------------------------
# Audit Report Generator
# ---------------------------------------------------------------------------

def generate_report(
    ticker: str,
    full_metrics: dict,
    scenario_results: dict,
    baselines: dict,
    audit: dict,
    bnh_return: float,
) -> str:
    """Format audit findings as a readable report section."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"STRATEGY AUDIT: {ticker}")
    lines.append(f"{'='*70}")

    # Full period summary
    m = full_metrics
    lines.append(f"\n--- Full Period ({m.get('bars', 'N/A')} bars) ---")
    lines.append(f"  Total Return:     {m['Total_Return_Pct']:>10.2f}%")
    lines.append(f"  Ann. Return:      {m['Annualized_Return_Pct']:>10.2f}%")
    lines.append(f"  Buy & Hold:       {bnh_return:>10.2f}%")
    lines.append(f"  Max Drawdown:     {m['Max_Drawdown_Pct']:>10.2f}%")
    lines.append(f"  Sharpe Ratio:     {m['Sharpe_Ratio']:>10.4f}")
    lines.append(f"  Sortino Ratio:    {m['Sortino_Ratio']:>10.4f}")
    lines.append(f"  Calmar Ratio:     {m['Calmar_Ratio']:>10.4f}")
    lines.append(f"  Hit Rate:         {m['Hit_Rate_Pct']:>10.2f}%")
    lines.append(f"  Profit Factor:    {m['Profit_Factor']:>10.2f}")
    lines.append(f"  Total Trades:     {m['Total_Trades']:>10}")
    lines.append(f"  Exposure:         {m['Exposure_Time_Pct']:>10.2f}%")

    # Scenario analysis
    lines.append(f"\n--- Scenario Analysis ---")
    lines.append(f"  {'Regime':<22} {'Type':<7} {'Bars':>5} {'Return':>8} {'DD':>7} {'Sharpe':>7} {'Calmar':>7} {'Hit%':>6}")
    lines.append(f"  {'-'*22} {'-'*7} {'-'*5} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    regime_scores = []
    for rname in REGIMES:
        r = scenario_results.get(rname, {})
        if "error" in r:
            lines.append(f"  {rname:<22} ERROR: {r['error']}")
        else:
            lines.append(f"  {rname:<22} {r['regime_type']:<7} {r['bars']:>5} "
                         f"{r['total_return']:>7.1f}% {r['max_dd']:>6.1f}% "
                         f"{r['sharpe']:>6.2f} {r['calmar']:>6.2f} {r['hit_rate']:>5.1f}%")
            regime_scores.append(r['sharpe'] if r['sharpe'] else 0)

    # Regime consistency stats
    if regime_scores:
        profitable = sum(1 for s in regime_scores if s > 0)
        total = len(regime_scores)
        lines.append(f"\n  Regime Consistency: {profitable}/{total} regimes profitable")
        lines.append(f"  Mean Regime Sharpe: {np.mean(regime_scores):.3f}")
        lines.append(f"  Worst Regime Sharpe: {np.min(regime_scores):.3f}")
        lines.append(f"  Best Regime Sharpe: {np.max(regime_scores):.3f}")

    # Bull vs Bear aggregate
    bull_returns = []
    bear_returns = []
    for rname in REGIMES:
        r = scenario_results.get(rname, {})
        if "error" not in r:
            rt = REGIME_TYPES.get(rname, "")
            if rt in ("Bull",):
                bull_returns.append(r.get("ann_return", 0))
            elif rt in ("Bear", "Crash"):
                bear_returns.append(r.get("ann_return", 0))

    lines.append(f"\n  Bull Markets (avg ann return): {np.mean(bull_returns):.1f}%")
    lines.append(f"  Bear/Crash Markets (avg ann return): {np.mean(bear_returns):.1f}%")
    lines.append(f"  Bull/Bear Ratio: {np.mean(bull_returns)/max(abs(np.mean(bear_returns)),1):.1f}x")

    # Peer comparison
    lines.append(f"\n--- Peer Comparison ---")
    lines.append(f"  {'Strategy':<30} {'Return':>8} {'DD':>7} {'Sharpe':>7} {'Calmar':>7}")
    lines.append(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
    for name, b in baselines.items():
        lines.append(f"  {name:<30} {b['total_return']:>7.1f}% {b['max_dd']:>6.1f}% "
                     f"{b['sharpe']:>6.2f} {b['calmar']:>6.2f}")
    lines.append(f"  {'Our Strategy':<30} {m['Total_Return_Pct']:>7.1f}% "
                 f"{m['Max_Drawdown_Pct']:>6.1f}% {m['Sharpe_Ratio']:>6.2f} "
                 f"{m['Calmar_Ratio']:>6.2f}")

    # Peer rating
    our_sharpe = m["Sharpe_Ratio"]
    lines.append(f"\n  Peer Rating:")
    if our_sharpe > 1.5: rating = "A+ (Institutional-grade)"
    elif our_sharpe > 1.0: rating = "A (Strong retail / Emerging institutional)"
    elif our_sharpe > 0.7: rating = "B+ (Solid retail strategy)"
    elif our_sharpe > 0.5: rating = "B (Average retail)"
    else: rating = "C (Below average)"
    lines.append(f"    Sharpe-based: {rating}")
    lines.append(f"    Industry context: Top quant hedge funds 1.0-2.0, Good retail 0.7-1.2, Average retail 0.3-0.6")

    # Live trading readiness
    lines.append(f"\n--- Live-Trading Readiness ---")
    lines.append(f"  Annual Turnover:       {audit.get('annual_turnover', 'N/A')} trades/year")
    lines.append(f"  Est. Cost Impact:      {audit.get('total_cost_estimate_pct', 'N/A')}% of gross return")
    lines.append(f"  Max Consecutive Losses:{audit.get('max_consecutive_losses', 'N/A')}")
    lines.append(f"  Worst Single Loss:     ${audit.get('worst_single_loss', 'N/A'):,.0f} ({audit.get('worst_single_loss_pct', 'N/A')}%)")
    lines.append(f"  Deepest Drawdown:      {audit.get('deepest_dd_pct', 'N/A')}%")
    lines.append(f"  Max Recovery Time:     {audit.get('max_recovery_days', 'N/A')} days")
    lines.append(f"  Signal Volatility:     {audit.get('signal_volatility', 'N/A')}")
    capacity = audit.get('capacity_ok', True)
    lines.append(f"  Capacity OK (1M):      {'Yes' if capacity else 'WARNING - may exceed 5% daily vol'}")

    # Overall rating
    lines.append(f"\n--- Overall Assessment ---")
    readiness_score = 0
    if audit.get("max_consecutive_losses", 99) < 8: readiness_score += 1
    if abs(audit.get("deepest_dd_pct", 99)) < 20: readiness_score += 1
    if audit.get("max_recovery_days", 999) < 126: readiness_score += 1
    if audit.get("capacity_ok", False): readiness_score += 1
    if regime_scores and sum(1 for s in regime_scores if s > 0) >= len(regime_scores) * 0.75:
        readiness_score += 1

    if readiness_score >= 5: ready = "READY for paper trading"
    elif readiness_score >= 3: ready = "CONDITIONALLY ready (address warnings first)"
    else: ready = "NOT ready — significant issues to address"

    lines.append(f"  Live Readiness Score: {readiness_score}/5")
    lines.append(f"  Verdict: {ready}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()

    config = BacktestConfig()
    tickers = [args.ticker] if args.ticker else sorted(
        [f[:-4] for f in os.listdir(config.data_dir) if f.endswith(".csv")]
    )

    os.makedirs("logs", exist_ok=True)
    report_path = "logs/audit_report.log"

    with open(report_path, "w", encoding="utf-8") as report_fh:
        report_fh.write(f"# Strategy Audit Report\n")
        report_fh.write(f"# Generated: {datetime.now().isoformat()}\n")
        report_fh.write(f"# Tickers: {tickers}\n\n")

        for ticker in tickers:
            print(f"\nAuditing {ticker}...")
            filepath = os.path.join(config.data_dir, f"{ticker}.csv")
            df = load_historical_data(filepath)

            # Full backtest
            print(f"  Running full backtest...")
            engine = BacktestEngine(config)
            result = engine.run(ticker, df)
            mc = MetricsCalculator(result["trades"], result["equity_curve"],
                                    config.risk.initial_capital)
            full_m = mc.compute_all()
            bnh = mc.benchmark_return(df)

            # Scenario analysis
            print(f"  Running scenario analysis...")
            scenarios = run_scenario_analysis(ticker, df, config)

            # Baselines
            print(f"  Running baselines...")
            ma_base = run_ma_crossover_baseline(df)
            port_6040 = run_6040_baseline(df)
            bnh_base = {
                "total_return": bnh,
                "ann_return": full_m["Annualized_Return_Pct"],  # placeholder
                "max_dd": full_m["Max_Drawdown_Pct"],  # placeholder
                "sharpe": 0.0,
                "calmar": 0.0,
            }
            # Recompute B&H properly
            bnh_series = df["Close"] / df["Close"].iloc[0] * config.risk.initial_capital
            bnh_peak = bnh_series.expanding().max()
            bnh_dd = (bnh_series - bnh_peak) / bnh_peak * 100
            bnh_ret = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
            bnh_daily = df["Close"].pct_change().dropna()
            bnh_sharpe = (bnh_daily.mean() / bnh_daily.std() * np.sqrt(252)) if bnh_daily.std() > 0 else 0
            days = (df.index[-1] - df.index[0]).days
            bnh_ann = ((1 + bnh_ret/100) ** (365/max(days,1)) - 1) * 100
            bnh_calmar = bnh_ann / abs(bnh_dd.min()) if bnh_dd.min() != 0 else 0
            bnh_base = {
                "total_return": round(bnh_ret, 2),
                "ann_return": round(bnh_ann, 2),
                "max_dd": round(bnh_dd.min(), 2),
                "sharpe": round(bnh_sharpe, 4),
                "calmar": round(bnh_calmar, 4),
            }

            baselines = {
                "Buy & Hold": bnh_base,
                "200-d MA Crossover": ma_base,
                "60/40 Monthly Rebalance": port_6040,
            }

            # Live trading audit
            print(f"  Running live-trading audit...")
            audit_results = audit_live_readiness(ticker, df, config)

            # Generate report
            report = generate_report(ticker, full_m, scenarios, baselines,
                                     audit_results, bnh)
            print(report)
            report_fh.write(report + "\n")

    print(f"\nAudit report saved to {report_path}")


if __name__ == "__main__":
    main()
