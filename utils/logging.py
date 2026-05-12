"""
Structured .log output for bar-level indicator data and trade actions.

Two loggers:

  BarLogger   — pipe-delimited per-bar log with all indicator values,
                signal components, position state, and equity.
  TradeLogger — multi-line structured blocks for entry and exit events
                with full signal context, ending with a per-stock summary.
"""

import os
from datetime import datetime


# ---------------------------------------------------------------------------
# BarLogger — per-bar indicator + equity log
# ---------------------------------------------------------------------------

class BarLogger:
    """
    Writes a pipe-delimited per-bar log with all indicator and signal values.

    One line per bar. Header comments identify columns for self-documentation.

    Usage:
        logger = BarLogger("logs/QQQ_bars.log")
        logger.write_header()
        for bar_data in result["bar_log"]:
            logger.write_bar(bar_data)
        logger.close()
    """

    # Column order matching bar_log dict keys
    COLUMNS = [
        "Date", "Close", "Open", "High", "Low", "Volume",
        "AMA", "ATR", "SMFI", "SMFI_Zone", "SMFI_Div",
        "DSMO_Fast", "DSMO_Slow", "DSMO_Zone", "ADX", "Choppiness", "ADX_Weight",
        "RawSignal", "SmoothSignal", "SignalMomentum", "RegimeWeight", "EffSignal",
        "State", "Position", "PositionValue", "Equity", "Cash",
    ]

    def __init__(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.fh = open(filepath, "w", encoding="utf-8")
        self.filepath = filepath

    def write_header(self, ticker: str) -> None:
        """Write comment header with generation info and column names."""
        self.fh.write(f"# Bar Log for {ticker}\n")
        self.fh.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.fh.write(f"# Columns: {'|'.join(self.COLUMNS)}\n")

    def write_bar(self, bar: dict) -> None:
        """
        Write a single bar as a pipe-delimited line.

        Missing values are written as '-' for clarity.
        """
        values = []
        for col in self.COLUMNS:
            val = bar.get(col, "-")
            if val is None or (isinstance(val, float) and str(val) == "nan"):
                val = "-"
            values.append(str(val))
        self.fh.write("|".join(values) + "\n")

    def close(self) -> None:
        """Flush and close the log file."""
        self.fh.flush()
        self.fh.close()


# ---------------------------------------------------------------------------
# TradeLogger — structured trade event log
# ---------------------------------------------------------------------------

class TradeLogger:
    """
    Writes structured, multi-line trade event blocks with full signal context.

    Entry and exit events are grouped with context lines showing the signal
    state at decision time. Ends with a per-stock performance summary.

    Usage:
        logger = TradeLogger("logs/QQQ_trades.log")
        logger.write_header("QQQ")
        for trade in result["trades"]:
            if trade["Action"] in ("BUY", "SELL_SHORT"):
                logger.log_entry(trade)
            else:
                logger.log_exit(trade)
        logger.log_summary(metrics_dict)
        logger.close()
    """

    SEPARATOR = "=" * 72

    def __init__(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.fh = open(filepath, "w", encoding="utf-8")

    def write_header(self, ticker: str) -> None:
        """Write header with ticker and generation timestamp."""
        self.fh.write(f"# Trade Log for {ticker}\n")
        self.fh.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.fh.write(f"#\n")

    def log_entry(self, trade: dict) -> None:
        """Log an entry (BUY or SELL_SHORT) with signal context."""
        direction = "LONG" if trade["Action"] == "BUY" else "SHORT"
        date_str = str(trade["Date"])[:10]  # strip time component
        self.fh.write(self.SEPARATOR + "\n")
        self.fh.write(
            f"ENTRY | {date_str} | {direction} | "
            f"Price: {trade['Price']} | Shares: {trade['Shares']} | "
            f"Notional: ${trade['Notional']:,.2f}\n"
        )
        self.fh.write(
            f"  Signal: {trade.get('Signal', '-')} | SMFI: {trade.get('SMFI', '-')} | "
            f"RealizedVol: {trade.get('RealizedVol', '-')}%\n"
        )
        self.fh.write(f"  Reason: {trade['Reason']}\n")

    def log_exit(self, trade: dict) -> None:
        """Log an exit (SELL or BUY_TO_COVER) with PnL and duration."""
        date_str = str(trade["Date"])[:10]  # strip time component
        self.fh.write(self.SEPARATOR + "\n")
        self.fh.write(
            f"EXIT  | {date_str} | "
            f"Price: {trade['Price']} | Shares: {trade['Shares']} | "
            f"Notional: ${trade['Notional']:,.2f}\n"
        )
        pnl_sign = "+" if trade.get("PnL", 0) >= 0 else ""
        pnl = trade.get("PnL", 0)
        pnl_pct = trade.get("PnL_Pct", 0)
        self.fh.write(
            f"  PnL: {pnl_sign}${pnl:,.2f} "
            f"({pnl_sign}{pnl_pct:.2f}%) | "
            f"Bars Held: {trade.get('BarsHeld', '-')} | "
            f"Signal: {trade.get('Signal', '-')} | SMFI: {trade.get('SMFI', '-')}\n"
        )
        self.fh.write(f"  Reason: {trade['Reason']}\n")

    def log_summary(self, metrics: dict) -> None:
        """Write per-stock performance summary."""
        self.fh.write(self.SEPARATOR + "\n")
        self.fh.write("PERFORMANCE SUMMARY\n")
        self.fh.write(self.SEPARATOR + "\n")

        rows = [
            ("Total Return", f"{metrics.get('Total_Return_Pct', 0):.2f}%"),
            ("Annualized Return", f"{metrics.get('Annualized_Return_Pct', 0):.2f}%"),
            ("Buy & Hold Return", f"{metrics.get('BuyHold_Return_Pct', 0):.2f}%"),
            ("Max Drawdown", f"{metrics.get('Max_Drawdown_Pct', 0):.2f}%"),
            ("Exposure Time", f"{metrics.get('Exposure_Time_Pct', 0):.2f}%"),
            ("Annualized Volatility", f"{metrics.get('Annualized_Volatility_Pct', 0):.2f}%"),
            ("Sharpe Ratio", f"{metrics.get('Sharpe_Ratio', 0):.4f}"),
            ("Sortino Ratio", f"{metrics.get('Sortino_Ratio', 0):.4f}"),
            ("Calmar Ratio", f"{metrics.get('Calmar_Ratio', 0):.4f}"),
            ("Total Trades", f"{metrics.get('Total_Trades', 0)}"),
            ("Hit Rate", f"{metrics.get('Hit_Rate_Pct', 0):.2f}%"),
            ("Profit Factor", f"{metrics.get('Profit_Factor', 0):.4f}"),
            ("Avg Win", f"${metrics.get('Avg_Win', 0):,.2f}"),
            ("Avg Loss", f"${metrics.get('Avg_Loss', 0):,.2f}"),
            ("Win/Loss Ratio", f"{metrics.get('Avg_Win_Loss_Ratio', 0):.4f}"),
        ]
        if metrics.get("Long_Hit_Rate_Pct") is not None:
            rows.append(("Long Hit Rate", f"{metrics['Long_Hit_Rate_Pct']:.2f}%"))
        if metrics.get("Short_Hit_Rate_Pct") is not None:
            rows.append(("Short Hit Rate", f"{metrics['Short_Hit_Rate_Pct']:.2f}%"))
        if metrics.get("Max_Leverage") is not None:
            rows.append(("Max Exposure", f"{metrics['Max_Leverage']:.2f}x"))
        if metrics.get("Avg_Bars_Held") is not None:
            rows.append(("Avg Bars Held", f"{metrics['Avg_Bars_Held']:.1f}"))

        label_width = max(len(label) for label, _ in rows) + 2
        for label, value in rows:
            self.fh.write(f"  {label:<{label_width}} {value}\n")

    def close(self) -> None:
        """Flush and close the log file."""
        self.fh.flush()
        self.fh.close()
