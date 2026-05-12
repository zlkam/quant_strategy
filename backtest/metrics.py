"""
Performance metrics for evaluating backtest results.

Computes standard quant-finance metrics from trade logs and equity curves.
Updated for bidirectional (long + short) trading with split hit rates,
exposure breakdown, and trade duration statistics.
"""

import numpy as np
import pandas as pd


class MetricsCalculator:
    """
    Calculate performance metrics from backtest output.

    Parameters
    ----------
    trades : list[dict]
        List of all executed trades.
    equity_curve : pd.DataFrame
        Daily equity with columns: Equity, Drawdown.
    initial_capital : float
        Starting capital for the backtest.
    risk_free_rate : float, default 0.0
        Annualised risk-free rate for Sharpe calculation.
    """

    def __init__(
        self,
        trades: list[dict],
        equity_curve: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.trades = trades
        self.equity = equity_curve
        self.init_cap = initial_capital
        self.rf = risk_free_rate

    # ------------------------------------------------------------------
    # Complete metrics dictionary
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """Return a dict of all performance metrics."""
        return {
            "Total_Return_Pct": self.total_return(),
            "Annualized_Return_Pct": self.annualized_return(),
            "BuyHold_Return_Pct": None,  # filled by caller
            "Max_Drawdown_Pct": self.max_drawdown(),
            "Exposure_Time_Pct": self.exposure_time(),
            "Long_Exposure_Pct": self.long_exposure_pct(),
            "Short_Exposure_Pct": self.short_exposure_pct(),
            "Annualized_Volatility_Pct": self.annualized_volatility(),
            "Sharpe_Ratio": self.sharpe_ratio(),
            "Sortino_Ratio": self.sortino_ratio(),
            "Calmar_Ratio": self.calmar_ratio(),
            "Total_Trades": self.total_trades(),
            "Hit_Rate_Pct": self.hit_rate(),
            "Long_Hit_Rate_Pct": self.long_hit_rate(),
            "Short_Hit_Rate_Pct": self.short_hit_rate(),
            "Profit_Factor": self.profit_factor(),
            "Profit_Factor_Long": self.profit_factor_long(),
            "Profit_Factor_Short": self.profit_factor_short(),
            "Avg_Win": self.avg_win(),
            "Avg_Loss": self.avg_loss(),
            "Avg_Win_Loss_Ratio": self.win_loss_ratio(),
            "Avg_Bars_Held": self.avg_bars_held(),
            "Max_Leverage": self.max_leverage(),
            "Total_Long_Trades": self.total_long_trades(),
            "Total_Short_Trades": self.total_short_trades(),
        }

    # ------------------------------------------------------------------
    # Return and volatility metrics
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        if self.equity.empty:
            return 0.0
        final_eq = self.equity["Equity"].iloc[-1]
        return round((final_eq - self.init_cap) / self.init_cap * 100, 4)

    def annualized_return(self) -> float:
        if self.equity.empty:
            return 0.0
        days = (self.equity.index[-1] - self.equity.index[0]).days
        if days <= 0:
            return 0.0
        total_ret = self.total_return() / 100.0
        ann_ret = (1.0 + total_ret) ** (365.0 / days) - 1.0
        return round(ann_ret * 100, 4)

    def max_drawdown(self) -> float:
        if self.equity.empty or "Drawdown" not in self.equity.columns:
            return 0.0
        return round(self.equity["Drawdown"].min(), 4)

    def annualized_volatility(self) -> float:
        if self.equity.empty or len(self.equity) < 2:
            return 0.0
        daily_ret = self.equity["Equity"].pct_change().dropna()
        return round(daily_ret.std() * np.sqrt(252) * 100, 4)

    # ------------------------------------------------------------------
    # Risk-adjusted ratios
    # ------------------------------------------------------------------

    def sharpe_ratio(self) -> float:
        vol = self.annualized_volatility()
        if vol == 0:
            return 0.0
        excess = (self.annualized_return() - self.rf * 100) / 100.0
        vol_dec = vol / 100.0
        return round(excess / vol_dec, 4)

    def sortino_ratio(self) -> float:
        if self.equity.empty or len(self.equity) < 2:
            return 0.0
        daily_ret = self.equity["Equity"].pct_change().dropna()
        downside = daily_ret[daily_ret < 0]
        if downside.empty or downside.std() == 0:
            return 0.0
        ann_down_vol = downside.std() * np.sqrt(252)
        ann_ret = self.annualized_return() / 100.0
        return round(ann_ret / ann_down_vol, 4)

    def calmar_ratio(self) -> float:
        dd = abs(self.max_drawdown())
        if dd == 0:
            return 0.0
        return round(self.annualized_return() / abs(dd), 4)

    # ------------------------------------------------------------------
    # Exposure metrics
    # ------------------------------------------------------------------

    def exposure_time(self) -> float:
        """Percentage of bars where any position (long or short) was held."""
        if self.equity.empty or "Position" not in self.equity.columns:
            return 0.0
        in_position = self.equity["Position"] != 0
        return round(in_position.mean() * 100, 4)

    def long_exposure_pct(self) -> float:
        """Percentage of bars with a long position."""
        if self.equity.empty or "Position" not in self.equity.columns:
            return 0.0
        is_long = self.equity["Position"] > 0
        return round(is_long.mean() * 100, 4)

    def short_exposure_pct(self) -> float:
        """Percentage of bars with a short position."""
        if self.equity.empty or "Position" not in self.equity.columns:
            return 0.0
        is_short = self.equity["Position"] < 0
        return round(is_short.mean() * 100, 4)

    def max_leverage(self) -> float:
        """Maximum absolute exposure (|position_value| / equity) observed."""
        if self.equity.empty or "Position" not in self.equity.columns or "PositionValue" not in self.equity.columns:
            return 0.0
        valid = self.equity["Equity"] > 0
        if not valid.any():
            return 0.0
        leverage = self.equity.loc[valid, "PositionValue"].abs() / self.equity.loc[valid, "Equity"]
        return round(leverage.max(), 4)

    # ------------------------------------------------------------------
    # Trade-level metrics — all trades
    # ------------------------------------------------------------------

    def _exit_trades(self) -> list[dict]:
        """Return only exit/cover trades (exclude entries)."""
        return [t for t in self.trades if t["Action"] in ("SELL", "BUY_TO_COVER")]

    def _long_exits(self) -> list[dict]:
        return [t for t in self.trades if t["Action"] == "SELL"]

    def _short_exits(self) -> list[dict]:
        return [t for t in self.trades if t["Action"] == "BUY_TO_COVER"]

    def total_trades(self) -> int:
        return sum(1 for t in self.trades if t["Action"] in ("BUY", "SELL_SHORT"))

    def total_long_trades(self) -> int:
        return sum(1 for t in self.trades if t["Action"] == "BUY")

    def total_short_trades(self) -> int:
        return sum(1 for t in self.trades if t["Action"] == "SELL_SHORT")

    def hit_rate(self) -> float:
        exits = self._exit_trades()
        if not exits:
            return 0.0
        wins = sum(1 for t in exits if t["PnL"] > 0)
        return round(wins / len(exits) * 100, 4)

    def long_hit_rate(self) -> float:
        exits = self._long_exits()
        if not exits:
            return None
        wins = sum(1 for t in exits if t["PnL"] > 0)
        return round(wins / len(exits) * 100, 4)

    def short_hit_rate(self) -> float:
        exits = self._short_exits()
        if not exits:
            return None
        wins = sum(1 for t in exits if t["PnL"] > 0)
        return round(wins / len(exits) * 100, 4)

    def profit_factor(self) -> float:
        exits = self._exit_trades()
        if not exits:
            return 0.0
        gross_profit = sum(t["PnL"] for t in exits if t["PnL"] > 0)
        gross_loss = abs(sum(t["PnL"] for t in exits if t["PnL"] < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 4)

    def profit_factor_long(self) -> float:
        exits = self._long_exits()
        if not exits:
            return None
        gross_profit = sum(t["PnL"] for t in exits if t["PnL"] > 0)
        gross_loss = abs(sum(t["PnL"] for t in exits if t["PnL"] < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 4)

    def profit_factor_short(self) -> float:
        exits = self._short_exits()
        if not exits:
            return None
        gross_profit = sum(t["PnL"] for t in exits if t["PnL"] > 0)
        gross_loss = abs(sum(t["PnL"] for t in exits if t["PnL"] < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 4)

    def avg_win(self) -> float:
        exits = [t for t in self._exit_trades() if t["PnL"] > 0]
        if not exits:
            return 0.0
        return round(np.mean([t["PnL"] for t in exits]), 2)

    def avg_loss(self) -> float:
        exits = [t for t in self._exit_trades() if t["PnL"] < 0]
        if not exits:
            return 0.0
        return round(np.mean([t["PnL"] for t in exits]), 2)

    def win_loss_ratio(self) -> float:
        al = self.avg_loss()
        if al == 0:
            return 0.0
        return round(abs(self.avg_win() / al), 4)

    def avg_bars_held(self) -> float:
        """Average trade duration in bars (across all closed exits)."""
        exits = self._exit_trades()
        if not exits:
            return 0.0
        durations = [t.get("BarsHeld", 0) for t in exits if t.get("BarsHeld", 0) > 0]
        if not durations:
            return 0.0
        return round(np.mean(durations), 1)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark_return(self, df: pd.DataFrame) -> float:
        """Buy-and-hold return for the same period."""
        if df.empty:
            return 0.0
        first_close = df["Close"].iloc[0]
        last_close = df["Close"].iloc[-1]
        if first_close <= 0:
            return 0.0
        return round((last_close - first_close) / first_close * 100, 4)
