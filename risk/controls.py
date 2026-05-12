"""
Risk management: volatility-targeted position sizing, dynamic trailing
stops gated by SMFI flow regime, and per-ticker drawdown circuit breakers.

Key design principle: risk-per-unit is held constant via inverse-vol
sizing. Stops widen when smart money is accumulating (let winners run)
and tighten when smart money is distributing (cut fast).
"""

import numpy as np
import pandas as pd

from config import RiskConfig


class RiskManager:
    """
    Manages position sizing, trailing stops, and drawdown limits.

    Parameters
    ----------
    config : RiskConfig
        All risk parameters from the centralised config.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Volatility-Targeted Position Sizing
    # ------------------------------------------------------------------

    def compute_position_size(
        self,
        capital: float,
        price: float,
        effective_signal: float,
        realized_vol_annual: float,
    ) -> tuple[float, float]:
        """
        Compute shares and notional for a target position.

        The sizing formula follows the risk-parity principle:
        size inversely proportional to realized volatility so that
        each unit of capital deployed carries roughly constant risk.

            target_exposure = (signal / 100) * (target_vol / realized_vol)
            target_exposure = clamp(target_exposure, min_exposure, max_exposure)
            notional = capital * |target_exposure|
            shares = notional / price

        Positive signal → long position.
        Negative signal → short position (shares returned as negative).

        A floor on realized_vol prevents position sizes from exploding
        during ultra-low-volatility regimes (e.g., holiday weeks).

        Parameters
        ----------
        capital : float
            Current total equity (cash + position value).
        price : float
            Current execution price.
        effective_signal : float
            Regime-gated, smoothed signal in [-100, +100].
        realized_vol_annual : float
            Annualized realized volatility (fraction, e.g. 0.18 = 18%).

        Returns
        -------
        tuple[float, float]
            (shares, notional). Shares is negative for short positions.
        """
        # Floor realized vol at 50% of target to prevent extreme sizing in low-vol
        vol_floor = self.cfg.target_vol_annual * 0.5
        vol_used = max(realized_vol_annual, vol_floor)

        # Target exposure: signal strength × vol scaling
        target_exposure = (effective_signal / 100.0) * (self.cfg.target_vol_annual / vol_used)
        target_exposure = float(np.clip(target_exposure, self.cfg.min_exposure, self.cfg.max_exposure))

        notional = capital * abs(target_exposure)

        # Shares: positive = long, negative = short
        if price <= 0:
            return 0.0, 0.0
        shares = notional / price
        if target_exposure < 0:
            shares = -shares

        return shares, notional

    # ------------------------------------------------------------------
    # Dynamic Trailing Stop (SMFI-gated)
    # ------------------------------------------------------------------

    def get_stop_multiplier(self, smfi_value: float) -> float:
        """
        Return the ATR multiplier for the trailing stop based on SMFI zone.

        SMFI > 60 (accumulation): 3.0x ATR → wider stop, let winners compound.
            Smart money is actively accumulating, so temporary drawdowns
            are likely to be bought — giving the trade room prevents
            premature exits during normal pullbacks.
        SMFI < 40 (distribution): 1.0x ATR → tighter stop, cut fast.
            Smart money is distributing — the trend is likely to fail.
            Tightening the stop preserves capital.
        SMFI 40–60 (neutral): 2.0x ATR → standard stop.
            Industry-standard 2x ATR for swing trading on daily bars.

        Parameters
        ----------
        smfi_value : float
            Current SMFI reading.

        Returns
        -------
        float
            ATR multiplier for stop distance.
        """
        if smfi_value > 60.0:
            return self.cfg.atr_accumulation
        elif smfi_value < 40.0:
            return self.cfg.atr_distribution
        return self.cfg.atr_base

    def compute_stop_level(
        self,
        reference_price: float,
        atr: float,
        smfi_value: float,
        is_long: bool,
    ) -> float:
        """
        Compute the trailing stop level.

        For longs:  stop = highest_close_since_entry - multiplier * ATR
        For shorts: stop = lowest_close_since_entry  + multiplier * ATR

        The reference price for longs is the highest close since entry;
        for shorts it's the lowest close since entry. Both are updated
        each bar by the engine.

        Parameters
        ----------
        reference_price : float
            Highest close (long) or lowest close (short) since entry.
        atr : float
            Current ATR value.
        smfi_value : float
            Current SMFI reading — determines stop width.
        is_long : bool
            True for long positions, False for short.

        Returns
        -------
        float
            Stop price level.
        """
        multiplier = self.get_stop_multiplier(smfi_value)
        distance = multiplier * atr

        if is_long:
            return reference_price - distance
        return reference_price + distance

    # ------------------------------------------------------------------
    # Multi-Stage Profit Targets (Improvement #1)
    # ------------------------------------------------------------------

    def compute_profit_targets(
        self,
        entry_price: float,
        entry_atr: float,
        is_long: bool,
    ) -> list[dict]:
        """
        Compute locked profit target levels at entry.

        Each target is a dict: {price, fraction_to_sell, label}
        Prices are ATR-multiples from the entry price, locked at entry
        to prevent dynamic recalculation mid-trade (avoids backtest-to-live
        divergence from shifting ATR).

        For longs:  TP = entry_price + level * entry_ATR
        For shorts: TP = entry_price - level * entry_ATR

        The remaining fraction (1 - sum of all tp_fractions) is left to
        trail with the dynamic stop.

        Research basis (trustdan 293-backtest study):
        Multi-stage profit targets with ATR-based levels consistently
        outperform trailing-only exits by capturing trend extensions
        without premature exits.

        Parameters
        ----------
        entry_price : float
            Execution price at entry.
        entry_atr : float
            ATR value at entry bar (locked, not dynamic).
        is_long : bool
            True for long positions, False for short.

        Returns
        -------
        list[dict]
            Sorted list of {price, fraction, label} dicts.
            For longs: sorted ascending (closest TP first).
            For shorts: sorted descending (closest TP first).
        """
        if not self.cfg.tp_enabled:
            return []

        levels = self.cfg.tp_levels
        fractions = self.cfg.tp_fractions

        targets = []
        for i, (level, frac) in enumerate(zip(levels, fractions)):
            if is_long:
                tp_price = entry_price + level * entry_atr
            else:
                tp_price = entry_price - level * entry_atr
            targets.append({
                "price": round(tp_price, 4),
                "fraction": frac,
                "filled": False,
                "label": f"TP{level}N",
            })

        # Sort: longs closest TP first, shorts closest TP first
        targets.sort(key=lambda t: t["price"] if is_long else -t["price"])

        return targets

    # ------------------------------------------------------------------
    # Drawdown Circuit Breaker
    # ------------------------------------------------------------------

    def check_drawdown_halt(
        self,
        ticker: str,
        equity_curve: pd.DataFrame,
    ) -> tuple[bool, float, float]:
        """
        Check if the max drawdown limit has been breached for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol (determines which DD limit applies).
        equity_curve : pd.DataFrame
            Must contain a 'Drawdown' column from the engine.

        Returns
        -------
        tuple[bool, float, float]
            (halted, current_max_dd_pct, max_dd_limit_pct)
        """
        if equity_curve.empty or "Drawdown" not in equity_curve.columns:
            return False, 0.0, self.cfg.get_max_dd(ticker) * 100

        current_max_dd = abs(equity_curve["Drawdown"].min())
        limit = self.cfg.get_max_dd(ticker)
        return current_max_dd / 100.0 >= limit, current_max_dd, limit * 100

    # ------------------------------------------------------------------
    # Realized Volatility
    # ------------------------------------------------------------------

    @staticmethod
    def compute_realized_vol(
        returns: pd.Series,
        lookback: int = 20,
        ann_factor: int = 252,
    ) -> pd.Series:
        """
        Compute rolling annualized realized volatility from daily returns.

        Uses sample standard deviation (ddof=1) over the lookback window.
        Annualised by sqrt(252) assuming i.i.d. daily returns.

        Parameters
        ----------
        returns : pd.Series
            Daily percentage returns (e.g. close.pct_change()).
        lookback : int, default 20
            Rolling window in bars (~1 month of daily data).
        ann_factor : int, default 252
            Trading days per year for annualisation.

        Returns
        -------
        pd.Series
            Annualized realized volatility as a fraction (e.g. 0.15 = 15%).
        """
        vol = returns.rolling(window=lookback, min_periods=5).std() * np.sqrt(ann_factor)
        # Fill NaN warmup with median vol to avoid zero-division in sizing
        median_vol = vol.median()
        if pd.isna(median_vol) or median_vol == 0:
            median_vol = 0.20  # fallback: 20% ann vol
        return vol.fillna(median_vol)
