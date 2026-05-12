"""
Backtesting engine with continuous-signal framework and 3-state execution.

Processes one ticker at a time through the full pipeline:
  indicators → signal construction → hysteresis state → execution.

Supports LONG, FLAT, and SHORT positions with volatility-targeted sizing,
dynamic trailing stops gated by SMFI flow regime, and per-ticker drawdown
circuit breakers.

All look-ahead bias is prevented: signals at bar t use information
available at bar t close. Execution occurs at bar t+1 Open.
"""

import pandas as pd
import numpy as np

from config import BacktestConfig
from indicators import calculate_ama, calculate_dsmo, calculate_smfi
from strategy.signal import (
    compute_adx,
    compute_raw_signal,
    smooth_signal,
    compute_choppiness_index,
    compute_regime_weight,
    compute_sigmoid_regime_weight,
    compute_dual_regime_weight,
    compute_signal_momentum,
    compute_effective_signal,
    compute_hysteresis_state,
)
from risk.controls import RiskManager


class BacktestEngine:
    """
    Runs a single-ticker backtest with the continuous-signal framework.

    Parameters
    ----------
    config : BacktestConfig
        All strategy, signal, regime, and risk parameters.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.sig_cfg = config.signals
        self.reg_cfg = config.regime
        self.ind_cfg = config.indicators
        self.risk_mgr = RiskManager(config.risk)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, ticker: str, df: pd.DataFrame) -> dict:
        """
        Execute a full backtest for a single ticker.

        Pipeline:
          1. Compute indicators (AMA, SMFI, DSMO, ADX)
          2. Compute signals (raw, smoothed, regime-weighted, hysteresis state)
          3. Walk-forward state machine (LONG / FLAT / SHORT)
          4. Close any open position at final bar

        Parameters
        ----------
        ticker : str
            Ticker symbol for logging.
        df : pd.DataFrame
            OHLCV data with columns: Open, High, Low, Close, Volume.

        Returns
        -------
        dict with keys:
            ticker       — str
            trades       — list[dict] of every entry and exit
            bar_log      — list[dict] of every bar's indicator values + equity
            equity_curve — pd.DataFrame with Date, Equity, Drawdown
        """
        df = df.copy()

        # Step 1 — Indicators
        df = self._compute_indicators(df)

        # Step 2 — Signals
        df = self._compute_signals(df)

        # Step 3 — Execution
        trades, bar_log, equity_df = self._run_state_machine(ticker, df)

        return {
            "ticker": ticker,
            "trades": trades,
            "bar_log": bar_log,
            "equity_curve": equity_df,
        }

    # ------------------------------------------------------------------
    # Step 1: Compute all indicators
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply AMA, SMFI, DSMO, and ADX to the dataframe."""
        ic = self.ind_cfg

        df = calculate_ama(
            df,
            bos_p=ic.ama_bos_period,
            slow_p=ic.ama_slow_period,
            fast_p=ic.ama_fast_period,
            push_fac=ic.ama_push_factor,
            anch_w=ic.ama_anchor_weight,
            smth_p=ic.ama_smooth_period,
            filter_th=ic.ama_filter_threshold,
        )

        df = calculate_smfi(
            df,
            flow_period=ic.smfi_flow_period,
            vol_period=ic.smfi_vol_period,
            inst_threshold=ic.smfi_inst_threshold,
            smth_period=ic.smfi_smooth_period,
            div_period=ic.smfi_div_period,
            div_th=ic.smfi_div_threshold,
            accum_th=ic.smfi_accum_threshold,
            dist_th=ic.smfi_dist_threshold,
        )

        df = calculate_dsmo(
            df,
            stoch_period=ic.dsmo_stoch_period,
            pre_smooth=ic.dsmo_pre_smooth,
            fast_smooth=ic.dsmo_fast_smooth,
            slow_smooth=ic.dsmo_slow_smooth,
            bottom_th=ic.dsmo_bottom_threshold,
            top_th=ic.dsmo_top_threshold,
        )

        df["ADX"] = compute_adx(df, period=ic.adx_period)

        # Choppiness Index — structural regime filter (improvement #3)
        df["choppiness"] = compute_choppiness_index(df, period=ic.ci_period)

        return df

    # ------------------------------------------------------------------
    # Step 2: Compute signals
    # ------------------------------------------------------------------

    def _compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the full signal pipeline and append columns to df.

        Columns added:
          raw_signal, smoothed_signal, adx_weight, regime_weight,
          effective_signal, signal_momentum, hysteresis_state
        """
        sc = self.sig_cfg
        rc = self.reg_cfg

        # Continuous raw signal from indicator components
        df["raw_signal"] = compute_raw_signal(
            df,
            ama_w=sc.ama_weight,
            smfi_w=sc.smfi_weight,
            dsmo_w=sc.dsmo_weight,
        )

        # EMA smoothing to dampen single-bar whipsaws
        df["smoothed_signal"] = smooth_signal(
            df["raw_signal"], period=sc.signal_ema_period
        )

        # Sigmoid ADX weight (improvement #5) — smooth continuous blending
        # or fall back to binary 3-zone if use_sigmoid=False in config
        if rc.use_sigmoid:
            df["adx_weight"] = compute_sigmoid_regime_weight(
                df["ADX"],
                midpoint=rc.sigmoid_midpoint,
                steepness=rc.sigmoid_steepness,
                floor=rc.adx_floor,
            )
        else:
            df["adx_weight"] = compute_regime_weight(
                df["ADX"],
                trend_threshold=self.ind_cfg.adx_trend_threshold,
                transition_low=self.ind_cfg.adx_transition_low,
                trending_weight=rc.trending_weight,
                transitional_weight=rc.transitional_weight,
                ranging_weight=rc.ranging_weight,
            )

        # Dual regime gate: ADX sigmoid + Choppiness Index (improvement #3)
        df["regime_weight"] = compute_dual_regime_weight(
            df["adx_weight"],
            df["choppiness"],
            ci_threshold=rc.ci_choppy_threshold,
            ci_enabled=rc.ci_gate_enabled,
        )

        # Effective signal = smoothed × regime weight
        df["effective_signal"] = compute_effective_signal(
            df["smoothed_signal"], df["regime_weight"]
        )

        # Signal momentum — rate-of-change for entry gating (improvement #2)
        df["signal_momentum"] = compute_signal_momentum(
            df["effective_signal"],
            lookback=sc.momentum_lookback,
        )

        # Hysteresis state machine (LONG / FLAT / SHORT)
        df["hysteresis_state"] = compute_hysteresis_state(
            df["effective_signal"],
            long_entry=sc.long_entry,
            long_exit=sc.long_exit,
            short_entry=sc.short_entry,
            short_exit=sc.short_exit,
        )

        return df

    # ------------------------------------------------------------------
    # Step 3: Walk-forward state machine
    # ------------------------------------------------------------------

    def _run_state_machine(
        self, ticker: str, df: pd.DataFrame
    ) -> tuple[list[dict], list[dict], pd.DataFrame]:
        """
        Walk forward through bars executing LONG / FLAT / SHORT transitions.

        Signal at bar i-1 (information through bar i-1 close) determines
        the action to take at bar i Open. No look-ahead bias.

        Also checks dynamic trailing stop each bar while in position.
        Stop width is adjusted by current SMFI reading.

        Returns (trades, bar_log, equity_df).
        """
        init_cap = self.config.risk.initial_capital
        cash = init_cap
        sc = self.sig_cfg

        # Position tracking
        shares = 0.0           # positive = long, negative = short
        avg_entry_price = 0.0   # weighted average entry price
        reference_price = 0.0   # highest close (long) or lowest close (short) for trailing stop
        bars_in_position = 0    # counter for trade duration

        # Pyramiding state (improvement #4)
        pyramid_count = 0           # number of additions made (0 = initial entry only)
        last_pyramid_signal = 0.0   # abs effective_signal at last entry/add for comparison
        entry_bar_idx = 0           # bar index of initial entry for time window
        target_shares = 0.0         # full target position size from vol-targeted sizing
        is_long_pos = True          # direction of current position

        # Multi-stage TP state (improvement #1)
        tp_targets: list[dict] = []  # list of {price, fraction, filled, label}

        trades: list[dict] = []
        bar_log: list[dict] = []

        # Realized vol from close-to-close returns
        daily_returns = df["Close"].pct_change()
        realized_vol = RiskManager.compute_realized_vol(
            daily_returns, lookback=self.config.risk.vol_lookback
        )

        n = len(df)

        for i in range(1, n):
            # Signal bar (bar i-1): information known at close of bar i-1
            sig_bar = df.iloc[i - 1]
            # Execution bar (bar i): trade at Open
            exec_bar = df.iloc[i]
            exec_date = df.index[i]
            exec_open = exec_bar["Open"]
            exec_close = exec_bar["Close"]
            exec_high = exec_bar["High"]
            exec_low = exec_bar["Low"]

            # Signal context at bar i-1 (the decision inputs)
            eff_signal = float(sig_bar.get("effective_signal", 0.0))
            raw_sig = float(sig_bar.get("raw_signal", 0.0))
            smooth_sig = float(sig_bar.get("smoothed_signal", 0.0))
            regime_w = float(sig_bar.get("regime_weight", 0.0))
            signal_mom = float(sig_bar.get("signal_momentum", 0.0))
            hyst_state = int(sig_bar.get("hysteresis_state", 0))
            smfi_val = float(sig_bar.get("SMFI", 50.0))
            atr_val = float(sig_bar.get("ATR", 0.0))
            vol = float(realized_vol.iloc[i - 1]) if i - 1 < len(realized_vol) else 0.20

            # --- State machine execution ---
            trade_action = None
            tp_fill = None  # TP target that got filled this bar (for logging)

            if shares == 0:
                # ---- FLAT ----
                # Compute full target position (used for pyramiding reference)
                sh_target, notional_target = self.risk_mgr.compute_position_size(
                    cash, exec_open, eff_signal, vol
                )

                # Signal momentum gate (improvement #2):
                # Only enter if conviction is BUILDING (rising for long, falling for short)
                mom_ok = True
                if sc.require_momentum_entry:
                    if hyst_state == 1 and signal_mom <= 0:
                        mom_ok = False  # signal not building bullish
                    elif hyst_state == -1 and signal_mom >= 0:
                        mom_ok = False  # signal not building bearish

                if hyst_state == 1 and mom_ok and sh_target > 0:
                    # Enter LONG — pyramid initial fraction (improvement #4)
                    entry_fraction = sc.pyramid_initial
                    sh = sh_target * entry_fraction
                    notional = abs(sh) * exec_open
                    cash -= notional
                    shares = sh
                    avg_entry_price = exec_open
                    reference_price = exec_close
                    bars_in_position = 0
                    is_long_pos = True
                    entry_bar_idx = i
                    pyramid_count = 0
                    last_pyramid_signal = abs(eff_signal)
                    target_shares = sh_target

                    # Locked TP targets at entry (improvement #1)
                    tp_targets = self.risk_mgr.compute_profit_targets(
                        exec_open, atr_val, is_long=True
                    )

                    trade_action = {
                        "Date": exec_date,
                        "Ticker": ticker,
                        "Action": "BUY",
                        "Price": round(exec_open, 4),
                        "Shares": round(shares, 4),
                        "Notional": round(notional, 2),
                        "Signal": round(eff_signal, 2),
                        "SignalMom": round(signal_mom, 2),
                        "SMFI": round(smfi_val, 2),
                        "RealizedVol": round(vol * 100, 2),
                        "PnL": 0.0,
                        "PnL_Pct": 0.0,
                        "Pyramid": "1/3",
                        "Reason": f"long entry (signal={eff_signal:.1f}, mom={signal_mom:+.1f})",
                    }
                    trades.append(trade_action)

                elif hyst_state == -1 and mom_ok and sh_target < 0:
                    # Enter SHORT — pyramid initial fraction
                    entry_fraction = sc.pyramid_initial
                    sh = sh_target * abs(entry_fraction)  # sh_target is negative, sh stays negative
                    # Actually sh_target is negative, fraction is positive. Need careful handling.
                    sh = sh_target * entry_fraction  # e.g., -1000 * 0.5 = -500
                    notional = abs(sh) * exec_open
                    short_proceeds = abs(sh) * exec_open
                    cash += short_proceeds
                    shares = sh
                    avg_entry_price = exec_open
                    reference_price = exec_close
                    bars_in_position = 0
                    is_long_pos = False
                    entry_bar_idx = i
                    pyramid_count = 0
                    last_pyramid_signal = abs(eff_signal)
                    target_shares = sh_target

                    # Locked TP targets at entry (improvement #1)
                    tp_targets = self.risk_mgr.compute_profit_targets(
                        exec_open, atr_val, is_long=False
                    )

                    trade_action = {
                        "Date": exec_date,
                        "Ticker": ticker,
                        "Action": "SELL_SHORT",
                        "Price": round(exec_open, 4),
                        "Shares": round(shares, 4),
                        "Notional": round(short_proceeds, 2),
                        "Signal": round(eff_signal, 2),
                        "SignalMom": round(signal_mom, 2),
                        "SMFI": round(smfi_val, 2),
                        "RealizedVol": round(vol * 100, 2),
                        "PnL": 0.0,
                        "PnL_Pct": 0.0,
                        "Pyramid": "1/3",
                        "Reason": f"short entry (signal={eff_signal:.1f}, mom={signal_mom:+.1f})",
                    }
                    trades.append(trade_action)

            elif shares > 0:
                # ---- LONG position ----
                bars_in_position += 1

                # Update trailing stop reference (highest close since entry)
                reference_price = max(reference_price, exec_close)

                # --- Multi-stage TP check (improvement #1) ---
                # Check if any TP level was hit using this bar's high
                tp_exit_shares = 0.0
                tp_exit_reason = ""
                for tp in tp_targets:
                    if not tp["filled"] and exec_high >= tp["price"]:
                        sell_frac = tp["fraction"]
                        tp_exit_shares += shares * sell_frac
                        tp["filled"] = True
                        tp_fill = tp
                        tp_exit_reason = f"TP {tp['label']} hit (target={tp['price']:.2f}, high={exec_high:.2f})"

                if tp_exit_shares > 0:
                    tp_exit_notional = tp_exit_shares * exec_open
                    cash += tp_exit_notional
                    tp_pnl = tp_exit_notional - tp_exit_shares * avg_entry_price
                    tp_pnl_pct = (
                        (exec_open - avg_entry_price) / avg_entry_price * 100.0
                        if avg_entry_price > 0 else 0.0
                    )
                    trades.append({
                        "Date": exec_date,
                        "Ticker": ticker,
                        "Action": "SELL",
                        "Price": round(exec_open, 4),
                        "Shares": round(tp_exit_shares, 4),
                        "Notional": round(tp_exit_notional, 2),
                        "Signal": round(eff_signal, 2),
                        "SMFI": round(smfi_val, 2),
                        "PnL": round(tp_pnl, 2),
                        "PnL_Pct": round(tp_pnl_pct, 4),
                        "BarsHeld": bars_in_position,
                        "Tier": tp_fill["label"] if tp_fill else "TP",
                        "Reason": tp_exit_reason,
                    })
                    trade_action = trades[-1]
                    shares -= tp_exit_shares
                    # If all shares sold via TP, reset position
                    if shares <= 1e-8:
                        shares = 0.0
                        avg_entry_price = 0.0
                        reference_price = 0.0
                        bars_in_position = 0
                        tp_targets = []

                # --- Pyramid add check (improvement #4) ---
                # Add to position if signal strengthens within the time window
                if shares > 0 and pyramid_count < 2:
                    bars_since_entry = i - entry_bar_idx
                    signal_improved = abs(eff_signal) - last_pyramid_signal >= sc.pyramid_signal_boost
                    if bars_since_entry <= sc.pyramid_max_bars and signal_improved and target_shares != 0:
                        add_shares = target_shares * sc.pyramid_add
                        add_notional = add_shares * exec_open
                        cash -= add_notional
                        # Weighted average entry price update
                        avg_entry_price = (
                            (avg_entry_price * shares + exec_open * add_shares)
                            / (shares + add_shares)
                        )
                        shares += add_shares
                        pyramid_count += 1
                        last_pyramid_signal = abs(eff_signal)
                        layer_label = f"{pyramid_count + 1}/3"
                        trades.append({
                            "Date": exec_date,
                            "Ticker": ticker,
                            "Action": "BUY",
                            "Price": round(exec_open, 4),
                            "Shares": round(add_shares, 4),
                            "Notional": round(add_notional, 2),
                            "Signal": round(eff_signal, 2),
                            "SignalMom": round(signal_mom, 2),
                            "SMFI": round(smfi_val, 2),
                            "PnL": 0.0,
                            "PnL_Pct": 0.0,
                            "Pyramid": layer_label,
                            "Reason": f"pyramid add (signal improved to {abs(eff_signal):.1f}, +{abs(eff_signal)-last_pyramid_signal+sc.pyramid_signal_boost:.1f}pts)",
                        })

                # --- Exit conditions (stop / reversal / hysteresis) ---
                if shares > 0:
                    stop_level = self.risk_mgr.compute_stop_level(
                        reference_price, atr_val, smfi_val, is_long=True
                    )
                    stop_hit = exec_close <= stop_level and atr_val > 0

                    exit_shares = 0.0
                    exit_reason = ""

                    if stop_hit:
                        exit_shares = shares
                        exit_reason = f"trailing stop hit (stop={stop_level:.2f}, close={exec_close:.2f})"
                    elif hyst_state == -1:
                        exit_shares = shares
                        exit_reason = f"signal reversal to SHORT (signal={eff_signal:.1f})"
                    elif hyst_state == 0:
                        exit_shares = shares
                        exit_reason = f"hysteresis exit to FLAT (signal={eff_signal:.1f})"

                    if exit_shares > 0:
                        exit_notional = exit_shares * exec_open
                        cash += exit_notional
                        pnl = exit_notional - exit_shares * avg_entry_price
                        pnl_pct = (
                            (exec_open - avg_entry_price) / avg_entry_price * 100.0
                            if avg_entry_price > 0
                            else 0.0
                        )
                        trades.append({
                            "Date": exec_date,
                            "Ticker": ticker,
                            "Action": "SELL",
                            "Price": round(exec_open, 4),
                            "Shares": round(exit_shares, 4),
                            "Notional": round(exit_notional, 2),
                            "Signal": round(eff_signal, 2),
                            "SMFI": round(smfi_val, 2),
                            "RealizedVol": round(vol * 100, 2),
                            "PnL": round(pnl, 2),
                            "PnL_Pct": round(pnl_pct, 4),
                            "BarsHeld": bars_in_position,
                            "Reason": exit_reason,
                        })
                        trade_action = trades[-1]
                        shares = 0.0
                        avg_entry_price = 0.0
                        reference_price = 0.0
                        bars_in_position = 0
                        tp_targets = []

            elif shares < 0:
                # ---- SHORT position ----
                bars_in_position += 1

                # Update trailing stop reference (lowest close since entry)
                if reference_price == 0:
                    reference_price = exec_close
                else:
                    reference_price = min(reference_price, exec_close)

                # --- Multi-stage TP check (improvement #1) ---
                # Check if any TP level was hit using this bar's low
                tp_cover_shares = 0.0
                tp_cover_reason = ""
                for tp in tp_targets:
                    if not tp["filled"] and exec_low <= tp["price"]:
                        cover_frac = tp["fraction"]
                        tp_cover_shares += abs(shares) * cover_frac
                        tp["filled"] = True
                        tp_fill = tp
                        tp_cover_reason = f"TP {tp['label']} hit (target={tp['price']:.2f}, low={exec_low:.2f})"

                if tp_cover_shares > 0:
                    tp_cover_cost = tp_cover_shares * exec_open
                    cash -= tp_cover_cost
                    tp_pnl = tp_cover_shares * avg_entry_price - tp_cover_cost
                    tp_pnl_pct = (
                        (avg_entry_price - exec_open) / avg_entry_price * 100.0
                        if avg_entry_price > 0 else 0.0
                    )
                    trades.append({
                        "Date": exec_date,
                        "Ticker": ticker,
                        "Action": "BUY_TO_COVER",
                        "Price": round(exec_open, 4),
                        "Shares": round(tp_cover_shares, 4),
                        "Notional": round(tp_cover_cost, 2),
                        "Signal": round(eff_signal, 2),
                        "SMFI": round(smfi_val, 2),
                        "PnL": round(tp_pnl, 2),
                        "PnL_Pct": round(tp_pnl_pct, 4),
                        "BarsHeld": bars_in_position,
                        "Tier": tp_fill["label"] if tp_fill else "TP",
                        "Reason": tp_cover_reason,
                    })
                    trade_action = trades[-1]
                    shares += tp_cover_shares  # shares is negative, so adding positive reduces |shares|
                    # If all shares covered via TP, reset position
                    if abs(shares) <= 1e-8:
                        shares = 0.0
                        avg_entry_price = 0.0
                        reference_price = 0.0
                        bars_in_position = 0
                        tp_targets = []

                # --- Pyramid add check (improvement #4) ---
                # Add to short if signal strengthens (more negative) within time window
                if shares < 0 and pyramid_count < 2:
                    bars_since_entry = i - entry_bar_idx
                    signal_improved = abs(eff_signal) - last_pyramid_signal >= sc.pyramid_signal_boost
                    if bars_since_entry <= sc.pyramid_max_bars and signal_improved and target_shares != 0:
                        add_shares = target_shares * sc.pyramid_add  # negative
                        add_notional = abs(add_shares) * exec_open
                        short_proceeds = abs(add_shares) * exec_open
                        cash += short_proceeds
                        # Weighted average entry price update
                        avg_entry_price = (
                            (avg_entry_price * abs(shares) + exec_open * abs(add_shares))
                            / (abs(shares) + abs(add_shares))
                        )
                        shares += add_shares  # more negative
                        pyramid_count += 1
                        last_pyramid_signal = abs(eff_signal)
                        layer_label = f"{pyramid_count + 1}/3"
                        trades.append({
                            "Date": exec_date,
                            "Ticker": ticker,
                            "Action": "SELL_SHORT",
                            "Price": round(exec_open, 4),
                            "Shares": round(add_shares, 4),
                            "Notional": round(short_proceeds, 2),
                            "Signal": round(eff_signal, 2),
                            "SignalMom": round(signal_mom, 2),
                            "SMFI": round(smfi_val, 2),
                            "PnL": 0.0,
                            "PnL_Pct": 0.0,
                            "Pyramid": layer_label,
                            "Reason": f"pyramid add short (signal improved to {abs(eff_signal):.1f})",
                        })

                # --- Cover conditions (stop / reversal / hysteresis) ---
                if shares < 0:
                    stop_level = self.risk_mgr.compute_stop_level(
                        reference_price, atr_val, smfi_val, is_long=False
                    )
                    stop_hit = exec_close >= stop_level and atr_val > 0

                    cover_shares = 0.0
                    cover_reason = ""

                    if stop_hit:
                        cover_shares = abs(shares)
                        cover_reason = f"trailing stop hit (stop={stop_level:.2f}, close={exec_close:.2f})"
                    elif hyst_state == 1:
                        cover_shares = abs(shares)
                        cover_reason = f"signal reversal to LONG (signal={eff_signal:.1f})"
                    elif hyst_state == 0:
                        cover_shares = abs(shares)
                        cover_reason = f"hysteresis cover to FLAT (signal={eff_signal:.1f})"

                    if cover_shares > 0:
                        cover_cost = cover_shares * exec_open
                        cash -= cover_cost
                        pnl = abs(shares) * avg_entry_price - cover_cost
                        pnl_pct = (
                            (avg_entry_price - exec_open) / avg_entry_price * 100.0
                            if avg_entry_price > 0
                            else 0.0
                        )
                        trades.append({
                            "Date": exec_date,
                            "Ticker": ticker,
                            "Action": "BUY_TO_COVER",
                            "Price": round(exec_open, 4),
                            "Shares": round(cover_shares, 4),
                            "Notional": round(cover_cost, 2),
                            "Signal": round(eff_signal, 2),
                            "SMFI": round(smfi_val, 2),
                            "RealizedVol": round(vol * 100, 2),
                            "PnL": round(pnl, 2),
                            "PnL_Pct": round(pnl_pct, 4),
                            "BarsHeld": bars_in_position,
                            "Reason": cover_reason,
                        })
                        trade_action = trades[-1]
                        shares = 0.0
                        avg_entry_price = 0.0
                        reference_price = 0.0
                        bars_in_position = 0
                        tp_targets = []

            # --- After exit reversal, check for opposite entry on same bar ---
            if shares == 0 and trade_action is not None:
                exit_action = trade_action["Action"]
                if hyst_state == 1 and exit_action in ("SELL", "BUY_TO_COVER"):
                    # Exit long and go short, or cover short and go long — already handled
                    pass

            # --- Daily mark-to-market ---
            # Position value: shares * close (long positive, short negative)
            pos_value = shares * exec_close
            equity = cash + pos_value

            bar_log.append({
                "Date": exec_date.strftime("%Y-%m-%d"),
                "Close": round(exec_close, 4),
                "Open": round(exec_open, 4),
                "High": round(exec_high, 4),
                "Low": round(exec_low, 4),
                "Volume": int(exec_bar.get("Volume", 0)) if not pd.isna(exec_bar.get("Volume", 0)) else 0,
                "AMA": round(float(sig_bar.get("AMA", 0)), 4),
                "ATR": round(atr_val, 4),
                "SMFI": round(smfi_val, 2),
                "SMFI_Zone": str(sig_bar.get("SMFI_Zone", "-")),
                "SMFI_Div": int(sig_bar.get("SMFI_Div", 0)),
                "DSMO_Fast": round(float(sig_bar.get("DSMO_Fast", 50)), 2),
                "DSMO_Slow": round(float(sig_bar.get("DSMO_Slow", 50)), 2),
                "DSMO_Zone": str(sig_bar.get("DSMO_Zone", "-")),
                "ADX": round(float(sig_bar.get("ADX", 0)), 2) if not pd.isna(sig_bar.get("ADX")) else 0.0,
                "Choppiness": round(float(sig_bar.get("choppiness", 0)), 2) if not pd.isna(sig_bar.get("choppiness")) else 0.0,
                "ADX_Weight": round(float(sig_bar.get("adx_weight", 0)), 4) if not pd.isna(sig_bar.get("adx_weight")) else 0.0,
                "RawSignal": round(raw_sig, 2),
                "SmoothSignal": round(smooth_sig, 2),
                "SignalMomentum": round(signal_mom, 2),
                "RegimeWeight": round(regime_w, 2),
                "EffSignal": round(eff_signal, 2),
                "State": {1: "LONG", 0: "FLAT", -1: "SHORT"}.get(int(sig_bar.get("hysteresis_state", 0)), "FLAT"),
                "Position": round(shares, 4),
                "PositionValue": round(pos_value, 2),
                "Equity": round(equity, 2),
                "Cash": round(cash, 2),
            })

        # --- Close any remaining position at final bar ---
        if shares != 0:
            final_bar = df.iloc[-1]
            final_date = df.index[-1]
            final_price = final_bar["Close"]

            if shares > 0:
                exit_notional = shares * final_price
                cash += exit_notional
                pnl = exit_notional - shares * avg_entry_price
                pnl_pct = (
                    (final_price - avg_entry_price) / avg_entry_price * 100.0
                    if avg_entry_price > 0 else 0.0
                )
                action = "SELL"
            else:
                cover_cost = abs(shares) * final_price
                cash -= cover_cost
                pnl = abs(shares) * avg_entry_price - cover_cost
                pnl_pct = (
                    (avg_entry_price - final_price) / avg_entry_price * 100.0
                    if avg_entry_price > 0 else 0.0
                )
                action = "BUY_TO_COVER"

            trades.append({
                "Date": final_date,
                "Ticker": ticker,
                "Action": action,
                "Price": round(final_price, 4),
                "Shares": round(abs(shares), 4),
                "Notional": round(abs(shares) * final_price, 2),
                "Signal": 0.0,
                "SMFI": 0.0,
                "RealizedVol": 0.0,
                "PnL": round(pnl, 2),
                "PnL_Pct": round(pnl_pct, 4),
                "BarsHeld": bars_in_position,
                "Reason": "end of data forced close",
            })

        # --- Build equity curve ---
        equity_df = pd.DataFrame(bar_log)
        if not equity_df.empty:
            equity_df["Date"] = pd.to_datetime(equity_df["Date"])
            equity_df = equity_df.set_index("Date")
            peak = equity_df["Equity"].expanding().max()
            equity_df["Drawdown"] = ((equity_df["Equity"] - peak) / peak * 100).round(4)

        return trades, bar_log, equity_df
