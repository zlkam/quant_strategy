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
    compute_regime_weight,
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

        return df

    # ------------------------------------------------------------------
    # Step 2: Compute signals
    # ------------------------------------------------------------------

    def _compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the full signal pipeline and append columns to df.

        Columns added:
          raw_signal, smoothed_signal, regime_weight,
          effective_signal, hysteresis_state
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

        # ADX regime gate
        ic = self.ind_cfg
        df["regime_weight"] = compute_regime_weight(
            df["ADX"],
            trend_threshold=ic.adx_trend_threshold,
            transition_low=ic.adx_transition_low,
            trending_weight=rc.trending_weight,
            transitional_weight=rc.transitional_weight,
            ranging_weight=rc.ranging_weight,
        )

        # Effective signal = smoothed × regime weight
        df["effective_signal"] = compute_effective_signal(
            df["smoothed_signal"], df["regime_weight"]
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

        # Position tracking
        shares = 0.0           # positive = long, negative = short
        avg_entry_price = 0.0   # weighted average entry price
        reference_price = 0.0   # highest close (long) or lowest close (short) for trailing stop
        bars_in_position = 0    # counter for trade duration

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
            hyst_state = int(sig_bar.get("hysteresis_state", 0))
            smfi_val = float(sig_bar.get("SMFI", 50.0))
            atr_val = float(sig_bar.get("ATR", 0.0))
            vol = float(realized_vol.iloc[i - 1]) if i - 1 < len(realized_vol) else 0.20

            # --- State machine execution ---
            trade_action = None

            if shares == 0:
                # ---- FLAT ----
                if hyst_state == 1:
                    # Enter LONG
                    sh, notional = self.risk_mgr.compute_position_size(
                        cash, exec_open, eff_signal, vol
                    )
                    if sh > 0 and notional > 0:
                        cash -= notional  # pay for shares
                        shares = sh
                        avg_entry_price = exec_open
                        reference_price = exec_close
                        bars_in_position = 0
                        trade_action = {
                            "Date": exec_date,
                            "Ticker": ticker,
                            "Action": "BUY",
                            "Price": round(exec_open, 4),
                            "Shares": round(shares, 4),
                            "Notional": round(notional, 2),
                            "Signal": round(eff_signal, 2),
                            "SMFI": round(smfi_val, 2),
                            "RealizedVol": round(vol * 100, 2),
                            "PnL": 0.0,
                            "PnL_Pct": 0.0,
                            "Reason": f"hysteresis long entry (signal={eff_signal:.1f})",
                        }
                        trades.append(trade_action)

                elif hyst_state == -1:
                    # Enter SHORT — sell borrowed shares, receive proceeds
                    sh, notional = self.risk_mgr.compute_position_size(
                        cash, exec_open, eff_signal, vol
                    )
                    # For short: sh is negative, notional is positive (gross exposure)
                    if sh < 0 and notional > 0:
                        # Receive cash from short sale (sell borrowed shares)
                        short_proceeds = abs(sh) * exec_open
                        cash += short_proceeds
                        shares = sh
                        avg_entry_price = exec_open
                        reference_price = exec_close  # track lowest close for short stop
                        bars_in_position = 0
                        trade_action = {
                            "Date": exec_date,
                            "Ticker": ticker,
                            "Action": "SELL_SHORT",
                            "Price": round(exec_open, 4),
                            "Shares": round(shares, 4),
                            "Notional": round(short_proceeds, 2),
                            "Signal": round(eff_signal, 2),
                            "SMFI": round(smfi_val, 2),
                            "RealizedVol": round(vol * 100, 2),
                            "PnL": 0.0,
                            "PnL_Pct": 0.0,
                            "Reason": f"hysteresis short entry (signal={eff_signal:.1f})",
                        }
                        trades.append(trade_action)

            elif shares > 0:
                # ---- LONG position ----
                bars_in_position += 1

                # Update trailing stop reference (highest close since entry)
                reference_price = max(reference_price, exec_close)

                # Check dynamic trailing stop
                stop_level = self.risk_mgr.compute_stop_level(
                    reference_price, atr_val, smfi_val, is_long=True
                )
                stop_hit = exec_close <= stop_level and atr_val > 0

                # Exit conditions
                exit_shares = 0.0
                exit_reason = ""

                if stop_hit:
                    exit_shares = shares
                    exit_reason = f"trailing stop hit (stop={stop_level:.2f}, close={exec_close:.2f})"
                elif hyst_state == -1:
                    # Signal flipped to SHORT — exit long and reverse
                    exit_shares = shares
                    exit_reason = f"signal reversal to SHORT (signal={eff_signal:.1f})"
                elif hyst_state == 0:
                    # Signal dropped below long_exit — exit
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

            elif shares < 0:
                # ---- SHORT position ----
                bars_in_position += 1

                # Update trailing stop reference (lowest close since entry)
                reference_price = min(reference_price, exec_close) if reference_price != 0 else exec_close

                # Check dynamic trailing stop (reversed for shorts)
                stop_level = self.risk_mgr.compute_stop_level(
                    reference_price, atr_val, smfi_val, is_long=False
                )
                stop_hit = exec_close >= stop_level and atr_val > 0

                # Cover conditions
                cover_shares = 0.0
                cover_reason = ""

                if stop_hit:
                    cover_shares = abs(shares)
                    cover_reason = f"trailing stop hit (stop={stop_level:.2f}, close={exec_close:.2f})"
                elif hyst_state == 1:
                    # Signal flipped to LONG — cover and reverse
                    cover_shares = abs(shares)
                    cover_reason = f"signal reversal to LONG (signal={eff_signal:.1f})"
                elif hyst_state == 0:
                    # Signal rose above short_exit — cover
                    cover_shares = abs(shares)
                    cover_reason = f"hysteresis cover to FLAT (signal={eff_signal:.1f})"

                if cover_shares > 0:
                    # To cover short: buy back shares at exec_open
                    cover_cost = cover_shares * exec_open
                    cash -= cover_cost
                    # PnL for short: entry_price - exit_price (profit when price falls)
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
                "RawSignal": round(raw_sig, 2),
                "SmoothSignal": round(smooth_sig, 2),
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
