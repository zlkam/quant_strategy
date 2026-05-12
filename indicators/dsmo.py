import numpy as np
import pandas as pd


def calculate_dsmo(
    df: pd.DataFrame,
    stoch_period: int = 34,
    pre_smooth: int = 5,
    fast_smooth: int = 5,
    slow_smooth: int = 13,
    bottom_th: float = 20.0,
    top_th: float = 80.0,
) -> pd.DataFrame:
    """
    Calculate the Dual-Smoothed Momentum Oscillator (DSMO) and its signals.

    The DSMO measures where price sits within its recent swing range via a
    triple-EMA-smoothed Stochastic %K, producing a smooth fast line and a
    lagged slow line. Crossovers of these two lines within defined extreme
    zones flag potential entry and exit points.

    Smoothing architecture (3 progressive EMA passes)
    --------------------------------------------------
    raw %K  ->  pre_smooth (noise floor reduction)
            ->  DSMO_Fast  (reactive momentum line)
            ->  DSMO_Slow  (lagged consensus baseline)

    A wider ``stoch_period`` (default 34 vs. Lane's original 14) widens the
    High-Low range denominator, keeping the oscillator away from the 0/100
    walls during normal market conditions and reserving the extreme zones for
    genuine trend exhaustion.

    Signal definition
    -----------------
    - ``DSMO_Signal`` : Zone-gated crossover signal.
                        +1 = golden cross inside bottom zone (potential long entry)
                        -1 = death cross inside top zone    (potential exit / short)
                         0 = no qualifying crossover

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe. Must contain columns: 'Close', 'High', 'Low'.
    stoch_period : int, default 34
        Lookback window for the raw Stochastic %K (highest-high / lowest-low range).
        Widened to 34 (from Lane's 14) so the H-L denominator is broad enough
        that price only approaches the 0/100 extremes during genuine trend moves,
        keeping the oscillator more centred during consolidation.
    pre_smooth : int, default 5
        EMA period for the first (pre-smoothing) pass applied directly to raw %K.
        Acts as a noise floor: removes bar-to-bar jitter before the fast/slow
        lines are derived, producing the flowing curve shape.
    fast_smooth : int, default 5
        EMA period for the second pass (applied to the pre-smoothed series),
        producing the fast line. Reacts to momentum turns with moderate lag.
    slow_smooth : int, default 13
        EMA period for the third pass (applied to the fast line), producing
        the slow line. Acts as the lagged consensus baseline for crossover detection.
    bottom_th : float, default 20.0
        Oscillator level below which both lines must sit for a golden cross to
        qualify as a buy signal. Corresponds to the oversold zone [0, bottom_th].
    top_th : float, default 80.0
        Oscillator level above which both lines must sit for a death cross to
        qualify as a sell signal. Corresponds to the overbought zone [top_th, 100].

    Returns
    -------
    pd.DataFrame
        Original dataframe with the following columns appended:

        - 'Stoch_K'     : Raw Stochastic %K in [0, 100]
        - 'Stoch_K_Pre' : Pre-smoothed %K -- EMA(Stoch_K, pre_smooth)
        - 'DSMO_Fast'   : Fast line -- EMA(Stoch_K_Pre, fast_smooth)
        - 'DSMO_Slow'   : Slow line -- EMA(DSMO_Fast, slow_smooth)
        - 'DSMO_Zone'   : Zone label: 'bottom', 'top', or 'neutral'
        - 'DSMO_Signal' : Zone-gated crossover signal (+1 / -1 / 0)

    Notes
    -----
    - All rolling operations are shifted by 1 bar before computing to avoid
      look-ahead bias. Signals at bar t use only information available at
      bar t close; trades execute at bar t+1 open.
    - The Stochastic range denominator is clamped to a minimum of 1e-10 to
      prevent division-by-zero in flat-price periods.

    Examples
    --------
    ::

        import pandas as pd
        from DSMO import calculate_dsmo

        df = pd.read_csv("ohlcv.csv", parse_dates=["Date"], index_col="Date")
        df = calculate_dsmo(df)
        entries = df[df["DSMO_Signal"] == 1]
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Step 1 — Raw Stochastic %K
    # Shift high/low by 1 bar so bar-t signal uses only bars 0..t-1 range,
    # preventing look-ahead bias.
    #
    # stoch_period is set to 34 by default (vs. Lane's 14). The wider window
    # means the H-L denominator captures a full intermediate swing, so price
    # only sits near 0/100 during genuine exhaustion rather than routine
    # consolidation noise.
    # ------------------------------------------------------------------
    high_shifted = df["High"].shift(1)
    low_shifted  = df["Low"].shift(1)

    highest_high = high_shifted.rolling(window=stoch_period, min_periods=1).max()
    lowest_low   = low_shifted.rolling(window=stoch_period, min_periods=1).min()

    # Clamp denominator to avoid division-by-zero in flat / illiquid markets
    hl_range = (highest_high - lowest_low).clip(lower=1e-10)

    raw_k = 100.0 * (df["Close"] - lowest_low) / hl_range

    # Stochastic is theoretically [0, 100]; clamp for numerical safety
    raw_k = raw_k.clip(0.0, 100.0)
    df["Stoch_K"] = raw_k

    # ------------------------------------------------------------------
    # Step 2 — Pre-Smoothing Pass (noise floor reduction)
    # Apply an initial EMA directly to raw %K before deriving the fast/slow
    # lines. This first pass absorbs bar-to-bar jitter and is the key change
    # that produces a flowing, XDDW-style curve instead of a jagged one.
    # The fast and slow lines are then built on top of this cleaner series.
    # ------------------------------------------------------------------
    pre_smoothed = raw_k.ewm(span=pre_smooth, adjust=False).mean()
    df["Stoch_K_Pre"] = pre_smoothed

    # ------------------------------------------------------------------
    # Step 3 — Fast Line (second EMA pass)
    # Applied to the pre-smoothed series rather than raw %K. The combination
    # of pre_smooth + fast_smooth gives effective double-smoothing at this
    # stage, making the fast line visually clean while still responding to
    # meaningful momentum shifts.
    # ------------------------------------------------------------------
    df["DSMO_Fast"] = pre_smoothed.ewm(span=fast_smooth, adjust=False).mean()

    # ------------------------------------------------------------------
    # Step 4 — Slow Line (third EMA pass)
    # Applied to the fast line, producing the lagged consensus baseline.
    # Three progressive EMA passes (Blau, 1993) substantially eliminate
    # whipsaw crossovers while preserving the shape of significant turns.
    # ------------------------------------------------------------------
    df["DSMO_Slow"] = df["DSMO_Fast"].ewm(span=slow_smooth, adjust=False).mean()

    # ------------------------------------------------------------------
    # Step 5 — Zone Classification
    # A golden cross is only meaningful in the bottom (oversold) zone;
    # a death cross only in the top (overbought) zone. Crossovers in the
    # neutral middle are treated as noise and ignored.
    # ------------------------------------------------------------------
    in_bottom = (df["DSMO_Fast"] < bottom_th) & (df["DSMO_Slow"] < bottom_th)
    in_top    = (df["DSMO_Fast"] > top_th)    & (df["DSMO_Slow"] > top_th)

    df["DSMO_Zone"] = "neutral"
    df.loc[in_bottom, "DSMO_Zone"] = "bottom"
    df.loc[in_top,    "DSMO_Zone"] = "top"

    # ------------------------------------------------------------------
    # Step 6 — Crossover Detection
    # Compare current and previous bar positions of fast vs slow.
    # golden_cross : fast was below slow, now above -> upward momentum turn
    # death_cross  : fast was above slow, now below -> downward momentum turn
    # ------------------------------------------------------------------
    fast_prev = df["DSMO_Fast"].shift(1)
    slow_prev = df["DSMO_Slow"].shift(1)

    golden_cross = (fast_prev < slow_prev) & (df["DSMO_Fast"] > df["DSMO_Slow"])
    death_cross  = (fast_prev > slow_prev) & (df["DSMO_Fast"] < df["DSMO_Slow"])

    # ------------------------------------------------------------------
    # Step 7 — DSMO Signal (zone-gated crossovers only)
    # +1 : golden cross inside bottom zone  -> potential long entry
    # -1 : death cross inside top zone      -> potential exit / short entry
    #  0 : no qualifying crossover
    # ------------------------------------------------------------------
    dsmo_signal = np.zeros(len(df), dtype=float)
    dsmo_signal[golden_cross & in_bottom] =  1.0
    dsmo_signal[death_cross  & in_top]    = -1.0
    df["DSMO_Signal"] = dsmo_signal

    return df
