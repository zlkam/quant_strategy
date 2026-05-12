import numpy as np
import pandas as pd


def calculate_smfi(
    df: pd.DataFrame,
    flow_period: int = 14,
    vol_period: int = 20,
    inst_threshold: float = 1.5,
    smth_period: int = 5,
    div_period: int = 10,
    div_th: float = 0.5,
    accum_th: float = 60.0,
    dist_th: float = 40.0,
) -> pd.DataFrame:
    """
    Calculate the Smart Money Flow Index (SMFI) and its signals.

    The SMFI is a volume-driven capital flow indicator that detects whether
    institutional ("smart money") participants are actively accumulating or
    distributing a position. It combines three analytical layers:

      1. Chaikin-Blended Money Flow (CBMF) — granular directional buying /
         selling pressure derived from the Close Location Value (CLV), which
         measures where within each bar's range the close landed, weighted by
         volume.

      2. Institutional Volume Signature (IVS) — a rolling Z-score of volume
         that identifies abnormally large-volume bars (≥ inst_threshold σ above
         the rolling mean). When these bars coincide with directional flow, the
         composite score is amplified, expressing greater conviction in the
         direction of smart money.

      3. Price-Flow Divergence (PFD) — detects sign disagreement between
         rolling price rate-of-change and rolling SMFI rate-of-change.
         Bullish divergence (price falling, SMFI rising) signals accumulation
         into weakness. Bearish divergence (price rising, SMFI falling) signals
         distribution into strength.

    This indicator is designed to complement the AMA (macro trend) and the
    DSMO (momentum entry/exit timing). Neither AMA nor DSMO uses volume; the
    SMFI fills this gap entirely. When all three align, a multi-confirmation
    framework covering trend, momentum, and capital flow is achieved.

    Signal outputs
    --------------
    - ``SMFI_Signal`` : Threshold-crossing signal based on SMFI momentum.
                        +1 = SMFI crosses above accum_th from below
                             (capital flowing in → potential accumulation)
                        -1 = SMFI crosses below dist_th from above
                             (capital flowing out → potential distribution)
                         0 = no qualifying crossover

    - ``SMFI_Div``    : Price-flow divergence signal (secondary, high conviction).
                        +1 = bullish divergence: price falling, SMFI rising
                             (smart money accumulating into weakness)
                        -1 = bearish divergence: price rising, SMFI falling
                             (smart money distributing into strength)
                         0 = no divergence detected

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe. Must contain columns:
        ``'Close'``, ``'High'``, ``'Low'``, ``'Volume'``.
    flow_period : int, default 14
        EMA lookback for smoothing the buying and selling pressure series
        before computing the flow ratio. A longer period produces a slower,
        more structural reading of capital flow.
    vol_period : int, default 20
        Rolling window for computing the volume mean and standard deviation
        used in the IVS Z-score. 20 bars (~1 month of daily data) captures
        a representative baseline of normal activity.
    inst_threshold : float, default 1.5
        Minimum volume Z-score to classify a bar as institutionally driven.
        1.5σ corresponds to roughly the top 7% of volume observations under
        a normal distribution — elevated but not extreme.
    smth_period : int, default 5
        Final EMA smoothing pass applied to the raw composite SMFI score.
        Reduces bar-to-bar noise without adding significant lag.
    div_period : int, default 10
        Lookback in bars for computing price ROC and SMFI ROC used in
        divergence detection. 10 bars captures a short-term swing cycle.
    div_th : float, default 0.5
        Minimum absolute % price move (over div_period bars) required to
        qualify for divergence detection. Prevents flagging divergence during
        flat, low-volatility consolidation where the signal is unreliable.
    accum_th : float, default 60.0
        SMFI level above which the indicator is considered to be in an
        accumulation zone. Crossing this from below triggers SMFI_Signal = +1.
    dist_th : float, default 40.0
        SMFI level below which the indicator is considered to be in a
        distribution zone. Crossing this from above triggers SMFI_Signal = -1.

    Returns
    -------
    pd.DataFrame
        Original dataframe with the following columns appended:

        - ``'CLV'``          : Close Location Value in [-1, +1] per bar
        - ``'MFV'``          : Signed Money Flow Volume (CLV × Volume)
        - ``'CBMF'``         : Chaikin-Blended Money Flow score in [0, 100]
        - ``'Vol_Z'``        : Volume Z-score (institutional volume signature)
        - ``'SMFI_Raw'``     : Raw composite SMFI before final smoothing
        - ``'SMFI'``         : Final smoothed SMFI score in [0, 100]
        - ``'SMFI_Zone'``    : Zone label: ``'accumulation'``, ``'distribution'``,
                               or ``'neutral'``
        - ``'SMFI_Signal'``  : Threshold-crossing signal (+1 / -1 / 0)
        - ``'SMFI_Div'``     : Price-flow divergence signal (+1 / -1 / 0)

    Notes
    -----
    - All signals at bar ``t`` use only information available at bar ``t``
      close. Trades execute at bar ``t+1`` open. No look-ahead bias is
      introduced.
    - The H-L range in the CLV denominator and the volume standard deviation
      are both clamped to 1e-10 to prevent division-by-zero in illiquid or
      halted markets.
    - Volume must be actual traded volume (not tick count or open interest).
      For continuous futures contracts, ensure volume is consistent across
      roll dates.

    Examples
    --------
    ::

        import pandas as pd
        from SMFI import calculate_smfi

        df = pd.read_csv("ohlcv.csv", parse_dates=["Date"], index_col="Date")
        df = calculate_smfi(df)

        # Accumulation threshold crossings
        entries = df[df["SMFI_Signal"] == 1]

        # High-conviction divergence signals
        div_entries = df[df["SMFI_Div"] == 1]
    """
    df = df.copy()

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ------------------------------------------------------------------
    # Step 1 — Close Location Value (CLV)
    #
    # Measures where within the bar's High-Low range the close landed.
    # CLV = +1 : close at the high  → buyers fully dominated
    # CLV = -1 : close at the low   → sellers fully dominated
    # CLV =  0 : close at midpoint  → neither side won
    #
    # This is more informative than a binary up/down classification (used
    # in the standard Money Flow Index) because it captures the *degree*
    # of buyer or seller dominance on each bar.
    #
    # The H-L denominator is clamped to avoid division-by-zero on inside
    # bars or halted markets where High == Low.
    # ------------------------------------------------------------------
    hl_range = (high - low).clip(lower=1e-10)
    clv = ((close - low) - (high - close)) / hl_range
    clv = clv.clip(-1.0, 1.0)
    df["CLV"] = clv

    # ------------------------------------------------------------------
    # Step 2 — Signed Money Flow Volume (MFV)
    #
    # Scales each bar's volume by its CLV, producing a signed pressure
    # reading: positive = buying pressure dominated, negative = selling.
    # Large positive MFV bars are hallmarks of institutional accumulation;
    # large negative bars indicate institutional distribution.
    # ------------------------------------------------------------------
    mfv = clv * volume
    df["MFV"] = mfv

    # ------------------------------------------------------------------
    # Step 3 — Chaikin-Blended Money Flow Score (CBMF)
    #
    # Separates MFV into buying pressure (positive MFV) and selling
    # pressure (absolute negative MFV), then applies independent EMA
    # smoothing to each. The ratio of smoothed buying to total smoothed
    # pressure gives a stable, bounded [0, 100] flow score.
    #
    # Using EMA rather than a simple rolling sum (as in the original
    # Chaikin MF) means recent bars are weighted more heavily, making
    # the score more responsive to regime changes.
    #
    # A minimum pressure floor of 1e-10 prevents 0/0 in the ratio when
    # volume is zero on both sides simultaneously.
    # ------------------------------------------------------------------
    buy_pressure  = mfv.clip(lower=0.0).ewm(span=flow_period, adjust=False).mean()
    sell_pressure = mfv.clip(upper=0.0).abs().ewm(span=flow_period, adjust=False).mean()

    total_pressure = (buy_pressure + sell_pressure).clip(lower=1e-10)
    cbmf = (buy_pressure / total_pressure) * 100.0
    df["CBMF"] = cbmf

    # ------------------------------------------------------------------
    # Step 4 — Institutional Volume Signature (IVS) via Z-score
    #
    # Standardises volume relative to its recent rolling mean and std.
    # A Z-score above inst_threshold indicates a statistically abnormal
    # volume bar — the kind most consistent with institutional block
    # trading rather than retail participation.
    #
    # The rolling window is shifted by 1 bar to avoid look-ahead bias:
    # the Z-score at bar t is computed using history up to bar t-1.
    # ------------------------------------------------------------------
    vol_mean = volume.shift(1).rolling(window=vol_period, min_periods=2).mean()
    vol_std  = volume.shift(1).rolling(window=vol_period, min_periods=2).std().clip(lower=1e-10)
    vol_z    = (volume - vol_mean) / vol_std
    df["Vol_Z"] = vol_z

    # ------------------------------------------------------------------
    # Step 5 — Composite SMFI Score (volume-amplified)
    #
    # When institutional volume is present (vol_z > 0), the composite
    # score is pulled further from 50 in the direction of the current
    # flow, expressing greater conviction. The amplifier only activates
    # on elevated volume (vol_z clipped at 0 as lower bound), so normal
    # or thin-volume bars are not penalised — they simply use the base
    # CBMF reading.
    #
    # inst_weight in [0, 1]:
    #   0 = normal volume  → no amplification, SMFI_raw = CBMF
    #   1 = vol_z >= 3σ   → maximum amplification (doubles the distance
    #                        from 50), e.g. CBMF=70 → SMFI_raw=90
    #
    # The directional_pull term preserves sign: positive when CBMF > 50
    # (buying dominant), negative when CBMF < 50 (selling dominant).
    # ------------------------------------------------------------------
    inst_weight     = vol_z.clip(lower=0.0, upper=3.0) / 3.0
    directional_pull = (cbmf - 50.0) * inst_weight
    smfi_raw        = (cbmf + directional_pull).clip(0.0, 100.0)
    df["SMFI_Raw"]  = smfi_raw

    # ------------------------------------------------------------------
    # Step 6 — Final EMA Smoothing
    #
    # A light EMA pass removes residual bar-to-bar noise from the raw
    # composite score. smth_period is intentionally short (default 5)
    # to preserve responsiveness while eliminating spike artefacts from
    # single abnormal bars.
    # ------------------------------------------------------------------
    smfi = smfi_raw.ewm(span=smth_period, adjust=False).mean()

    # Fill the first few warm-up bars (where vol_std has insufficient history)
    # with 50 — the neutral midpoint — so downstream code never receives a NaN.
    smfi = smfi.fillna(50.0)
    df["SMFI"] = smfi

    # ------------------------------------------------------------------
    # Step 7 — Zone Classification
    #
    # Labels each bar according to which regime the SMFI is in.
    # accumulation : SMFI > accum_th → net buying / smart money flowing in
    # distribution : SMFI < dist_th  → net selling / smart money flowing out
    # neutral      : SMFI between dist_th and accum_th → no clear conviction
    # ------------------------------------------------------------------
    df["SMFI_Zone"] = "neutral"
    df.loc[smfi > accum_th, "SMFI_Zone"] = "accumulation"
    df.loc[smfi < dist_th,  "SMFI_Zone"] = "distribution"

    # ------------------------------------------------------------------
    # Step 8 — Threshold-Crossing Signal (SMFI_Signal)
    #
    # Fires when the SMFI crosses into the accumulation or distribution
    # zone with momentum — i.e., was outside the zone on the prior bar
    # and has now entered it. This is structurally analogous to the DSMO
    # golden/death cross: a zone entry under momentum, not a sustained
    # reading within the zone.
    #
    # +1 : SMFI crosses above accum_th → capital flowing in (long entry)
    # -1 : SMFI crosses below dist_th  → capital flowing out (exit / short)
    #  0 : no qualifying crossover
    # ------------------------------------------------------------------
    smfi_prev = smfi.shift(1)

    cross_into_accum = (smfi_prev <= accum_th) & (smfi > accum_th)
    cross_into_dist  = (smfi_prev >= dist_th)  & (smfi < dist_th)

    smfi_signal = np.zeros(len(df), dtype=float)
    smfi_signal[cross_into_accum] =  1.0
    smfi_signal[cross_into_dist]  = -1.0
    df["SMFI_Signal"] = smfi_signal

    # ------------------------------------------------------------------
    # Step 9 — Price-Flow Divergence Signal (SMFI_Div)
    #
    # Compares the direction of price movement with the direction of
    # SMFI movement over div_period bars. Divergence occurs when these
    # disagree — a classic Wyckoff / Elder signal of smart money acting
    # contrary to visible price action.
    #
    # Bullish divergence (+1):
    #   Price has fallen by more than div_th % BUT the SMFI has risen.
    #   Smart money is absorbing supply into weakness — accumulation.
    #
    # Bearish divergence (-1):
    #   Price has risen by more than div_th % BUT the SMFI has fallen.
    #   Smart money is selling into strength — distribution.
    #
    # The div_th price move filter prevents flagging divergence during
    # flat/low-volatility regimes where the signal has no meaning.
    # ------------------------------------------------------------------
    price_roc = close.pct_change(div_period) * 100.0   # % price change
    smfi_roc  = smfi.diff(div_period)                   # SMFI change (points)

    bull_div = (price_roc < -div_th) & (smfi_roc > 0)
    bear_div = (price_roc > div_th)  & (smfi_roc < 0)

    smfi_div = np.zeros(len(df), dtype=float)
    smfi_div[bull_div] =  1.0
    smfi_div[bear_div] = -1.0
    df["SMFI_Div"] = smfi_div

    return df
