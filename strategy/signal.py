"""
Continuous signal construction from AMA, SMFI, and DSMO indicator outputs.

Replaces the old binary checklist (0/1 per indicator, capped at 100) with
intensity-scaled contributions in [-100, +100] that capture *how strong*
each signal is, not just whether it fired.

Also provides ADX computation, EMA smoothing, regime gating, and a
hysteresis state machine for LONG / FLAT / SHORT positioning.
"""

import numpy as np
import pandas as pd

from config import SignalConfig, RegimeConfig


# ---------------------------------------------------------------------------
# ADX — Average Directional Index
# ---------------------------------------------------------------------------

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Wilder's Average Directional Index (ADX).

    ADX quantifies trend strength regardless of direction. Values above
    20 indicate a trending market; below 20 indicate ranging/consolidation
    where trend-following indicators lose edge.

    Uses EMA (not Wilder's SMA) for the smoothing passes — this is the
    modern convention and produces a slightly more responsive ADX.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'High', 'Low', 'Close'.
    period : int, default 14
        Smoothing period for +DI, -DI, and the final ADX line.
        Wilder's original used 14 bars.

    Returns
    -------
    pd.Series
        ADX values. First (period * 2) bars are NaN due to warmup.

    Notes
    -----
    +DM and -DM are clamped to zero — a bar cannot have both positive
    directional movement simultaneously by definition.
    The DX denominator is clamped to 1e-10 to prevent division by zero
    when +DI and -DI are both zero (flat price).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range — the maximum of three potential ranges
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift(1)).abs(),
        "lc": (low - close.shift(1)).abs(),
    }).max(axis=1)

    # Directional Movement
    up_move = high.diff()        # H[t] - H[t-1]
    down_move = -(low.diff())    # L[t-1] - L[t] (positive when price moved lower)

    # +DM: upward move that exceeds downward move, else zero
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    # -DM: downward move that exceeds upward move, else zero
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smooth TR and DMs with EMA (equivalent to Wilder's smoothing in the limit)
    atr_smooth = tr.ewm(span=period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean()

    # Directional Indicators (+DI, -DI) as percentage of ATR
    plus_di = 100.0 * plus_dm_smooth / atr_smooth.clip(lower=1e-10)
    minus_di = 100.0 * minus_dm_smooth / atr_smooth.clip(lower=1e-10)

    # Directional Movement Index (DX)
    di_sum = (plus_di + minus_di).clip(lower=1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum

    # ADX = smoothed DX
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx


# ---------------------------------------------------------------------------
# Raw Signal — intensity-scaled composite
# ---------------------------------------------------------------------------

def compute_raw_signal(
    df: pd.DataFrame,
    ama_w: float = 0.45,
    smfi_w: float = 0.35,
    dsmo_w: float = 0.20,
) -> pd.Series:
    """
    Compute the continuous, intensity-scaled composite raw signal.

    Each indicator contributes a component normalised to roughly [-1, +1],
    scaled by its weight, then summed and multiplied by 100 to produce a
    signal in the range [-100, +100].

    Positive = bullish conviction across indicators.
    Negative = bearish conviction across indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns from all three indicators:
        'AMA', 'ATR', 'SMFI', 'SMFI_Div', 'DSMO_Fast', 'DSMO_Slow'.
    ama_w, smfi_w, dsmo_w : float
        Weights for each indicator's contribution. Should sum to ~1.0
        for interpretability.

    Returns
    -------
    pd.Series
        Raw signal in [-100, +100].

    Notes
    -----
    Component 1 — AMA Trend Strength
        (AMA[t] - AMA[t-5]) / ATR[t]
        The 5-bar lookback captures meaningful trend drift without being
        overly jittery (1 bar) or excessively lagged (10+ bars).
        Dividing by ATR normalises for volatility so a 1% AMA rise in
        a low-vol environment scores similarly to a 3% rise in high vol.
        The result is passed through tanh() for soft-bounding.

    Component 2 — SMFI Flow Conviction
        (SMFI[t] - 50) / 25
        Deviation from the neutral midpoint (50). SMFI at 75 → +1.0,
        SMFI at 25 → -1.0. Divergence signals (SMFI_Div) amplify or
        dampen: bullish divergence (SMFI_Div=+1) multiplies by 1.5x
        because it represents smart money accumulating into weakness —
        a higher-conviction signal. Bearish divergence (SMFI_Div=-1)
        multiplies by 0.5x — dampening bullishness / amplifying bearishness.

    Component 3 — DSMO Momentum Position
        (DSMO_Fast[t] - 50) / 30
        Continuous position of the fast line in oscillator space.
        DSMO_Fast at 80 → +1.0, DSMO_Fast at 20 → -1.0.
        Golden cross (fast crosses above slow) adds +0.5 to capture
        the momentum turn event. Death cross subtracts -0.5.
        This fixes the binary bottleneck — the DSMO contributes
        directionally on EVERY bar, not just on rare zone-gated crosses.

    All components are passed through np.tanh() for soft-bounding to
    [-1, +1] before weighting. This prevents extreme values during
    flash crashes or short squeezes from dominating the composite.
    """
    # ---- AMA trend strength ----
    # 5-bar rate-of-change normalised by ATR
    ama = df["AMA"]
    atr = df["ATR"]
    ama_trend = (ama - ama.shift(5)) / atr.clip(lower=1e-10)
    # Soft-bound to [-1, +1] via tanh
    ama_component = np.tanh(ama_trend)

    # ---- SMFI flow conviction ----
    smfi = df["SMFI"]
    smfi_div = df.get("SMFI_Div", pd.Series(0.0, index=df.index))
    # Deviation from neutral midpoint, scaled
    smfi_dev = (smfi - 50.0) / 25.0
    # Divergence modulation: bullish div → 1.5x boost, bearish div → 0.5x dampen
    div_mod = np.ones(len(df))
    div_mod[smfi_div == 1.0] = 1.5   # bullish divergence → higher conviction
    div_mod[smfi_div == -1.0] = 0.5  # bearish divergence → dampened bullish / amplified bearish
    smfi_adjusted = smfi_dev * div_mod
    smfi_component = np.tanh(smfi_adjusted)

    # ---- DSMO momentum position ----
    dsmo_fast = df["DSMO_Fast"]
    dsmo_slow = df["DSMO_Slow"]
    # Continuous position of fast line relative to oscillator midpoint
    dsmo_pos = (dsmo_fast - 50.0) / 30.0
    # Crossover event boost: golden cross = +0.5, death cross = -0.5
    # Use shift(1) to avoid look-ahead — cross is detected using prior bar
    golden_cross = (dsmo_fast.shift(1) < dsmo_slow.shift(1)) & (dsmo_fast > dsmo_slow)
    death_cross = (dsmo_fast.shift(1) > dsmo_slow.shift(1)) & (dsmo_fast < dsmo_slow)
    cross_boost = np.zeros(len(df))
    cross_boost[golden_cross] = 0.5
    cross_boost[death_cross] = -0.5
    dsmo_adjusted = dsmo_pos + cross_boost
    dsmo_component = np.tanh(dsmo_adjusted)

    # ---- Weighted composite ----
    raw = (ama_w * ama_component + smfi_w * smfi_component + dsmo_w * dsmo_component) * 100.0

    return pd.Series(raw, index=df.index)


# ---------------------------------------------------------------------------
# Signal Smoothing
# ---------------------------------------------------------------------------

def smooth_signal(raw_signal: pd.Series, period: int = 3) -> pd.Series:
    """
    Apply EMA smoothing to reduce single-bar whipsaws.

    A raw signal that spikes from 0 to 80 and back to 0 in three bars
    will be smoothed into a more gradual rise and fall. This prevents
    entry/exit on isolated noise bars while preserving the overall
    conviction trajectory.

    Parameters
    ----------
    raw_signal : pd.Series
        The raw composite signal from compute_raw_signal().
    period : int, default 3
        EMA span. 3 bars balances responsiveness with noise reduction.

    Returns
    -------
    pd.Series
        Smoothed signal.
    """
    smoothed = raw_signal.ewm(span=period, adjust=False).mean()
    # Fill warmup NaNs with the raw value to avoid losing early bars
    return smoothed.fillna(raw_signal)


# ---------------------------------------------------------------------------
# Regime Weight (ADX-based)
# ---------------------------------------------------------------------------

def compute_regime_weight(
    adx: pd.Series,
    trend_threshold: float = 20.0,
    transition_low: float = 15.0,
    trending_weight: float = 1.0,
    transitional_weight: float = 0.50,
    ranging_weight: float = 0.0,
) -> pd.Series:
    """
    Compute a regime weight multiplier based on ADX trend strength.

    Three zones:
      ADX >  trend_threshold  → trending:    full signal weight
      ADX between thresholds  → transitional: half signal weight
      ADX <  transition_low   → ranging:      flat (no exposure)

    The transitional zone provides gradualism — it prevents the strategy
    from flipping from full exposure to zero on a single ADX tick.
    The ranging weight of 0.0 reflects that trend/momentum indicators
    produce only noise in non-trending markets.

    Parameters
    ----------
    adx : pd.Series
        ADX values from compute_adx().
    trend_threshold : float, default 20.0
        ADX above this → trending (Wilder's canonical threshold).
    transition_low : float, default 15.0
        ADX below this → ranging.
    trending_weight, transitional_weight, ranging_weight : float
        Multipliers applied to the effective signal.

    Returns
    -------
    pd.Series
        Regime weight in [0.0, 1.0].
    """
    weight = pd.Series(ranging_weight, index=adx.index, dtype=float)

    # Transitional zone: ADX between transition_low and trend_threshold
    in_transition = (adx >= transition_low) & (adx <= trend_threshold)
    weight[in_transition] = transitional_weight

    # Trending zone: ADX above trend_threshold
    in_trend = adx > trend_threshold
    weight[in_trend] = trending_weight

    # NaN ADX (warmup bars) → conservative, treat as ranging
    weight[adx.isna()] = ranging_weight

    return weight


# ---------------------------------------------------------------------------
# Choppiness Index (structural regime filter)
# ---------------------------------------------------------------------------

def compute_choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute the Choppiness Index (CI) — a structural regime indicator.

    CI measures whether the market is trending or choppy (ranging),
    regardless of direction. It complements ADX by capturing the
    *structural* quality of price movement rather than directional strength.

    Formula (Dreiss, 1992):
      CI = 100 * log10(SUM(ATR, n) / (HHV_n - LLV_n)) / log10(n)

    Where:
      - SUM(ATR, n) = sum of True Range over n bars (total path length)
      - HHV_n - LLV_n = range over n bars (net distance traveled)
      - The ratio compares "how far price traveled in total" vs "how far
        it got"

    Interpretation:
      CI > 61.8  → market is choppy / sideways (Fibonacci ratio)
      CI < 38.2  → market is strongly trending
      CI 38.2-61.8 → transitional

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'High', 'Low', 'Close'.
    period : int, default 14
        Lookback window for CI calculation.

    Returns
    -------
    pd.Series
        Choppiness Index values in [0, 100].

    Notes
    -----
    During strong trends, price moves far (large HHV-LLV range) relative
    to the total path length → CI is low. During consolidation, price
    oscillates within a narrow range while ATR accumulates → CI is high.
    The denominator log10(period) normalises for the lookback window size.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift(1)).abs(),
        "lc": (low - close.shift(1)).abs(),
    }).max(axis=1)

    # Total path length = rolling sum of TR over period bars
    atr_sum = tr.rolling(window=period, min_periods=period).sum()

    # Net distance = highest high - lowest low over period bars
    hhv = high.rolling(window=period, min_periods=period).max()
    llv = low.rolling(window=period, min_periods=period).min()
    price_range = hhv - llv

    # CI formula — clamp denominator to prevent division by zero
    ratio = atr_sum / price_range.clip(lower=1e-10)
    ci = 100.0 * np.log10(ratio) / np.log10(period)

    return ci.clip(0.0, 100.0)


# ---------------------------------------------------------------------------
# Sigmoid Regime Weight (continuous blending)
# ---------------------------------------------------------------------------

def compute_sigmoid_regime_weight(
    adx: pd.Series,
    midpoint: float = 17.5,
    steepness: float = 2.5,
    floor: float = 12.0,
) -> pd.Series:
    """
    Compute a continuous regime weight using sigmoid blending of ADX.

    Improvement #5: replaces binary 3-zone ADX with a smooth sigmoid curve.
    No hard boundaries → fewer whipsaw entries/exits at zone edges.

      weight = 1 / (1 + exp(-(ADX - midpoint) / steepness))

    At ADX = midpoint: weight ≈ 0.5
    At ADX >> midpoint: weight → 1.0 (strongly trending)
    At ADX << midpoint: weight → 0.0 (ranging)

    A floor parameter forces weight to zero below a minimum ADX level
    to prevent very weak trends from generating any exposure.

    Parameters
    ----------
    adx : pd.Series
        ADX values from compute_adx().
    midpoint : float, default 17.5
        ADX level at which weight = 0.5. Centered between the traditional
        15 (ranging) and 20 (trending) thresholds.
    steepness : float, default 2.5
        Controls transition speed. Higher = sharper step. 2.5 gives a
        smooth transition spanning roughly ±5 ADX points around midpoint.
    floor : float, default 12.0
        ADX below this is forced to weight = 0 regardless of sigmoid output.

    Returns
    -------
    pd.Series
        Continuous regime weight in [0.0, 1.0].
    """
    # Sigmoid: 1 / (1 + exp(-x))
    x = (adx - midpoint) / steepness
    adx_weight = 1.0 / (1.0 + np.exp(-x))

    # Force zero below floor (prevent tiny exposure in clearly ranging markets)
    adx_weight[adx < floor] = 0.0

    # NaN ADX (warmup) → conservative, treat as ranging
    adx_weight[adx.isna()] = 0.0

    return adx_weight


# ---------------------------------------------------------------------------
# Dual Regime Weight (ADX sigmoid + Choppiness Index gate)
# ---------------------------------------------------------------------------

def compute_dual_regime_weight(
    adx_weight: pd.Series,
    choppiness: pd.Series,
    ci_threshold: float = 61.8,
    ci_enabled: bool = True,
) -> pd.Series:
    """
    Combine ADX-based regime weight with Choppiness Index structural filter.

    Improvement #3: the dual gate requires BOTH:
      1. ADX sigmoid weight > 0 (trend has directional strength)
      2. Choppiness Index < threshold (price is structurally trending,
         not choppy)

    When CI indicates choppy/ranging (CI > 61.8), the regime weight is
    multiplied by 0.0 (blocked) regardless of ADX. This is the key
    combination that OxfordStrat research found superior to ADX alone.

    Parameters
    ----------
    adx_weight : pd.Series
        Sigmoid ADX weight from compute_sigmoid_regime_weight().
    choppiness : pd.Series
        Choppiness Index from compute_choppiness_index().
    ci_threshold : float, default 61.8
        CI above this → market is structurally choppy → block exposure.
        Fibonacci ratio 61.8 is the standard threshold per Dreiss (1992).
    ci_enabled : bool, default True
        If False, the CI gate is bypassed (used for A/B testing).

    Returns
    -------
    pd.Series
        Final dual regime weight in [0.0, 1.0].
    """
    weight = adx_weight.copy()

    if ci_enabled:
        # Block exposure when market is structurally choppy
        is_choppy = choppiness > ci_threshold
        weight[is_choppy] = 0.0

    # NaN CI (warmup) → conservative, block until we have data
    weight[choppiness.isna()] = 0.0

    return weight


# ---------------------------------------------------------------------------
# Signal Momentum (ROC of effective signal, for entry gating)
# ---------------------------------------------------------------------------

def compute_signal_momentum(
    effective_signal: pd.Series,
    lookback: int = 3,
) -> pd.Series:
    """
    Compute the rate-of-change of the effective signal over a lookback.

    Used as an entry gate: only enter when conviction is BUILDING
    (rising for longs, falling for shorts), not when it's fading.

    This is the "rising ADX slope" concept applied to our composite
    signal. OxfordStrat found entering when momentum is building
    outperforms entering on level alone.

    Parameters
    ----------
    effective_signal : pd.Series
        The regime-gated, smoothed signal.
    lookback : int, default 3
        Bars to look back for ROC calculation.

    Returns
    -------
    pd.Series
        Signal momentum (positive = building bullish, negative = building bearish).
    """
    roc = effective_signal - effective_signal.shift(lookback)
    return roc


# ---------------------------------------------------------------------------
# Effective Signal (regime-gated)
# ---------------------------------------------------------------------------

def compute_effective_signal(
    smoothed_signal: pd.Series,
    regime_weight: pd.Series,
) -> pd.Series:
    """
    Apply the regime weight to the smoothed signal.

    effective_signal = smoothed_signal * regime_weight

    When ADX indicates ranging (weight=0), the effective signal is 0
    regardless of indicator readings — the strategy stays flat.
    When ADX indicates trending (weight=1), the full smoothed signal
    drives positioning.

    Parameters
    ----------
    smoothed_signal : pd.Series
        The EMA-smoothed raw signal.
    regime_weight : pd.Series
        Regime multiplier from compute_regime_weight().

    Returns
    -------
    pd.Series
        Effective signal in [-100, +100].
    """
    return smoothed_signal * regime_weight


# ---------------------------------------------------------------------------
# Hysteresis State Machine
# ---------------------------------------------------------------------------

def compute_hysteresis_state(
    effective_signal: pd.Series,
    long_entry: float = 40.0,
    long_exit: float = 15.0,
    short_entry: float = -40.0,
    short_exit: float = -15.0,
) -> np.ndarray:
    """
    Hysteresis state machine for LONG / FLAT / SHORT positioning.

    Uses cross-over logic with memory: entry requires a stronger signal
    than exit, creating a "hold zone" that prevents whipsaw.

    State encoding:
      +1 = LONG
       0 = FLAT
      -1 = SHORT

    Transitions (evaluated sequentially, bar by bar):
      FLAT  → LONG:  effective_signal crosses ABOVE long_entry
      LONG  → FLAT:  effective_signal crosses BELOW long_exit
      FLAT  → SHORT: effective_signal crosses BELOW short_entry
      SHORT → FLAT:  effective_signal crosses ABOVE short_exit

    The 25-point gap between entry (+40) and exit (+15) is calibrated
    to be ~0.8σ above the typical signal noise floor (~10), preventing
    the classic "enter today, exit tomorrow" on minor fluctuations.

    Cross-over detection uses shift(1) comparison — the signal at bar t
    is compared to bar t-1 to determine if a threshold was crossed.
    No look-ahead bias.

    Parameters
    ----------
    effective_signal : pd.Series
        The regime-gated, smoothed signal.
    long_entry, long_exit : float
        Thresholds for long entry and exit.
    short_entry, short_exit : float
        Thresholds for short entry and exit (cover).

    Returns
    -------
    np.ndarray of int
        State array: +1 (long), 0 (flat), -1 (short).
    """
    n = len(effective_signal)
    state = np.zeros(n, dtype=int)

    # prev signal and state (for cross-over detection and memory)
    prev_signal = effective_signal.iloc[0] if n > 0 else 0.0
    prev_state = 0

    for i in range(n):
        curr_signal = effective_signal.iloc[i]
        if pd.isna(curr_signal):
            curr_signal = 0.0

        if prev_state == 0:
            # FLAT — check for entry signals
            if prev_signal <= long_entry and curr_signal > long_entry:
                prev_state = 1   # cross above long entry → go LONG
            elif prev_signal >= short_entry and curr_signal < short_entry:
                prev_state = -1  # cross below short entry → go SHORT

        elif prev_state == 1:
            # LONG — check for exit signal only
            if prev_signal >= long_exit and curr_signal < long_exit:
                prev_state = 0   # cross below long exit → go FLAT

        elif prev_state == -1:
            # SHORT — check for cover signal only
            if prev_signal <= short_exit and curr_signal > short_exit:
                prev_state = 0   # cross above short exit → go FLAT (cover)

        state[i] = prev_state
        prev_signal = curr_signal

    return state
