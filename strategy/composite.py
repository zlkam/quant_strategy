"""
Continuous composite signal construction from AMA, SMFI, and DSMO.

Three layers:
  1. Per-indicator component extraction (normalised, tanh-bounded)
  2. Weighted combination into a raw signal in [-100, +100]
  3. EMA smoothing to dampen single-bar whipsaws
"""

import numpy as np
import pandas as pd


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
# Per-Indicator Component Extraction (Phase A)
# ---------------------------------------------------------------------------

def compute_indicator_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the per-bar, unweighted indicator components for dynamic weighting.

    Returns a DataFrame with columns:
      ama_component  — AMA trend strength, tanh-bounded to [-1, +1]
      smfi_component — SMFI flow conviction, tanh-bounded to [-1, +1]
      dsmo_component — DSMO momentum position, tanh-bounded to [-1, +1]

    These components are the building blocks that get weighted and summed
    to produce the composite signal. By exposing them, dynamic weighting
    can optimize per-bar indicator contributions without re-extraction.

    The computations match those in compute_raw_signal() exactly — this
    function just separates the extraction from the weighted combination.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: 'AMA', 'ATR', 'SMFI', 'SMFI_Div', 'DSMO_Fast', 'DSMO_Slow'.

    Returns
    -------
    pd.DataFrame
        Columns: ama_component, smfi_component, dsmo_component.
        Index matches df.index.
    """
    # ---- AMA trend strength ----
    ama = df["AMA"]
    atr = df["ATR"]
    ama_trend = (ama - ama.shift(5)) / atr.clip(lower=1e-10)
    ama_comp = np.tanh(ama_trend)

    # ---- SMFI flow conviction ----
    smfi = df["SMFI"]
    smfi_div = df.get("SMFI_Div", pd.Series(0.0, index=df.index))
    smfi_dev = (smfi - 50.0) / 25.0
    div_mod = np.ones(len(df))
    div_mod[smfi_div == 1.0] = 1.5
    div_mod[smfi_div == -1.0] = 0.5
    smfi_comp = np.tanh(smfi_dev * div_mod)

    # ---- DSMO momentum position ----
    dsmo_fast = df["DSMO_Fast"]
    dsmo_slow = df["DSMO_Slow"]
    dsmo_pos = (dsmo_fast - 50.0) / 30.0
    golden_cross = (dsmo_fast.shift(1) < dsmo_slow.shift(1)) & (dsmo_fast > dsmo_slow)
    death_cross = (dsmo_fast.shift(1) > dsmo_slow.shift(1)) & (dsmo_fast < dsmo_slow)
    cross_boost = np.zeros(len(df))
    cross_boost[golden_cross] = 0.5
    cross_boost[death_cross] = -0.5
    dsmo_comp = np.tanh(dsmo_pos + cross_boost)

    return pd.DataFrame({
        "ama_component": ama_comp,
        "smfi_component": smfi_comp,
        "dsmo_component": dsmo_comp,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Weighted Signal Combination (Phase A)
# ---------------------------------------------------------------------------

def compute_weighted_signal(
    components: pd.DataFrame,
    ama_w: float = 0.45,
    smfi_w: float = 0.35,
    dsmo_w: float = 0.20,
) -> pd.Series:
    """
    Combine per-bar indicator components with weights into a composite signal.

    signal = (ama_w * ama_component + smfi_w * smfi_component
              + dsmo_w * dsmo_component) * 100.0

    The result is in [-100, +100]. Positive = bullish, negative = bearish.

    Parameters
    ----------
    components : pd.DataFrame
        From compute_indicator_components(). Must have columns:
        ama_component, smfi_component, dsmo_component.
    ama_w, smfi_w, dsmo_w : float
        Weights (should sum to ~1.0 for interpretability).

    Returns
    -------
    pd.Series
        Weighted composite signal in [-100, +100].
    """
    raw = (ama_w * components["ama_component"].values
           + smfi_w * components["smfi_component"].values
           + dsmo_w * components["dsmo_component"].values) * 100.0
    return pd.Series(raw, index=components.index)


# ---------------------------------------------------------------------------
# EMA Smoothing
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
