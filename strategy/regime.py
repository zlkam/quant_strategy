"""
Regime weight computation: ADX-based binary, sigmoid, and dual (ADX + CI) gates.

Three approaches to gating signal exposure:
  1. Binary 3-zone (trending / transitional / ranging)
  2. Sigmoid continuous blending (smooth 0→1, no sharp boundaries)
  3. Dual gate = sigmoid ADX + Choppiness Index structural filter
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regime Weight (ADX-based, binary zones)
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
