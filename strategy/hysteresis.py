"""
Hysteresis state machine and signal gating.

The hysteresis state machine maps a continuous effective signal into
discrete LONG / FLAT / SHORT states using cross-over logic with memory.
Entry requires a stronger signal than exit, creating a "hold zone" that
prevents whipsaw on minor signal fluctuations.

Also includes signal momentum (rate-of-change filter) and the effective
signal computation that gates the smoothed signal by regime weight.
"""

import numpy as np
import pandas as pd


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


# ---------------------------------------------------------------------------
# HMM-Aware Effective Signal
# ---------------------------------------------------------------------------

def compute_hmm_effective_signal(
    smoothed_signal: pd.Series,
    hmm_weight: np.ndarray,
) -> pd.Series:
    """
    Apply HMM regime weight to the smoothed signal.

    Unlike the ADX sigmoid which gates only on trend strength, HMM weight
    captures the full return distribution regime. Bull regime (1.0) gives
    full signal, transitional (0.5) halves it, bear/ranging (0.0) flattens.

    Parameters
    ----------
    smoothed_signal : pd.Series
        EMA-smoothed raw signal.
    hmm_weight : np.ndarray
        HMM regime weights from compute_hmm_regime().

    Returns
    -------
    pd.Series
        Effective signal in [-100, +100].
    """
    return smoothed_signal * pd.Series(hmm_weight, index=smoothed_signal.index)
