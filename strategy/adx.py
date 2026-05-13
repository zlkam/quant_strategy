"""
ADX (Average Directional Index) and Choppiness Index.

ADX quantifies trend strength regardless of direction. CI measures whether
price is structurally trending or choppy/ranging. Together they form the
foundation of the regime detection system.
"""

import numpy as np
import pandas as pd


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
