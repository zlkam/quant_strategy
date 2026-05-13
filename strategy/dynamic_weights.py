"""
Dynamic indicator weight optimization via rolling grid search.

For each bar, evaluates all weight combinations over a rolling window,
scoring by the Sharpe ratio of: sign(weighted_signal) × next_bar_return.
The best weights are applied to the current bar's signal.

This is one of three weight methods alongside rolling-sharpe and MLP
(see ml_weights.py).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dynamic Signal Weights (Improvement #7) — grid search
# ---------------------------------------------------------------------------

def compute_dynamic_weights(
    components: pd.DataFrame,
    df: pd.DataFrame,
    lookback: int = 252,
    min_weight: float = 0.15,
) -> pd.DataFrame:
    """
    Compute rolling Sharpe-optimized per-bar indicator weights.

    Uses true per-component data for accurate optimization: for each bar,
    evaluates all weight combinations over a rolling window, scoring by
    the Sharpe ratio of: (weighted_signal) × (next_bar_return).

    Parameters
    ----------
    components : pd.DataFrame
        From compute_indicator_components(). Must have columns:
        ama_component, smfi_component, dsmo_component.
    df : pd.DataFrame
        Must contain 'Close' for forward returns computation.
    lookback : int, default 252
        Rolling window for weight optimization.
    min_weight : float, default 0.15
        Floor per indicator weight.

    Returns
    -------
    pd.DataFrame
        Columns: ama_w, smfi_w, dsmo_w — optimized weights per bar.
        First (lookback) bars use default 0.45/0.35/0.20.
    """
    n = len(components)
    close = df["Close"].values

    # Default weights for warmup period
    ama_w_arr = np.full(n, 0.45)
    smfi_w_arr = np.full(n, 0.35)
    dsmo_w_arr = np.full(n, 0.20)

    if n < lookback + 20:
        return pd.DataFrame({
            "ama_w": ama_w_arr, "smfi_w": smfi_w_arr, "dsmo_w": dsmo_w_arr,
        }, index=components.index)

    # Get component arrays
    ama_c = components["ama_component"].values
    smfi_c = components["smfi_component"].values
    dsmo_c = components["dsmo_component"].values

    # Forward returns: bar t's signal predicts bar t+1's return
    fwd_ret = np.zeros(n)
    ret = np.diff(close) / np.maximum(close[:-1], 1e-10)
    fwd_ret[:len(ret)] = ret  # fwd_ret[t] = return from t to t+1

    # Weight grid: step 0.05, each >= min_weight, sum to 1.0
    weight_steps = np.arange(min_weight, 1.0 - 2 * min_weight + 0.005, 0.05)
    weight_combos = [(wa, ws, 1.0 - wa - ws)
                     for wa in weight_steps
                     for ws in weight_steps
                     if 1.0 - wa - ws >= min_weight]

    if not weight_combos:
        weight_combos = [(0.45, 0.35, 0.20)]

    # Pre-compute weighted signals for all combos (n × k matrix)
    n_combos = len(weight_combos)
    weighted_signals = np.zeros((n, n_combos))
    for c, (wa, ws, wd) in enumerate(weight_combos):
        weighted_signals[:, c] = (wa * ama_c + ws * smfi_c + wd * dsmo_c)

    # Rolling optimization: for each bar, find best weights over lookback window
    sqrt252 = np.sqrt(252)
    for i in range(lookback, n):
        start = max(0, i - lookback)
        sig_slice = weighted_signals[start:i, :]  # (window, n_combos)
        ret_slice = fwd_ret[start:i]              # (window,)

        # Strategy return for each combo: signal * fwd_return
        # We use sign(signal) * return (directional), not raw signal * return
        strat_rets = np.sign(sig_slice) * ret_slice[:, np.newaxis]  # (window, n_combos)
        mu = np.mean(strat_rets, axis=0)       # (n_combos,)
        sigma = np.std(strat_rets, axis=0)      # (n_combos,)
        sigma = np.maximum(sigma, 1e-8)
        sharpes = mu / sigma * sqrt252          # (n_combos,)

        best_idx = np.argmax(sharpes)
        ama_w_arr[i] = weight_combos[best_idx][0]
        smfi_w_arr[i] = weight_combos[best_idx][1]
        dsmo_w_arr[i] = weight_combos[best_idx][2]

    return pd.DataFrame({
        "ama_w": ama_w_arr, "smfi_w": smfi_w_arr, "dsmo_w": dsmo_w_arr,
    }, index=components.index)
