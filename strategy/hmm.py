"""
Hidden Markov Model (HMM) regime detection.

Trains a rolling GaussianHMM on log-returns and volume to classify each
bar into one of three regimes: bull trend, transitional, or bear/ranging.
Outputs a continuous regime weight in [0, 1] used to gate signal exposure.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# HMM Regime Detection (Improvement #6)
# ---------------------------------------------------------------------------

def compute_hmm_regime(
    df: pd.DataFrame,
    lookback: int = 252,
    retrain_freq: int = 21,
    n_components: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """
    Detect market regimes using a rolling Hidden Markov Model.

    Trains a GaussianHMM on log-returns and volume every retrain_freq bars
    using a rolling lookback window. Outputs a continuous regime weight
    for each bar in [0, 1] representing how "trend-friendly" the regime is.

    Three regimes are detected (ordered by mean return):
      - Regime 0 (highest return): Bull trend → weight = 1.0
      - Regime 1 (middle):        Transitional / weak trend → weight = 0.5
      - Regime 2 (lowest):        Bear / ranging → weight = 0.0 (or bear exposure)

    Unlike ADX which only measures trend strength, HMM captures the full
    return distribution structure — including volatility clustering and
    regime persistence — providing smoother, less whipsaw-prone signals.

    Research basis: HMM regime detection + MPT allocation boosted SPY
    Sharpe from 0.53 to 0.79 (bspreston10, 2024).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close' and 'Volume'.
    lookback : int, default 252
        Rolling window (bars) for HMM training (~1 year of daily data).
    retrain_freq : int, default 21
        Retrain HMM every N bars to balance responsiveness vs compute cost.
    n_components : int, default 3
        Number of hidden states (bull / transition / bear-ranging).
    random_state : int, default 42
        Seed for reproducible HMM initialization.

    Returns
    -------
    np.ndarray
        Regime weight in [0, 1] per bar. 1.0 = bull-trend, 0.5 = transition,
        0.0 = bear/ranging. First (lookback) bars return 0.5 (neutral).
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("WARNING: hmmlearn not installed. Falling back to neutral HMM weight.")
        return np.full(len(df), 0.5)

    close = df["Close"].values
    volume = df["Volume"].values

    # Features: log returns + volume change (standardized)
    log_ret = np.diff(np.log(np.maximum(close, 1e-10)), prepend=np.nan)
    log_ret[0] = 0.0
    vol_change = np.diff(volume, prepend=0) / (volume + 1)
    vol_change[0] = 0.0

    # Stack features and drop NaN
    features = np.column_stack([log_ret, vol_change])
    features = np.nan_to_num(features, nan=0.0)

    n = len(df)
    hmm_weight = np.full(n, 0.5)  # default: neutral

    if n < lookback + 10:
        return hmm_weight  # not enough data

    # Rolling HMM: retrain every retrain_freq bars, predict forward
    last_state_probs = None
    for start in range(0, n - lookback, retrain_freq):
        train_end = start + lookback
        if train_end > n:
            break

        train_data = features[start:train_end]

        # Standardize training window
        tr_mean = np.mean(train_data, axis=0)
        tr_std = np.std(train_data, axis=0).clip(min=1e-8)
        train_norm = (train_data - tr_mean) / tr_std

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model = GaussianHMM(
                    n_components=n_components,
                    covariance_type="full",
                    n_iter=1000,
                    random_state=random_state,
                    tol=1e-4,
                    init_params="stmc",  # initialize all params
                )
                model.fit(train_norm)

            # Predict regime for the training window
            states = model.predict(train_norm)
            state_means = []
            for s in range(n_components):
                mask = states == s
                if mask.sum() > 0:
                    state_means.append(log_ret[start:train_end][mask].mean())
                else:
                    state_means.append(0.0)

            # Order regimes by mean return: highest = bull
            regime_order = np.argsort(state_means)  # ascending
            # regime_order[0] = lowest return (bear/ranging) → weight 0.0
            # regime_order[1] = middle (transitional) → weight 0.5
            # regime_order[2] = highest return (bull) → weight 1.0

            # Predict forward from train_end to min(n, next retrain point)
            predict_end = min(start + lookback + retrain_freq, n)
            predict_data = features[train_end:predict_end]
            if len(predict_data) > 0:
                predict_norm = (predict_data - tr_mean) / tr_std
                pred_states = model.predict(predict_norm)

                for j, s in enumerate(pred_states):
                    idx = train_end + j
                    if idx >= n:
                        break
                    rank = np.where(regime_order == s)[0][0]
                    if rank == 2:
                        hmm_weight[idx] = 1.0   # bull → full exposure
                    elif rank == 1:
                        hmm_weight[idx] = 0.5   # transitional → half
                    else:
                        hmm_weight[idx] = 0.0   # bear/ranging → flat

        except Exception:
            # HMM fit failed (e.g., singular covariance) — keep neutral
            pass

    return hmm_weight
