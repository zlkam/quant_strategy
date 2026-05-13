"""
Continuous signal construction — re-export shim.

All functions have been split into logical submodules:
  adx              — ADX, Choppiness Index
  composite        — raw signal, indicator components, weighted combo, smoothing
  regime           — binary/sigmoid/dual regime weights
  hysteresis       — signal momentum, effective signal, hysteresis state machine
  hmm              — HMM regime detection
  dynamic_weights  — grid-search dynamic weight optimization

This module re-exports everything so existing callers (engine.py, server.py,
cross_validate.py) continue to work unchanged.
"""

from .adx import compute_adx, compute_choppiness_index
from .composite import (
    compute_raw_signal,
    compute_indicator_components,
    compute_weighted_signal,
    smooth_signal,
)
from .regime import (
    compute_regime_weight,
    compute_sigmoid_regime_weight,
    compute_dual_regime_weight,
)
from .hysteresis import (
    compute_signal_momentum,
    compute_effective_signal,
    compute_hysteresis_state,
    compute_hmm_effective_signal,
)
from .hmm import compute_hmm_regime
from .dynamic_weights import compute_dynamic_weights
