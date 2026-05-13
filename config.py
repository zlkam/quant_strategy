"""
Centralised configuration for the Continuous-Signal Quant Strategy.

All tuneable parameters live here. Five dataclasses organised by domain:
  IndicatorConfig  — lookbacks and thresholds for AMA, SMFI, DSMO, ADX
  SignalConfig     — continuous signal weights, smoothing, hysteresis
  RegimeConfig     — ADX-based regime filter multipliers
  RiskConfig       — vol targeting, dynamic stops, per-ticker DD limits
  BacktestConfig   — top-level orchestrator config

No magic numbers anywhere else in the codebase.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Indicator Parameters
# ---------------------------------------------------------------------------

@dataclass
class IndicatorConfig:
    """
    Lookback windows and thresholds for the four indicators.

    AMA, SMFI, and DSMO parameters are carried forward from the original
    indicator implementations and are left unchanged — the improvement
    comes from how we *combine* their outputs, not from re-tuning them.
    """

    # --- AMA (Adaptive Moving Average) ---
    ama_bos_period: int = 20
    ama_slow_period: int = 40
    ama_fast_period: int = 6
    ama_push_factor: float = 0.5
    ama_anchor_weight: float = 0.1
    ama_smooth_period: int = 3
    ama_filter_threshold: float = 0.0001

    # --- SMFI (Smart Money Flow Index) ---
    smfi_flow_period: int = 14
    smfi_vol_period: int = 20
    smfi_inst_threshold: float = 1.5
    smfi_smooth_period: int = 5
    smfi_div_period: int = 10
    smfi_div_threshold: float = 0.5
    smfi_accum_threshold: float = 60.0
    smfi_dist_threshold: float = 40.0

    # --- DSMO (Dual-Smoothed Momentum Oscillator) ---
    dsmo_stoch_period: int = 34
    dsmo_pre_smooth: int = 5
    dsmo_fast_smooth: int = 5
    dsmo_slow_smooth: int = 13
    dsmo_bottom_threshold: float = 20.0
    dsmo_top_threshold: float = 80.0

    # --- ADX (Average Directional Index) ---
    # ADX quantifies trend strength regardless of direction.
    # Wilder (1978) canonical threshold: ADX > 20 = trending.
    adx_period: int = 14
    adx_trend_threshold: float = 20.0    # above this → trending regime
    adx_transition_low: float = 15.0     # below this → ranging regime

    # --- Choppiness Index (structural regime filter) ---
    ci_period: int = 14
    ci_choppy_threshold: float = 61.8    # above this → market is choppy/ranging

    # --- HMM Regime Detection (improvement #6) ---
    # Hidden Markov Model for probabilistic 3-regime classification.
    # Replaces ADX+CI binary gating with soft, transition-aware detection.
    # Research: HMM regime detection + MPT boosted SPY Sharpe 0.53→0.79.
    hmm_enabled: bool = True             # True = use HMM, False = fallback to ADX+CI
    hmm_lookback: int = 504              # rolling window for HMM training (2 years)
    hmm_retrain_freq: int = 63           # retrain HMM every N bars (~1 quarter)
    hmm_n_components: int = 3            # bull / bear / ranging


# ---------------------------------------------------------------------------
# Signal Construction & Hysteresis
# ---------------------------------------------------------------------------

@dataclass
class SignalConfig:
    """
    Continuous signal construction weights, smoothing, and hysteresis
    thresholds for the state machine.

    The three indicators contribute directionally to a raw signal in
    [-100, +100]. Positive = bullish conviction, negative = bearish.

    Weights are chosen so that:
      - AMA (trend direction) is the primary driver (0.45) because it
        fires on 60% of bars — it's the trend backbone.
      - SMFI (capital flow) provides volume/flow confirmation (0.35)
        that neither AMA nor DSMO captures.
      - DSMO (momentum timing) refines entry/exit precision (0.20)
        without being the binding constraint it was in the binary model.

    The 25-point hysteresis gap (+40 entry, +15 exit) is calibrated to
    be ~0.8σ above the typical signal noise floor (~10). This prevents
    the "enter today, exit tomorrow" whipsaw while still responding to
    genuine conviction shifts within a few bars.
    """

    # Continuous signal weights (contribution to raw_signal in [-100, +100])
    # When dynamic_weights=True, these are starting values only — they get
    # overridden each bar by rolling Sharpe-optimized weights.
    ama_weight: float = 0.45
    smfi_weight: float = 0.35
    dsmo_weight: float = 0.20

    # --- Dynamic signal weighting (improvement #7) ---
    # Three weight methods: "fixed" (default 0.45/0.35/0.20), "grid" (rolling
    # Sharpe-optimized grid search), "mlp" (neural network predictor).
    # Research: Deep Momentum Networks — dynamic weights beat fixed by +110% Sharpe.
    dynamic_weights: bool = True         # True = use weight_method, False = fixed
    weight_method: str = "grid"           # "fixed" | "grid" | "rolling_sharpe"
    dw_lookback: int = 252               # rolling window for weight optimization
    dw_min_weight: float = 0.15          # floor per indicator (prevents zeroing out)
    # MLP-specific (weight_method="mlp")
    mlp_train_lookback: int = 504        # training window in bars (~2 years)
    mlp_retrain_freq: int = 63           # retrain every N bars (~1 quarter)
    mlp_lr: float = 0.002               # learning rate
    mlp_seed: int = 42                   # random seed for reproducibility

    # EMA smoothing applied to raw_signal to dampen single-bar spikes.
    # 3 bars is the sweet spot: 1 bar = too jittery, 5 bars = excessive lag.
    signal_ema_period: int = 3

    # Hysteresis state machine thresholds (effective signal range is [-100, +100])
    long_entry: float = 40.0       # cross above → enter long
    long_exit: float = 15.0        # cross below → exit long
    short_entry: float = -50.0     # cross below → enter short
    short_exit: float = -20.0      # cross above → cover short

    # --- Short-side quality filters (improvement #2) ---
    # Shorts underperform longs in equities due to upward drift.
    # These filters ensure we only short in high-conviction bearish regimes.
    short_require_adx: bool = True     # require ADX > 20 for short entry (must be trending)
    short_require_smfi: bool = True    # require SMFI < 45 for short entry (distribution zone)
    short_smfi_max: float = 45.0       # SMFI must be below this to confirm distribution

    # --- Signal momentum filter (improvement #2) ---
    # Enter only when conviction is BUILDING (rising for long, falling for short).
    # Lookback for signal rate-of-change check at entry bar.
    # OxfordStrat research: entering when ADX slope is rising outperforms
    # entering on ADX level alone. Applied here to the composite signal.
    momentum_lookback: int = 3         # bars for signal ROC calculation
    require_momentum_entry: bool = False  # gate entries on signal momentum (disabled: too restrictive)

    # --- Pyramiding entry (improvement #4) ---
    # Scale into positions rather than all-in at first signal.
    # Alt26 strategy (trustdan): pyramiding delivered +33.5% on SPY.
    # Only the initial fraction is deployed at first entry.
    pyramid_initial: float = 1.00      # 100% of target at first entry (no pyramiding)
    pyramid_add: float = 0.0           # no additional layers
    pyramid_signal_boost: float = 8.0  # signal must improve by this many points
    pyramid_max_bars: int = 30         # within this many bars to add a layer


# ---------------------------------------------------------------------------
# Regime Filter (ADX-based)
# ---------------------------------------------------------------------------

@dataclass
class RegimeConfig:
    """
    Dual regime filter: ADX (sigmoid-blended) + Choppiness Index.

    Improvement #3 (CI) + #5 (sigmoid): replaces binary ADX zones with
    a smooth sigmoid transition AND adds a structural filter that gates
    exposure when the market is structurally choppy regardless of ADX.

    Sigmoid blending (improvement #5):
      weight = sigmoid((ADX - midpoint) / steepness)
      Smooth 0→1 transition, no sharp boundaries → fewer whipsaws.

    Choppiness Index gate (improvement #3):
      CI > 61.8 → market is choppy/sideways → block entries
      OxfordStrat finding: ADX alone may reduce performance. Its value
      emerges when combined with structural filters.

    The dual gate requires BOTH conditions: ADX-based weight > 0 AND CI below threshold.
    """

    # Sigmoid blending (improvement #5) — replaces binary trending/transitional/ranging
    sigmoid_midpoint: float = 17.5     # ADX level where weight = 0.5
    sigmoid_steepness: float = 2.5     # higher = sharper transition
    use_sigmoid: bool = True           # False → revert to binary 3-zone ADX
    trending_weight: float = 1.0       # used only if use_sigmoid=False
    transitional_weight: float = 0.50  # used only if use_sigmoid=False
    ranging_weight: float = 0.0        # used only if use_sigmoid=False
    # Minimum ADX for sigmoid to give weight (floor — prevents zero exposure
    # in very low ADX where we want to be flat)
    adx_floor: float = 8.0             # ADX below this → forced weight = 0

    # Choppiness Index dual gate (improvement #3)
    ci_choppy_threshold: float = 75.0  # above this → market is choppy → reduce exposure (further relaxed)
    ci_gate_enabled: bool = True       # False → revert to ADX-only filter

    # --- HMM Regime Detection (improvement #6) ---
    # Uses hmmlearn GaussianHMM for probabilistic 3-state classification.
    # Regime weights are continuous probabilities, not binary gates.
    use_hmm: bool = True               # True = HMM regime, False = ADX sigmoid + CI
    # HMM regime → position scaling (multipliers per detected regime)
    hmm_bull_weight: float = 1.0       # full exposure in bull regime
    hmm_bear_weight: float = 1.0       # full exposure in bear regime (for shorts)
    hmm_ranging_weight: float = 0.0    # flat in ranging (same as ADX ranging)


# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """
    Volatility-targeted position sizing, dynamic trailing stops, and
    per-ticker drawdown circuit breakers.

    Position sizing follows the risk-parity principle: size inversely
    to realized volatility so that each unit of risk is constant.

      target_exposure = (signal / 100) * (target_vol / realized_vol)
      clamped to [min_exposure, max_exposure]

    Trailing stop multipliers are SMFI-gated:
      - SMFI > 60 (accumulation): 3.0x ATR — let winners compound
      - SMFI < 40 (distribution): 1.0x ATR — cut fast
      - SMFI 40-60 (neutral):     2.0x ATR — standard

    Per-ticker max drawdown limits:
      - ETFs (QQQ, SPY): 15% — tighter because they're benchmarks
      - Individual stocks: 25% — wider because single-name vol is higher
    """

    # Capital
    initial_capital: float = 1_000_000.0

    # Volatility targeting
    target_vol_annual: float = 0.20     # 20% annualized target vol
    vol_lookback: int = 20              # bars for realized vol computation
    max_exposure: float = 1.0           # cap long exposure at 100% (no leverage)
    min_exposure: float = -1.0          # cap short exposure at 100% (no leverage)

    # Dynamic trailing stop ATR multipliers (SMFI-gated)
    atr_accumulation: float = 3.0       # SMFI > 60 — let winners run
    atr_base: float = 2.0               # SMFI 40-60 — standard
    atr_distribution: float = 1.0       # SMFI < 40 — cut fast
    atr_short: float = 1.5              # tighter base stop for shorts (equity drift asymmetry)

    # --- Dynamic exit adaptation (improvement #3) ---
    tp_adx_adaptive: bool = True        # adapt TP levels to ADX strength
    tp_adx_boost: float = 1.5           # multiply TP levels by this when ADX > 30
    stop_signal_adaptive: bool = True   # widen stop when conviction is strong
    time_exit_enabled: bool = True      # exit if position stagnates
    time_exit_bars: int = 30            # max bars with < 1% cumulative return
    time_exit_min_return: float = 0.01  # minimum return over time_exit_bars to stay in

    # --- Multi-stage profit targets (improvement #1) ---
    # At entry, lock ATR-based take-profit levels. Sell fractions at each.
    # The remaining fraction trails with the dynamic stop.
    # Research (trustdan 293-backtest study): multi-stage TP >> trailing-only.
    # Levels based on ATR multiples above entry (long) / below entry (short).
    # Wider levels (6N/12N/20N) so only genuine trend extensions trigger TP.
    # Smaller fractions (15% each) so 55% of position continues to trail.
    tp_levels: tuple[float, ...] = (6.0, 12.0, 20.0)     # ATR multiples
    tp_fractions: tuple[float, ...] = (0.15, 0.15, 0.15)  # sell fraction at each level
    # Remaining fraction (1 - 0.45) = 0.55 trails with dynamic stop
    tp_enabled: bool = True

    # Per-ticker max drawdown circuit breakers
    # Keyed by ticker symbol; 'default' used for any ticker not listed.
    max_dd_by_ticker: dict[str, float] = field(default_factory=lambda: {
        "QQQ": 0.15,
        "SPY": 0.15,
        "default": 0.25,
    })

    def get_max_dd(self, ticker: str) -> float:
        """Return the max drawdown limit for a given ticker."""
        return self.max_dd_by_ticker.get(ticker, self.max_dd_by_ticker["default"])


# ---------------------------------------------------------------------------
# Top-Level Backtest Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Top-level configuration aggregating all sub-configs."""

    data_dir: str = "historical_data"
    log_dir: str = "logs"
    tickers: list[str] = field(default_factory=list)  # empty = auto-discover

    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
