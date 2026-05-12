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

    # --- ADX (Average Directional Index) — NEW ---
    # ADX quantifies trend strength regardless of direction.
    # Wilder (1978) canonical threshold: ADX > 20 = trending.
    adx_period: int = 14
    adx_trend_threshold: float = 20.0    # above this → trending regime
    adx_transition_low: float = 15.0     # below this → ranging regime


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
    ama_weight: float = 0.45
    smfi_weight: float = 0.35
    dsmo_weight: float = 0.20

    # EMA smoothing applied to raw_signal to dampen single-bar spikes.
    # 3 bars is the sweet spot: 1 bar = too jittery, 5 bars = excessive lag.
    signal_ema_period: int = 3

    # Hysteresis state machine thresholds (effective signal range is [-100, +100])
    long_entry: float = 40.0       # cross above → enter long
    long_exit: float = 15.0        # cross below → exit long
    short_entry: float = -40.0     # cross below → enter short
    short_exit: float = -15.0      # cross above → exit (cover) short


# ---------------------------------------------------------------------------
# Regime Filter (ADX-based)
# ---------------------------------------------------------------------------

@dataclass
class RegimeConfig:
    """
    ADX-based regime filter that scales the effective signal weight.

    Trend-following indicators (AMA, SMFI, DSMO) have no edge in
    ranging / consolidating markets — their signals are noise.
    This filter gates exposure accordingly:

      ADX > 20  → trending    → full signal weight (1.0)
      ADX 15-20 → transitional → half signal weight (0.50)
      ADX < 15  → ranging     → flat, no exposure (0.0)

    The transitional zone (15-20, rather than a hard cutoff at 20)
    provides gradualism and reduces whipsaw at regime boundaries.
    """

    trending_weight: float = 1.0
    transitional_weight: float = 0.50
    ranging_weight: float = 0.0


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
    target_vol_annual: float = 0.15     # 15% annualized target vol
    vol_lookback: int = 20              # bars for realized vol computation
    max_exposure: float = 1.0           # cap long exposure at 100%
    min_exposure: float = -1.0          # cap short exposure at 100%

    # Dynamic trailing stop ATR multipliers (SMFI-gated)
    atr_accumulation: float = 3.0       # SMFI > 60 — let winners run
    atr_base: float = 2.0               # SMFI 40-60 — standard
    atr_distribution: float = 1.0       # SMFI < 40 — cut fast

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
