# Quantitative Strategy Backtester — Implementation Plan

## Architecture: Continuous-Signal Framework v2

This plan documents the redesigned strategy architecture that replaces the original binary conviction score system with a continuous, intensity-scaled signal framework supporting bidirectional (long/short) positioning.

## Strategy Overview

Three indicators — AMA (trend direction), SMFI (capital flow), and DSMO (momentum timing) — contribute directionally to a continuous composite signal in [-100, +100]. The signal is smoothed, gated by an ADX regime filter, and fed into a hysteresis state machine that determines LONG / FLAT / SHORT positioning. Volatility-targeted sizing and dynamic trailing stops (SMFI-gated) manage risk.

## Signal Construction

### Continuous Raw Signal (3 components, weighted sum)

1. **AMA Trend Strength** = tanh((AMA[t] - AMA[t-5]) / ATR[t])
   - 5-bar rate-of-change normalized by volatility
   - Weight: 0.45 (trend backbone — AMA fires on 60% of bars)

2. **SMFI Flow Conviction** = tanh((SMFI[t] - 50) / 25 × div_mod)
   - Deviation from neutral (50) scaled to [-1, +1]
   - Divergence boost: ×1.5 (bullish div), ×0.5 (bearish div)
   - Weight: 0.35 (volume/flow dimension unique to SMFI)

3. **DSMO Momentum Position** = tanh((DSMO_Fast[t] - 50) / 30 + cross_boost)
   - Continuous oscillator position + crossover event boost (+/-0.5)
   - Weight: 0.20 (timing refinement layer)

### Signal Pipeline

```
Raw Signal [-100,+100] → EMA(3) → Smoothed Signal
                                         ↓
ADX(14) → Regime Weight [0, 0.5, 1.0] → Effective Signal
                                         ↓
                          Hysteresis State Machine → LONG/FLAT/SHORT
```

### ADX Regime Filter

| ADX Zone | Weight | Rationale |
|----------|--------|-----------|
| ADX > 20 | 1.0 | Trending — full signal weight |
| ADX 15–20 | 0.50 | Transitional — half weight |
| ADX < 15 | 0.0 | Ranging — flat (no edge for trend indicators) |

### Hysteresis State Machine

| Transition | Condition |
|------------|-----------|
| FLAT → LONG | Effective signal crosses ABOVE +40 |
| LONG → FLAT | Effective signal crosses BELOW +15 |
| FLAT → SHORT | Effective signal crosses BELOW -40 |
| SHORT → FLAT | Effective signal crosses ABOVE -15 |

25-point gap between entry and exit prevents single-bar whipsaws.

## Risk Management

### Volatility-Targeted Position Sizing

```
target_exposure = (effective_signal / 100) × (target_vol / realized_vol)
clamped to [-max_exposure, +max_exposure]
```

- Target annual vol: 15%
- Max exposure: ±100% (no leverage)
- Vol floor: 50% of target (prevents extreme sizing in low vol)

### Dynamic Trailing Stops (SMFI-gated)

| SMFI Zone | ATR Multiplier | Rationale |
|-----------|----------------|-----------|
| > 60 (accumulation) | 3.0× | Let winners compound |
| 40–60 (neutral) | 2.0× | Standard swing stop |
| < 40 (distribution) | 1.0× | Cut fast, preserve capital |

### Per-Ticker Drawdown Limits

- ETFs (QQQ, SPY): 15% max DD
- Individual stocks: 25% max DD

## Performance Targets

| Metric | Target |
|--------|--------|
| Annualized Return | > 15% |
| Max DD (QQQ/SPY) | < 15% |
| Max DD (stocks) | < 25% |
| Sharpe Ratio | > 1.5 |
| Calmar Ratio | > 1.5 |
| vs Buy & Hold | Competitive (not a big loss) |

## Project Structure

```
quant_strategy/
├── config.py              # Centralised parameters (5 dataclasses)
├── main.py                # Entry point orchestrator
├── data/
│   └── loader.py          # CSV ingestion and normalisation
├── indicators/
│   ├── ama.py             # Adaptive Moving Average
│   ├── smfi.py            # Smart Money Flow Index
│   └── dsmo.py            # Dual-Smoothed Momentum Oscillator
├── strategy/
│   └── signal.py          # Continuous signal, ADX, hysteresis
├── backtest/
│   ├── engine.py          # 3-state execution engine
│   └── metrics.py         # Performance metrics (bidirectional)
├── risk/
│   └── controls.py        # Vol targeting, dynamic stops, DD limits
├── utils/
│   └── logging.py         # BarLogger + TradeLogger (.log format)
├── historical_data/       # Input CSV files (gitignored)
└── logs/                  # Output logs (gitignored)
```

## Output Logs

- `<TICKER>_bars.log` — Pipe-delimited per-bar log with all indicator values, signal components, position state, and equity (one line per bar)
- `<TICKER>_trades.log` — Structured trade blocks with entry/exit details, signal context, PnL, and per-stock performance summary
- `portfolio_summary.log` — Combined metrics for all tickers

## Requirements

```
pandas>=1.5.0
numpy>=1.21.0
```

Install: `pip install pandas numpy`

## Usage

```bash
python main.py
```

## Verification

1. Run `python main.py` — all tickers process without errors
2. Check `logs/<TICKER>_bars.log` — verify per-bar indicator values present
3. Check `logs/<TICKER>_trades.log` — verify entry/exit with signal context
4. Review `logs/portfolio_summary.log` — verify metrics
5. Key checks: exposure > 50%, max DD < limits, Sharpe trending upward
