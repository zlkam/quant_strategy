# Quantitative Strategy Backtester — Implementation Plan v3

## Architecture: Continuous-Signal with 5 Research-Backed Improvements

This plan documents the current strategy architecture incorporating research findings from:
- trustdan/trend-following (293 backtests, pyramiding + multi-stage TP)
- OxfordStrat (42 futures, 34 years: ADX + structural filters)
- Alpha TRIX / fxer (multi-layer regime gating)
- Valeyre (2025): simple EMA optimal, don't over-engineer

## Strategy Overview

Three indicators (AMA trend, SMFI flow, DSMO momentum) combine into a continuous composite signal [-100, +100]. The signal is smoothed, dual-regime-gated (ADX sigmoid + Choppiness Index), and fed into a hysteresis state machine. Five improvements over the original:

1. **Multi-Stage Profit Targets** — 3 ATR-locked TP levels (6N/12N/20N) with 15% fractions each; 55% trails
2. **Signal Momentum Filter** — optional entry gate requiring building conviction
3. **Choppiness Index + ADX Dual Regime** — structural + directional regime detection
4. **Pyramiding Entry** — 70% initial + 30% add when signal strengthens (max 130%)
5. **Sigmoid Regime Blending** — smooth continuous ADX weight instead of binary zones

## Signal Pipeline

```
Indicators → Raw Signal [-100,+100] → EMA(3) → ADX Sigmoid → × CI Gate → Effective Signal
                                                                           ↓
                                                          Hysteresis → LONG/FLAT/SHORT
                                                                           ↓
                                            Position: 70% initial + 30% pyramid add
                                            Exit: 3-stage TP (15% each) + trailing stop
```

## Current Performance (1.5x leverage, 20% target vol)

| Ticker | Total Return | Ann. | Max DD | Sharpe | Calmar | vs B&H |
|--------|-------------|------|--------|--------|--------|--------|
| QQQ | +574% | 7.6% | -10.7% | 1.06 | 0.71 | Beats B&H |
| SPY | +446% | 6.7% | -8.1% | 1.00 | 0.83 | Beats B&H |
| NVDA | +857% | 9.1% | -11.3% | 1.12 | 0.80 | — |
| GOOGL | +575% | 9.3% | -10.5% | 1.10 | 0.89 | — |
| TSLA | +322% | 9.7% | -8.4% | 1.20 | 1.16 | — |
| META | +230% | 9.2% | -11.0% | 0.96 | 0.83 | — |
| AAPL | +721% | 8.4% | -17.3% | 1.07 | 0.49 | — |
| AMZN | +394% | 6.3% | -11.2% | 0.72 | 0.56 | — |
| MSFT | +369% | 6.1% | -13.6% | 0.74 | 0.45 | — |

### Targets Status

| Target | Status | Best |
|--------|--------|------|
| Max DD < 15% (QQQ/SPY) | ✅ Met | QQQ -10.7%, SPY -8.1% |
| Max DD < 25% (stocks) | ✅ Met | Worst: AAPL -17.3% |
| Beat/compete B&H (QQQ/SPY) | ✅ Met | QQQ +574% vs +551%, SPY +446% vs +369% |
| Sortino > 1.5 | ✅ Met (NVDA) | NVDA 1.51 |
| Sharpe > 1.5 | 🔶 Close | TSLA 1.20, NVDA 1.12 |
| Calmar > 1.5 | 🔶 Close | TSLA 1.16 |
| Ann. Return > 15% | ⬜ Working | Best: TSLA 9.7% |

### Levers to Close Remaining Gap

1. Increase `target_vol_annual` from 20% → 25%
2. Increase `max_exposure` from 1.5 → 2.0
3. Reduce `ci_choppy_threshold` to 80 (less restrictive)
4. Enable `require_momentum_entry` for better hit rate

## Parameters

See `config.py` for all tuneable parameters in 5 dataclasses.
