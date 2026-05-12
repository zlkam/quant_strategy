# Continuous-Signal Quant Strategy

Multi-indicator directional strategy combining AMA (trend), SMFI (flow), and DSMO (momentum) into a continuous composite signal supporting LONG / FLAT / SHORT positioning with volatility-targeted sizing and dynamic trailing stops.

## Strategy Logic

### Signal Pipeline

```
Indicators → Raw Signal [-100,+100] → EMA(3) → ADX Regime Gate → Hysteresis → Position
```

1. **AMA** measures trend direction via adaptive smoothing — the directional backbone
2. **SMFI** detects institutional capital flow via volume analysis — confirms or dampens
3. **DSMO** provides momentum timing via triple-smoothed stochastic — refines entries/exits

### Execution

- **LONG entry**: Smoothed, regime-gated signal crosses above +40
- **LONG exit**: Signal crosses below +15 (or trailing stop hit)
- **SHORT entry**: Signal crosses below -40
- **SHORT cover**: Signal crosses above -15 (or trailing stop hit)

25-point hysteresis gap prevents whipsaw.

### Risk Controls

- **Position sizing**: Volatility-targeted — larger in low vol, smaller in high vol
- **Trailing stops**: SMFI-gated — wider during accumulation (let winners run), tighter during distribution (cut fast)
- **Regime filter**: ADX(14) keeps strategy flat in ranging markets
- **DD circuit breakers**: 15% max DD for ETFs, 25% for stocks

## Project Structure

```
quant_strategy/
├── config.py              # All tuneable parameters
├── main.py                # Entry point
├── strategy/              # Signal construction + hysteresis
├── backtest/              # Engine + metrics
├── risk/                  # Position sizing + stops
├── indicators/            # AMA, SMFI, DSMO
├── data/                  # CSV loader
├── utils/                 # .log output
├── historical_data/       # Input CSVs
└── logs/                  # Output logs
```

## Parameters

See `config.py` — all parameters are in five dataclasses:
- `IndicatorConfig` — lookback windows and thresholds
- `SignalConfig` — continuous weights, smoothing, hysteresis thresholds
- `RegimeConfig` — ADX zone multipliers
- `RiskConfig` — vol targeting, dynamic stops, DD limits
- `BacktestConfig` — top-level orchestrator

## Requirements

```
pandas>=1.5.0
numpy>=1.21.0
```

## Usage

```bash
python main.py
```

### Output

| File | Content |
|------|---------|
| `logs/<TICKER>_bars.log` | Per-bar indicator values + equity (pipe-delimited) |
| `logs/<TICKER>_trades.log` | Structured trade blocks with signal context + summary |
| `logs/portfolio_summary.log` | Combined metrics across all tickers |
