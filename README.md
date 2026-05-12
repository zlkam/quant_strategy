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
yfinance>=0.2.30
requests>=2.28.0
```

## Usage

### Backtest
```bash
python main.py
```

### Daily Automation
```bash
python daily_signal.py
```
Runs the signal pipeline on yesterday's closing data and sends a bilingual (EN/ZH) Telegram report.

### Output

| File | Content |
|------|---------|
| `logs/<TICKER>_bars.log` | Per-bar indicator values + equity (pipe-delimited) |
| `logs/<TICKER>_trades.log` | Structured trade blocks with signal context + summary |
| `logs/portfolio_summary.log` | Combined metrics across all tickers |
| `logs/daily_runs/run_YYYY-MM-DD.log` | Daily automation signal log |

## GitHub Actions Automation

A workflow runs Mon–Fri at 12:30 UTC (8:30 AM ET, ~1 hour before US market open):

1. Fetches yesterday's OHLCV data from Yahoo Finance
2. Computes the full signal pipeline for all 9 tickers
3. Compares against tracked positions
4. Sends a bilingual Telegram report with actionable signals
5. Logs the run to `logs/daily_runs/`

### Setup

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather) and get the token
2. Get your chat ID (send a message to [@userinfobot](https://t.me/userinfobot))
3. Add to GitHub Secrets:
   - `TELEGRAM_BOT_TOKEN` — your bot token
   - `TELEGRAM_CHAT_ID` — your chat ID
4. Enable the workflow in GitHub Actions tab
