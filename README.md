# Continuous-Signal Quant Strategy

Multi-indicator directional strategy combining AMA (trend), SMFI (flow), and DSMO (momentum) into a continuous composite signal supporting LONG / FLAT / SHORT positioning with volatility-targeted sizing and dynamic trailing stops.

## Live Dashboard

```bash
uv run python web/server.py
# or: uv run uvicorn web.server:app --host 0.0.0.0 --port 8501
```

Opens at **http://localhost:8501**. FastAPI backend serves the HTML/JS frontend
and a JSON API. The frontend uses Plotly.js for candlestick + indicator charts
with pan/zoom, signal cards, position tracker, and auto-refresh.

Deploy for free on [Render](https://render.com) or [Railway](https://railway.app)
— see [Deployment](#deployment).

## Strategy Logic

### Signal Pipeline

```
Indicators → Raw Signal [-100,+100] → EMA(3) → ADX/HMM Regime Gate → Hysteresis → Position
```

1. **AMA** — trend direction via adaptive smoothing
2. **SMFI** — institutional capital flow via volume analysis
3. **DSMO** — momentum timing via triple-smoothed stochastic

### Execution

- **LONG entry**: Smoothed, regime-gated signal crosses above +40
- **LONG exit**: Signal crosses below +15 (or trailing stop hit)
- **SHORT entry**: Signal crosses below -50
- **SHORT cover**: Signal crosses above -20 (or trailing stop hit)

25-point hysteresis gap prevents whipsaw.

### Risk Controls

- **Position sizing**: Volatility-targeted — larger in low vol, smaller in high vol
- **Trailing stops**: SMFI-gated — wider during accumulation, tighter during distribution
- **Regime filter**: HMM + ADX complementary blend keeps strategy flat in ranging markets
- **DD circuit breakers**: 15% max DD for ETFs, 25% for stocks

## Project Structure

```
quant_strategy/
├── .github/workflows/         # CI/CD
│   └── daily_signal.yml       # Mon-Fri automated signal run
├── analysis/                  # Research & validation tools
│   ├── audit.py               # Strategy audit: scenario analysis, peer comparison
│   ├── optimize.py            # Walk-forward parameter optimization
│   └── cross_validate.py      # Cross-validation vs backtesting.py engine
├── automation/                # Scheduled jobs
│   └── daily_signal.py        # Daily signal pipeline + Telegram report
├── backtest/                  # Vectorized backtest engine + output
│   ├── engine.py              # State machine: signal → execution → equity
│   ├── metrics.py             # Sharpe, Sortino, Calmar, hit rates
│   └── logging.py             # Bar-level and trade-level log writers
├── data/
│   └── loader.py              # CSV ingestion (handles BOM, M/B/K suffixes)
├── indicators/                # Technical indicators
│   ├── ama.py                 # Adaptive Moving Average
│   ├── dsmo.py                # Double Smoothed Momentum Oscillator
│   └── smfi.py                # Smart Money Flow Index
├── risk/
│   └── controls.py            # Vol targeting, dynamic stops, TP levels
├── strategy/                  # Signal pipeline (continuous composite)
│   ├── adx.py                 # ADX + Choppiness Index
│   ├── composite.py           # Raw signal, indicator components, smoothing
│   ├── regime.py              # Binary, sigmoid & dual (ADX+CI) regime gates
│   ├── hysteresis.py          # Signal momentum, hysteresis state machine
│   ├── hmm.py                 # HMM regime detection (3-state Gaussian)
│   ├── dynamic_weights.py     # Grid-search dynamic weight optimization
│   ├── ml_weights.py          # Rolling-Sharpe + MLP weight predictors
│   └── signal.py              # Re-export shim (all public API)
├── web/                       # Web dashboard (FastAPI + HTML/JS)
│   ├── server.py              # FastAPI backend serving JSON API
│   └── static/
│       ├── index.html         # Frontend layout
│       ├── style.css          # Dark theme
│       └── app.js             # Plotly.js charts + interactivity
├── config.py                  # All tuneable parameters (5 dataclasses)
├── main.py                    # Backtest entry point
├── pyproject.toml             # uv project config + dependencies
└── README.md
```

## Parameters

See `config.py` — all parameters in five dataclasses:
- `IndicatorConfig` — lookback windows and thresholds
- `SignalConfig` — weights, smoothing, hysteresis thresholds
- `RegimeConfig` — ADX/HMM blending, CI gate
- `RiskConfig` — vol targeting, dynamic stops, DD limits
- `BacktestConfig` — top-level orchestrator

## Setup

### Install uv (fast Python package manager)

```bash
pip install uv          # or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Sync dependencies

```bash
uv sync                 # creates .venv + installs all packages
```

All dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

## Usage

### Backtest

```bash
uv run python main.py
```


### Parameter Optimization

```bash
uv run python analysis/optimize.py                    # all tickers
uv run python analysis/optimize.py --ticker QQQ       # single ticker
uv run python analysis/optimize.py --metric calmar    # score by Calmar ratio
uv run python analysis/optimize.py --fast             # reduced grid for quick runs
```

Best parameters saved to `logs/optimal_params.json`.

### Cross-Validation

```bash
uv run python analysis/cross_validate.py              # all tickers
uv run python analysis/cross_validate.py --ticker QQQ # single ticker
```

Compares our vectorized engine against `backtesting.py` (event-driven).

### Strategy Audit

```bash
uv run python analysis/audit.py
```

Runs 8-regime scenario analysis, peer comparison (B&H, 200-d MA, 60/40), and live-trading readiness checks.

### Daily Automation

```bash
uv run python automation/daily_signal.py
```

Runs the signal pipeline on yesterday's closing data and sends a bilingual (EN/ZH) Telegram report via GitHub Actions.

## Deployment

### Web Dashboard (free)

**Render** (easiest):
1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect repo, set:
   - **Build command**: `pip install uv && uv sync`
   - **Start command**: `uv run uvicorn web.server:app --host 0.0.0.0 --port $PORT`
4. Deploy — live at `https://your-app.onrender.com`

**Railway** ([railway.app](https://railway.app)):
1. Push to GitHub
2. New project → Deploy from GitHub
3. Set start command: `uv run uvicorn web.server:app --host 0.0.0.0 --port $PORT`
4. Deploy

### GitHub Actions (daily signal)

The workflow at `.github/workflows/daily_signal.yml` runs Mon–Fri at 02:00 UTC.

**Setup:**
1. Create a Telegram bot via [@BotFather](https://t.me/BotFather)
2. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
3. Add to GitHub Secrets:
   - `TELEGRAM_BOT_TOKEN` — your bot token
   - `TELEGRAM_CHAT_ID` — your chat ID

## Output

| File | Content |
|------|---------|
| `logs/<TICKER>_bars.log` | Per-bar indicator values + equity |
| `logs/<TICKER>_trades.log` | Structured trade blocks with summary |
| `logs/portfolio_summary.log` | Combined metrics across all tickers |
| `logs/daily_runs/run_YYYY-MM-DD.log` | Daily automation signal log |
| `logs/audit_report.log` | Strategy audit results |
| `logs/wfo_results.json` | Walk-forward optimization results |
