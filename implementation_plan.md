# Quantitative Strategy Backtester Implementation Plan

This plan outlines the architecture and tasks required to build the backtesting framework as per your `Quantitative_Deployment_Proposal.md`. The framework will evaluate the Conviction Score using AMA, SMFI, and DSMO indicators, handle tiered exits, calculate advanced metrics, and output the required logs.

## Backtesting Initialization

> 1. **Starting Capital**: $1,000,000
> 2. **Allocation**: 100% of available capital per trade
> 3. **Scale-Out Size**: 50% (Depends on the signals, for example if a sudden drop in conviction score happens, we sell 50% of the position or even more.)

## Proposed Changes

### 1. Project Restructuring
To maintain a neat and scalable folder structure:
#### [MODIFY] [__init__.py](file:///e:/Projects/quant_strategy/indicators/__init__.py)
#### [MODIFY] [__init__.py](file:///e:/Projects/quant_strategy/backtest/__init__.py)
#### [DELETE] [DSMO.py](file:///e:/Projects/quant_strategy/DSMO.py)
#### [NEW] [DSMO.py](file:///e:/Projects/quant_strategy/indicators/DSMO.py)
#### [DELETE] [SMFI.py](file:///e:/Projects/quant_strategy/SMFI.py)
#### [NEW] [SMFI.py](file:///e:/Projects/quant_strategy/indicators/SMFI.py)

### 2. Backtesting Engine
#### [NEW] [engine.py](file:///e:/Projects/quant_strategy/backtest/engine.py)
- **Data Ingestion**: Load historical csv data from `historical_data/` for all tickers.
- **Signal Calculation**: Compute AMA, SMFI, and DSMO series.
- **Conviction Score System**: 
  - `AMA`: 40 points if `AMA_Signal == 1`.
  - `SMFI`: 25 points if `SMFI_Signal == 1` OR 40 points if `SMFI_Div == 1`.
  - `DSMO`: 20 points if `DSMO_Signal == 1`.
- **Execution State Machine**: 
  - **Entry**: Buy when Conviction Score $\ge$ 80.
  - **Exit Tier 1 (Momentum Decay)**: Sell 50% when Conviction Score < 60.
  - **Exit Tier 2 (Trend Invalidation)**: Sell 100% when `AMA_Signal == -1` or `AMA < AMA.shift(1)`.
  - **Exit Tier 3 (Failsafe)**: Sell 100% when Price $\le$ Highest Close - (1.5 * ATR).

### 3. Metrics & Logging
#### [NEW] [metrics.py](file:///e:/Projects/quant_strategy/backtest/metrics.py)
- Functions to calculate: Total Return, Annualized Return, Max Drawdown, Exposure Time (%), Volatility, Sharpe, Sortino, Calmar Ratio.
- Benchmarking logic for Buy & Hold return.

#### [NEW] [main.py](file:///e:/Projects/quant_strategy/main.py)
- Serves as the entry point script to run the backtest across all tickers.
- Generates `logs/detailed_trades.log` (all buys/sells/signals per stock).
- Generates `logs/metrics_summary.log` (portfolio and per-stock metrics).

### 4. Final Output Generation
#### [NEW] [README.md](file:///e:/Projects/quant_strategy/README.md)
#### [MODIFY] [requirements.txt](file:///e:/Projects/quant_strategy/requirements.txt)
- Clean up dependencies based only on what's utilized by the engine (e.g., pandas, numpy, etc.).

## Verification Plan

### Automated Tests
- Execute `python main.py` and ensure the engine traverses all `.csv` files inside `historical_data/` without errors.

### Manual Verification
- Review `logs/detailed_trades.log` to confirm entry prices align with signal timestamps, and check that Tier 1/2/3 exits trigger at the correct conditions.
- Ensure the metrics calculated logically match the resulting equity curve.
