# Proposal: Quantitative Deployment via Weighted Scoring Matrix

## 1. System Architecture: The Conviction Score
The system evaluates the real-time data from the three sub-systems and assigns a dynamic **Conviction Score** from 0 to 100. Trade execution is treated as a continuous state machine rather than a binary toggle. 

The entry logic is governed by:
$\text{Score} = (w_1 \cdot \text{AMA}) + (w_2 \cdot \text{SMFI}) + (w_3 \cdot \text{DSMO})$
**Entry Condition:** Execute long when $\text{Score} \ge 80$.

### 1.1 Indicator Weighting Logic (Entry)
* **AMA (Macro Trend) — Weight: 40 points**
    * **Logic:** Fighting the macro trend is statistically dangerous. The AMA, utilizing its Break of Structure (BOS) and volatility push factors, acts as the foundation.
    * *Scoring:* Grants 40 points if `AMA_Signal == 1` (confirmed uptrend). Grants 0 points if in a downtrend.
* **SMFI (Capital Flow & Conviction) — Weight: Up to 40 points**
    * **Logic:** Institutional backing provides the highest safety margin.
    * *Scoring:* Grants **25 points** for a standard cross into the accumulation zone (`SMFI_Signal == 1`). Grants the full **40 points** if a bullish price-flow divergence is detected (`SMFI_Div == 1`).
* **DSMO (Momentum Timing) — Weight: 20 points**
    * **Logic:** The DSMO measures where the price sits within its swing range. 
    * *Scoring:* Grants 20 points for a "golden cross" occurring inside the bottom zone (`DSMO_Signal == 1`).

## 2. Dynamic Exit Matrix: The State Machine
To avoid getting shaken out by normal market noise while still protecting capital from a macro regime shift, the exit logic is divided into three tiers. It responds mathematically to the decay of the Conviction Score.

### Tier 1: Momentum Decay (Scale-Out)
* **Trigger:** The Conviction Score drops below a maintenance threshold (e.g., $\text{Score} < 60$). This typically occurs when the DSMO fires a "death cross" (`DSMO_Signal == -1`) or the SMFI begins distributing, but the AMA is still technically in an uptrend.
* **Action:** **Liquidate 30% to 50% of the position.**
* **Logic:** This is the flexibility mechanism. It locks in profits when momentum fades, protecting against a deeper pullback. By keeping a "runner" position alive, you avoid a complete shakeout if the macro trend resumes.

### Tier 2: Trend Invalidation (The Bear Market Shield)
* **Trigger:** The AMA registers a confirmed downtrend (`AMA_Signal == -1` or `AMA < AMA.shift(1)`).
* **Action:** **Liquidate 100% of remaining position.**
* **Logic:** This is the strict cut-off. The AMA is designed to filter out chop and only turn when the structure breaks. If the AMA rolls over, the macro environment has shifted. Exiting here preserves capital before the start of a prolonged bear market. 

### Tier 3: Catastrophic Volatility Stop (The Failsafe)
* **Trigger:** Price breaches a dynamic volatility boundary, utilizing the Average True Range calculated by the AMA module. 
* **Formula:** $\text{Stop Price} = \text{Highest Close} - (1.5 \cdot \text{ATR})$
* **Action:** **Liquidate 100% immediately.**
* **Logic:** The ultimate failsafe. If a black-swan event or sudden institutional dump occurs faster than the moving averages can track, this trailing ATR stop severs the risk instantly.

## 3. Phase 1: Backtesting Protocol
Conquering this complex system requires rigorous historical simulation to debug the weighting parameters and exit thresholds.

* **Data Acquisition:** Source high-quality OHLCV data. Ensure the volume data reflects actual traded volume, not tick count, as the SMFI's validity entirely depends on this.
* **Grid Search Optimization:** Run multi-variable grid searches to calibrate the internal smoothing parameters of the indicators (e.g., `slow_p` for AMA, `stoch_period` for DSMO). Optimize strictly for risk-adjusted metrics like the Sortino ratio to measure the effectiveness of the Tier 1 Scale-Out logic against downside deviation.
* **Threshold Calibration:** Test varying entry thresholds ($T$) and scale-out thresholds to find the optimal balance between win rate, trade frequency, and drawdown minimization.
* **Ledger and Accounting:** When structuring the performance reporting module of the backtester, engineer the logging system to calculate and isolate monthly profit and loss strictly in the original currency of the deposit and withdrawal (e.g., SGD and USD). Do not force a conversion to a single base currency in the logs; maintaining distinct currency arrays prevents fluctuating exchange rates from artificially skewing the raw performance metrics of your trading logic.
* **Divergence Lag Accounting:** Factor in the inherent lag of the SMFI divergence signal, as it requires a lookback period to confirm price-flow disagreement. The backtester must simulate reality without look-ahead bias.
