"""
Quant Strategy Web Dashboard — FastAPI backend.

Serves the HTML/JS frontend and a JSON API that the frontend calls
to fetch signals, chart data, and positions.

Usage:
    uv run python web/server.py
    uv run uvicorn web.server:app --host 0.0.0.0 --port 8501 --reload
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import BacktestConfig
from indicators import calculate_ama, calculate_dsmo, calculate_smfi
from strategy.signal import (
    compute_adx, compute_raw_signal, smooth_signal,
    compute_choppiness_index, compute_sigmoid_regime_weight,
    compute_dual_regime_weight, compute_effective_signal,
    compute_hysteresis_state, compute_hmm_regime, compute_signal_momentum,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Quant Strategy Dashboard API", version="0.1.0")

# Serve static frontend files from web/static/
STATIC = ROOT / "web" / "static"
STATIC.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

TICKERS = ["QQQ", "SPY", "AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
STATE_MAP = {1: "LONG", 0: "FLAT", -1: "SHORT"}

# Cache signals in memory (refresh each request for now — yfinance is fast)
_signal_cache: dict = {}
_cache_time: datetime | None = None
CACHE_TTL_SEC = 60  # refresh data at most once per minute


def _compute_signals_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full signal pipeline on a DataFrame."""
    cfg = BacktestConfig()
    ic, sc, rc = cfg.indicators, cfg.signals, cfg.regime
    df = df.copy()
    df = calculate_ama(df, bos_p=ic.ama_bos_period, slow_p=ic.ama_slow_period,
                       fast_p=ic.ama_fast_period, push_fac=ic.ama_push_factor,
                       anch_w=ic.ama_anchor_weight, smth_p=ic.ama_smooth_period,
                       filter_th=ic.ama_filter_threshold)
    df = calculate_smfi(df, flow_period=ic.smfi_flow_period, vol_period=ic.smfi_vol_period,
                        inst_threshold=ic.smfi_inst_threshold, smth_period=ic.smfi_smooth_period,
                        div_period=ic.smfi_div_period, div_th=ic.smfi_div_threshold,
                        accum_th=ic.smfi_accum_threshold, dist_th=ic.smfi_dist_threshold)
    df = calculate_dsmo(df, stoch_period=ic.dsmo_stoch_period, pre_smooth=ic.dsmo_pre_smooth,
                        fast_smooth=ic.dsmo_fast_smooth, slow_smooth=ic.dsmo_slow_smooth,
                        bottom_th=ic.dsmo_bottom_threshold, top_th=ic.dsmo_top_threshold)
    df["ADX"] = compute_adx(df, period=ic.adx_period)
    df["choppiness"] = compute_choppiness_index(df, period=ic.ci_period)
    df["raw_signal"] = compute_raw_signal(df, ama_w=sc.ama_weight,
                                          smfi_w=sc.smfi_weight, dsmo_w=sc.dsmo_weight)
    df["smoothed_signal"] = smooth_signal(df["raw_signal"], period=sc.signal_ema_period)
    df["adx_weight"] = compute_sigmoid_regime_weight(
        df["ADX"], midpoint=rc.sigmoid_midpoint,
        steepness=rc.sigmoid_steepness, floor=rc.adx_floor)
    if rc.use_hmm and ic.hmm_enabled:
        hmm_regime = compute_hmm_regime(
            df, lookback=ic.hmm_lookback, retrain_freq=ic.hmm_retrain_freq,
            n_components=ic.hmm_n_components)
        df["regime_weight"] = np.maximum(df["adx_weight"].values, hmm_regime)
    else:
        df["regime_weight"] = compute_dual_regime_weight(
            df["adx_weight"], df["choppiness"],
            ci_threshold=rc.ci_choppy_threshold, ci_enabled=rc.ci_gate_enabled)
    df["effective_signal"] = compute_effective_signal(df["smoothed_signal"], df["regime_weight"])
    df["signal_momentum"] = compute_signal_momentum(df["effective_signal"],
                                                     lookback=sc.momentum_lookback)
    df["hysteresis_state"] = compute_hysteresis_state(
        df["effective_signal"], long_entry=sc.long_entry, long_exit=sc.long_exit,
        short_entry=sc.short_entry, short_exit=sc.short_exit)
    return df


def _fetch_ticker(ticker: str, period: str = "3mo") -> pd.DataFrame:
    period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
    days = period_days.get(period, 90)
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]]
    df.index = pd.to_datetime(df.index).normalize()
    return df.sort_index()


def _build_all_signals(period: str) -> dict:
    """Fetch + compute signals for every ticker."""
    out = {}
    for t in TICKERS:
        try:
            df = _fetch_ticker(t, period)
            df_sig = _compute_signals_df(df)
            last = df_sig.iloc[-1]
            prev = df_sig.iloc[-2] if len(df_sig) >= 2 else last
            state = int(last.get("hysteresis_state", 0))
            out[t] = {
                "state": STATE_MAP.get(state, "FLAT"),
                "signal": round(float(last.get("effective_signal", 0)), 1),
                "prev_signal": round(float(prev.get("effective_signal", 0)), 1),
                "close": round(float(last["Close"]), 2),
                "adx": round(float(last.get("ADX", 0)), 1) if not pd.isna(last.get("ADX")) else 0,
                "smfi": round(float(last.get("SMFI", 50)), 1),
                "choppiness": round(float(last.get("choppiness", 50)), 1) if not pd.isna(last.get("choppiness")) else 50,
                "ama": round(float(last.get("AMA", 0)), 2),
                "atr": round(float(last.get("ATR", 0)), 2),
                "regime_weight": round(float(last.get("regime_weight", 0)), 2),
            }
        except Exception as e:
            out[t] = {"state": "ERROR", "signal": 0, "prev_signal": 0,
                      "close": 0, "adx": 0, "smfi": 0, "choppiness": 0,
                      "ama": 0, "atr": 0, "regime_weight": 0, "error": str(e)}
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(str(STATIC / "index.html"))


@app.get("/api/tickers")
async def api_tickers():
    return {"tickers": TICKERS}


@app.get("/api/thresholds")
async def api_thresholds():
    cfg = BacktestConfig().signals
    return {
        "long_entry": cfg.long_entry,
        "long_exit": cfg.long_exit,
        "short_entry": cfg.short_entry,
        "short_exit": cfg.short_exit,
    }


@app.get("/api/signals")
async def api_signals(period: str = Query("3mo", pattern="^(1mo|3mo|6mo|1y|2y)$")):
    return _build_all_signals(period)


@app.get("/api/chart/{ticker}")
async def api_chart(ticker: str, period: str = Query("3mo", pattern="^(1mo|3mo|6mo|1y|2y)$")):
    if ticker not in TICKERS:
        return {"error": f"Unknown ticker: {ticker}"}
    try:
        df = _fetch_ticker(ticker, period)
        df_sig = _compute_signals_df(df)
        # Return only the columns the frontend needs, as ISO-dated JSON arrays
        cols = ["Open", "High", "Low", "Close", "Volume",
                "AMA", "ATR", "SMFI", "ADX", "choppiness",
                "effective_signal", "hysteresis_state", "regime_weight"]
        available = [c for c in cols if c in df_sig.columns]
        result = {"ticker": ticker, "period": period, "bars": len(df_sig),
                  "start": df_sig.index[0].isoformat(),
                  "end": df_sig.index[-1].isoformat()}
        result["data"] = {
            col: [None if (isinstance(v, float) and np.isnan(v)) else v
                  for v in df_sig[col].values.tolist()]
            for col in available
        }
        result["dates"] = [d.isoformat() for d in df_sig.index]
        result["data"]["dates"] = result["dates"]
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/chart_lightweight/{ticker}")
async def api_chart_lightweight(ticker: str, period: str = Query("3mo", pattern="^(1mo|3mo|6mo|1y|2y)$")):
    """Same as /api/chart but returns row-oriented JSON (smaller payload for Plotly)."""
    if ticker not in TICKERS:
        return {"error": f"Unknown ticker: {ticker}"}
    try:
        df = _fetch_ticker(ticker, period)
        df_sig = _compute_signals_df(df)
        cols = ["Open", "High", "Low", "Close", "Volume",
                "AMA", "ATR", "SMFI", "ADX", "choppiness",
                "effective_signal", "hysteresis_state", "regime_weight"]
        available = [c for c in cols if c in df_sig.columns]
        rows = []
        for i, (idx, row) in enumerate(df_sig.iterrows()):
            r = {"date": idx.isoformat()}
            for c in available:
                v = row[c]
                if isinstance(v, float) and np.isnan(v):
                    r[c] = None
                elif isinstance(v, (np.integer,)):
                    r[c] = int(v)
                elif isinstance(v, (np.floating,)):
                    r[c] = round(float(v), 6)
                else:
                    r[c] = v
            rows.append(r)
        return {"ticker": ticker, "period": period, "bars": len(rows),
                "start": df_sig.index[0].isoformat(),
                "end": df_sig.index[-1].isoformat(),
                "data": rows}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/positions")
async def api_positions():
    pos_file = ROOT / "logs" / "current_positions.json"
    if pos_file.exists():
        with open(pos_file) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print(f"Starting Quant Strategy Dashboard at http://localhost:8501")
    uvicorn.run("web.server:app", host="0.0.0.0", port=8501, reload=True)
