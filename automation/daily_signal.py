"""
Daily signal automation script.

Fetches yesterday's OHLCV data from Yahoo Finance, runs the continuous-signal
pipeline for all configured tickers, compares against tracked positions, and
sends a bilingual (EN/ZH) actionable report via Telegram Bot API.

Designed to run via GitHub Actions (M-F, 2am UTC) but also executable locally:
    uv run python automation/daily_signal.py

Environment variables (set in GitHub Secrets):
    TELEGRAM_BOT_TOKEN   — Telegram Bot API token
    TELEGRAM_CHAT_ID     — Target chat/group ID for report delivery
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Find project root (parent of automation/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import BacktestConfig
from backtest.engine import BacktestEngine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tickers to track (must match the ones we have indicator calibrations for)
TICKERS = ["QQQ", "SPY", "AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

# State file to persist positions between runs
POSITIONS_FILE = Path("logs/current_positions.json")

# Daily run log
DAILY_LOG_DIR = Path("logs/daily_runs")
DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Warmup bars needed for reliable indicator computation (~60 bars = ~3 months)
WARMUP_BARS = 120

# Encoding-safe print helper for Windows terminals that can't handle Unicode
def _safe_print(text: str) -> None:
    """Print, replacing unencodable characters on legacy terminals."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))

# State mapping for display
STATE_MAP = {1: "LONG", 0: "FLAT", -1: "SHORT"}

# Emoji/display mapping
STATE_EMOJI = {"LONG": "🟢", "FLAT": "⚪", "SHORT": "🔴"}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_recent_data(ticker: str, days: int = WARMUP_BARS + 5) -> pd.DataFrame:
    """
    Fetch recent daily OHLCV data from Yahoo Finance.

    Requests enough bars for indicator warmup plus yesterday's close.
    Retries up to 3 times with exponential backoff on failure.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    days : int
        Number of calendar days to fetch (includes non-trading days).

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume.
    Index is DatetimeIndex (dates only).
    """
    for attempt in range(3):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Flatten multi-level columns if any
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standardize column names
            df = df.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low",
                "Close": "Close", "Volume": "Volume",
            })

            # Keep only needed columns
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.index = pd.to_datetime(df.index).normalize()
            df = df.sort_index()

            return df

        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1} for {ticker}: {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to fetch {ticker} after 3 attempts: {e}")


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

def load_positions() -> dict:
    """
    Load tracked positions from the JSON state file.

    Returns empty dict on first run (assumes we hold all tickers at market).

    Position entry format:
        {"QQQ": {"state": "LONG", "entry_date": "2026-05-10",
                  "entry_price": 610.00, "shares": 1639,
                  "stop_level": 595.00, "capital": 1000000.00}}
    """
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_positions(positions: dict) -> None:
    """Persist current positions to JSON."""
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Signal pipeline (thin wrapper around existing engine)
# ---------------------------------------------------------------------------

def compute_current_signals() -> dict[str, dict]:
    """
    Fetch data and compute current signal state for all tickers.

    Returns a dict keyed by ticker with the last bar's signal state:
        {ticker: {close, raw_signal, smoothed_signal, effective_signal,
                  hysteresis_state, regime_weight, adx, choppiness,
                  ama, atr, smfi, smfi_zone, dsmo_fast, dsmo_slow,
                  signal_momentum, state_str, conviction_str}}
    """
    config = BacktestConfig()
    engine = BacktestEngine(config)

    results = {}

    for ticker in TICKERS:
        try:
            # Fetch data with warmup
            df = fetch_recent_data(ticker)

            # Run full pipeline
            df = engine._compute_indicators(df)
            df = engine._compute_signals(df)

            # Get last bar's signal state
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            hyst_state = int(last.get("hysteresis_state", 0))
            eff_signal = float(last.get("effective_signal", 0.0))
            prev_eff_signal = float(prev.get("effective_signal", 0.0))
            regime_w = float(last.get("regime_weight", 0.0))

            # Conviction label
            abs_sig = abs(eff_signal)
            if abs_sig >= 60:
                conviction = "Strong | 强"
            elif abs_sig >= 30:
                conviction = "Moderate | 中"
            elif abs_sig >= 10:
                conviction = "Weak | 弱"
            else:
                conviction = "None | 无"

            # Regime label
            if regime_w >= 0.8:
                regime = "Trending | 趋势"
            elif regime_w >= 0.3:
                regime = "Transitional | 过渡"
            else:
                regime = "Ranging | 震荡"

            print(f"  {ticker}: signal={eff_signal:+.1f}, state={STATE_MAP.get(hyst_state, 'FLAT')}")

            results[ticker] = {
                "close": round(float(last["Close"]), 2),
                "raw_signal": round(float(last.get("raw_signal", 0)), 2),
                "smoothed_signal": round(float(last.get("smoothed_signal", 0)), 2),
                "effective_signal": round(eff_signal, 2),
                "prev_effective_signal": round(prev_eff_signal, 2),
                "hysteresis_state": hyst_state,
                "state_str": STATE_MAP.get(hyst_state, "FLAT"),
                "regime_weight": round(regime_w, 2),
                "adx": round(float(last.get("ADX", 0)), 2) if not pd.isna(last.get("ADX")) else 0.0,
                "choppiness": round(float(last.get("choppiness", 0)), 2) if not pd.isna(last.get("choppiness")) else 0.0,
                "ama": round(float(last.get("AMA", 0)), 2),
                "atr": round(float(last.get("ATR", 0)), 2),
                "smfi": round(float(last.get("SMFI", 50)), 2),
                "smfi_zone": str(last.get("SMFI_Zone", "-")),
                "dsmo_fast": round(float(last.get("DSMO_Fast", 50)), 2),
                "dsmo_slow": round(float(last.get("DSMO_Slow", 50)), 2),
                "signal_momentum": round(float(last.get("signal_momentum", 0)), 2),
                "conviction_str": conviction,
                "regime_str": regime,
            }

        except Exception as e:
            print(f"  ERROR computing signals for {ticker}: {e}")
            results[ticker] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Action determination
# ---------------------------------------------------------------------------

def determine_actions(
    signals: dict[str, dict],
    positions: dict,
    is_first_run: bool = False,
) -> list[dict]:
    """
    Compare current signals against tracked positions to produce actions.

    Returns a list of action dicts with keys:
        ticker, action, action_cn, signal_state, details

    Action logic:
      - FLAT + LONG signal  → BUY
      - FLAT + SHORT signal → SELL_SHORT
      - FLAT + FLAT signal  → HOLD_CASH (no action)
      - LONG + LONG signal  → HOLD (or RAISE_STOP if stop moved)
      - LONG + FLAT signal  → SELL (exit)
      - LONG + SHORT signal → REVERSE_TO_SHORT
      - SHORT + LONG signal → REVERSE_TO_LONG
      - SHORT + FLAT signal → COVER
      - SHORT + SHORT signal → HOLD

    On first run (no saved positions), initializes positions based on
    current signal state instead of assuming LONG for all.
    """
    actions = []
    first_run = is_first_run or not positions

    for ticker in TICKERS:
        sig = signals.get(ticker, {})
        if "error" in sig:
            actions.append({
                "ticker": ticker,
                "action": "ERROR",
                "action_cn": "cn-error",
                "signal_state": "-",
                "details": sig["error"],
            })
            continue

        signal_state = sig["state_str"]
        pos = positions.get(ticker, {})
        current_pos_state = pos.get("state", "FLAT") if pos else "FLAT"

        # On first run, match position state to current signal
        if first_run:
            current_pos_state = signal_state if signal_state != "FLAT" else "FLAT"
            pos = {
                "state": current_pos_state,
                "entry_date": str(datetime.now().date()),
                "entry_price": sig["close"],
                "shares": 0,
                "stop_level": sig["close"] * (0.85 if current_pos_state == "LONG" else 1.15),
                "capital": "initial",
            }

        action = None
        action_cn = None
        details = ""
        action_data = {}

        # First run: just initialize, no action
        if first_run:
            if current_pos_state != "FLAT":
                action = "INIT"
                action_cn = "初始化持仓"
                details = (f"Initialized {current_pos_state} position | "
                          f"初始化{current_pos_state}仓位 @ {sig['close']}")
            else:
                action = "INIT_CASH"
                action_cn = "初始化现金"
                details = "Initialized — no position | 初始化 — 空仓"

        elif current_pos_state == "FLAT":
            if signal_state == "LONG":
                action = "BUY"
                action_cn = "买入"
                details = (f"Signal {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. "
                           f"Enter LONG. | 信号 {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. 买入做多.")
            elif signal_state == "SHORT":
                action = "SELL_SHORT"
                action_cn = "做空"
                details = (f"Signal {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. "
                           f"Enter SHORT. | 信号 {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. 开盘做空.")
            else:
                action = "HOLD_CASH"
                action_cn = "持有现金"
                details = f"No signal — stay in cash | 无信号 — 持有现金"

        elif current_pos_state == "LONG":
            if signal_state == "LONG":
                old_ref_price = pos.get("reference_price", sig["close"])
                new_ref_price = max(old_ref_price, sig["close"])
                new_stop = new_ref_price - sig["atr"] * 2.0
                old_stop = pos.get("stop_level", 0)

                action_data["new_ref_price"] = new_ref_price

                if new_stop > old_stop:
                    action = "RAISE_STOP"
                    action_cn = "上调止损"
                    details = f"Hold LONG — raise stop to {new_stop:.2f} | 持多仓 — 上调止损至 {new_stop:.2f}"
                    action_data["new_stop"] = new_stop
                else:
                    action = "HOLD"
                    action_cn = "持有多仓"
                    details = f"Hold LONG — conviction intact | 持有多仓 — 信号确认"
            elif signal_state == "FLAT":
                action = "SELL"
                action_cn = "卖出平多"
                details = (f"Signal {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. "
                           f"Exit LONG. | 信号 {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. 平多仓.")
            elif signal_state == "SHORT":
                action = "REVERSE_TO_SHORT"
                action_cn = "多翻空"
                details = (f"Signal {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. "
                           f"Exit LONG & enter SHORT. | 信号 {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. 平多仓并做空.")

        elif current_pos_state == "SHORT":
            if signal_state == "SHORT":
                old_ref_price = pos.get("reference_price", sig["close"])
                new_ref_price = min(old_ref_price, sig["close"])
                new_stop = new_ref_price + sig["atr"] * 2.0
                old_stop = pos.get("stop_level", float("inf"))

                action_data["new_ref_price"] = new_ref_price

                if new_stop < old_stop:
                    action = "LOWER_STOP"
                    action_cn = "下调止损"
                    details = f"Hold SHORT — lower stop to {new_stop:.2f} | 持空仓 — 下调止损至 {new_stop:.2f}"
                    action_data["new_stop"] = new_stop
                else:
                    action = "HOLD"
                    action_cn = "持有空仓"
                    details = f"Hold SHORT — conviction intact | 持有空仓 — 信号确认"
            elif signal_state == "FLAT":
                action = "COVER"
                action_cn = "买入平空"
                details = (f"Signal {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. "
                           f"Cover SHORT. | 信号 {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. 平空仓.")
            elif signal_state == "LONG":
                action = "REVERSE_TO_LONG"
                action_cn = "空翻多"
                details = (f"Signal {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. "
                           f"Cover SHORT & enter LONG. | 信号 {sig['prev_effective_signal']:+.1f} → {sig['effective_signal']:+.1f}. 平空仓并做多.")

        actions.append({
            "ticker": ticker,
            "action": action,
            "action_cn": action_cn,
            "signal_state": signal_state,
            "current_state": current_pos_state,
            "close": sig["close"],
            "effective_signal": sig["effective_signal"],
            "prev_effective_signal": sig.get("prev_effective_signal"),
            "conviction_str": sig["conviction_str"],
            "regime_str": sig["regime_str"],
            "atr": sig["atr"],
            "details": details,
            **action_data,
        })

    return actions


# ---------------------------------------------------------------------------
# Position update
# ---------------------------------------------------------------------------

def update_positions(actions: list[dict], positions: dict, signals: dict) -> dict:
    """
    Update the positions state file based on today's actions.

    Returns the updated positions dict.
    """
    today = datetime.now().date()

    for act in actions:
        ticker = act["ticker"]
        sig = signals.get(ticker, {})

        if act["action"] in ("INIT", "BUY", "REVERSE_TO_LONG"):
            positions[ticker] = {
                "state": "LONG",
                "entry_date": str(today),
                "entry_price": sig.get("close", 0),
                "reference_price": sig.get("close", 0),
                "shares": 0,  # not computed in daily mode
                "stop_level": round(sig.get("close", 0) - sig.get("atr", 0) * 2.0, 2),
                "capital": "auto",
            }
        elif act["action"] in ("SELL_SHORT", "REVERSE_TO_SHORT"):
            positions[ticker] = {
                "state": "SHORT",
                "entry_date": str(today),
                "entry_price": sig.get("close", 0),
                "reference_price": sig.get("close", 0),
                "shares": 0,
                "stop_level": round(sig.get("close", 0) + sig.get("atr", 0) * 2.0, 2),
                "capital": "auto",
            }
        elif act["action"] in ("SELL", "COVER", "HOLD_CASH", "INIT_CASH"):
            positions[ticker] = {
                "state": "FLAT",
                "entry_date": str(today),
                "entry_price": 0,
                "shares": 0,
                "stop_level": 0,
                "capital": "auto",
            }
        elif act["action"] in ("RAISE_STOP", "LOWER_STOP"):
            if ticker in positions:
                positions[ticker]["stop_level"] = act["new_stop"]
                positions[ticker]["reference_price"] = act["new_ref_price"]
        elif act["action"] == "HOLD":
            if ticker in positions and "new_ref_price" in act:
                positions[ticker]["reference_price"] = act["new_ref_price"]

    save_positions(positions)
    return positions


# ---------------------------------------------------------------------------
# Telegram message formatting and delivery
# ---------------------------------------------------------------------------

def format_telegram_message(
    actions: list[dict],
    signals: dict[str, dict],
    positions: dict,
) -> str:
    """
    Format the daily report as a bilingual Telegram message.

    Uses emoji indicators for readability. Keeps within Telegram's
    4096-character message limit by keeping content concise.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    weekday = datetime.now().strftime("%A")

    lines = []
    lines.append(f"📊 Daily Quant Signal | 每日量化信号")
    lines.append(f"📅 {today} ({weekday})")
    lines.append("")

    # --- Summary of changes ---
    opens = [a for a in actions if a['action'] in ('BUY', 'SELL_SHORT')]
    closes = [a for a in actions if a['action'] in ('SELL', 'COVER')]
    reversals = [a for a in actions if 'REVERSE' in a['action']]

    if opens or closes or reversals:
        lines.append("🔥 Today's Actions | 今日操作")
        if opens:
            for a in opens:
                lines.append(f"  ➡️ Open {a['signal_state']} on {a['ticker']} | 建立 {a['ticker']} {a['action_cn']}仓位")
        if closes:
            for a in closes:
                lines.append(f"  ⬅️ Close {a['current_state']} on {a['ticker']} | 平掉 {a['ticker']} {a['action_cn']}仓位")
        if reversals:
            for a in reversals:
                lines.append(f"  🔄 Reverse {a['ticker']} to {a['signal_state']} | {a['ticker']} {a['action_cn']}")
        lines.append("")

    # --- Actionable signals table ---
    lines.append("━" * 30)
    lines.append("📈 SIGNALS | 信号")
    lines.append("━" * 30)

    # Group by action type
    urgent_actions = [
        "BUY", "SELL_SHORT", "SELL", "COVER",
        "REVERSE_TO_LONG", "REVERSE_TO_SHORT"
    ]
    hold_actions = ["HOLD", "HOLD_CASH", "RAISE_STOP", "LOWER_STOP"]

    # Init / first-run setup is no longer shown in the report

    # Urgent actions first
    urgent = [a for a in actions if a["action"] in urgent_actions]
    hold = [a for a in actions if a["action"] in hold_actions]

    if not urgent and not hold:
        lines.append("No actionable signals today | 今日无操作信号")
        lines.append("")

    if urgent:
        for a in urgent:
            emoji = STATE_EMOJI.get(a["signal_state"], "⚪")
            lines.append(
                f"{emoji} {a['ticker']:<6} | {a['action']:<18} | {a['action_cn']}"
            )
            lines.append(
                f"   Close: {a['close']:<10} Signal: {a['effective_signal']:+.1f}"
            )
            lines.append(
                f"   {a['conviction_str']:<20} | {a['regime_str']}"
            )
            lines.append(f"   → {a['details']}")
            lines.append("")

    # Hold/stable positions (condensed)
    if hold:
        lines.append("━" * 30)
        lines.append("📋 HOLDING | 持仓不变")
        lines.append("━" * 30)
        for a in hold:
            emoji = STATE_EMOJI.get(a["current_state"], "⚪")
            short_action = a["action"].replace("_", " ").title()
            lines.append(
                f"{emoji} {a['ticker']:<6} | {short_action:<14} | "
                f"{a['action_cn']:<10} | Sig: {a['effective_signal']:+.1f}"
            )

    # --- Current positions summary ---
    lines.append("")
    lines.append("━" * 30)
    lines.append("💼 POSITIONS | 当前持仓")
    lines.append("━" * 30)
    active_positions = {k: v for k, v in positions.items() if v.get("state") != "FLAT"}
    if active_positions:
        for ticker, pos in active_positions.items():
            emoji = STATE_EMOJI.get(pos["state"], "⚪")
            lines.append(
                f"{emoji} {ticker:<6} | {pos['state']:<6} | "
                f"Entry: {pos.get('entry_price', '-')} | Stop: {pos.get('stop_level', '-')}"
            )
    else:
        lines.append("No active positions | 无持仓")

    # --- Risk check ---
    lines.append("")
    lines.append("━" * 30)
    lines.append("⚠️  RISK | 风险提示")
    lines.append("━" * 30)

    warnings = []
    for a in actions:
        if a.get("action") == "ERROR":
            warnings.append(f"❌ {a['ticker']}: Data error — {a.get('details', 'unknown')}")
            continue

        is_in_pos = a.get("current_state") in ("LONG", "SHORT")
        eff_sig = a.get("effective_signal", 0)

        if is_in_pos and a.get("action") in ("HOLD", "RAISE_STOP", "LOWER_STOP"):
            if a.get("current_state") == "LONG" and 15.0 <= eff_sig < 25.0:
                warnings.append(f"⚠️ {a['ticker']}: LONG signal {eff_sig:+.1f} is approaching exit level (15.0). | 多头信号 {eff_sig:+.1f} 接近平仓水平 (15.0).")
            elif a.get("current_state") == "SHORT" and -30.0 < eff_sig <= -20.0:
                warnings.append(f"⚠️ {a['ticker']}: SHORT signal {eff_sig:+.1f} is approaching cover level (-20.0). | 空头信号 {eff_sig:+.1f} 接近平仓水平 (-20.0).")

    if warnings:
        for w in warnings:
            lines.append(w)
    else:
        lines.append("No risk alerts | 无风险警报")

    lines.append("")
    lines.append("━" * 30)
    lines.append("🤖 Auto-generated by Quant Strategy Bot")
    lines.append("每日自动生成 | 仅供参考 | For reference only")

    return "\n".join(lines)


def send_telegram_message(message: str) -> bool:
    """
    Send a message via Telegram Bot API.

    Returns True on success, False on failure.
    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. "
              f"Message ({len(message)} chars) not sent.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        resp.raise_for_status()
        result = resp.json()
        if not result.get("ok"):
            print(f"Telegram API error: {result}")
            return False
        print("Telegram message sent successfully.")
        return True

    except requests.RequestException as e:
        print(f"Failed to send Telegram message: {e}")
        return False


# ---------------------------------------------------------------------------
# Daily run log
# ---------------------------------------------------------------------------

def write_daily_log(
    actions: list[dict],
    signals: dict[str, dict],
    positions: dict,
) -> None:
    """Append today's signal data to a daily run log file."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = DAILY_LOG_DIR / f"run_{today}.log"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Daily Signal Run — {today}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")

        # Per-ticker signal details
        f.write(f"{'Ticker':<8} {'Close':>10} {'EffSig':>8} {'State':>6} "
                 f"{'ADX':>6} {'CI':>6} {'SMFI':>6} {'Regime':>12} {'AMA':>10} {'ATR':>6}\n")
        f.write("-" * 90 + "\n")
        for ticker in TICKERS:
            s = signals.get(ticker, {})
            if "error" in s:
                f.write(f"{ticker:<8} ERROR: {s['error']}\n")
            else:
                f.write(
                    f"{ticker:<8} {s.get('close', 0):>10.2f} {s.get('effective_signal', 0):>+8.1f} "
                    f"{s.get('state_str', '-'):>6} {s.get('adx', 0):>6.1f} "
                    f"{s.get('choppiness', 0):>6.1f} {s.get('smfi', 0):>6.1f} "
                    f"{s.get('regime_str', '-'):>12} {s.get('ama', 0):>10.2f} {s.get('atr', 0):>6.2f}\n"
                )

        f.write("\n# Actions\n")
        for a in actions:
            f.write(f"  {a['ticker']:<8} {a['action']:<20} | {a['action_cn']}\n")
            f.write(f"    {a['details']}\n")

        f.write("\n# Positions\n")
        for ticker, pos in positions.items():
            f.write(f"  {ticker:<8} {pos.get('state', '-'):<8} "
                     f"Entry: {pos.get('entry_price', '-')} Stop: {pos.get('stop_level', '-')}\n")

    print(f"Daily log written to {log_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"=== Daily Quant Signal Run: {datetime.now().isoformat()} ===")
    print(f"Tickers: {TICKERS}")

    # Step 1 — Fetch data and compute signals
    print("\n[1/5] Computing signals...")
    signals = compute_current_signals()

    errors = [t for t, s in signals.items() if "error" in s]
    if errors:
        print(f"  WARNING: {len(errors)} ticker(s) had errors: {errors}")

    # Step 2 — Load current positions
    print("\n[2/5] Loading positions...")
    positions = load_positions()
    print(f"  Active positions: {len([p for p in positions.values() if p.get('state') != 'FLAT'])}")

    # Step 3 — Determine actions
    print("\n[3/5] Determining actions...")
    is_first = not POSITIONS_FILE.exists()
    actions = determine_actions(signals, positions, is_first_run=is_first)

    for a in actions:
        state_icon = {"LONG": "+", "SHORT": "-", "FLAT": "0"}.get(a["signal_state"], "?")
        _safe_print(f"  {a['ticker']:<6} [{state_icon}] -> {a['action']:<20} {a['action_cn']}")

    # Step 4 — Write daily log
    print("\n[4/5] Writing daily log...")
    write_daily_log(actions, signals, positions)

    # Step 5 — Format and send Telegram message
    print("\n[5/5] Sending Telegram report...")
    # For first run, show positions that WILL be set (not the empty state)
    display_positions = positions if positions else update_positions(actions, {}, signals)
    message = format_telegram_message(actions, signals, display_positions)
    print(f"  Message length: {len(message)} chars")

    # Print message locally for debugging
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        _safe_print("\n" + "=" * 50)
        _safe_print("TELEGRAM MESSAGE PREVIEW (no token set):")
        _safe_print("=" * 50)
        _safe_print(message)
        _safe_print("=" * 50)

    success = send_telegram_message(message)
    if not success:
        print("  Telegram delivery FAILED — check token/chat_id.")
        sys.exit(1)

    # Update positions AFTER successful send (so we don't lose state on send failure)
    print("\n[Post-run] Updating positions...")
    positions = update_positions(actions, positions, signals)
    print("Done.")


if __name__ == "__main__":
    main()
