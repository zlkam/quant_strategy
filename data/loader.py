"""
Data ingestion and cleaning for historical CSV price data.

Handles column-name normalisation (Price → Close, Vol. → Volume),
volume-suffix parsing (M / B / K), BOM-stripped headers, and quoted
numeric values.
"""

import os

import numpy as np
import pandas as pd


def _is_missing(value) -> bool:
    """True if value is NaN, NaT, None, or pd.NA."""
    try:
        return pd.isna(value)
    except TypeError:
        return False


def _parse_volume(value) -> float:
    """Convert a volume string like '42.50M', '1.2B', '500K' to a numeric float."""
    if _is_missing(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip().strip('"').replace(",", "")
    if s in ("", ".", "-"):
        return np.nan

    multipliers = {"B": 1e9, "M": 1e6, "K": 1e3}
    suffix = s[-1].upper()
    if suffix in multipliers:
        return float(s[:-1]) * multipliers[suffix]
    # Handles malformed values like "0.45%" leaking into the volume column
    if suffix == "%":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_numeric(value) -> float:
    """Parse a potentially quoted numeric value (e.g. '"0.93"', '614.31')."""
    if _is_missing(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip().strip('"').replace(",", "")
    if s in ("", ".", "-"):
        return np.nan
    return float(s)


def load_historical_data(filepath: str) -> pd.DataFrame:
    """
    Load a single historical CSV and normalise it for the backtest engine.

    Returns a DataFrame with columns:
        Date (datetime index), Open, High, Low, Close, Volume, ChangePct
    """
    # Read CSV; strip BOM via encoding="utf-8-sig"
    df = pd.read_csv(filepath, encoding="utf-8-sig")

    # Normalise column names: strip quotes and whitespace
    df.columns = df.columns.str.strip().str.strip('"')

    # Rename to canonical form
    rename_map = {
        "Price": "Close",
        "Vol.": "Volume",
        "Vol": "Volume",
        "Change%": "ChangePct",
        "Change %": "ChangePct",
    }
    df = df.rename(columns=rename_map)

    # Parse numeric columns
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_numeric)

    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].apply(_parse_volume)

    if "ChangePct" in df.columns:
        df["ChangePct"] = df["ChangePct"].apply(
            lambda x: np.nan if _is_missing(x)
            else float(str(x).strip().strip('"').replace("%", "").replace(",", ""))
        )

    # Parse dates and set as index
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.sort_values("Date").set_index("Date")

    # Keep only needed columns
    keep_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    return df


def load_all_tickers(data_dir: str = "historical_data") -> dict[str, pd.DataFrame]:
    """
    Load all CSV files from the data directory.

    Returns a dict mapping ticker name (filename stem) to its DataFrame.
    """
    tickers = {}
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            ticker = os.path.splitext(fname)[0]
            filepath = os.path.join(data_dir, fname)
            tickers[ticker] = load_historical_data(filepath)
    return tickers
