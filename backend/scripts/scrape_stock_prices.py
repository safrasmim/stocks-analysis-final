"""
Fixed Stock Price Scraper - yfinance 1.2.0 compatible
Handles MultiIndex columns returned by new yfinance
"""

import sys
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, TICKERS

# Suppress yfinance noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ALT_SYMBOLS = {
    "1120": ["1120.SR", "RJHI.SR"],
    "2010": ["2010.SR", "SABIC.SR"],
    "7010": ["7010.SR", "STC.SR"],
    "1150": ["1150.SR", "ALINMA.SR"],
    "4325": ["4325.SR", "MASAR.SR"],
}


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance 1.2.0 returns MultiIndex columns like:
      ('Close', '1120.SR'), ('Open', '1120.SR'), ...
    This flattens them to: Close, Open, ...
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col
                      for col in df.columns]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def try_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    # Approach 1: standard download
    try:
        raw = yf.download(symbol, start=start, end=end,
                          progress=False, auto_adjust=True, timeout=30)
        if not raw.empty:
            df = flatten_columns(raw.copy())
            if "Close" in df.columns:
                logger.info("  ✓ yf.download OK for %s (%d rows)", symbol, len(df))
                return df.reset_index()
    except Exception as e:
        logger.debug("  Approach 1 failed: %s", e)

    # Approach 2: Ticker.history (usually clean columns)
    try:
        raw = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
        if not raw.empty:
            df = flatten_columns(raw.copy()).reset_index()
            if "Datetime" in df.columns:
                df.rename(columns={"Datetime": "Date"}, inplace=True)
            if "Close" in df.columns:
                logger.info("  ✓ Ticker.history OK for %s (%d rows)", symbol, len(df))
                return df
    except Exception as e:
        logger.debug("  Approach 2 failed: %s", e)

    # Approach 3: period-based
    try:
        raw = yf.download(symbol, period="2y", progress=False,
                          auto_adjust=True, timeout=30)
        if not raw.empty:
            df = flatten_columns(raw.copy())
            if "Close" in df.columns:
                logger.info("  ✓ period=2y OK for %s (%d rows)", symbol, len(df))
                return df.reset_index()
    except Exception as e:
        logger.debug("  Approach 3 failed: %s", e)

    return pd.DataFrame()


def generate_synthetic_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    BASE = {"1120": 95.0, "2010": 110.0, "7010": 52.0, "1150": 23.5, "4325": 45.0}
    base  = BASE.get(ticker, 50.0)
    dates = pd.bdate_range(start=start, end=end)
    n     = len(dates)
    np.random.seed(hash(ticker) % (2**31))
    returns = np.random.normal(0.0003, 0.012, n)
    close   = base * np.cumprod(1 + returns)
    high    = close * (1 + np.abs(np.random.normal(0.005, 0.004, n)))
    low     = close * (1 - np.abs(np.random.normal(0.005, 0.004, n)))
    open_   = np.roll(close, 1); open_[0] = base
    volume  = np.random.randint(500_000, 5_000_000, n)
    return pd.DataFrame({
        "Date":   pd.to_datetime(dates),
        "Open":   np.round(open_,  2),
        "High":   np.round(high,   2),
        "Low":    np.round(low,    2),
        "Close":  np.round(close,  2),
        "Volume": volume,
    })


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Date and Close columns exist and are clean."""
    df = flatten_columns(df)
    if "Date" not in df.columns:
        if df.index.name in ("Date", "Datetime", "index"):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    keep = [c for c in ["Date","Open","High","Low","Close","Volume"] if c in df.columns]
    return df[keep].copy()


def main():
    start   = "2022-01-01"
    end     = datetime.now().strftime("%Y-%m-%d")
    all_dfs = []

    print("=" * 55)
    print("DOWNLOADING STOCK PRICES")
    print("=" * 55)

    for code, info in TICKERS.items():
        print(f"\n[{code}] {info['name']}")
        df     = pd.DataFrame()
        source = "synthetic"

        for sym in ALT_SYMBOLS.get(code, [f"{code}.SR"]):
            print(f"  Trying {sym}...")
            raw = try_yfinance(sym, start, end)
            if not raw.empty:
                df     = clean_df(raw)
                source = "yahoo"
                break
            time.sleep(1)

        if df.empty or "Close" not in df.columns:
            print(f"  ⚠ Using synthetic data (GBM model)")
            df     = generate_synthetic_prices(code, start, end)
            source = "synthetic"

        df["ticker"]     = code
        df["name"]       = info["name"]
        df["source"]     = source
        df["prev_close"] = df["Close"].shift(1)
        df["label"]      = (df["Close"] > df["prev_close"]).astype(int)
        df.dropna(subset=["prev_close"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        up  = int(df["label"].sum())
        dn  = int((df["label"] == 0).sum())
        print(f"  ✓ {len(df)} days | Up={up} Down={dn} | source={source}")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "stock_prices.csv"
    combined.to_csv(out, index=False)

    print("\n" + "=" * 55)
    print(f"SAVED → {out}")
    print(f"Total rows : {len(combined)}")
    print(f"Tickers    : {combined['ticker'].nunique()}")
    print(f"Date range : {combined['Date'].min()} → {combined['Date'].max()}")
    print("=" * 55)


if __name__ == "__main__":
    main()
