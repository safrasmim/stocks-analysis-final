"""
build_training_from_ohlc.py
Joins Mubasher news/announcements with OHLC price data
to produce a labelled training dataset that passes data_contracts validation.

All fixes permanently baked in:
  - ticker dtype normalised to str
  - date / text alias columns added
  - data_source set to "real"
  - duplicates on (date, ticker, text) dropped
  - label cast to int

Usage:
    python scripts/build_training_from_ohlc.py
    python scripts/build_training_from_ohlc.py --label same_day
    python scripts/build_training_from_ohlc.py --label forward_3d
"""
import argparse
from pathlib import Path
import pandas as pd

OHLC_DEFAULT = "data/processed/ohlc_tdwl.csv"
NEWS_DEFAULT = "data/ingested/tdwl_news_announcements.parquet"
OUT_DEFAULT  = "data/processed/real_training_data.csv"


def add_labels(ohlc, strategy):
    df = ohlc.sort_values(["ticker", "date"]).copy()
    if strategy == "next_day":
        df["label"] = (df.groupby("ticker")["close"].shift(-1) > df["close"]).astype(int)
    elif strategy == "same_day":
        df["label"] = (df["close"] > df["open"]).astype(int)
    elif strategy == "forward_3d":
        df["label"] = (df.groupby("ticker")["close"].shift(-3) > df["close"] * 1.005).astype(int)
    else:
        raise ValueError("Unknown strategy: " + strategy)
    df["intraday_vol"]  = (df["high"] - df["low"]) / df["open"].replace(0, float("nan"))
    df["body_strength"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["vol_surge"]     = df.groupby("ticker")["volume"].transform(
        lambda x: x / x.rolling(10, min_periods=1).mean()
    )
    return df


def assign_event_date(pub_date, pub_hour, trading_days_set, trading_days_sorted):
    """After-hours (>=12 UTC = 15:00 AST) -> assign to next trading day."""
    if pub_hour >= 12:
        for td in trading_days_sorted:
            if td > pub_date:
                return td
    if pub_date in trading_days_set:
        return pub_date
    for td in trading_days_sorted:
        if td > pub_date:
            return td
    return pub_date


def join_news_to_ohlc(news, ohlc, label_col):
    news = news.copy()
    ohlc = ohlc.copy()

    # ── FIX 1: normalise ticker dtype ────────────────────
    news["ticker"] = news["ticker"].astype(str).str.strip()
    ohlc["ticker"] = ohlc["ticker"].astype(str).str.strip()

    news["pub_ts"]   = pd.to_datetime(news["published_at"], utc=True)
    news["pub_hour"] = news["pub_ts"].dt.hour
    news["pub_date"] = news["pub_ts"].dt.date

    trading_days_set    = set(ohlc["date"].dt.date.unique())
    trading_days_sorted = sorted(trading_days_set)

    news["event_date"] = news.apply(
        lambda r: assign_event_date(r["pub_date"], r["pub_hour"],
                                    trading_days_set, trading_days_sorted),
        axis=1
    )
    ohlc["event_date"] = ohlc["date"].dt.date

    keep = ["ticker", "event_date", label_col,
            "intraday_vol", "body_strength", "vol_surge", "close"]
    merged = news.merge(ohlc[keep], on=["ticker", "event_date"], how="inner")

    # ── FIX 2: add alias columns required by data_contracts ──
    merged["date"]  = merged["event_date"].astype(str)
    merged["text"]  = merged["headline"].astype(str)

    # ── FIX 3: set data_source = "real" ──────────────────
    merged["data_source"] = "real"

    # ── FIX 4: cast label to int ─────────────────────────
    merged["label"] = merged[label_col].astype(int)

    # ── FIX 5: drop duplicates on contract key ───────────
    before = len(merged)
    merged = merged.drop_duplicates(subset=["date", "ticker", "text"])
    dropped = before - len(merged)
    if dropped:
        print(f"  Dropped {dropped} duplicate (date, ticker, text) rows")

    # ── FIX 6: direction label ────────────────────────────
    merged["direction"] = merged["label"].map({1: "UP", 0: "DOWN"})

    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ohlc",  default=OHLC_DEFAULT)
    p.add_argument("--news",  default=NEWS_DEFAULT)
    p.add_argument("--out",   default=OUT_DEFAULT)
    p.add_argument("--label", default="next_day",
                   choices=["next_day", "same_day", "forward_3d"])
    a = p.parse_args()

    print("Loading OHLC  :", a.ohlc)
    ohlc = pd.read_csv(a.ohlc, parse_dates=["date"])
    print(f"  {len(ohlc)} rows | {ohlc.ticker.nunique()} tickers")

    print("Loading news  :", a.news)
    news_path = Path(a.news)
    if news_path.suffix == ".parquet":
        news = pd.read_parquet(news_path)
    else:
        news = pd.read_csv(news_path)
    print(f"  {len(news)} news/announcement rows")

    print(f"Adding labels (strategy: {a.label})...")
    ohlc = add_labels(ohlc, a.label)

    print("Joining + applying all data_contracts fixes...")
    dataset = join_news_to_ohlc(news, ohlc, label_col="label")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(a.out, index=False)

    sep = "=" * 55
    print(sep)
    print(f"Real labelled rows : {len(dataset)}")
    print(f"Label balance      : {dataset.direction.value_counts().to_dict()}")
    print(f"data_source values : {dataset.data_source.unique().tolist()}")
    print("By ticker          :")
    print(dataset.ticker.value_counts().to_string())
    print("By item_type       :")
    if "item_type" in dataset.columns:
        print(dataset.item_type.value_counts().to_string())
    print(f"Saved to           : {a.out}")
    print(sep)
    print()
    print(f"Next: python scripts/train_all_models.py --data {a.out}")


if __name__ == "__main__":
    main()
