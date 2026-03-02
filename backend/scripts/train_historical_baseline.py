"""Train a true historical-price baseline (OHLCV + technical indicators only)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import DATA_DIR, MODELS_DIR
from src.data_contracts import validate_stock_dataframe


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    g = df.groupby("ticker", group_keys=False)
    df["return_1d"] = g["Close"].pct_change()
    df["return_5d"] = g["Close"].pct_change(5)
    df["volatility_5d"] = g["return_1d"].rolling(5).std().reset_index(level=0, drop=True)
    df["sma_5"] = g["Close"].rolling(5).mean().reset_index(level=0, drop=True)
    df["sma_20"] = g["Close"].rolling(20).mean().reset_index(level=0, drop=True)
    df["price_vs_sma_20"] = (df["Close"] - df["sma_20"]) / (df["sma_20"] + 1e-9)

    # RSI(14)
    delta = g["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.groupby(df["ticker"]).rolling(14).mean().reset_index(level=0, drop=True)
    roll_down = down.groupby(df["ticker"]).rolling(14).mean().reset_index(level=0, drop=True)
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD(12,26)
    ema12 = g["Close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g["Close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = ema12 - ema26

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "return_1d", "return_5d", "volatility_5d", "sma_5", "sma_20",
        "price_vs_sma_20", "rsi_14", "macd",
    ]
    out = df.dropna(subset=feature_cols + ["label"]).copy()
    return out


def main() -> None:
    stock_path = DATA_DIR / "stock_prices.csv"
    if not stock_path.exists():
        raise FileNotFoundError(f"Missing stock prices file: {stock_path}")

    df = pd.read_csv(stock_path)
    validate_stock_dataframe(df)
    feat = build_features(df)

    feat = feat.sort_values("Date")
    split = int(len(feat) * 0.8)
    train, test = feat.iloc[:split], feat.iloc[split:]

    feature_cols = [c for c in feat.columns if c not in {"Date", "ticker", "name", "source", "prev_close", "label"}]
    X_train, y_train = train[feature_cols].values, train["label"].astype(int).values
    X_test, y_test = test[feature_cols].values, test["label"].astype(int).values

    model = RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "model": "historical_price_baseline",
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": feature_cols,
    }

    out_dir = MODELS_DIR / "historical_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "historical_baseline_rf.joblib")
    (out_dir / "historical_baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
