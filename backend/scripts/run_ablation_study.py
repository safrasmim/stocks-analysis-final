"""Ablation study for news-based features (sentiment/topics/entities/macro)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import DATA_DIR, MODELS_DIR
from src.data_contracts import validate_news_dataframe
from src.features.feature_engineering import ALL_FEATURE_COLS, extract_all_features
from src.features.macro import load_macro_data


def get_groups(cols):
    return {
        "full": cols,
        "no_sentiment": [c for c in cols if not c.startswith("sentiment_")],
        "no_topics": [c for c in cols if not c.startswith("topic_")],
        "no_entities_events": [c for c in cols if c not in {
            "company_mention_count", "location_mention_count", "money_mention_count",
            "is_earnings_news", "is_policy_news", "is_merger_news", "is_inflation_news", "is_interest_rate_news",
            "is_rate_cut", "is_rate_hike", "is_oil_bullish", "is_oil_bearish",
        }],
        "no_macro": [c for c in cols if c not in {
            "gdp_growth", "inflation_rate", "interest_rate", "unemployment_rate", "oil_price_change", "currency_rate_change",
            "sector_sentiment_score", "is_macro_news",
        }],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATA_DIR / "raw_news_sample.csv"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    if "data_source" not in df.columns:
        df["data_source"] = "synthetic"
    validate_news_dataframe(df, allow_synthetic=args.allow_synthetic)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    macro_df = load_macro_data(DATA_DIR / "macro_indicators.csv")
    feat = extract_all_features(df.copy(), macro_df=macro_df, model_dir=MODELS_DIR)

    split = int(len(feat) * 0.8)
    y = feat["label"].astype(int).values

    groups = get_groups(ALL_FEATURE_COLS)
    results = []
    for name, cols in groups.items():
        X = feat[cols].fillna(0).values.astype(np.float32)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=8, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        results.append({
            "variant": name,
            "feature_count": len(cols),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else 0.5, 4),
        })

    results = sorted(results, key=lambda r: r["f1_score"], reverse=True)
    out = DATA_DIR / "evaluation" / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
