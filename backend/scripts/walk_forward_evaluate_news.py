"""Walk-forward evaluation for macro-news models (chronology-safe)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import DATA_DIR, MODELS_DIR
from src.data_contracts import validate_news_dataframe
from src.features.feature_engineering import extract_all_features, get_feature_matrix
from src.features.macro import load_macro_data


def _score(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATA_DIR / "raw_news_sample.csv"))
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    if "data_source" not in df.columns:
        df["data_source"] = "synthetic"
    validate_news_dataframe(df, allow_synthetic=args.allow_synthetic)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    macro_df = load_macro_data(DATA_DIR / "macro_indicators.csv")
    all_feat = extract_all_features(df.copy(), macro_df=macro_df, model_dir=MODELS_DIR)
    X = get_feature_matrix(all_feat)
    y = df["label"].astype(int).values

    n = len(df)
    folds = max(3, args.n_folds)
    test_size = n // (folds + 1)
    train_min = max(test_size, int(n * 0.4))

    fold_results = []
    for i in range(folds):
        train_end = train_min + i * test_size
        test_end = min(train_end + test_size, n)
        if test_end - train_end < 50:
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]

        rf = RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=8, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(random_state=42)
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)

        rf_prob = rf.predict_proba(X_test)[:, 1]
        gb_prob = gb.predict_proba(X_test)[:, 1]
        ens_prob = (rf_prob + gb_prob) / 2.0

        fold_results.append({
            "fold": i + 1,
            "rf": _score(y_test, (rf_prob >= 0.5).astype(int), rf_prob),
            "gradient_boosting": _score(y_test, (gb_prob >= 0.5).astype(int), gb_prob),
            "ensemble": _score(y_test, (ens_prob >= 0.5).astype(int), ens_prob),
            "train_rows": int(train_end),
            "test_rows": int(test_end - train_end),
        })

    def summarize(model_key: str):
        vals = {m: [f[model_key][m] for f in fold_results] for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]}
        return {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4), "worst": round(float(np.min(v)), 4)} for k, v in vals.items()}

    report = {
        "dataset": args.dataset,
        "folds_executed": len(fold_results),
        "fold_results": fold_results,
        "summary": {
            "rf": summarize("rf") if fold_results else {},
            "gradient_boosting": summarize("gradient_boosting") if fold_results else {},
            "ensemble": summarize("ensemble") if fold_results else {},
        },
    }

    out = DATA_DIR / "evaluation" / "walk_forward_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
