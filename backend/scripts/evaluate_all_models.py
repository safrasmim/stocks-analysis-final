"""
evaluate_all_models.py  — v5
Uses the project's model WRAPPER classes (RandomForestModel, XGBoostModel)
instead of raw joblib sklearn objects — this ensures any internal scaling
applied during training is also applied during evaluation.
"""
import sys, csv, json, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from src.config import DATA_DIR, MODELS_DIR

EVAL_DIR     = DATA_DIR / "evaluation"
METRICS_CSV  = EVAL_DIR / "evaluation_metrics.csv"
METRICS_JSON = EVAL_DIR / "evaluation_metrics.json"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────
def _metrics(name, y_true, y_pred, y_proba):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = (roc_auc_score(y_true, y_proba)
            if (y_proba is not None and len(y_proba) > 0) else 0.0)
    cm   = confusion_matrix(y_true, y_pred).tolist()
    return {"model": name,
            "accuracy":  round(acc,  4), "precision": round(prec, 4),
            "recall":    round(rec,  4), "f1_score":  round(f1,   4),
            "roc_auc":   round(auc,  4), "confusion_matrix": cm}


def _print_row(m):
    print(f"  {m['model']:<20} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
          f"{m['recall']:>7.4f} {m['f1_score']:>7.4f} {m['roc_auc']:>7.4f}")


def _get_pred_prob(result):
    """Extract predictions + probabilities from a model wrapper result dict."""
    y_pred = np.array(result.get("predictions", []))
    y_prob = np.array(result.get("probabilities_up", []))
    if len(y_prob) == 0:
        y_prob = None
    return y_pred, y_prob


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("EVALUATING ALL MODELS  (held-out test set)")
    print("=" * 62)

    # ── Load test set saved by train_all_models.py ─────────────────
    xp = MODELS_DIR / "X_test.joblib"
    yp = MODELS_DIR / "y_test.joblib"

    if not xp.exists() or not yp.exists():
        print("❌ X_test.joblib / y_test.joblib not found.")
        print("   Run: python scripts\train_all_models.py")
        sys.exit(1)

    X_test = joblib.load(xp)
    y_test = joblib.load(yp)
    n      = len(y_test)
    n_up   = int(y_test.sum())
    n_down = n - n_up
    print(f"Test set: {n} samples  (UP={n_up}  DOWN={n_down})")
    print(f"Feature matrix: {X_test.shape}")
    print()
    print(f"  {'Model':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("  " + "-" * 53)

    all_metrics = []

    # ── Random Forest — use wrapper (handles internal scaler) ─────
    rf_dir = MODELS_DIR / "random_forest"
    if rf_dir.exists():
        try:
            from src.models.random_forest import RandomForestModel
            rf = RandomForestModel()
            rf.load(rf_dir)
            result = rf.predict(X_test)
            y_pred, y_prob = _get_pred_prob(result)
            # fallback proba if wrapper doesn't return it
            if y_prob is None:
                raw = joblib.load(rf_dir / "random_forest.joblib")
                y_prob = raw.predict_proba(X_test)[:, 1]
            m = _metrics("Random Forest", y_test, y_pred, y_prob)
            all_metrics.append(m); _print_row(m)
        except Exception as e:
            print(f"  ⚠️  RF error: {e}")
    else:
        print("  ⚠️  RF model dir not found")

    # ── XGBoost — use wrapper ─────────────────────────────────────
    xgb_dir = MODELS_DIR / "xgboost"
    if xgb_dir.exists():
        try:
            from src.models.xgboost_model import XGBoostModel
            xgb = XGBoostModel()
            xgb.load(xgb_dir)
            result = xgb.predict(X_test)
            y_pred, y_prob = _get_pred_prob(result)
            if y_prob is None:
                raw = joblib.load(xgb_dir / "xgboost.joblib")
                y_prob = raw.predict_proba(X_test)[:, 1]
            m = _metrics("XGBoost", y_test, y_pred, y_prob)
            all_metrics.append(m); _print_row(m)
        except Exception as e:
            print(f"  ⚠️  XGB error: {e}")
    else:
        print("  ⚠️  XGB model dir not found")

    # ── LSTM ──────────────────────────────────────────────────────
    lstm_dir = MODELS_DIR / "lstm"
    if lstm_dir.exists():
        try:
            from src.models.lstm_model import LSTMModel
            lstm = LSTMModel()
            lstm.load(lstm_dir)
            result = lstm.predict(X_test)
            y_pred, y_prob = _get_pred_prob(result)
            m = _metrics("LSTM", y_test, y_pred, y_prob)
            all_metrics.append(m); _print_row(m)
        except Exception as e:
            print(f"  ⚠️  LSTM skipped: {e}")

    # ── Ensemble — soft vote via wrappers ─────────────────────────
    if len(all_metrics) >= 2:
        try:
            from src.models.random_forest import RandomForestModel
            from src.models.xgboost_model import XGBoostModel

            rf2  = RandomForestModel(); rf2.load(MODELS_DIR / "random_forest")
            xgb2 = XGBoostModel();     xgb2.load(MODELS_DIR / "xgboost")

            res_rf  = rf2.predict(X_test)
            res_xgb = xgb2.predict(X_test)

            p_rf  = np.array(res_rf.get("probabilities_up",  []))
            p_xgb = np.array(res_xgb.get("probabilities_up", []))

            # Fallback to raw proba if wrapper doesn't return it
            if len(p_rf)  == 0:
                p_rf  = joblib.load(MODELS_DIR / "random_forest" / "random_forest.joblib").predict_proba(X_test)[:, 1]
            if len(p_xgb) == 0:
                p_xgb = joblib.load(MODELS_DIR / "xgboost" / "xgboost.joblib").predict_proba(X_test)[:, 1]

            p_ens  = (p_rf + p_xgb) / 2.0
            y_pred = (p_ens >= 0.5).astype(int)
            m = _metrics("Ensemble", y_test, y_pred, p_ens)
            all_metrics.append(m); _print_row(m)
        except Exception as e:
            print(f"  ⚠️  Ensemble skipped: {e}")

    # ── Baseline ──────────────────────────────────────────────────
    majority = int(np.bincount(y_test.astype(int)).argmax())
    y_pred_b = np.full(n, majority)
    y_prob_b = np.full(n, float(n_up) / n)
    m = _metrics("Baseline", y_test, y_pred_b, y_prob_b)
    all_metrics.append(m); _print_row(m)

    # ── Save ──────────────────────────────────────────────────────
    fields = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in all_metrics:
            w.writerow({k: m[k] for k in fields})

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print()
    print(f"  CSV  → {METRICS_CSV}")
    print(f"  JSON → {METRICS_JSON}")
    print()
    print("  Model health:")
    for m in all_metrics:
        if m["model"] == "Baseline":
            continue
        acc = m["accuracy"]
        if   acc >= 0.75: icon, label = "✅", "good generalisation"
        elif acc >= 0.60: icon, label = "🟡", "acceptable"
        else:             icon, label = "🔴", "underperforming"
        print(f"  {icon} {m['model']:<20}: {acc:.4f}  — {label}")

    print()
    print("=" * 62)
    print("EVALUATION COMPLETE")
    print("=" * 62)


if __name__ == "__main__":
    main()