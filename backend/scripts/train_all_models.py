"""
train_all_models.py
Full training pipeline: data → features → RF + XGB + LSTM + Ensemble

Run: python scripts/train_all_models.py
"""

# ═══════════════════════════════════════════════════════════════
# SILENCE ALL COSMETIC WARNINGS — must be first, before imports
# ═══════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)
logging.getLogger("gensim").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════
# Standard imports
# ═══════════════════════════════════════════════════════════════
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.config import DATA_DIR, MODELS_DIR
from src.features.feature_engineering import extract_all_features, get_feature_matrix
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model    import LSTMModel
from src.models.ensemble      import EnsembleModel
from src.models.baseline      import BaselineModel

logger = logging.getLogger(__name__)


def train_lda(texts: list) -> tuple:
    """Train LDA topic model and save to models/lda/."""
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.utils import simple_preprocess

    print("  Training LDA topic model...")
    processed  = [simple_preprocess(t, deacc=True) for t in texts]
    dictionary = corpora.Dictionary(processed)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    corpus = [dictionary.doc2bow(doc) for doc in processed]

    lda = LdaModel(
        corpus=corpus, id2word=dictionary,
        num_topics=10, passes=10, iterations=50,
        alpha="auto", eta="auto", random_state=42,
    )

    lda_dir = MODELS_DIR / "lda"
    lda_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lda,        lda_dir / "lda_model.joblib")
    joblib.dump(dictionary, lda_dir / "lda_dictionary.joblib")
    print(f"  LDA saved → {lda_dir}")
    return lda, dictionary


def main():
    print("=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────
    data_path = DATA_DIR / "raw_news_sample.csv"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        print("Run: python scripts\generate_sample_data.py")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")

    # ── 2. Train LDA on ALL data (unsupervised) ───────────────────
    lda_model, lda_dict = train_lda(df["text"].tolist())

    # ── 3. Train / validation split (80/20 stratified) ────────────
    df_train, df_val = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )
    print(f"  Train: {len(df_train)}  |  Validation: {len(df_val)}")

    # ── 4. Extract features ───────────────────────────────────────
    print("  Extracting features...")

    df_train_f = extract_all_features(
        df_train.copy(), lda_model=lda_model,
        lda_dict=lda_dict, model_dir=MODELS_DIR
    )
    df_val_f = extract_all_features(
        df_val.copy(), lda_model=lda_model,
        lda_dict=lda_dict, model_dir=MODELS_DIR
    )

    X_train = get_feature_matrix(df_train_f)
    y_train = df_train["label"].values
    X_val   = get_feature_matrix(df_val_f)
    y_val   = df_val["label"].values

    print(f"  Feature matrix: {X_train.shape}")

    # ── 5. Save train/test arrays for evaluate_all_models.py ──────
    #    This guarantees evaluate uses EXACTLY the same features
    #    as training — prevents score drift between runs.
    print("  Saving train/test arrays...")
    joblib.dump(X_train, MODELS_DIR / "X_train.joblib")
    joblib.dump(X_val,   MODELS_DIR / "X_test.joblib")   # val = held-out test
    joblib.dump(y_train, MODELS_DIR / "y_train.joblib")
    joblib.dump(y_val,   MODELS_DIR / "y_test.joblib")

    # Also save full matrix for reference / re-splitting
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    joblib.dump(X_all, MODELS_DIR / "feature_matrix.joblib")
    joblib.dump(y_all, MODELS_DIR / "feature_labels.joblib")

    # Sentinel flag — tells evaluate the saved X_test is trustworthy
    (MODELS_DIR / ".split_saved_by_train").write_text("ok")
    print("  Train/test arrays saved ✅")

    # ── 6. Random Forest ──────────────────────────────────────────
    print()
    print("[1/5] Training Random Forest...")
    rf = RandomForestModel()
    rf.train(X_train, y_train, X_val, y_val)
    rf.save(MODELS_DIR / "random_forest")
    print("  RF saved.")

    # ── 7. XGBoost ────────────────────────────────────────────────
    print()
    print("[2/5] Training XGBoost...")
    xgb = XGBoostModel()
    xgb.train(X_train, y_train, X_val, y_val)
    xgb.save(MODELS_DIR / "xgboost")
    print("  XGBoost saved.")

    # ── 8. LSTM ───────────────────────────────────────────────────
    print()
    print("[3/5] Training LSTM...")
    lstm    = LSTMModel()
    lstm_ok = False
    try:
        lstm.train(X_train, y_train, X_val, y_val)
        lstm.save(MODELS_DIR / "lstm")
        print("  LSTM saved.")
        lstm_ok = True
    except Exception as e:
        print(f"  LSTM skipped: {e}")

    # ── 9. Ensemble ───────────────────────────────────────────────
    print()
    print("[4/5] Configuring Ensemble...")
    ensemble = EnsembleModel()
    ensemble.set_models(rf, xgb, lstm if lstm_ok else None)
    ensemble.save(MODELS_DIR / "ensemble")
    print("  Ensemble saved.")

    # ── 10. Baseline ──────────────────────────────────────────────
    print()
    print("[5/5] Training Baseline...")
    baseline = BaselineModel()
    baseline.train(X_train, y_train)
    baseline.save(MODELS_DIR / "baseline")
    print("  Baseline saved.")

    # ── 11. Quick validation summary ──────────────────────────────
    print()
    print("  Validation accuracy (training set — for sanity check only):")
    for name, model in [("RF", rf), ("XGBoost", xgb)]:
        result = model.predict(X_val)
        acc    = accuracy_score(y_val, result["predictions"])
        note   = "  ← run evaluate_all_models.py for honest test score" if acc > 0.95 else ""
        print(f"    {name:<10}: {acc:.4f}{note}")

    print()
    print("=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print(f"Saved to: {MODELS_DIR}")
    print("=" * 60)
    print()
    print("Next step:")
    print("  python scripts\evaluate_all_models.py")


if __name__ == "__main__":
    main()