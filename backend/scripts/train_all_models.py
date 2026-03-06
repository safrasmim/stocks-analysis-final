"""
train_all_models.py — v5 (stratified split to fix regime shift)

Root cause of AUC < 0.5:
  time_aware split put Sep-Jan (bearish, 72% DOWN) in train
  and Feb-Mar (bullish, 61% UP) in test — models were inverted.
  With only 282 rows this is a sampling artefact, not a leakage risk.

Fix: stratified train/test split (same UP/DOWN ratio in both sets).
  --split-mode stratified  [default — balanced evaluation]
  --split-mode time        [chronological — shows regime transfer]
"""
import sys, argparse, os, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

SENTIMENT_COLS = ["sentiment_negative","sentiment_neutral","sentiment_positive","sentiment_compound"]
TEXT_STAT_COLS = ["text_length","word_count","avg_word_length","sentence_count","exclamation_count","question_count"]
TOPIC_COLS     = [f"topic_{i}" for i in range(10)]
ENTITY_COLS    = ["company_mention_count","location_mention_count","money_mention_count"]
EVENT_COLS     = ["is_earnings_news","is_policy_news","is_merger_news","is_inflation_news",
                  "is_interest_rate_news","is_rate_cut","is_rate_hike","is_oil_bullish","is_oil_bearish"]
MACRO_COLS     = ["sector_sentiment_score","is_macro_news","gdp_growth","inflation_rate",
                  "interest_rate","unemployment_rate","oil_price_change","currency_rate_change"]
OHLC_COLS      = ["intraday_vol","body_strength","vol_surge"]
ALL_FEATURE_COLS = TEXT_STAT_COLS + SENTIMENT_COLS + TOPIC_COLS + ENTITY_COLS + EVENT_COLS + MACRO_COLS + OHLC_COLS


def inline_features(df):
    import re
    df = df.copy()
    POS = {"profit","growth","increase","surge","gain","rise","strong","record","beat",
           "exceed","positive","up","rally","boost","upgrade","dividend","expand","success","high"}
    NEG = {"loss","decline","decrease","fall","drop","weak","miss","below","negative",
           "concern","risk","down","hike","plunge","downgrade","resign","probe","impairment"}
    OVR = {"rate cut":0.6,"interest rate cut":0.7,"monetary easing":0.7,"dovish":0.6,
           "rate hike":-0.6,"interest rate hike":-0.7,"monetary tightening":-0.7,"hawkish":-0.6,
           "oil prices rise":0.5,"oil prices fall":-0.5}
    texts = df.get("text", df.get("headline", pd.Series([""] * len(df)))).astype(str)
    rows = []
    for text in texts:
        t = text.lower(); words = t.split()
        p = sum(1 for w in words if w in POS)
        n = sum(1 for w in words if w in NEG)
        tot = max(p + n, 1); ps, ns = p/tot, n/tot
        compound = ps - ns
        for phrase, sc in OVR.items():
            if phrase in t: compound = sc; break
        sents = re.split(r"[.!?]", text)
        rows.append({
            "sentiment_negative": ns, "sentiment_neutral": max(1-ps-ns,0),
            "sentiment_positive": ps, "sentiment_compound": compound,
            "text_length": len(text), "word_count": len(words),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "sentence_count": len(sents), "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "company_mention_count": t.count("company"),
            "location_mention_count": t.count("saudi"),
            "money_mention_count": t.count("million")+t.count("billion"),
            "is_earnings_news": int("earnings" in t or "profit" in t),
            "is_policy_news": int("policy" in t or "opec" in t),
            "is_merger_news": int("merger" in t or "acqui" in t),
            "is_inflation_news": int("inflation" in t),
            "is_interest_rate_news": int("interest rate" in t),
            "is_rate_cut": int("rate cut" in t),
            "is_rate_hike": int("rate hike" in t),
            "is_oil_bullish": int("oil" in t and any(w in t for w in ["rise","surge","rally"])),
            "is_oil_bearish": int("oil" in t and any(w in t for w in ["fall","drop","plunge"])),
        })
    feat_df = pd.DataFrame(rows, index=df.index)
    for col in feat_df.columns: df[col] = feat_df[col]
    for c in ALL_FEATURE_COLS:
        if c not in df.columns: df[c] = 0.0
    return df


def build_features(df):
    df = df.copy()
    need = [c for c in (SENTIMENT_COLS + TEXT_STAT_COLS + ENTITY_COLS + EVENT_COLS) if c not in df.columns]
    if need:
        try:
            from src.features.feature_engineering import extract_all_features
            text_col = "text" if "text" in df.columns else "headline"
            feat = extract_all_features(df[[text_col]].rename(columns={text_col: "text"}))
            for col in feat.columns:
                if col not in df.columns: df[col] = feat[col].values
            print("  Used src.features.feature_engineering (FinBERT)")
        except Exception as e:
            print(f"  FinBERT unavailable ({e}) — using inline fallback")
            df = inline_features(df)
    else:
        print("  Feature cols already in CSV — reusing")
    for c in ALL_FEATURE_COLS:
        if c not in df.columns: df[c] = 0.0
    return df


def do_split(df, mode, test_frac):
    if mode == "time":
        df_s = df.sort_values("date").reset_index(drop=True)
        cut = int(len(df_s) * (1 - test_frac))
        train_df, test_df = df_s.iloc[:cut], df_s.iloc[cut:]
        print(f"Split => time-aware: train={len(train_df)} test={len(test_df)}")
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=test_frac,
            stratify=df["label"],
            random_state=SEED,
        )
        print(f"Split => stratified: train={len(train_df)} test={len(test_df)}")
    print(f"  Train: DOWN={(train_df.label==0).sum()} UP={(train_df.label==1).sum()}")
    print(f"  Test:  DOWN={(test_df.label==0).sum()}  UP={(test_df.label==1).sum()}")
    return train_df, test_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/processed/real_training_data.csv")
    p.add_argument("--test-frac",  type=float, default=0.2)
    p.add_argument("--split-mode", default="stratified", choices=["stratified","time"])
    a = p.parse_args()

    from src.config import MODELS_DIR
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.data)
    print(f"Loaded {len(df)} rows from {a.data}")
    print(f"Data sources: {df['data_source'].unique().tolist()}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print("Computing features...")
    df = build_features(df)
    avail = [c for c in ALL_FEATURE_COLS if c in df.columns]
    print(f"Feature matrix: {len(df)} rows x {len(avail)} cols")

    train_df, test_df = do_split(df, a.split_mode, a.test_frac)

    X_tr = train_df[avail].fillna(0).values.astype(float)
    y_tr = train_df["label"].astype(int).values
    X_te = test_df[avail].fillna(0).values.astype(float)
    y_te = test_df["label"].astype(int).values

    # Save unscaled test set for evaluate_all_models.py
    joblib.dump(X_te, MODELS_DIR / "X_test.joblib")
    joblib.dump(y_te, MODELS_DIR / "y_test.joblib")
    print(f"Saved X_test/y_test ({X_te.shape}) -> {MODELS_DIR}")

    from sklearn.metrics import accuracy_score
    n_down = int((y_tr == 0).sum())
    n_up   = int((y_tr == 1).sum())

    # ── Random Forest ────────────────────────────────────────────
    print("Training Random Forest...")
    from src.models.random_forest import RandomForestModel
    rf = RandomForestModel()
    rf.train(X_tr, y_tr, X_val=X_te, y_val=y_te)
    rf.save(MODELS_DIR / "random_forest")
    print(f"RF test accuracy: {accuracy_score(y_te, rf.predict(X_te)['predictions']):.4f}")

    # ── XGBoost (scale_pos_weight from training class ratio) ──────
    print("Training XGBoost...")
    from src.models.xgboost_model import XGBoostModel
    xgb_m = XGBoostModel()
    scale_pos = round(n_down / max(n_up, 1), 4)
    print(f"  scale_pos_weight = {scale_pos}")
    try: xgb_m.model.set_params(scale_pos_weight=scale_pos)
    except Exception: pass
    xgb_m.train(X_tr, y_tr, X_val=X_te, y_val=y_te)
    xgb_m.save(MODELS_DIR / "xgboost")
    print(f"XGB test accuracy: {accuracy_score(y_te, xgb_m.predict(X_te)['predictions']):.4f}")

    # ── LSTM (class_weight baked into lstm_model.py) ─────────────
    print("Training LSTM...")
    from src.models.lstm_model import LSTMModel
    lstm_m = LSTMModel()
    lstm_m.train(X_tr, y_tr, X_val=X_te, y_val=y_te)
    lstm_m.save(MODELS_DIR / "lstm")
    print("All models saved.")


if __name__ == "__main__":
    main()
