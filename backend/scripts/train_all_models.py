"""Train all models with thesis-alignment guards (data contracts + time-aware split)."""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, MODELS_DIR, ALLOW_SYNTHETIC_BY_DEFAULT
from src.data_contracts import validate_news_dataframe
from src.features.feature_engineering import extract_all_features, get_feature_matrix
from src.models.baseline import BaselineModel
from src.models.ensemble import EnsembleModel
from src.models.lstm_model import LSTMModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel


def train_lda(texts: list[str]) -> tuple:
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.utils import simple_preprocess

    processed = [simple_preprocess(t, deacc=True) for t in texts]
    dictionary = corpora.Dictionary(processed)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    corpus = [dictionary.doc2bow(doc) for doc in processed]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=10,
        passes=10,
        iterations=50,
        alpha="auto",
        eta="auto",
        random_state=42,
    )

    lda_dir = MODELS_DIR / "lda"
    lda_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lda, lda_dir / "lda_model.joblib")
    joblib.dump(dictionary, lda_dir / "lda_dictionary.joblib")
    return lda, dictionary


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _split_data(df: pd.DataFrame, time_aware: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if time_aware:
        df = df.sort_values("date").reset_index(drop=True)
        cut = int(len(df) * 0.8)
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()

    return train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATA_DIR / "raw_news_sample.csv"))
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=ALLOW_SYNTHETIC_BY_DEFAULT,
        help=("Allow synthetic/mixed data sources. Keep OFF for thesis final experiments."),
    )
    parser.add_argument("--time-aware", action="store_true", help="Use chronological split (recommended for thesis)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.dataset)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if "data_source" not in df.columns:
        df["data_source"] = "synthetic"
    try:
        contract = validate_news_dataframe(df, allow_synthetic=args.allow_synthetic)
    except ValueError as exc:
        if "Synthetic data detected" in str(exc):
            raise ValueError(
                f"{exc}\nHint: rerun with --allow-synthetic for demo data, or switch --dataset to a real-news file."
            ) from exc
        raise

    print(f"Loaded {contract.rows} rows from {data_path}")
    print(f"Data sources: {', '.join(contract.sources)}")

    lda_model, lda_dict = train_lda(df["text"].astype(str).tolist())

    df_train, df_val = _split_data(df, time_aware=args.time_aware)
    print(f"Split => train={len(df_train)} test={len(df_val)} (time_aware={args.time_aware})")

    df_train_f = extract_all_features(df_train.copy(), lda_model=lda_model, lda_dict=lda_dict, model_dir=MODELS_DIR)
    df_val_f = extract_all_features(df_val.copy(), lda_model=lda_model, lda_dict=lda_dict, model_dir=MODELS_DIR)

    X_train = get_feature_matrix(df_train_f)
    y_train = df_train["label"].values
    X_val = get_feature_matrix(df_val_f)
    y_val = df_val["label"].values

    joblib.dump(X_train, MODELS_DIR / "X_train.joblib")
    joblib.dump(X_val, MODELS_DIR / "X_test.joblib")
    joblib.dump(y_train, MODELS_DIR / "y_train.joblib")
    joblib.dump(y_val, MODELS_DIR / "y_test.joblib")

    rf = RandomForestModel(); rf.train(X_train, y_train, X_val, y_val); rf.save(MODELS_DIR / "random_forest")
    xgb = XGBoostModel(); xgb.train(X_train, y_train, X_val, y_val); xgb.save(MODELS_DIR / "xgboost")

    lstm = LSTMModel(); lstm_ok = False
    try:
        lstm.train(X_train, y_train, X_val, y_val)
        lstm.save(MODELS_DIR / "lstm")
        lstm_ok = True
    except Exception as exc:
        print(f"LSTM skipped: {exc}")

    ens = EnsembleModel(); ens.set_models(rf, xgb, lstm if lstm_ok else None); ens.save(MODELS_DIR / "ensemble")
    baseline = BaselineModel(); baseline.train(X_train, y_train); baseline.save(MODELS_DIR / "baseline")

    manifest = {
        "dataset": str(data_path),
        "rows": int(len(df)),
        "sources": contract.sources,
        "allow_synthetic": args.allow_synthetic,
        "time_aware_split": args.time_aware,
        "git_sha": _git_sha(),
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_val)),
    }
    (MODELS_DIR / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for name, model in [("RF", rf), ("XGB", xgb)]:
        acc = accuracy_score(y_val, model.predict(X_val)["predictions"])
        print(f"{name} test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
