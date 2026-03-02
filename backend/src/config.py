"""
config.py
Central configuration for the Tadawul Stock Prediction System.
MSc Thesis Project — February 2026
"""
from pathlib import Path
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings

# ── Directory structure ────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
SRC_DIR     = BASE_DIR / "src"
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
MODEL_DIR   = MODELS_DIR          # alias — both names work
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR    = BASE_DIR / "logs"

for _dir in [DATA_DIR, MODELS_DIR, SCRIPTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── API settings ───────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    api_host:   str  = "0.0.0.0"
    api_port:   int  = 8000
    api_reload: bool = True
    log_level:  str  = "INFO"

    class Config:
        env_file          = BASE_DIR / ".env"
        env_file_encoding = "utf-8"

settings = Settings()

API_TITLE       = "Tadawul Stock Movement Prediction API"
API_VERSION     = "2.0.0"
API_DESCRIPTION = """
**News-Driven Stock Movement Prediction System**

Predicts stock price movements (Up/Down) based on financial news analysis.

**Features:**
- FinBERT sentiment analysis
- LDA topic modelling
- Macro-economic sector impact scoring
- Multiple ML models: Random Forest, XGBoost, LSTM, Ensemble

**Target Market:** Tadawul (Saudi Stock Exchange)
**MSc Thesis Project** — February 2026
"""

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# ── Stock tickers — Tadawul ────────────────────────────────────────────────────
TICKERS: Dict[str, Dict[str, str]] = {
    "1120": {
        "name":        "Al Rajhi Bank",
        "sector":      "Banking",
        "description": "Largest Islamic bank in Saudi Arabia",
    },
    "2010": {
        "name":        "SABIC",
        "sector":      "Petrochemicals",
        "description": "Saudi Basic Industries Corporation",
    },
    "7010": {
        "name":        "STC",
        "sector":      "Telecommunications",
        "description": "Saudi Telecom Company",
    },
    "1150": {
        "name":        "Alinma Bank",
        "sector":      "Banking",
        "description": "Islamic banking and financial services",
    },
    "4325": {
        "name":        "MASAR",
        "sector":      "Financial Services",
        "description": "MASAR Leasing and Financing Company",
    },
    "2222": {
        "name":        "Saudi Aramco",
        "sector":      "Energy",
        "description": "Largest company on Tadawul by market cap",
    },
    "1211": {
        "name":        "Ma'aden",
        "sector":      "Mining",
        "description": "Saudi Arabian Mining Company",
    },
    "4110": {
        "name":        "TAWUNIYA",
        "sector":      "Insurance",
        "description": "Largest insurance company on Tadawul",
    },
}

TICKER_NAMES: Dict[str, str] = {
    code: info["name"] for code, info in TICKERS.items()
}

# ── Data schema column names ───────────────────────────────────────────────────
DATE_COLUMN     = "date"
TICKER_COLUMN   = "ticker"
HEADLINE_COLUMN = "headline"
TEXT_COLUMN     = "text"
LABEL_COLUMN    = "label"
RETURN_COLUMN   = "actual_return"
SOURCE_COLUMN   = "source"
URL_COLUMN      = "url"

REQUIRED_COLUMNS = [DATE_COLUMN, TICKER_COLUMN, TEXT_COLUMN, LABEL_COLUMN]

# ── Model file paths ───────────────────────────────────────────────────────────
RF_MODEL_PATH       = MODELS_DIR / "random_forest" / "random_forest.joblib"
XGB_MODEL_PATH      = MODELS_DIR / "xgboost"       / "xgboost.joblib"
LSTM_MODEL_PATH     = MODELS_DIR / "lstm"           / "lstm_model.h5"
LSTM_TOKENIZER_PATH = MODELS_DIR / "lstm"           / "lstm_tokenizer.pkl"
SCALER_PATH         = MODELS_DIR / "feature_scaler.joblib"
LABEL_ENCODER_PATH  = MODELS_DIR / "label_encoder.joblib"
LDA_MODEL_PATH      = MODELS_DIR / "lda"            / "lda_model.joblib"
DICTIONARY_PATH     = MODELS_DIR / "lda"            / "lda_dictionary.joblib"

# ── NLP / Feature engineering ──────────────────────────────────────────────────
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
FINBERT_LABELS     = ["negative", "neutral", "positive"]
FINBERT_BATCH_SIZE = 16
FINBERT_MAX_LENGTH = 512

NUM_TOPICS       = 10
LDA_PASSES       = 10
LDA_ITERATIONS   = 50
LDA_RANDOM_STATE = 42

MIN_TEXT_LENGTH  = 10
MAX_TEXT_LENGTH  = 5000
STOPWORDS_LANG   = "english"
REMOVE_URLS      = True
REMOVE_NUMBERS   = False
LOWERCASE        = True

SENTIMENT_FEATURES = [
    "sentiment_negative", "sentiment_neutral",
    "sentiment_positive", "sentiment_compound",
]
TOPIC_FEATURES     = [f"topic_{i}" for i in range(NUM_TOPICS)]
TEXT_FEATURES      = [
    "text_length", "word_count", "avg_word_length",
    "sentence_count", "exclamation_count", "question_count",
]
ALL_FEATURE_COLUMNS = SENTIMENT_FEATURES + TOPIC_FEATURES + TEXT_FEATURES

# ── Train / validation / test split ───────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Random Forest — constrained to prevent overfitting ────────────────────────
RF_CONFIG = {
    "n_estimators":      300,
    "max_depth":         5,       # was 10 — reduced to prevent memorisation
    "min_samples_split": 20,      # was 5
    "min_samples_leaf":  10,      # was 2
    "max_features":      "sqrt",
    "random_state":      42,
    "n_jobs":            -1,
    "class_weight":      "balanced",
}

# ── XGBoost — regularised to prevent overfitting ──────────────────────────────
XGB_CONFIG = {
    "n_estimators":     300,
    "max_depth":        4,        # was 8 — reduced
    "learning_rate":    0.05,     # was 0.1 — slower learning
    "subsample":        0.7,      # was 0.8
    "colsample_bytree": 0.7,      # was 0.8
    "min_child_weight": 10,       # NEW — prevents over-specific splits
    "reg_alpha":        0.1,      # NEW — L1 regularisation
    "reg_lambda":       1.0,      # NEW — L2 regularisation
    "random_state":     42,
    "n_jobs":           -1,
    "eval_metric":      "logloss",
}

# ── LSTM ───────────────────────────────────────────────────────────────────────
LSTM_CONFIG = {
    "units":                   64,
    "dropout":                 0.3,
    "recurrent_dropout":       0.2,
    "epochs":                  50,
    "batch_size":              32,
    "sequence_length":         10,
    "validation_split":        0.2,
    "early_stopping_patience": 5,
}

# ── Evaluation metrics ─────────────────────────────────────────────────────────
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]


# ── Data governance / thesis-alignment guards ───────────────────────────────
ALLOW_SYNTHETIC_BY_DEFAULT = False
MACRO_MAX_STALENESS_DAYS   = 45
REQUIRE_EVENT_DATE         = True

# ── Sample data generation config ─────────────────────────────────────────────
SAMPLE_DATA_CONFIG = {
    "num_articles_per_stock": 400,
    "date_range_days":        730,
    "positive_ratio":         0.50,
    "start_date":             "2021-01-01",
    "include_weekends":       False,
    "add_noise":              True,
    "noise_ratio":            0.12,
}

# ── News sources (future web scraping) ────────────────────────────────────────
NEWS_SOURCES = {
    "alarabiya": {"url": "https://english.alarabiya.net/business",                "enabled": False},
    "arabnews":  {"url": "https://www.arabnews.com/saudi-arabia/business-economy", "enabled": False},
    "argaam":    {"url": "https://www.argaam.com/en/market/market-news",           "enabled": False},
}

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE   = LOGS_DIR / "app.log"
LOG_LEVEL  = settings.log_level

# ── Helper functions ───────────────────────────────────────────────────────────
def get_data_path(filename: str) -> Path:
    return DATA_DIR / filename

def get_model_path(model_type: str) -> Path:
    model_map = {
        "rf":            RF_MODEL_PATH,
        "random_forest": RF_MODEL_PATH,
        "xgb":           XGB_MODEL_PATH,
        "xgboost":       XGB_MODEL_PATH,
        "lstm":          LSTM_MODEL_PATH,
    }
    return model_map.get(model_type.lower(), RF_MODEL_PATH)

def validate_ticker(ticker: str) -> bool:
    return ticker in TICKERS

def get_ticker_info(ticker: str) -> Optional[Dict]:
    return TICKERS.get(ticker)

__all__ = [
    "BASE_DIR", "DATA_DIR", "MODELS_DIR", "MODEL_DIR",
    "TICKERS", "TICKER_NAMES",
    "DATE_COLUMN", "TICKER_COLUMN", "TEXT_COLUMN", "LABEL_COLUMN",
    "RF_CONFIG", "XGB_CONFIG", "LSTM_CONFIG",
    "ALL_FEATURE_COLUMNS", "SAMPLE_DATA_CONFIG",
    "settings",
]
