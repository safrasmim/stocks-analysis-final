"""
predictor.py
Loads all trained models and exposes a unified .predict() interface.
Used by app.py to serve predictions via the FastAPI endpoints.
"""
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Predictor:
    """
    Unified prediction interface for all trained models.
    Loads RF, XGBoost, LSTM, and Ensemble from the models directory.
    Falls back gracefully if any model is missing.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._rf        = None
        self._xgb       = None
        self._lstm      = None
        self._ensemble  = None
        self._lda       = None
        self._lda_dict  = None
        self._macro_df  = None
        self._loaded    = False

    def load(self):
        """Load all available models from model_dir."""
        errors = []

        # Random Forest
        rf_path = self.model_dir / "random_forest" / "random_forest.joblib"
        if rf_path.exists():
            try:
                from src.models.random_forest import RandomForestModel
                self._rf = RandomForestModel()
                self._rf.load(self.model_dir / "random_forest")
                logger.info("RF loaded.")
            except Exception as e:
                errors.append(f"RF: {e}")
        else:
            errors.append(f"RF model not found at {rf_path}")

        # XGBoost
        xgb_path = self.model_dir / "xgboost" / "xgboost.joblib"
        if xgb_path.exists():
            try:
                from src.models.xgboost_model import XGBoostModel
                self._xgb = XGBoostModel()
                self._xgb.load(self.model_dir / "xgboost")
                logger.info("XGB loaded.")
            except Exception as e:
                errors.append(f"XGB: {e}")
        else:
            errors.append(f"XGB model not found at {xgb_path}")

        # LSTM (optional — graceful skip)
        try:
            from src.models.lstm_model import LSTMModel
            lstm_path = self.model_dir / "lstm" / "lstm_model.h5"
            if lstm_path.exists():
                self._lstm = LSTMModel()
                self._lstm.load(self.model_dir / "lstm")
                logger.info("LSTM loaded.")
        except Exception as e:
            logger.warning("LSTM not loaded (non-fatal): %s", e)

        # Ensemble
        try:
            from src.models.ensemble import EnsembleModel
            self._ensemble = EnsembleModel()
            self._ensemble.set_models(self._rf, self._xgb, self._lstm)
            logger.info("Ensemble configured.")
        except Exception as e:
            errors.append(f"Ensemble: {e}")

        # LDA (for feature extraction)
        lda_path  = self.model_dir / "lda" / "lda_model.joblib"
        dict_path = self.model_dir / "lda" / "lda_dictionary.joblib"
        if lda_path.exists() and dict_path.exists():
            try:
                self._lda      = joblib.load(lda_path)
                self._lda_dict = joblib.load(dict_path)
                logger.info("LDA loaded.")
            except Exception as e:
                errors.append(f"LDA: {e}")

        # Macro data (optional, but used when available for realistic inference)
        try:
            from src.config import DATA_DIR
            from src.features.macro import load_macro_data
            self._macro_df = load_macro_data(DATA_DIR / "macro_indicators.csv")
            logger.info("Macro indicators loaded: %d rows", len(self._macro_df))
        except Exception as e:
            logger.warning("Macro data not loaded (non-fatal): %s", e)
            self._macro_df = None

        if errors:
            for err in errors:
                logger.warning("Load warning: %s", err)

        self._loaded = (self._rf is not None or self._xgb is not None)
        if not self._loaded:
            raise RuntimeError(
                "No models could be loaded from: " + str(self.model_dir) +
                "\nRun: python scripts/train_all_models.py"
            )

    def models_loaded(self) -> bool:
        return self._loaded and (self._rf is not None or self._xgb is not None)

    def _extract_features(
        self,
        texts: List[str],
        ticker: str = "",
        event_dates: Optional[List[str]] = None,
    ) -> np.ndarray:
        import pandas as pd
        from src.features.feature_engineering import extract_all_features, get_feature_matrix

        if event_dates and len(event_dates) == len(texts):
            dates = event_dates
        else:
            dates = [datetime.utcnow().strftime("%Y-%m-%d")] * len(texts)

        df = pd.DataFrame({
            "text":   texts,
            "ticker": [ticker] * len(texts),
            "date":   dates,
            "label":  [0] * len(texts),
        })
        df_feat = extract_all_features(
            df,
            macro_df=self._macro_df,
            lda_model=self._lda,
            lda_dict=self._lda_dict,
            model_dir=self.model_dir,
        )
        return get_feature_matrix(df_feat)

    def predict(
        self,
        texts: List[str],
        ticker: str = "",
        model: str = "ensemble",
        event_dates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.models_loaded():
            raise RuntimeError("Models not loaded. Call .load() first.")

        X = self._extract_features(texts, ticker, event_dates=event_dates)
        model_lower = model.lower()

        if model_lower in ("ensemble", "auto"):
            chosen = self._ensemble or self._rf or self._xgb
        elif model_lower in ("rf", "random_forest"):
            chosen = self._rf
        elif model_lower in ("xgb", "xgboost"):
            chosen = self._xgb
        elif model_lower == "lstm":
            chosen = self._lstm
        else:
            chosen = self._ensemble or self._rf or self._xgb

        if chosen is None:
            raise RuntimeError(f"Requested model '{model}' is not available.")

        result           = chosen.predict(X)
        predictions      = result.get("predictions", [])
        probabilities_up = result.get("probabilities_up", [])

        if len(probabilities_up) == 0:
            probabilities_up = [0.75 if p == 1 else 0.25 for p in predictions]

        return {
            "predictions":      [int(p) for p in predictions],
            "probabilities_up": [float(p) for p in probabilities_up],
        }

    def predict_single(self, text: str, ticker: str = "") -> Dict[str, Any]:
        result = self.predict([text], ticker=ticker)
        return {
            "prediction":     result["predictions"][0],
            "probability_up": result["probabilities_up"][0],
            "direction":      "UP" if result["predictions"][0] == 1 else "DOWN",
            "confidence":     round(
                (result["probabilities_up"][0]
                 if result["predictions"][0] == 1
                 else 1 - result["probabilities_up"][0]) * 100, 1
            ),
        }
