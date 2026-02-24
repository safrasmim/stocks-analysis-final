"""
Ensemble - RF+XGB for single inputs, adds LSTM only for batch >= 10
"""
import numpy as np
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EnsembleModel:
    def __init__(self, weights=None):
        self.weights    = weights or {"random_forest": 0.45, "xgboost": 0.45, "lstm": 0.10}
        self.rf         = None
        self.xgb        = None
        self.lstm       = None
        self.is_trained = False

    def set_models(self, rf, xgb, lstm):
        self.rf         = rf
        self.xgb        = xgb
        self.lstm       = lstm
        self.is_trained = (
            (rf  is not None and rf.is_trained) or
            (xgb is not None and xgb.is_trained)
        )

    def predict(self, X):
        n            = len(X)
        probs        = np.zeros(n)
        total_weight = 0.0

        if self.rf and self.rf.is_trained:
            p             = self.rf.predict(X)["probabilities_up"]
            probs        += self.weights["random_forest"] * p
            total_weight += self.weights["random_forest"]

        if self.xgb and self.xgb.is_trained:
            p             = self.xgb.predict(X)["probabilities_up"]
            probs        += self.weights["xgboost"] * p
            total_weight += self.weights["xgboost"]

        # LSTM: only for batches of 10+ AND only if signal > 5% from neutral
        if self.lstm and self.lstm.is_trained and n >= 10:
            p      = self.lstm.predict(X)["probabilities_up"]
            signal = float(np.abs(p.mean() - 0.5))
            if signal > 0.05:
                probs        += self.weights["lstm"] * p
                total_weight += self.weights["lstm"]

        if total_weight > 0:
            probs /= total_weight

        return {
            "predictions":      (probs >= 0.5).astype(int),
            "probabilities_up": probs,
        }

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.weights, path / "ensemble_weights.joblib")

    def load(self, path: Path):
        try:
            self.weights = joblib.load(path / "ensemble_weights.joblib")
        except Exception:
            pass
