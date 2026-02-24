import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class BaselineModel:
    """Historical price-only model — comparison baseline"""
    def __init__(self):
        self.model  = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_price, y):
        Xs = self.scaler.fit_transform(X_price)
        self.model.fit(Xs, y)
        self.is_trained = True
        logger.info("Baseline model trained.")

    def predict(self, X_price):
        Xs = self.scaler.transform(X_price)
        return {"predictions": self.model.predict(Xs),
                "probabilities_up": self.model.predict_proba(Xs)[:,1]}

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model,  path / "baseline.joblib")
        joblib.dump(self.scaler, path / "baseline_scaler.joblib")

    def load(self, path: Path):
        self.model  = joblib.load(path / "baseline.joblib")
        self.scaler = joblib.load(path / "baseline_scaler.joblib")
        self.is_trained = True
