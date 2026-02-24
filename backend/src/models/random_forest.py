import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class RandomForestModel:
    def __init__(self):
        self.model  = RandomForestClassifier(n_estimators=300, max_depth=15,
                        min_samples_split=5, min_samples_leaf=2,
                        class_weight="balanced", random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train, y_train, X_val=None, y_val=None):
        Xs = self.scaler.fit_transform(X_train)
        self.model.fit(Xs, y_train)
        self.is_trained = True
        if X_val is not None:
            acc = self.model.score(self.scaler.transform(X_val), y_val)
            logger.info("RF Validation Accuracy: %.4f", acc)
            return acc

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return {"predictions": self.model.predict(Xs),
                "probabilities_up": self.model.predict_proba(Xs)[:,1]}

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model,  path / "random_forest.joblib")
        joblib.dump(self.scaler, path / "rf_scaler.joblib")

    def load(self, path: Path):
        self.model  = joblib.load(path / "random_forest.joblib")
        self.scaler = joblib.load(path / "rf_scaler.joblib")
        self.is_trained = True

    def feature_importance(self, feature_names):
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)
