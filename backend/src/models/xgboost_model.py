import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class XGBoostModel:
    def __init__(self):
        self.model  = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train, y_train, X_val=None, y_val=None):
        Xs = self.scaler.fit_transform(X_train)
        eval_set = [(self.scaler.transform(X_val), y_val)] if X_val is not None else None
        self.model.fit(Xs, y_train, eval_set=eval_set, verbose=False)
        self.is_trained = True
        if X_val is not None:
            acc = self.model.score(self.scaler.transform(X_val), y_val)
            logger.info("XGB Validation Accuracy: %.4f", acc)
            return acc

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return {"predictions": self.model.predict(Xs),
                "probabilities_up": self.model.predict_proba(Xs)[:,1]}

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model,  path / "xgboost.joblib")
        joblib.dump(self.scaler, path / "xgb_scaler.joblib")

    def load(self, path: Path):
        self.model  = joblib.load(path / "xgboost.joblib")
        self.scaler = joblib.load(path / "xgb_scaler.joblib")
        self.is_trained = True

    def feature_importance(self, feature_names):
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)
