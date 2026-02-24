"""
LSTM Model - Fixed for single-sample prediction
"""
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class LSTMModel:
    def __init__(self, units=128, dropout=0.3, sequence_length=10,
                 epochs=50, batch_size=32, patience=10):
        self.units           = units
        self.dropout         = dropout
        self.sequence_length = sequence_length
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.patience        = patience
        self.model           = None
        self.scaler          = StandardScaler()
        self.is_trained      = False

    def _build_model(self, input_shape):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        m = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape,
                 dropout=self.dropout, recurrent_dropout=0.2),
            BatchNormalization(),
            LSTM(self.units // 2, dropout=self.dropout, recurrent_dropout=0.2),
            BatchNormalization(),
            Dense(64, activation="relu"),
            Dropout(self.dropout),
            Dense(1, activation="sigmoid"),
        ])
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return m

    def _make_sequences(self, X, y=None):
        Xs, ys = [], []
        for i in range(self.sequence_length, len(X)):
            Xs.append(X[i - self.sequence_length:i])
            if y is not None:
                ys.append(y[i])
        if not Xs:
            empty = np.array([]).reshape(0, self.sequence_length, X.shape[1])
            return (empty, np.array([])) if y is not None else empty
        return (np.array(Xs), np.array(ys)) if y is not None else np.array(Xs)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        Xs = self.scaler.fit_transform(X_train)
        X_seq, y_seq = self._make_sequences(Xs, y_train)
        if len(X_seq) == 0:
            logger.warning("Not enough sequences for LSTM training")
            return None
        self.model = self._build_model((self.sequence_length, X_train.shape[1]))
        val_data = None
        if X_val is not None and len(X_val) > self.sequence_length:
            Xv = self.scaler.transform(X_val)
            Xv_seq, yv_seq = self._make_sequences(Xv, y_val)
            if len(Xv_seq) > 0:
                val_data = (Xv_seq, yv_seq)
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs, batch_size=self.batch_size,
            validation_data=val_data,
            callbacks=[
                EarlyStopping(patience=self.patience, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5),
            ],
            verbose=0,
        )
        self.is_trained = True
        logger.info("LSTM training complete")

    def predict(self, X):
        n  = len(X)
        Xs = self.scaler.transform(X)

        # Tile rows to fill sequence window for small inputs
        if n < self.sequence_length:
            repeats = int(np.ceil(self.sequence_length / n))
            Xs_full = np.tile(Xs, (repeats, 1))[:self.sequence_length]
            X_seq   = Xs_full[np.newaxis, :, :]
            prob    = float(self.model.predict(X_seq, verbose=0)[0, 0])
            return {
                "predictions":      np.array([int(prob >= 0.5)] * n),
                "probabilities_up": np.array([prob] * n),
            }

        X_seq = self._make_sequences(Xs)
        if len(X_seq) == 0:
            return {"predictions": np.zeros(n, int),
                    "probabilities_up": np.full(n, 0.5)}
        probs = self.model.predict(X_seq, verbose=0).flatten()
        preds = (probs >= 0.5).astype(int)
        pad   = n - len(probs)
        if pad > 0:
            probs = np.concatenate([np.full(pad, float(probs[0])), probs])
            preds = (probs >= 0.5).astype(int)
        return {"predictions": preds, "probabilities_up": probs}

    def save(self, path: Path):
        import joblib
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path / "lstm_model.h5"))
        joblib.dump(self.scaler, path / "lstm_scaler.joblib")

    def load(self, path: Path):
        import joblib
        from tensorflow.keras.models import load_model
        self.model      = load_model(str(path / "lstm_model.h5"))
        self.scaler     = joblib.load(path / "lstm_scaler.joblib")
        self.is_trained = True
