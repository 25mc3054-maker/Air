import os
import joblib
import numpy as np
import xgboost as xgb
from typing import Dict, Any, Optional

class XGBoostForecastModel:
    """
    Chronologically validated XGBoost Tabular Baseline for 72-hour forecasting.
    Predicts multi-horizon trajectories and residual offsets.
    """
    def __init__(self, model_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = model_dir or os.path.join(base_dir, "models", "xgboost")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "xgboost_tabular.joblib")
        self.model: Optional[xgb.XGBRegressor] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        self.model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        joblib.dump(self.model, self.model_path)
        return self

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()
        if self.model is not None:
            return self.model.predict(X)
        # Persistence fallback
        return X[:, 0]
