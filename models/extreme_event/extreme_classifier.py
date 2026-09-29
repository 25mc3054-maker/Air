import os
import joblib
import numpy as np
import xgboost as xgb
from typing import Dict, Any, Tuple, Optional

class ExtremePollutionClassifier:
    """
    Imbalance-aware classifier for severe air pollution episodes in Delhi NCR.
    Predicts probability of breaching critical thresholds:
      - Threshold 200: Moderate/Poor boundary (PM2.5 >= 90 µg/m³)
      - Threshold 300: Poor/Very Poor boundary (PM2.5 >= 120 µg/m³)
      - Threshold 400: Severe Smog Episode (PM2.5 >= 250 µg/m³)
      - Threshold 500: Hazardous / Emergency Episode (PM2.5 >= 345 µg/m³)
    
    Employs scale_pos_weight to counteract class imbalance and optimize recall
    during winter stubble-burning and cold-pool trapping events.
    """
    def __init__(self, model_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = model_dir or os.path.join(base_dir, "models", "extreme_event")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "extreme_xgb_classifier.joblib")
        self.model: Optional[xgb.XGBClassifier] = None

    def fit(self, X_train: np.ndarray, y_train_pm25: np.ndarray, threshold: float = 250.0):
        """
        Trains extreme event classifier with balanced class weighting.
        X_train: [N, D] summary features
        y_train_pm25: [N] ground truth PM2.5 values
        """
        y_binary = (y_train_pm25 >= threshold).astype(int)
        pos = np.sum(y_binary == 1)
        neg = np.sum(y_binary == 0)
        scale_pos = max(1.0, float(neg) / max(1.0, float(pos)))

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
        self.model.fit(X_train, y_binary)
        joblib.dump(self.model, self.model_path)
        return self

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()
        if self.model is not None:
            return self.model.predict_proba(X)[:, 1]
        
        # Physics-based baseline heuristic if model not yet fitted
        # Severe risk increases when current PM2.5 > 150, wind < 2.0, fire FRP > 50
        probs = []
        for row in X:
            # Assuming row[0] is recent PM2.5, row[1] is wind, row[2] is fire
            cur_pm = row[0] if len(row) > 0 else 100.0
            prob = 1.0 / (1.0 + np.exp(-(cur_pm - 200.0) / 40.0))
            probs.append(float(np.clip(prob, 0.01, 0.99)))
        return np.array(probs)
