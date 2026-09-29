import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, List, Optional, Tuple

class ObservationResidualCorrector:
    """
    Observation-Driven Residual / Bias Correction Engine.
    Learns systematic forecast error patterns:
        residual_{t+h} = observed_PM25_{t+h} - raw_forecast_{t+h}
        
    Predicts residual offsets using:
      - Recent observation levels & immediate forecast error
      - Forecast horizon step (1 to 72 hours)
      - Future WRF meteorology (wind speed, direction, temperature, humidity, PBL height)
      - Upwind biomass burning indicators (FIRMS FRP & fire counts)
      - Regional atmospheric background (CAMS PM2.5 & AOD550)
      - Cyclic temporal encodings (hour, day of week, day of year)
    
    Corrected forecast:
        corrected_{t+h} = max(0, raw_forecast_{t+h} + predicted_residual_{t+h})
        
    STRICT LEAKAGE AUDIT:
    Only observations available at or before issue time T are used as antecedent error features.
    Future meteorology strictly uses NWP/WRF forecast fields.
    """
    def __init__(self, model_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = model_dir or os.path.join(base_dir, "models", "bias_correction")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "residual_xgb_corrector.joblib")
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names = [
            "raw_forecast", "horizon_hour", "recent_observed_pm25", "recent_forecast_error",
            "wrf_temperature_c", "wrf_relative_humidity", "wrf_wind_speed", "wrf_wind_u", "wrf_wind_v",
            "wrf_pbl_height_m", "cams_pm25", "cams_aod550", "fire_frp_50km", "fire_count_50km",
            "hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos"
        ]

    def fit(self, X_features: np.ndarray, y_residuals: np.ndarray, val_features: Optional[np.ndarray] = None, val_residuals: Optional[np.ndarray] = None):
        """
        Trains the residual correction model using chronological training/validation sets.
        """
        self.model = xgb.XGBRegressor(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1
        )
        eval_set = [(val_features, val_residuals)] if (val_features is not None and val_residuals is not None) else None
        self.model.fit(X_features, y_residuals, eval_set=eval_set, verbose=False)
        joblib.dump(self.model, self.model_path)
        return self

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        return self

    def predict_residual(self, features_matrix: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()
        if self.model is not None:
            return self.model.predict(features_matrix)
        
        # Physics-based baseline offset if model not yet fitted
        # Severe episodes with low wind and high CAMS have persistent negative bias
        offsets = []
        for row in features_matrix:
            raw_val = row[0]
            ws = row[6] if len(row) > 6 else 2.0
            pbl = row[9] if len(row) > 9 else 400.0
            if raw_val > 250.0 and ws < 2.0:
                # Cold-pool inversion trapping bonus to relieve negative compression
                offset = (raw_val - 200.0) * 0.35 * (1.0 + max(0.0, (500.0 - pbl) / 500.0))
            elif raw_val > 150.0:
                offset = (raw_val - 120.0) * 0.15
            else:
                offset = 0.0
            offsets.append(float(offset))
        return np.array(offsets)

    def correct_forecast(
        self,
        raw_forecast_72h: np.ndarray,
        recent_obs_pm25: float,
        recent_error: float,
        future_wrf_df: pd.DataFrame,
        cams_pm25: float = 78.4,
        cams_aod: float = 0.65,
        fire_frp: float = 0.0,
        fire_count: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies learned residual correction over the 72-hour forecast horizon.
        Returns: (corrected_forecast_72h, predicted_residuals_72h)
        """
        rows = []
        for h in range(72):
            wrf_row = future_wrf_df.iloc[h] if len(future_wrf_df) > h else {}
            feature_vec = [
                float(raw_forecast_72h[h]),
                float(h + 1),
                float(recent_obs_pm25),
                float(recent_error),
                float(wrf_row.get("temperature_c", 22.0)),
                float(wrf_row.get("relative_humidity", 60.0)),
                float(wrf_row.get("wind_speed", 2.0)),
                float(wrf_row.get("wind_u_local", 0.0)),
                float(wrf_row.get("wind_v_local", 0.0)),
                float(wrf_row.get("pbl_height_m", 400.0)),
                float(cams_pm25),
                float(cams_aod),
                float(fire_frp),
                float(fire_count),
                float(wrf_row.get("hour_sin", 0.0)),
                float(wrf_row.get("hour_cos", 0.0)),
                float(wrf_row.get("dayofyear_sin", 0.0)),
                float(wrf_row.get("dayofyear_cos", 0.0))
            ]
            rows.append(feature_vec)

        X_mat = np.array(rows, dtype=np.float32)
        residuals = self.predict_residual(X_mat)
        
        # Apply additive correction clamped to physical reality >= 0
        corrected = np.maximum(0.0, raw_forecast_72h + residuals)
        return corrected, residuals

bias_corrector = ObservationResidualCorrector()
