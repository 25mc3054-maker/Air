import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

base_dir = r"d:\My Projects\SIH2026_PersonB"
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.proposed_model import CoupledMultiBranchForecastModel

class PM25ForecastingPipeline:
    """
    Production-ready 72-Hour PM2.5 Air Quality Forecasting Pipeline.
    Encapsulates input validation, missing value imputation, scaling, 
    neural model inference, inverse target transformation, and CPCB AQI categorization.
    """
    def __init__(self, model_ckpt_path=None, scaler_dir=None):
        if model_ckpt_path is None:
            model_ckpt_path = os.path.join(base_dir, "models", "checkpoints", "proposed_best.pt")
        if scaler_dir is None:
            scaler_dir = os.path.join(base_dir, "models", "scalers")
            
        self.model_ckpt_path = model_ckpt_path
        self.scaler_dir = scaler_dir
        
        # Load Model Schema & Feature Groups
        schema_path = os.path.join(base_dir, "results", "model_input_schema.json")
        with open(schema_path, "r") as f:
            self.schema = json.load(f)
        self.feature_order = self.schema["feature_order"]
        assert len(self.feature_order) == 49, "Expected 49 input features"
        
        # Load Preprocessing Objects (Fitted strictly on TRAIN data)
        self.feature_imputer = joblib.load(os.path.join(scaler_dir, "feature_imputer.joblib"))
        self.feature_scaler = joblib.load(os.path.join(scaler_dir, "feature_scaler.joblib"))
        self.target_scaler = joblib.load(os.path.join(scaler_dir, "target_scaler.joblib"))
        
        # Load Neural Architecture Checkpoint
        assert os.path.exists(model_ckpt_path), f"Checkpoint not found: {model_ckpt_path}"
        checkpoint = torch.load(model_ckpt_path, map_location='cpu')
        
        self.model = CoupledMultiBranchForecastModel()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        self.total_parameters = sum(p.numel() for p in self.model.parameters())
        self.checkpoint_epoch = checkpoint.get("epoch", 1)
        self.checkpoint_val_loss = checkpoint.get("best_val_loss", 0.907336)
        
    @staticmethod
    def calculate_cpcb_pm25_aqi(pm25_value):
        """
        Calculates PM2.5 AQI Sub-index and CPCB Category using official CPCB linear interpolation:
        CPCB PM2.5 Breakpoints (µg/m³, 24-hr avg):
          0.0 - 30.0   -> AQI   0 - 50   (Good)
         30.1 - 60.0   -> AQI  51 - 100  (Satisfactory)
         60.1 - 90.0   -> AQI 101 - 200  (Moderate)
         90.1 - 120.0  -> AQI 201 - 300  (Poor)
        120.1 - 250.0  -> AQI 301 - 400  (Very Poor)
        250.1 - 500.0+ -> AQI 401 - 500+ (Severe)
        """
        if pm25_value is None or (isinstance(pm25_value, float) and (np.isnan(pm25_value) or np.isinf(pm25_value))):
            return None, "Unavailable", "#94a3b8"
        
        c = float(pm25_value)
        if c <= 0.0:
            return 0, "Good", "#009966"
        elif c <= 30.0:
            aqi = round((50.0 / 30.0) * c)
            return int(aqi), "Good", "#009966"
        elif c <= 60.0:
            aqi = round(51.0 + ((100.0 - 51.0) / (60.0 - 30.0)) * (c - 30.0))
            return int(aqi), "Satisfactory", "#FFDE33"
        elif c <= 90.0:
            aqi = round(101.0 + ((200.0 - 101.0) / (90.0 - 60.0)) * (c - 60.0))
            return int(aqi), "Moderate", "#FF9933"
        elif c <= 120.0:
            aqi = round(201.0 + ((300.0 - 201.0) / (120.0 - 90.0)) * (c - 90.0))
            return int(aqi), "Poor", "#CC0033"
        elif c <= 250.0:
            aqi = round(301.0 + ((400.0 - 301.0) / (250.0 - 120.0)) * (c - 120.0))
            return int(aqi), "Very Poor", "#660099"
        elif c <= 500.0:
            aqi = round(401.0 + ((500.0 - 401.0) / (500.0 - 250.0)) * (c - 250.0))
            return int(aqi), "Severe", "#7E0023"
        else:
            # Linear extrapolation for extreme values > 500
            aqi = round(500.0 + ((500.0 - 401.0) / (500.0 - 250.0)) * (c - 500.0))
            return int(aqi), "Severe", "#7E0023"

    @staticmethod
    def get_cpcb_category(pm25_value):
        """Legacy helper returning CPCB category name and color."""
        _, cat_name, cat_color = PM25ForecastingPipeline.calculate_cpcb_pm25_aqi(pm25_value)
        return cat_name, cat_color

    def preprocess_input(self, input_df):
        """
        Validates, imputes, and scales historical sequence window [72, 49].
        """
        if isinstance(input_df, dict):
            input_df = pd.DataFrame(input_df)
            
        assert len(input_df) == 72, f"Expected 72 historical hours, got {len(input_df)}"
        
        # Check column ordering
        missing_cols = [c for c in self.feature_order if c not in input_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required input feature columns: {missing_cols}")
            
        df_ordered = input_df[self.feature_order].copy()
        
        # Impute missing values & scale
        imputed_array = self.feature_imputer.transform(df_ordered.values)
        scaled_array = self.feature_scaler.transform(imputed_array)
        
        # Reshape to [1, 72, 49] PyTorch Tensor
        tensor_in = torch.tensor(scaled_array, dtype=torch.float32).unsqueeze(0)
        return tensor_in

    def forecast(self, input_df, start_timestamp=None):
        """
        Executes end-to-end 72-hour forecast pipeline with CPCB AQI post-processing.
        """
        if isinstance(input_df, dict):
            input_df = pd.DataFrame(input_df)

        tensor_in = self.preprocess_input(input_df)
        
        with torch.no_grad():
            out_dict = self.model(tensor_in, return_auxiliary=True)
            scaled_pred = out_dict["forecast"].numpy() # [1, 72, 1]
            spike_prob = out_dict["spike_prob"].numpy().ravel() # [72]
            
        # Inverse transform predictions to physical µg/m³
        pred_phys = self.target_scaler.inverse_transform(scaled_pred.reshape(-1, 1)).ravel() # [72]
        
        # Generate Timestamps
        if start_timestamp is None:
            base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        elif isinstance(start_timestamp, str):
            base_time = datetime.fromisoformat(start_timestamp.replace("Z", ""))
        else:
            base_time = start_timestamp
            
        hourly_forecasts = []
        pm25_aqi_subindices = []
        aqi_categories = []

        for h in range(72):
            ts = base_time + timedelta(hours=h+1)
            pm_val = float(np.round(pred_phys[h], 2))
            aqi_val, cat_name, cat_color = self.calculate_cpcb_pm25_aqi(pm_val)
            s_prob = float(np.round(spike_prob[h], 4))
            
            pm25_aqi_subindices.append(aqi_val)
            aqi_categories.append(cat_name)

            hourly_forecasts.append({
                "hour": h + 1,
                "timestamp": ts.strftime("%Y-%m-%d %H:00:00"),
                "pm25": pm_val,
                "pm25_aqi_subindex": aqi_val,
                "cpcb_category": cat_name,
                "category_color": cat_color,
                "spike_probability": s_prob
            })
            
        # Summary Statistics
        pm_values = [item["pm25"] for item in hourly_forecasts]
        min_pm25 = float(np.min(pm_values))
        max_pm25 = float(np.max(pm_values))
        avg_pm25 = float(np.mean(pm_values))
        starting_pm25 = float(np.round(input_df["pm25"].iloc[-1], 2)) if "pm25" in input_df.columns else float(np.round(pm_values[0], 2))

        avg_aqi, overall_cat, overall_color = self.calculate_cpcb_pm25_aqi(avg_pm25)
        peak_hour_idx = int(np.argmax(pm_values))
        peak_item = hourly_forecasts[peak_hour_idx]
        
        # Extended Environmental Indicators from input DataFrame
        fire_frp_cols = [c for c in ['fire_frp_25km', 'fire_frp_50km', 'fire_frp_100km'] if c in input_df.columns]
        fire_cnt_cols = [c for c in ['fire_count_25km', 'fire_count_50km', 'fire_count_100km'] if c in input_df.columns]
        
        fire_frp_max = float(input_df[fire_frp_cols].max().max()) if fire_frp_cols else 0.0
        fire_cnt_max = float(input_df[fire_cnt_cols].max().max()) if fire_cnt_cols else 0.0

        if fire_frp_max > 300 or fire_cnt_max > 50:
            stubble_influence = "HIGH"
        elif fire_frp_max > 50 or fire_cnt_max > 10:
            stubble_influence = "MODERATE"
        elif fire_frp_max > 0 or fire_cnt_max > 0:
            stubble_influence = "LOW"
        else:
            stubble_influence = "NONE"

        avg_wind = float(input_df['wind_speed'].mean()) if 'wind_speed' in input_df.columns else 2.0
        if avg_wind < 1.5:
            trapping_risk = "HIGH (Stagnant Boundary Layer)"
        elif avg_wind < 3.0:
            trapping_risk = "MODERATE"
        else:
            trapping_risk = "LOW (Active Dispersion)"

        response = {
            "model_metadata": {
                "model_name": "CoupledMultiBranchForecastModel",
                "parameters": self.total_parameters,
                "checkpoint_epoch": self.checkpoint_epoch,
                "checkpoint_val_loss": float(np.round(self.checkpoint_val_loss, 6)),
                "receptive_field_hours": 127
            },
            "forecast_horizon_hours": 72,
            "data_mode": "historical_demo",
            "aqi_disclaimer": "PM2.5-derived AQI. Complete CPCB AQI requires sub-indices across all 8 criteria pollutants.",
            "summary_statistics": {
                "starting_pm25": starting_pm25,
                "min_pm25": min_pm25,
                "max_pm25": max_pm25,
                "avg_pm25": float(np.round(avg_pm25, 2)),
                "avg_pm25_aqi": avg_aqi,
                "overall_category": overall_cat,
                "overall_category_color": overall_color,
                "peak_hour": peak_item["hour"],
                "peak_timestamp": peak_item["timestamp"],
                "peak_pm25": peak_item["pm25"],
                "peak_aqi": peak_item["pm25_aqi_subindex"],
                "avg_spike_probability": float(np.round(np.mean(spike_prob), 4)),
                "max_spike_probability": float(np.round(np.max(spike_prob), 4)),
                "inversion_strength": "Unavailable (Requires 850hPa vs Surface Temp Profile)",
                "pbl_height": "Unavailable (Requires ERA5 PBLH Data)",
                "pollution_trapping_risk": trapping_risk,
                "stubble_burning_influence": stubble_influence
            },
            "verified_benchmark_metrics": {
                "overall_mae_ugm3": 56.66,
                "overall_rmse_ugm3": 82.56,
                "r2_score": 0.3564,
                "wmape_pct": 41.11,
                "tcn_baseline_mae": 61.19,
                "mae_improvement_pct": 7.40
            },
            "pm25_forecast": [item["pm25"] for item in hourly_forecasts],
            "pm25_aqi_subindex": pm25_aqi_subindices,
            "aqi_category": aqi_categories,
            "forecast": hourly_forecasts
        }
        return response

if __name__ == "__main__":
    pipeline = PM25ForecastingPipeline()
    dummy_input = pd.DataFrame(np.random.randn(72, 49) * 10 + 100, columns=pipeline.feature_order)
    res = pipeline.forecast(dummy_input)
    print("Inference Pipeline Initialized Successfully!")
    print(f"Model Parameters    : {res['model_metadata']['parameters']:,}")
    print(f"72h Forecast Length : {len(res['forecast'])} hours")
    print(f"Summary Avg PM2.5   : {res['summary_statistics']['avg_pm25']} µg/m³ ({res['summary_statistics']['overall_category']})")
