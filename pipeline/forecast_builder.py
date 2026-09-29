import os
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

from pipeline.normalization import (
    compute_temporal_features,
    wind_to_uv,
    sanitize_and_clamp_variable
)
from data_sources.base_adapter import DataRecord

class ForecastBuilder:
    """
    Constructs leak-free feature tensors and tabular representations for inference.
    Strictly enforces temporal causality:
      - Input history: T-71 ... T (only observations <= T)
      - Forecast horizon: T+1 ... T+72 (future meteorology from NWP/WRF only)
    """
    def __init__(self, feature_order: Optional[List[str]] = None):
        if feature_order is None:
            # Default to canonical 49 features
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            schema_path = os.path.join(base_dir, "results", "model_input_schema.json")
            if os.path.exists(schema_path):
                with open(schema_path, "r") as f:
                    schema = json.load(f)
                self.feature_order = schema["feature_order"]
            else:
                cfg_path = os.path.join(base_dir, "configs", "feature_groups.json")
                with open(cfg_path, "r") as f:
                    groups = json.load(f)
                self.feature_order = (groups["pollution"] + groups["meteorology"] + 
                                      groups["atmospheric_external"] + groups["temporal"] + groups["other"])
        else:
            self.feature_order = feature_order

        assert len(self.feature_order) == 49, f"Expected 49 features, got {len(self.feature_order)}"

    def audit_leakage(self, history_df: pd.DataFrame, issue_time: datetime) -> bool:
        """
        Validates that history dataframe contains NO timestamps or observations > issue_time.
        """
        if "timestamp" in history_df.columns:
            ts_series = pd.to_datetime(history_df["timestamp"])
            if (ts_series > issue_time).any():
                future_ts = ts_series[ts_series > issue_time].tolist()
                raise ValueError(f"LEAKAGE DETECTED: Input sequence contains timestamps beyond issue time {issue_time}: {future_ts}")
        return True

    def build_history_tensor(self, history_df: pd.DataFrame, issue_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Validates, reorders, and ensures complete 49-feature historical sequence [72, 49].
        """
        if issue_time is not None:
            self.audit_leakage(history_df, issue_time)

        assert len(history_df) == 72, f"Expected 72 historical timesteps, got {len(history_df)}"

        df_clean = history_df.copy()

        # Fill missing required columns with reasonable physical defaults
        for col in self.feature_order:
            if col not in df_clean.columns:
                if col.endswith("_available"):
                    df_clean[col] = 0.0
                elif "fire" in col:
                    df_clean[col] = 0.0
                elif col in ["wind_u_local", "wind_v_local"]:
                    df_clean[col] = 0.0
                else:
                    df_clean[col] = np.nan

        # Reorder columns strictly to match model schema
        return df_clean[self.feature_order]

    def build_future_wrf_features(
        self,
        wrf_records: List[DataRecord],
        issue_time: datetime,
        horizon_hours: int = 72
    ) -> pd.DataFrame:
        """
        Extracts future NWP/WRF meteorological features for horizons T+1 ... T+72.
        Guaranteed to contain only forecast variables (no ground truth leakage).
        """
        rows = []
        for h in range(horizon_hours):
            t_valid = issue_time + timedelta(hours=h + 1)
            t_iso = t_valid.strftime("%Y-%m-%d %H:00:00Z")
            
            # Find matching WRF records for this horizon step
            step_records = {r.variable: r.value for r in wrf_records if r.timestamp == t_iso}
            
            # Also derive temporal cyclic features for this future hour
            t_feats = compute_temporal_features(pd.Timestamp(t_valid))
            
            row = {
                "horizon_hour": h + 1,
                "timestamp": t_iso,
                "temperature_c": step_records.get("temperature_c", 22.0),
                "relative_humidity": step_records.get("relative_humidity", 60.0),
                "wind_speed": step_records.get("wind_speed", 2.0),
                "wind_direction": step_records.get("wind_direction", 280.0),
                "wind_u_local": step_records.get("wind_u_local", 0.0),
                "wind_v_local": step_records.get("wind_v_local", 0.0),
                "pressure_mmhg": step_records.get("pressure_mmhg", 745.0),
                "pbl_height_m": step_records.get("pbl_height_m", 400.0),
                **t_feats
            }
            rows.append(row)

        return pd.DataFrame(rows)

forecast_builder = ForecastBuilder()
