import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from pipeline.forecast_builder import ForecastBuilder
from data_sources.wrf_adapter import WRFAdapter

def test_leakage_audit_rejects_future_timestamps():
    builder = ForecastBuilder()
    issue_time = datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create invalid sequence where last 2 hours are > issue_time
    timestamps = [issue_time - timedelta(hours=72 - i - 1) + timedelta(hours=2) for i in range(72)]
    data = {c: np.random.randn(72) for c in builder.feature_order}
    data["timestamp"] = timestamps
    df_invalid = pd.DataFrame(data)

    with pytest.raises(ValueError) as excinfo:
        builder.audit_leakage(df_invalid, issue_time)
    assert "LEAKAGE DETECTED" in str(excinfo.value)

def test_leakage_audit_accepts_strictly_past_timestamps():
    builder = ForecastBuilder()
    issue_time = datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
    
    # Valid strictly past timestamps: T-71 ... T
    timestamps = [issue_time - timedelta(hours=71 - i) for i in range(72)]
    data = {c: np.random.randn(72) for c in builder.feature_order}
    data["timestamp"] = timestamps
    df_valid = pd.DataFrame(data)

    assert builder.audit_leakage(df_valid, issue_time) is True

def test_future_forcing_contains_no_ground_truth_targets():
    builder = ForecastBuilder()
    wrf_adapter = WRFAdapter()
    issue_time = datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
    
    wrf_records = wrf_adapter.fetch(issue_time, issue_time + timedelta(hours=72))
    future_df = builder.build_future_wrf_features(wrf_records, issue_time, horizon_hours=72)
    
    assert len(future_df) == 72
    # Ensure none of the ground truth criteria pollutants are present in future forcing features
    forbidden_targets = ["pm25", "pm10", "no", "no2", "nox", "nh3", "so2", "co", "o3"]
    for target in forbidden_targets:
        assert target not in future_df.columns, f"Target {target} leaked into future forecast forcing!"

    # Ensure allowed meteorological drivers and temporal cyclic encodings are present
    allowed_met = ["temperature_c", "relative_humidity", "wind_speed", "pbl_height_m", "hour_sin", "hour_cos"]
    for met in allowed_met:
        assert met in future_df.columns

if __name__ == "__main__":
    test_leakage_audit_rejects_future_timestamps()
    test_leakage_audit_accepts_strictly_past_timestamps()
    test_future_forcing_contains_no_ground_truth_targets()
    print("All automated leakage audit tests passed cleanly!")
