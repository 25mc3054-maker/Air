import os
import sys
import pytest
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from api.app import (
    app,
    health_check,
    get_model_info,
    get_metrics,
    get_spatial_forecast,
    get_scenario_outlook,
    get_source_status,
    demo_predict,
    ForecastRequest,
    predict_pm25
)

def test_versioned_forecast_schema():
    res = demo_predict(1493)
    
    # 1. Check Versioning Envelope
    assert "forecast_versioning" in res, "Missing forecast_versioning metadata"
    ver = res["forecast_versioning"]
    assert "forecast_id" in ver and ver["forecast_id"].startswith("ATMOSAIR-")
    assert "issue_time" in ver
    assert "model_version" in ver
    assert "data_version" in ver

    # 2. Check Calibrated Uncertainty Intervals
    assert "uncertainty_intervals" in res, "Missing uncertainty_intervals"
    unc = res["uncertainty_intervals"]
    assert "p10_lower_80" in unc and len(unc["p10_lower_80"]) == 72
    assert "p90_upper_80" in unc and len(unc["p90_upper_80"]) == 72
    assert "p05_lower_95" in unc and len(unc["p05_lower_95"]) == 72
    assert "p95_upper_95" in unc and len(unc["p95_upper_95"]) == 72

    # 3. Check Multi-Model Forecast Curves
    assert "pm25_raw_deep" in res and len(res["pm25_raw_deep"]) == 72
    assert "pm25_corrected" in res and len(res["pm25_corrected"]) == 72
    assert "pm25_forecast" in res and len(res["pm25_forecast"]) == 72

    # 4. Check Official AQI & Dominant Pollutant
    assert "aqi_compliance" in res
    assert "dominant_pollutant" in res["aqi_compliance"]
    assert "dominant_pollutants" in res and len(res["dominant_pollutants"]) == 72

def test_spatial_endpoint():
    spatial = get_spatial_forecast(wind_speed=2.5, wind_direction=280.0, fire_frp=20.0)
    assert spatial["domain"] is not None
    assert len(spatial["grid_data"]) > 50
    cell_sample = spatial["grid_data"][0]
    assert "latitude" in cell_sample and "longitude" in cell_sample
    assert "pm25" in cell_sample and "aqi" in cell_sample
    assert "dominant_pollutant" in cell_sample

def test_scenario_endpoint():
    scenario = get_scenario_outlook(base_year=2024, baseline_pm25=108.5)
    assert "Long-Range Scenario Outlook" in scenario["outlook_type"]
    assert "scenarios" in scenario
    assert "bau" in scenario["scenarios"]
    assert "ncap_strict" in scenario["scenarios"]
    assert len(scenario["horizon_years"]) == 5

def test_source_health_endpoint():
    sources = get_source_status()
    assert sources["status"] == "OPERATIONAL"
    assert len(sources["sources"]) == 6
    names = [s["source"] for s in sources["sources"]]
    assert any("CPCB" in n for n in names)
    assert any("FIRMS" in n for n in names)
    assert any("CAMS" in n for n in names)
    assert any("WRF" in n for n in names)

if __name__ == "__main__":
    test_versioned_forecast_schema()
    test_spatial_endpoint()
    test_scenario_endpoint()
    test_source_health_endpoint()
    print("All forecast schema & API endpoint tests passed cleanly!")
