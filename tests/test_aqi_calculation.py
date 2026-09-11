import os
import sys
import pytest
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from pipeline.inference_pipeline import PM25ForecastingPipeline

def test_cpcb_pm25_aqi_breakpoint_interpolation():
    calc = PM25ForecastingPipeline.calculate_cpcb_pm25_aqi
    
    # 1. Good Category (0 - 30 µg/m³ -> AQI 0 - 50)
    aqi, cat, color = calc(0.0)
    assert aqi == 0 and cat == "Good" and color == "#009966"
    
    aqi, cat, color = calc(15.0)
    assert aqi == 25 and cat == "Good"
    
    aqi, cat, color = calc(30.0)
    assert aqi == 50 and cat == "Good"

    # 2. Satisfactory Category (30.1 - 60 µg/m³ -> AQI 51 - 100)
    aqi, cat, color = calc(30.1)
    assert aqi == 51 and cat == "Satisfactory" and color == "#FFDE33"
    
    aqi, cat, color = calc(45.0)
    assert aqi == 76 and cat == "Satisfactory"
    
    aqi, cat, color = calc(60.0)
    assert aqi == 100 and cat == "Satisfactory"

    # 3. Moderate Category (60.1 - 90 µg/m³ -> AQI 101 - 200)
    aqi, cat, color = calc(60.1)
    assert aqi == 101 and cat == "Moderate" and color == "#FF9933"
    
    aqi, cat, color = calc(75.0)
    assert aqi in [150, 151] and cat == "Moderate"
    
    aqi, cat, color = calc(90.0)
    assert aqi == 200 and cat == "Moderate"

    # 4. Poor Category (90.1 - 120 µg/m³ -> AQI 201 - 300)
    aqi, cat, color = calc(90.1)
    assert aqi == 201 and cat == "Poor" and color == "#CC0033"
    
    aqi, cat, color = calc(99.42)
    assert aqi == 232 and cat == "Poor"
    
    aqi, cat, color = calc(120.0)
    assert aqi == 300 and cat == "Poor"

    # 5. Very Poor Category (120.1 - 250 µg/m³ -> AQI 301 - 400)
    aqi, cat, color = calc(120.1)
    assert aqi == 301 and cat == "Very Poor" and color == "#660099"
    
    aqi, cat, color = calc(185.0)
    assert aqi in [350, 351] and cat == "Very Poor"
    
    aqi, cat, color = calc(250.0)
    assert aqi == 400 and cat == "Very Poor"

    # 6. Severe Category (250.1 - 500+ µg/m³ -> AQI 401 - 500+)
    aqi, cat, color = calc(250.1)
    assert aqi == 401 and cat == "Severe" and color == "#7E0023"
    
    aqi, cat, color = calc(375.0)
    assert aqi in [450, 451] and cat == "Severe"
    
    aqi, cat, color = calc(500.0)
    assert aqi == 500 and cat == "Severe"

    aqi, cat, color = calc(750.0)
    assert aqi > 500 and cat == "Severe"

def test_cpcb_pm25_aqi_edge_cases():
    calc = PM25ForecastingPipeline.calculate_cpcb_pm25_aqi
    
    # Negative values
    aqi, cat, _ = calc(-10.0)
    assert aqi == 0 and cat == "Good"
    
    # NaN
    aqi, cat, _ = calc(np.nan)
    assert aqi is None and cat == "Unavailable"
    
    # None
    aqi, cat, _ = calc(None)
    assert aqi is None and cat == "Unavailable"
    
    # Infinity
    aqi, cat, _ = calc(np.inf)
    assert aqi is None and cat == "Unavailable"

def test_pipeline_aqi_array_generation():
    pipeline = PM25ForecastingPipeline()
    dummy_input = pd.DataFrame(np.random.randn(72, 49) * 10 + 100, columns=pipeline.feature_order)
    res = pipeline.forecast(dummy_input)
    
    assert "pm25_forecast" in res
    assert "pm25_aqi_subindex" in res
    assert "aqi_category" in res
    assert len(res["pm25_forecast"]) == 72
    assert len(res["pm25_aqi_subindex"]) == 72
    assert len(res["aqi_category"]) == 72
    assert "peak_aqi" in res["summary_statistics"]
    assert res["summary_statistics"]["peak_aqi"] >= 0

if __name__ == "__main__":
    test_cpcb_pm25_aqi_breakpoint_interpolation()
    test_cpcb_pm25_aqi_edge_cases()
    test_pipeline_aqi_array_generation()
    print("All AQI Unit Tests Passed Cleanly!")
