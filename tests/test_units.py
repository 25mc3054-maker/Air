import os
import sys
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from pipeline.normalization import (
    kelvin_to_celsius,
    celsius_to_kelvin,
    pa_to_mmhg,
    mmhg_to_pa,
    wind_to_uv,
    uv_to_wind,
    compute_temporal_features,
    sanitize_and_clamp_variable
)

def test_temperature_conversion():
    assert abs(kelvin_to_celsius(273.15) - 0.0) < 1e-4
    assert abs(kelvin_to_celsius(300.0) - 26.85) < 1e-4
    assert abs(celsius_to_kelvin(25.0) - 298.15) < 1e-4

def test_pressure_conversion():
    # 101325 Pa = 760 mmHg approx
    assert abs(pa_to_mmhg(101325.0) - 760.0) < 0.1
    assert abs(mmhg_to_pa(760.0) - 101325.0) < 20.0

def test_wind_vector_conversion():
    # North wind (from 0 deg / 360 deg) blowing south
    u, v = wind_to_uv(speed=10.0, direction_deg=0.0)
    assert abs(u) < 1e-4
    assert abs(v - (-10.0)) < 1e-4
    
    # West wind (from 270 deg) blowing east
    u, v = wind_to_uv(speed=10.0, direction_deg=270.0)
    assert abs(u - 10.0) < 1e-4
    assert abs(v) < 1e-4

    # Roundtrip check
    speed, dir_deg = uv_to_wind(u, v)
    assert abs(speed - 10.0) < 1e-3
    assert abs(dir_deg - 270.0) < 1e-3

def test_temporal_features():
    dt = pd.Timestamp("2024-06-15 12:00:00")
    tf = compute_temporal_features(dt)
    assert len(tf) == 8
    for k, v in tf.items():
        assert -1.0 <= v <= 1.0

def test_clamping():
    assert sanitize_and_clamp_variable("pm25", -5.0) == 0.0
    assert sanitize_and_clamp_variable("pm25", 2500.0) == 1500.0
    assert sanitize_and_clamp_variable("relative_humidity", 120.0) == 100.0
    assert sanitize_and_clamp_variable("pm25", np.nan) is None

if __name__ == "__main__":
    test_temperature_conversion()
    test_pressure_conversion()
    test_wind_vector_conversion()
    test_temporal_features()
    test_clamping()
    print("Unit & normalization tests passed successfully!")
