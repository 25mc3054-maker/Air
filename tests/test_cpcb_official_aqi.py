import os
import sys
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from pipeline.aqi import OfficialCPCBAQIEngine

def test_pm25_breakpoints():
    engine = OfficialCPCBAQIEngine()
    # Good
    assert engine.calculate_subindex("pm25", 0.0) == 0
    assert engine.calculate_subindex("pm25", 15.0) == 25
    assert engine.calculate_subindex("pm25", 30.0) == 50
    # Satisfactory
    assert engine.calculate_subindex("pm25", 30.1) == 51
    assert engine.calculate_subindex("pm25", 60.0) == 100
    # Moderate
    assert engine.calculate_subindex("pm25", 60.1) == 101
    assert engine.calculate_subindex("pm25", 90.0) == 200
    # Poor
    assert engine.calculate_subindex("pm25", 90.1) == 201
    assert engine.calculate_subindex("pm25", 120.0) == 300
    # Very Poor
    assert engine.calculate_subindex("pm25", 120.1) == 301
    assert engine.calculate_subindex("pm25", 250.0) == 400
    # Severe
    assert engine.calculate_subindex("pm25", 250.1) == 401
    assert engine.calculate_subindex("pm25", 500.0) == 500

def test_pm10_breakpoints():
    engine = OfficialCPCBAQIEngine()
    assert engine.calculate_subindex("pm10", 25.0) == 25
    assert engine.calculate_subindex("pm10", 75.0) in [75, 76]
    assert engine.calculate_subindex("pm10", 250.0) == 200
    assert engine.calculate_subindex("pm10", 350.0) == 300
    assert engine.calculate_subindex("pm10", 430.0) == 400

def test_gaseous_pollutants():
    engine = OfficialCPCBAQIEngine()
    # NO2: 40 -> 50, 80 -> 100
    assert engine.calculate_subindex("no2", 40.0) == 50
    assert engine.calculate_subindex("no2", 80.0) == 100
    # SO2: 40 -> 50
    assert engine.calculate_subindex("so2", 40.0) == 50
    # CO: 1.0 -> 50, 2.0 -> 100
    assert engine.calculate_subindex("co", 1.0) == 50
    assert engine.calculate_subindex("co", 2.0) == 100

def test_governing_pollutant_and_compliance():
    engine = OfficialCPCBAQIEngine()
    # Scenario: PM2.5 is high, PM10 is moderate, NO2 is moderate
    concentrations = {
        "pm25": 135.0,  # Sub-index ~ 312 (Very Poor)
        "pm10": 180.0,  # Sub-index ~ 153 (Moderate)
        "no2": 65.0,    # Sub-index ~ 82 (Satisfactory)
        "so2": 15.0     # Sub-index ~ 19 (Good)
    }
    res = engine.calculate_comprehensive_aqi(concentrations)
    assert res["dominant_pollutant"] == "PM25"
    assert res["category"] == "Very Poor"
    assert res["aqi"] >= 310
    assert res["regulatory_compliant"] is True

def test_dominant_pollutant_switch():
    engine = OfficialCPCBAQIEngine()
    # Scenario: Ozone is highest on a summer afternoon
    concentrations = {
        "pm25": 25.0,   # Sub-index ~ 42 (Good)
        "pm10": 45.0,   # Sub-index ~ 45 (Good)
        "o3": 180.0     # Sub-index ~ 230 (Poor)
    }
    res = engine.calculate_comprehensive_aqi(concentrations)
    assert res["dominant_pollutant"] == "O3"
    assert res["category"] == "Poor"
    assert res["regulatory_compliant"] is True

if __name__ == "__main__":
    test_pm25_breakpoints()
    test_pm10_breakpoints()
    test_gaseous_pollutants()
    test_governing_pollutant_and_compliance()
    test_dominant_pollutant_switch()
    print("All official CPCB AQI engine tests passed cleanly!")
