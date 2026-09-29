import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Physical validation thresholds for Delhi NCR ambient conditions
PHYSICAL_BOUNDS = {
    "pm25": (0.0, 1500.0),
    "pm10": (0.0, 3000.0),
    "no": (0.0, 500.0),
    "no2": (0.0, 500.0),
    "nox": (0.0, 800.0),
    "nh3": (0.0, 500.0),
    "so2": (0.0, 400.0),
    "co": (0.0, 50.0),
    "o3": (0.0, 600.0),
    "temperature_c": (-5.0, 55.0),
    "relative_humidity": (0.0, 100.0),
    "wind_speed": (0.0, 40.0),
    "wind_direction": (0.0, 360.0),
    "rainfall": (0.0, 300.0),
    "solar_radiation": (0.0, 1500.0),
    "pressure_mmhg": (650.0, 850.0),
    "cams_pm25_ugm3": (0.0, 1000.0),
    "cams_pm10_ugm3": (0.0, 2000.0),
    "cams_aod550": (0.0, 5.0)
}

def kelvin_to_celsius(k: float) -> float:
    return float(k - 273.15)

def celsius_to_kelvin(c: float) -> float:
    return float(c + 273.15)

def pa_to_mmhg(pa: float) -> float:
    # 1 mmHg = 133.322387415 Pa
    return float(pa / 133.322387415)

def mmhg_to_pa(mmhg: float) -> float:
    return float(mmhg * 133.322387415)

def wind_to_uv(speed: float, direction_deg: float) -> Tuple[float, float]:
    """
    Converts meteorological wind speed (m/s) and direction (degrees from North)
    into local Cartesian U (zonal, East-West) and V (meridional, North-South) vectors.
    """
    rad = math.radians(direction_deg)
    u = -speed * math.sin(rad)
    v = -speed * math.cos(rad)
    return float(round(u, 4)), float(round(v, 4))

def uv_to_wind(u: float, v: float) -> Tuple[float, float]:
    """Converts local U and V components back to speed (m/s) and direction (degrees)."""
    speed = math.sqrt(u * u + v * v)
    direction = (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0
    return float(round(speed, 4)), float(round(direction, 2))

def compute_temporal_features(dt: pd.Timestamp) -> Dict[str, float]:
    """
    Generates exact 8 cyclic trigonometric features matching model schema.
    """
    hour = dt.hour
    dow = dt.dayofweek
    doy = dt.dayofyear
    month = dt.month

    return {
        "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
        "dayofweek_sin": float(np.sin(2.0 * np.pi * dow / 7.0)),
        "dayofweek_cos": float(np.cos(2.0 * np.pi * dow / 7.0)),
        "dayofyear_sin": float(np.sin(2.0 * np.pi * doy / 365.25)),
        "dayofyear_cos": float(np.cos(2.0 * np.pi * doy / 365.25)),
        "month_sin": float(np.sin(2.0 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * month / 12.0))
    }

def sanitize_and_clamp_variable(var_name: str, val: Optional[float]) -> Optional[float]:
    """Enforces physical reality bounds on measurements and models."""
    if val is None or np.isnan(val) or np.isinf(val):
        return None
    if var_name in PHYSICAL_BOUNDS:
        low, high = PHYSICAL_BOUNDS[var_name]
        return float(np.clip(val, low, high))
    return float(val)
