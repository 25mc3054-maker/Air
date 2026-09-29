import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Official CPCB Breakpoints Table (CPCB 2014 Standard)
# Format: (C_low, C_high, I_low, I_high)
CPCB_BREAKPOINTS = {
    "pm25": [
        (0.0, 30.0, 0, 50),
        (30.0, 60.0, 51, 100),
        (60.0, 90.0, 101, 200),
        (90.0, 120.0, 201, 300),
        (120.0, 250.0, 301, 400),
        (250.0, 500.0, 401, 500)
    ],
    "pm10": [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 250.0, 101, 200),
        (250.0, 350.0, 201, 300),
        (350.0, 430.0, 301, 400),
        (430.0, 500.0, 401, 500)
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 180.0, 101, 200),
        (180.0, 280.0, 201, 300),
        (280.0, 400.0, 301, 400),
        (400.0, 500.0, 401, 500)
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 380.0, 101, 200),
        (380.0, 800.0, 201, 300),
        (800.0, 1600.0, 301, 400),
        (1600.0, 2000.0, 401, 500)
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.0, 2.0, 51, 100),
        (2.0, 10.0, 101, 200),
        (10.0, 17.0, 201, 300),
        (17.0, 34.0, 301, 400),
        (34.0, 50.0, 401, 500)
    ],
    "o3": [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 168.0, 101, 200),
        (168.0, 208.0, 201, 300),
        (208.0, 748.0, 301, 400),
        (748.0, 1000.0, 401, 500)
    ],
    "nh3": [
        (0.0, 200.0, 0, 50),
        (200.0, 400.0, 51, 100),
        (400.0, 800.0, 101, 200),
        (800.0, 1200.0, 201, 300),
        (1200.0, 1800.0, 301, 400),
        (1800.0, 2400.0, 401, 500)
    ]
}

CPCB_CATEGORIES = [
    (0, 50, "Good", "#009966"),
    (51, 100, "Satisfactory", "#FFDE33"),
    (101, 200, "Moderate", "#FF9933"),
    (201, 300, "Poor", "#CC0033"),
    (301, 400, "Very Poor", "#660099"),
    (401, 10000, "Severe", "#7E0023")
]

class OfficialCPCBAQIEngine:
    """
    Official Indian CPCB National Air Quality Index (IND-AQI) Engine.
    Implements piecewise linear interpolation across all 7 criteria pollutants,
    sub-index calculation, governing pollutant resolution, and regulatory category assignment.
    """

    @staticmethod
    def calculate_subindex(pollutant: str, concentration: Optional[float]) -> Optional[int]:
        """
        Calculates pollutant-specific AQI sub-index using official CPCB piecewise linear formula:
            I = I_low + ((I_high - I_low) / (C_high - C_low)) * (C - C_low)
        """
        if concentration is None or np.isnan(concentration) or np.isinf(concentration):
            return None
        
        c = float(concentration)
        if c <= 0.0:
            return 0

        p = pollutant.lower()
        if p not in CPCB_BREAKPOINTS:
            return None

        ranges = CPCB_BREAKPOINTS[p]

        # Check in which range concentration falls
        for c_low, c_high, i_low, i_high in ranges:
            if c <= c_high:
                aqi = i_low + ((i_high - i_low) / (c_high - c_low)) * (c - c_low)
                return int(round(aqi + 1e-9))

        # Extrapolation for concentrations exceeding the severe threshold
        last_c_low, last_c_high, last_i_low, last_i_high = ranges[-1]
        aqi = last_i_high + ((last_i_high - last_i_low) / (last_c_high - last_c_low)) * (c - last_c_high)
        return int(round(aqi + 1e-9))

    @staticmethod
    def get_category_and_color(aqi_value: Optional[int]) -> Tuple[str, str]:
        if aqi_value is None:
            return "Unavailable", "#94a3b8"
        for low, high, cat_name, hex_color in CPCB_CATEGORIES:
            if low <= aqi_value <= high:
                return cat_name, hex_color
        return "Severe", "#7E0023"

    @classmethod
    def calculate_comprehensive_aqi(cls, pollutant_concentrations: Dict[str, Optional[float]]) -> Dict[str, Any]:
        """
        Calculates governing CPCB AQI and dominant pollutant according to official methodology:
          1. Calculate sub-index for every available criteria pollutant.
          2. Check compliance: requires at least 3 pollutants, including PM2.5 or PM10.
             (If fewer are available, calculates governing AQI from available with clear advisory).
          3. Overall AQI = max(sub-indices).
          4. Governing / Dominant pollutant = pollutant with the highest sub-index.
        """
        subindices: Dict[str, Optional[int]] = {}
        valid_subindices: Dict[str, int] = {}

        for p, conc in pollutant_concentrations.items():
            sub = cls.calculate_subindex(p, conc)
            subindices[p] = sub
            if sub is not None:
                valid_subindices[p] = sub

        if not valid_subindices:
            return {
                "aqi": None,
                "category": "Unavailable",
                "color": "#94a3b8",
                "dominant_pollutant": "None",
                "subindices": subindices,
                "regulatory_compliant": False,
                "compliance_note": "No valid criteria pollutant concentrations available."
            }

        # Regulatory compliance check: requires >= 3 pollutants including PM2.5 or PM10
        has_particulate = ("pm25" in valid_subindices) or ("pm10" in valid_subindices)
        is_compliant = (len(valid_subindices) >= 3) and has_particulate

        # Governing AQI is the maximum of valid sub-indices
        dominant_p = max(valid_subindices, key=valid_subindices.get)
        governing_aqi = valid_subindices[dominant_p]
        cat_name, hex_color = cls.get_category_and_color(governing_aqi)

        compliance_note = (
            "Fully compliant with CPCB National AQI standard (>= 3 pollutants including particulate matter)."
            if is_compliant else
            f"Derived AQI based on {len(valid_subindices)} available pollutant(s). Complete official CPCB AQI requires >= 3 pollutants."
        )

        return {
            "aqi": governing_aqi,
            "category": cat_name,
            "color": hex_color,
            "dominant_pollutant": dominant_p.upper(),
            "dominant_subindex": governing_aqi,
            "subindices": subindices,
            "regulatory_compliant": is_compliant,
            "compliance_note": compliance_note
        }

aqi_engine = OfficialCPCBAQIEngine()
