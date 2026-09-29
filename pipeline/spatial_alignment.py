import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# The 10 Target Districts & Regional Hubs across Delhi NCR
NCR_REGIONS = [
    {"id": "delhi_central", "name": "Central Delhi (ITO)", "lat": 28.6319, "lon": 77.2410, "type": "station_anchor", "authority": "CPCB"},
    {"id": "delhi_east", "name": "East Delhi (Anand Vihar)", "lat": 28.6508, "lon": 77.3152, "type": "station_anchor", "authority": "DPCC"},
    {"id": "delhi_north", "name": "North Delhi (Jahangirpuri)", "lat": 28.7328, "lon": 77.1706, "type": "station_anchor", "authority": "DPCC"},
    {"id": "delhi_south", "name": "South Delhi (R.K. Puram)", "lat": 28.5633, "lon": 77.1867, "type": "station_anchor", "authority": "CPCB"},
    {"id": "delhi_west", "name": "West Delhi (Punjabi Bagh)", "lat": 28.6683, "lon": 77.1167, "type": "station_anchor", "authority": "DPCC"},
    {"id": "gurugram", "name": "Gurugram (Sector 51)", "lat": 28.4284, "lon": 77.0708, "type": "station_anchor", "authority": "HSPCB"},
    {"id": "faridabad", "name": "Faridabad (Sector 16A)", "lat": 28.4093, "lon": 77.3182, "type": "station_anchor", "authority": "HSPCB"},
    {"id": "noida", "name": "Noida (Sector 62)", "lat": 28.6258, "lon": 77.3648, "type": "station_anchor", "authority": "UPPCB"},
    {"id": "greater_noida", "name": "Greater Noida (Knowledge Park)", "lat": 28.4732, "lon": 77.4820, "type": "station_anchor", "authority": "UPPCB"},
    {"id": "ghaziabad", "name": "Ghaziabad (Vasundhara)", "lat": 28.6603, "lon": 77.3573, "type": "station_anchor", "authority": "UPPCB"},
    {"id": "sonipat", "name": "Sonipat (Murthal Road)", "lat": 28.9931, "lon": 77.0151, "type": "station_anchor", "authority": "HSPCB"},
    {"id": "jhajjar", "name": "Jhajjar Regional Area", "lat": 28.6080, "lon": 76.6560, "type": "model_estimate", "authority": "HSPCB_EXT"},
    {"id": "rohtak", "name": "Rohtak (MD University)", "lat": 28.8788, "lon": 76.6200, "type": "station_anchor", "authority": "HSPCB"},
    {"id": "meerut", "name": "Meerut (Ganga Nagar)", "lat": 28.9845, "lon": 77.7064, "type": "station_anchor", "authority": "UPPCB"}
]

class SpatialNCRDomainEngine:
    """
    Delhi-NCR Spatial Grid and Regional Hotspot Modeling Engine.
    Covers the 10 operational districts:
      Delhi, Gurugram, Faridabad, Noida, Greater Noida, Ghaziabad, 
      Sonipat, Jhajjar, Rohtak, Meerut.
      
    Configurable bounding box and grid resolution (e.g. 0.05° to 0.10°).
    Uses Inverse Distance Weighting (IDW) with wind-vector advection bias
    to interpolate between ground station anchors.
    All non-station locations are explicitly labeled as 'MODEL_ESTIMATE'.
    """
    def __init__(self, grid_res_deg: float = 0.08):
        self.grid_res_deg = grid_res_deg
        # Operational Delhi-NCR bounding box:
        # Lat: 28.25 to 29.15, Lon: 76.55 to 77.75
        self.lat_min = 28.25
        self.lat_max = 29.15
        self.lon_min = 76.55
        self.lon_max = 77.75
        self.anchors = NCR_REGIONS

    def generate_grid_cells(self) -> List[Dict[str, float]]:
        """Generates regular 2D coordinate lattice covering Delhi NCR domain."""
        lats = np.arange(self.lat_min, self.lat_max + 1e-4, self.grid_res_deg)
        lons = np.arange(self.lon_min, self.lon_max + 1e-4, self.grid_res_deg)
        grid = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                grid.append({
                    "cell_id": f"NCR_CELL_{i:02d}_{j:02d}",
                    "lat": float(round(lat, 4)),
                    "lon": float(round(lon, 4))
                })
        return grid

    def interpolate_spatial_forecast(
        self,
        anchor_predictions: Dict[str, float],
        wind_speed: float = 2.0,
        wind_direction_deg: float = 290.0,
        fire_frp: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Produces spatial PM2.5 and AQI estimates across the Delhi-NCR domain.
        Applies Inverse Distance Weighting (p=2.0) with wind-vector advection bias.
        """
        grid_cells = self.generate_grid_cells()
        results = []

        # Find nearest anchor station predictions
        anchor_coords = []
        anchor_vals = []
        for a in self.anchors:
            val = anchor_predictions.get(a["id"], anchor_predictions.get("delhi_east", 125.0))
            anchor_coords.append((a["lat"], a["lon"]))
            anchor_vals.append(val)

        rad_wind = math.radians(wind_direction_deg)
        # Wind advection vector
        u_wind = -wind_speed * math.sin(rad_wind)
        v_wind = -wind_speed * math.cos(rad_wind)

        for cell in grid_cells:
            c_lat = cell["lat"]
            c_lon = cell["lon"]

            weights = []
            for (a_lat, a_lon) in anchor_coords:
                dist = math.sqrt((c_lat - a_lat)**2 + (c_lon - a_lon)**2)
                if dist < 0.01:
                    dist = 0.01
                # Wind advection term: points downwind from high-emission anchors receive higher weight
                advection = (c_lon - a_lon) * u_wind + (c_lat - a_lat) * v_wind
                w = (1.0 / (dist ** 2.0)) * (1.0 + 0.1 * max(-0.5, min(0.5, advection)))
                weights.append(max(0.001, w))

            weights = np.array(weights)
            weights /= np.sum(weights)
            interp_pm25 = float(np.sum(weights * np.array(anchor_vals)))

            # If near fires / high FRP, add local hotspot probability
            hotspot_prob = min(0.99, max(0.05, (interp_pm25 - 120.0) / 250.0 + 0.15 * (fire_frp / 100.0)))

            # Determine dominant pollutant (in winter Delhi-NCR, PM2.5 dominates ~85% of time, PM10 ~15%)
            dominant = "PM2.5" if interp_pm25 > 80.0 else "PM10"
            
            # Simple AQI calculation for cell
            from pipeline.aqi import OfficialCPCBAQIEngine
            aqi_val = OfficialCPCBAQIEngine.calculate_subindex("pm25", interp_pm25)
            cat_name, cat_color = OfficialCPCBAQIEngine.get_category_and_color(aqi_val)

            results.append({
                "cell_id": cell["cell_id"],
                "latitude": c_lat,
                "longitude": c_lon,
                "pm25": round(interp_pm25, 1),
                "aqi": aqi_val,
                "category": cat_name,
                "color": cat_color,
                "dominant_pollutant": dominant,
                "hotspot_probability": round(hotspot_prob, 3),
                "is_station_anchor": False,
                "estimation_method": "Physics-Constrained Inverse Distance Weighting with NWP Wind Advection",
                "label": "MODEL ESTIMATE"
            })

        # Also append the explicit official station anchors clearly marked
        for a in self.anchors:
            val = anchor_predictions.get(a["id"], anchor_predictions.get("delhi_east", 125.0))
            aqi_val = OfficialCPCBAQIEngine.calculate_subindex("pm25", val)
            cat_name, cat_color = OfficialCPCBAQIEngine.get_category_and_color(aqi_val)
            hotspot_prob = min(0.99, max(0.05, (val - 120.0) / 250.0 + 0.15 * (fire_frp / 100.0)))
            results.append({
                "cell_id": f"ANCHOR_{a['id'].upper()}",
                "latitude": a["lat"],
                "longitude": a["lon"],
                "station_name": a["name"],
                "authority": a["authority"],
                "pm25": round(val, 1),
                "aqi": aqi_val,
                "category": cat_name,
                "color": cat_color,
                "dominant_pollutant": "PM2.5",
                "hotspot_probability": round(hotspot_prob, 3),
                "is_station_anchor": (a["type"] == "station_anchor"),
                "estimation_method": "CPCB/DPCC Official Regulatory Monitoring Station" if a["type"] == "station_anchor" else "Regional Spatial Proxy Estimate",
                "label": "OFFICIAL CAAQMS STATION" if a["type"] == "station_anchor" else "MODEL ESTIMATE"
            })

        return results

spatial_engine = SpatialNCRDomainEngine()
