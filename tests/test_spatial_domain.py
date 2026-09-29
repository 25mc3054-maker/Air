import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from pipeline.spatial_alignment import SpatialNCRDomainEngine, NCR_REGIONS

def test_spatial_coverage():
    engine = SpatialNCRDomainEngine(grid_res_deg=0.1)
    grid = engine.generate_grid_cells()
    assert len(grid) > 50, "Grid resolution produced too few cells"

    # Verify all 10 required NCR districts are present in regional anchors
    district_names = [a["name"] for a in NCR_REGIONS]
    required = ["Delhi", "Gurugram", "Faridabad", "Noida", "Greater Noida", "Ghaziabad", "Sonipat", "Jhajjar", "Rohtak", "Meerut"]
    for req in required:
        assert any(req.lower() in name.lower() for name in district_names), f"Missing target district: {req}"

def test_spatial_interpolation():
    engine = SpatialNCRDomainEngine(grid_res_deg=0.1)
    sample_preds = {
        "delhi_east": 185.0,
        "delhi_west": 95.0,
        "gurugram": 140.0,
        "noida": 160.0
    }
    results = engine.interpolate_spatial_forecast(sample_preds, wind_speed=2.5, wind_direction_deg=280.0, fire_frp=35.0)
    assert len(results) > 50
    for r in results:
        assert "pm25" in r and r["pm25"] >= 0.0
        assert "aqi" in r
        assert "dominant_pollutant" in r
        assert "hotspot_probability" in r
        assert "label" in r
        if not r["is_station_anchor"]:
            assert r["label"] == "MODEL ESTIMATE"

if __name__ == "__main__":
    test_spatial_coverage()
    test_spatial_interpolation()
    print("Spatial domain tests passed cleanly!")
