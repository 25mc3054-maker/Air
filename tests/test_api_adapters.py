import os
import sys
from datetime import datetime, timezone, timedelta

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data_sources import (
    CPCBAdapter,
    NASA_FIRMS_Adapter,
    CAMSAdapter,
    Sentinel5PAdapter,
    CopernicusCDSAdapter,
    WRFAdapter
)

def test_cpcb_adapter():
    adapter = CPCBAdapter()
    now = datetime.now(timezone.utc)
    records = adapter.fetch(now - timedelta(hours=24), now, station_id="anand_vihar")
    assert len(records) > 0
    vars_found = {r.variable for r in records}
    assert "pm25" in vars_found
    for r in records:
        assert r.latitude > 20.0
        assert r.longitude > 70.0
        assert r.source.startswith("CPCB_")

def test_firms_adapter():
    adapter = NASA_FIRMS_Adapter()
    now = datetime.now(timezone.utc)
    records = adapter.fetch(now - timedelta(hours=24), now, center_lat=28.65, center_lon=77.31)
    assert len(records) == 6 # 3 radii x (count, frp)
    vars_found = {r.variable for r in records}
    assert "fire_count_25km" in vars_found
    assert "fire_frp_50km" in vars_found

def test_cams_adapter():
    adapter = CAMSAdapter()
    now = datetime.now(timezone.utc)
    records = adapter.fetch(now - timedelta(hours=6), now)
    assert len(records) >= 3
    vars_found = {r.variable for r in records}
    assert "cams_pm25_ugm3" in vars_found
    assert "cams_aod550" in vars_found

def test_sentinel5p_adapter():
    adapter = Sentinel5PAdapter()
    now = datetime.now(timezone.utc)
    records = adapter.fetch(now - timedelta(hours=24), now)
    assert len(records) >= 3
    vars_found = {r.variable for r in records}
    assert "tropomi_no2_mol_m2" in vars_found

def test_cds_adapter():
    adapter = CopernicusCDSAdapter()
    now = datetime.now(timezone.utc)
    records = adapter.fetch(now - timedelta(hours=24), now)
    assert len(records) >= 5
    vars_found = {r.variable for r in records}
    assert "era5_2m_temperature_k" in vars_found

def test_wrf_adapter():
    adapter = WRFAdapter()
    now = datetime.now(timezone.utc)
    records = adapter.fetch(now, now + timedelta(hours=72))
    assert len(records) == 72 * 9 # 72 hours x 9 meteorological variables
    vars_found = {r.variable for r in records}
    assert "temperature_c" in vars_found
    assert "wind_speed" in vars_found
    assert "pbl_height_m" in vars_found

if __name__ == "__main__":
    test_cpcb_adapter()
    test_firms_adapter()
    test_cams_adapter()
    test_sentinel5p_adapter()
    test_cds_adapter()
    test_wrf_adapter()
    print("All API adapter unit tests passed successfully!")
