# ATMOSAIR — WRF / WPS Regional Numerical Weather Prediction Integration Guide
**Smart India Hackathon 2026 | Problem Statement 26082**
**Target Domain:** Delhi NCR Regional Atmospheric Boundary Layer

---

## 1. Executive Hardware & Environment Reality Check

The operational host runs **Windows 11 / x86_64**. The Weather Research and Forecasting (WRF-ARW) model is a native Linux POSIX Fortran/C MPI-compiled physics engine. 

### Implementation Strategy:
1. **Host Integration via Modular WRF Adapter (`data_sources/wrf_adapter.py`):**
   ATMOSAIR connects to WRF outputs (`wrfout_d03_*`) through a decoupled filesystem/API interface.
2. **Computational Architecture:**
   * WRF/WPS executes inside Linux containers (Docker / Singularity), a WSL2 Ubuntu 22.04 LTS instance, or a dedicated Linux cluster node.
   * ATMOSAIR ingests the resulting NetCDF4 fields, extracts grid-point timeseries, transforms units, and feeds them as future meteorological forecast forcing ($T+1 \dots T+72$) into the machine learning ensemble.
3. **Realistic Resolution Statement:**
   * Outer regional India domain (`d01`): **27 km** resolution.
   * Intermediate Indo-Gangetic Plains domain (`d02`): **9 km** resolution.
   * Inner Delhi-NCR operational domain (`d03`): **3 km** resolution (computationally feasible for daily 72h cycles on 16-32 vCPU workstations).
   * **Note:** Sub-kilometer claims (e.g. 400m) require massive supercomputing facilities (1000+ MPI cores) and are not claimed here.

---

## 2. WPS Namelist Configuration (`namelist.wps`)

Place the following configuration in your WPS directory:

```fortran
&share
 wrf_core = 'ARW',
 max_dom = 3,
 start_date = '__START_DATE__', '__START_DATE__', '__START_DATE__',
 end_date   = '__END_DATE__',   '__END_DATE__',   '__END_DATE__',
 interval_seconds = 21600,
 io_form_geogrid = 2,
/

&geogrid
 parent_id         =   1,   1,   2,
 parent_grid_ratio =   1,   3,   3,
 i_parent_start    =   1,  35,  40,
 j_parent_start    =   1,  25,  35,
 e_we              = 100, 121, 151,
 e_sn              = 100, 121, 151,
 geog_data_res     = 'default','default','default',
 dx = 27000,
 dy = 27000,
 map_proj = 'lambert',
 ref_lat   =  28.65,
 ref_lon   =  77.25,
 truelat1  =  20.0,
 truelat2  =  35.0,
 stand_lon =  77.25,
 geog_data_path = '/opt/wrf/geog'
/

&ungrib
 out_format = 'WPS',
 prefix = 'FILE',
/

&metgrid
 fg_name = 'FILE'
 io_form_metgrid = 2,
/
```

---

## 3. WRF Model Namelist Configuration (`namelist.input`)

Physics suite optimized for planetary boundary layer and winter smog dynamics in Delhi-NCR:

```fortran
&time_control
 run_days                            = 3,
 run_hours                           = 0,
 start_year                          = __YEAR__, __YEAR__, __YEAR__,
 start_month                         = __MONTH__, __MONTH__, __MONTH__,
 start_day                           = __DAY__, __DAY__, __DAY__,
 start_hour                          = __HOUR__, __HOUR__, __HOUR__,
 end_year                            = __END_YEAR__, __END_YEAR__, __END_YEAR__,
 end_month                           = __END_MONTH__, __END_MONTH__, __END_MONTH__,
 end_day                             = __END_DAY__, __END_DAY__, __END_DAY__,
 end_hour                            = __END_HOUR__, __END_HOUR__, __END_HOUR__,
 interval_seconds                    = 21600,
 history_interval                    = 60, 60, 60,
 frames_per_outfile                  = 72, 72, 72,
 restart                             = .false.,
 io_form_history                     = 2,
/

&domains
 time_step                           = 150,
 max_dom                             = 3,
 e_we                                = 100, 121, 151,
 e_sn                                = 100, 121, 151,
 e_vert                              = 45, 45, 45,
 p_top_requested                     = 5000,
 dx                                  = 27000, 9000, 3000,
 dy                                  = 27000, 9000, 3000,
 grid_id                             = 1, 2, 3,
 parent_id                           = 0, 1, 2,
 i_parent_start                      = 1, 35, 40,
 j_parent_start                      = 1, 25, 35,
 parent_grid_ratio                   = 1, 3, 3,
/

&physics
 mp_physics                          = 6, 6, 6,        ! WSM 6-class graupel scheme
 ra_lw_physics                       = 4, 4, 4,        ! RRTMG Longwave
 ra_sw_physics                       = 4, 4, 4,        ! RRTMG Shortwave
 radt                                = 27, 9, 3,
 sf_sfclay_physics                   = 1, 1, 1,        ! Revised MM5 surface layer
 sf_surface_physics                  = 2, 2, 2,        ! Noah Land Surface Model
 bl_pbl_physics                      = 1, 1, 1,        ! YSU Planetary Boundary Layer scheme (critical for PBLH)
 bldt                                = 0, 0, 0,
 cu_physics                          = 1, 1, 0,        ! Kain-Fritsch (off in 3km domain)
/
```

---

## 4. NWP Initial and Boundary Conditions

> [!IMPORTANT]
> **Leakage Prevention Rule:**
> Future forecast meteorological inputs MUST come from genuine NWP forecast products (e.g., GFS 0.25° or ECMWF IFS Open Data), NEVER from future ERA5 reanalysis!

To automate boundary condition retrieval for daily operational runs:
```bash
#!/bin/bash
# Download GFS 0.25-degree operational forecast GRIB2 files
DATE=$(date -u +%Y%m%d)
CYCLE="00"
BASE_URL="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.${DATE}/${CYCLE}/atmos"

mkdir -p /opt/wrf/data/gfs
for H in $(seq -w 0 3 72); do
    curl -s -O "${BASE_URL}/gfs.t${CYCLE}z.pgrb2.0p25.f${H}"
done
```

---

## 5. Output Extraction Pipeline to ATMOSAIR Feature Schema

Use the modular script `scripts/extract_wrf_to_atmosair.py` to extract 1D hourly timeseries at Delhi NCR monitoring stations:

```python
import xarray as xr
import pandas as pd
import numpy as np

def extract_wrf_station(wrfout_path: str, target_lat: float, target_lon: float):
    ds = xr.open_dataset(wrfout_path)
    # Find nearest grid index
    dist = (ds.XLAT.isel(Time=0) - target_lat)**2 + (ds.XLONG.isel(Time=0) - target_lon)**2
    j, i = np.unravel_index(dist.argmin(), dist.shape)
    
    # Extract variables
    t2_c = ds.T2.isel(south_north=j, west_east=i).values - 273.15
    psfc_pa = ds.PSFC.isel(south_north=j, west_east=i).values
    u10 = ds.U10.isel(south_north=j, west_east=i).values
    v10 = ds.V10.isel(south_north=j, west_east=i).values
    pblh = ds.PBLH.isel(south_north=j, west_east=i).values
    rainc = ds.RAINC.isel(south_north=j, west_east=i).values
    rainnc = ds.RAINNC.isel(south_north=j, west_east=i).values
    
    wind_spd = np.sqrt(u10**2 + v10**2)
    wind_dir = (np.degrees(np.arctan2(-u10, -v10)) + 360.0) % 360.0
    
    df = pd.DataFrame({
        "temperature_c": t2_c,
        "pressure_mmhg": psfc_pa / 133.322,
        "wind_speed": wind_spd,
        "wind_direction": wind_dir,
        "wind_u_local": u10,
        "wind_v_local": v10,
        "pbl_height_m": pblh,
        "precipitation_m": (rainc + rainnc) / 1000.0
    })
    return df
```

---

## 6. Phased WRF-Chem Roadmap

* **Phase 1 (Active):** WRF regional meteorology interface operational, extracting temperature, RH, winds, surface pressure, and boundary layer height (PBLH).
* **Phase 2 (Active):** Direct coupling of WRF meteorology into ATMOSAIR feature pipeline as future forecast forcing ($T+1 \dots T+72$).
* **Phase 3 (Optional Advanced):** WRF-Chem chemical transport modeling with EDGAR/CAMS emission inventories, enabled when HPC Linux infrastructure is allocated.
