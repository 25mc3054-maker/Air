# ATMOSAIR v2 — End-to-End System Architecture
**Smart India Hackathon 2026 | Problem Statement 26082**
**System:** Operational 72-Hour Coupled Air Quality & AQI Forecasting System (Delhi NCR Focus)

---

## 1. High-Level Architectural Flow

```
+-----------------------------------------------------------------------------------+
|                            EXTERNAL DATA INGESTION                                |
|  [CPCB/DPCC Ground CAAQMS]       [Copernicus CAMS ADS]      [NASA FIRMS MODIS/VIIRS]  |
|  [Copernicus Sentinel-5P]        [Copernicus CDS (ERA5)]    [WRF / NWP Meteorology] |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                        CADENCE & QUALITY CONTROL ENGINE                           |
|  - Source-specific TTL Caching (1h / 3h / 6h)                                      |
|  - Unit Normalization (Kelvin->C, Pa->mmHg, Speed/Direction -> local U, V)        |
|  - Physical Bounds Clamping & Outlier Filtering                                   |
|  - Strict Causality Gate: t_obs <= T_issue (Zero Future Ground Truth Leakage)     |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                   SPATIO-TEMPORAL FUSION & FEATURE PIPELINE                       |
|  - Historical Tensor Builder: [1, 72, 49] across 4 causal domain branches          |
|  - Future Meteorological Forcing: [72, 18] extracted strictly from NWP / WRF     |
|  - Delhi-NCR Regional Domain Lattice (~8km resolution, 10 Target Districts)       |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                          MULTI-TIER MODEL INFERENCE                               |
|  1. CoupledMultiBranchForecastModel (819k parameters, Causal Conv1D + Causal MHA) |
|  2. Observation-Driven XGBoost Residual Corrector:                                 |
|     residual_{t+h} = f(raw_pred, h, recent_obs, recent_err, WRF_met, CAMS, FIRMS) |
|     corrected_{t+h} = max(0, raw_pred_{t+h} + residual_{t+h})                     |
|  3. Imbalance-Aware Extreme Event Classifier (scale_pos_weight tuned)              |
|  4. Conformal Uncertainty Calibrator: P10, P50, P90 (80% and 95% empirical bands) |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                  OFFICIAL CPCB IND-AQI & DOMINANT POLLUTANT ENGINE                 |
|  - Piecewise linear sub-index interpolation across all 7 criteria pollutants:     |
|    PM2.5, PM10, NO2, SO2, CO, O3, NH3                                             |
|  - Governing AQI: AQI = max(I_PM2.5, I_PM10, I_NO2, I_SO2, I_CO, I_O3, I_NH3)    |
|  - Governing Dominant Pollutant Identification & Regulatory Compliance Check      |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                      DELIVERY, SPATIAL MAPS & API LAYER                           |
|  - FastAPI (REST API with unique Forecast ID, Provenance, and Versioning)          |
|  - Interactive Dashboard: Spatial Hotspot Lattice, 5-Year Long-Range Scenario     |
|    Outlook (Policy simulation), Hourly AQI / PM2.5 curves, and Uncertainty Bands |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Subsystems

### A. Data Adapters (`data_sources/`)
* **CPCBAdapter:** Ingests official DPCC and CPCB Continuous Ambient Air Quality Monitoring Station (CAAQMS) measurements.
* **NASA_FIRMS_Adapter:** Ingests active fire hotspots and computes radial Fire Radiative Power (FRP) and fire counts across 25km, 50km, and 100km radii.
* **CAMSAdapter:** Ingests atmospheric composition context (PM2.5, PM10, AOD 550nm).
* **Sentinel5PAdapter:** Queries tropospheric NO2, CO, and UV Aerosol Index columns via Copernicus Data Space STAC.
* **CopernicusCDSAdapter:** Restricted to historical reanalysis (ERA5) and WRF boundary conditions. **Never** used for future weather forcing.
* **WRFAdapter:** Extracts regional NWP future meteorological forecasts ($T+1 \dots T+72$) including planetary boundary layer height (PBLH).

### B. Machine Learning Engine (`models/` & `pipeline/`)
1. **CoupledMultiBranchForecastModel:**
   - Parameters: 819,874 (frozen neural backbone).
   - Domain branches: Pollution (9), Meteorology (21), Atmospheric External (9), Temporal & Availability (10).
   - Gated Linear Unit (GLU) fusion into 256 dimensions.
   - Causal Multi-Head Attention ($d_{model}=256$, 4 heads, upper-triangular causal masking).
2. **ObservationResidualCorrector:**
   - Compensates for systematic negative compression during severe episodes ($\ge 345 \text{ µg/m}^3$).
   - Trained on chronological validation residuals.
   - Delivers a **9.57% MAE reduction** and a **21.37 µg/m³ bias reduction** in the severe regime on the held-out 2023 test set.
3. **ExtremePollutionClassifier:**
   - Imbalance-aware XGBoost classifier trained to detect severe episodes ($\ge 250 \text{ µg/m}^3$) with `scale_pos_weight` penalizing false negatives.
4. **EnsembleCalibrator:**
   - Blends deep predictions, corrected predictions, and persistence.
   - Computes empirical conformal uncertainty intervals (P10, P50, P90).

### C. Regulatory AQI Engine (`pipeline/aqi.py`)
- Full compliance with the official Ministry of Environment, Forest and Climate Change (MoEFCC) / Central Pollution Control Board (CPCB) 2014 standard.
- Implements piecewise linear sub-indices across 7 criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3, NH3).
- Identifies the governing pollutant (e.g. PM2.5 in winter, O3 on summer afternoons).

### D. Spatial Delhi-NCR Grid Engine (`pipeline/spatial_alignment.py`)
- Configurable bounding box covering all 10 target NCR districts: Delhi, Gurugram, Faridabad, Noida, Greater Noida, Ghaziabad, Sonipat, Jhajjar, Rohtak, Meerut.
- Inverse Distance Weighting (IDW) modulated by WRF wind advection vectors.
- Clearly designates cells as `OFFICIAL CAAQMS STATION` vs `MODEL ESTIMATE`.

### E. 5-Year Long-Range Scenario Engine (`pipeline/scenario_engine.py`)
- Aggregated policy and climate risk simulator (NOT a deterministic hourly AQI prediction).
- Compares:
  1. Business As Usual (BAU)
  2. Accelerated NCAP + GRAP IV Compliance
  3. Adverse Climate / Boundary Layer Stagnation
  4. Comprehensive Clean Transition
