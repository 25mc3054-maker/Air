# ATMOSAIR v2 — Model Card

## Model Details
* **Model Name:** ATMOSAIR Coupled Multi-Branch Deep Neural Ensemble (`CoupledMultiBranchForecastModel_v2_Ensemble`)
* **Model Architecture:** Multi-Branch Causal Dilated Convolutions + Gated Linear Unit (GLU) Fusion + Causal Multi-Head Self-Attention + Dual Forecast/Spike Conv1D Heads + Observation-Driven XGBoost Residual Correction.
* **Model Parameters:** `819,874` trainable parameters in base neural backbone + auxiliary XGBoost residual/spike estimators.
* **Developers:** WINNERZWINNERZ Team (Smart India Hackathon 2026, Problem Statement 26082).
* **Model Version:** `v2.0.0-coupled-deep-ensemble` (Release: September 2026).

---

## Intended Use
* **Primary Operational Task:** 72-hour continuous hourly forecasting of PM2.5 concentrations and official CPCB National Air Quality Index (IND-AQI) across the Delhi-NCR metropolitan region.
* **Secondary Capabilities:**
  - Regional spatial hotspot detection across 10 NCR districts.
  - Multi-year aggregated policy and climate scenario outlook (5-Year Long-Range Simulation).
* **Out-of-Scope Uses:**
  - Deterministic minute-level AQI predictions (data cadence does not support minute precision).
  - Deterministic hourly forecasts 5 years into the future.
  - Medical advice for acute individual respiratory emergencies.

---

## Geographic & Temporal Specifications
* **Geographic Domain:** Delhi-NCR operational bounding box:
  - Latitude: `28.25° N` to `29.15° N`
  - Longitude: `76.55° E` to `77.75° E`
  - 10 Target Districts: Delhi (Central, East, North, South, West), Gurugram, Faridabad, Noida, Greater Noida, Ghaziabad, Sonipat, Jhajjar, Rohtak, Meerut.
* **Input Window:** 72 consecutive historical hours ($T-71 \dots T$).
* **Forecast Horizon:** 72 future hours ($T+1 \dots T+72$).
* **Temporal Resolution:** 1-hour continuous discrete steps.

---

## Chronological Partitioning & Datasets
* **Training Period:** April 4, 2015 10:00 UTC to December 31, 2021 23:00 UTC (`19,005` sequence samples).
* **Validation Period:** January 1, 2022 00:00 UTC to December 31, 2022 23:00 UTC (`4,493` sequence samples).
* **Test Period:** January 1, 2023 00:00 UTC to December 31, 2023 23:00 UTC (`3,393` sequence samples, untouched held-out evaluation).
* **Data Sources:**
  1. CPCB / DPCC Continuous Ambient Air Quality Monitoring Stations (CAAQMS).
  2. Copernicus CAMS EAC4 / ADS Atmospheric Composition.
  3. NASA FIRMS MODIS & VIIRS active fire hotspots and Fire Radiative Power (FRP).
  4. Copernicus Sentinel-5P / TROPOMI Level-2 atmospheric products.
  5. Copernicus CDS ERA5 historical reanalysis (historical training and WRF boundary conditions only).
  6. Regional Weather Research and Forecasting (WRF) / NWP forecast fields.

---

## Leakage Prevention Controls
1. **Strict Timestamp Causality:** At forecast issue time $T$, only observations with $t_{obs} \le T$ are permitted in the input tensor.
2. **No Future Ground Truth Leakage:** Horizons $T+1 \dots T+72$ never consume observed criteria pollutant values from ground stations.
3. **Future Weather Sourcing:** Future meteorology strictly originates from numerical forecast products (NWP/WRF), never from future ERA5 reanalysis or retrospective products.
4. **Automated Continuous Audit:** Enforced via `tests/test_leakage.py`.

---

## Performance Metrics (Held-Out 2023 Test Set)

| Model Configuration | Overall MAE | Overall RMSE | $R^2$ Score | Severe MAE ($\ge 345 \text{ µg/m}^3$) | Severe Mean Bias |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ATMOSAIR v2 Ensemble** | **56.66** | **81.24** | **0.3768** | **198.74** | **-198.29** |
| Deep Base Model | 56.66 | 82.56 | 0.3564 | 219.78 | -219.66 |
| TCN Baseline | 61.19 | 91.74 | 0.2053 | 223.01 | -222.80 |
| Naive Persistence | 75.42 | 107.00 | -0.0810 | 199.79 | -199.40 |

* **Severe Regime Improvements:** Residual correction reduces severe pollution MAE by **`9.57%`** and reduces severe underestimation bias by **`21.37 µg/m³`**.

---

## Uncertainty & Calibration
* Predictions are equipped with empirical conformal prediction intervals:
  - 80% Confidence Interval: P10 to P90
  - 95% Confidence Interval: P05 to P95
* Uncertainty bounds widen dynamically from +1h to +72h to account for growing meteorological forecast dispersion.

---

## Known Limitations
1. **Peak Extreme Underestimation:** Although residual correction substantially reduces negative bias, peak smog spikes exceeding $600 \text{ µg/m}^3$ (e.g. Diwali post-midnight spikes) remain partially smoothed due to target variance regularization.
2. **Station Density:** Non-station grid cells rely on inverse distance weighting and WRF advection; direct measurements require expanding the CAAQMS physical sensor grid.
3. **WRF Computational Latency:** Full regional WRF runs take 20–45 minutes on 16 vCPU nodes; the system relies on cache-aware scheduling.
