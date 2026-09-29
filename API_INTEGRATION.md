# ATMOSAIR v2 — REST API & Integration Specification
**Smart India Hackathon 2026 | Problem Statement 26082**
**Base URL:** `http://127.0.0.1:8000`

---

## 1. Authentication & Security
* API keys are configured via environment variables (`.env`).
* In production, client calls must include `X-API-Key` or standard OAuth2 Bearer tokens in headers.
* Secrets and API tokens are never exposed in log outputs.

---

## 2. API Endpoints

### `GET /health`
Verifies backend service and model readiness.
* **Response (200 OK):**
```json
{
  "status": "HEALTHY",
  "service": "ATMOSAIR v2 Air Quality Forecasting API",
  "model_loaded": true,
  "parameters": 819874,
  "version": "2.0.0"
}
```

### `GET /model-info`
Returns architectural metadata, parameter counts, receptive field, and v2 components.
* **Response (200 OK):**
```json
{
  "model_name": "CoupledMultiBranchForecastModel",
  "total_parameters": 819874,
  "parameters_formatted": "819,874",
  "input_window_hours": 72,
  "forecast_horizon_hours": 72,
  "input_features": 49,
  "receptive_field_hours": 127,
  "v2_enhancements": [
    "Observation-Driven XGBoost Residual/Bias Corrector (Severe Episode MAE -9.57%)",
    "Official CPCB IND-AQI Multi-Pollutant Calculation Engine (PM2.5, PM10, NO2, SO2, CO, O3, NH3)",
    "Empirical Conformal Prediction Intervals (P10, P50, P90)",
    "Delhi NCR 10-District Spatial Grid Hotspot Modeling",
    "Modular WRF/WPS Numerical Weather Prediction Integration",
    "Five-Year Long-Range Scenario Policy Engine"
  ]
}
```

### `GET /metrics`
Reports test evaluation benchmarks across models and forecast horizons on the untouched 2023 test set.
* **Response (200 OK):**
```json
{
  "proposed_model_metrics": {
    "mae_ugm3": 56.66,
    "rmse_ugm3": 82.56,
    "r2_score": 0.3564,
    "wmape_pct": 41.11,
    "mean_bias_ugm3": -14.98
  },
  "v2_residual_corrected_metrics": {
    "mae_ugm3": 59.24,
    "rmse_ugm3": 81.24,
    "r2_score": 0.3768,
    "mean_bias_ugm3": 4.26,
    "severe_regime_mae_ugm3": 198.74,
    "severe_regime_bias_reduction_ugm3": 21.37,
    "severe_regime_mae_improvement_pct": 9.57
  }
}
```

### `POST /predict`
Executes 72-hour forecast inference on custom 72-hour historical sequences.
* **Request Body:**
```json
{
  "start_timestamp": "2024-01-15T00:00:00Z",
  "station_id": "anand_vihar",
  "sequence": [
    {"pm25": 145.2, "temperature_c": 18.5, "...": 0.0}
  ]
}
```
* **Response (200 OK):** Contains `forecast_versioning`, `forecast` (72 entries with hourly `pm25`, `governing_aqi`, `dominant_pollutant`, `uncertainty_p10`, `uncertainty_p90`), and `summary_statistics`.

### `GET /demo-predict?sample_id={id}`
Executes forecast inference using real held-out test sequence samples.
* **Parameters:** `sample_id` (e.g. `1493` for Anand Vihar, `2708` for Punjabi Bagh, `1686` for R.K. Puram, `1927` for Jahangirpuri, `123` for ITO).

### `GET /forecast/spatial`
Returns spatial grid estimates covering the 10 operational Delhi-NCR target districts.
* **Query Parameters:** `wind_speed` (float, m/s), `wind_direction` (float, degrees), `fire_frp` (float).
* **Response (200 OK):**
```json
{
  "domain": "Delhi-NCR Regional Air Quality Domain (10 Districts)",
  "bounding_box": {"lat_min": 28.25, "lat_max": 29.15, "lon_min": 76.55, "lon_max": 77.75},
  "grid_resolution_deg": 0.08,
  "total_cells": 133,
  "grid_data": [
    {
      "cell_id": "NCR_CELL_00_00",
      "latitude": 28.25,
      "longitude": 76.55,
      "pm25": 118.4,
      "aqi": 295,
      "category": "Poor",
      "dominant_pollutant": "PM2.5",
      "hotspot_probability": 0.32,
      "is_station_anchor": false,
      "label": "MODEL ESTIMATE"
    }
  ]
}
```

### `GET /forecast/scenario`
Returns the 5-Year Long-Range Policy & Climate Scenario Outlook.
* **Query Parameters:** `base_year` (default 2024), `baseline_pm25` (default 108.5).
* **Response (200 OK):** Contains 4 scenarios (`bau`, `ncap_strict`, `climate_stagnation`, `clean_transition`) with annual PM2.5 projections and expected severe smog days per year.

### `GET /forecast/sources`
Returns real-time connectivity status, rate limits, and latency across external environmental data sources.
