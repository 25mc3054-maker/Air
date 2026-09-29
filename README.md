# ATMOSAIR v2 — 72-Hour Coupled Air Quality & AQI Forecasting System
> **Smart India Hackathon (SIH) 2026 | Problem Statement ID: 26082**  
> **Team:** WINNERZWINNERZ | **Product:** ATMOSAIR  
> **Domain Focus:** Delhi NCR Regional Air Quality Basin (10 Operational Districts)

---

## 1. Executive Summary & Problem Overview
Air pollution in the Delhi National Capital Region (NCR) is a severe public health challenge driven by complex seasonal meteorological trapping, regional biomass burning, and heavy urban emissions. Standard forecasting prototypes degrade sharply beyond 12 hours and systematically underestimate extreme winter episodes.

**ATMOSAIR v2** is an operational, scientifically grounded 72-hour air quality forecasting system. Built around a **Coupled Multi-Branch Deep Neural Network (`CoupledMultiBranchForecastModel`, 819,874 parameters)**, the system incorporates:
1. **Observation-Driven XGBoost Residual Correction:** Resolves negative bias during severe pollution episodes ($\ge 345 \text{ µg/m}^3$), reducing severe MAE by **9.57%** and bias by **21.37 µg/m³**.
2. **Official CPCB National AQI (IND-AQI) Engine:** Full piecewise linear sub-indices across 7 criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3, NH3) with governing pollutant tracking.
3. **Regional Spatial Lattice:** Covers 10 NCR districts (Delhi, Gurugram, Faridabad, Noida, Greater Noida, Ghaziabad, Sonipat, Jhajjar, Rohtak, Meerut).
4. **WRF NWP Regional Meteorology Integration:** Decoupled WPS/WRF numerical weather prediction interface feeding future meteorological forcing ($T+1 \dots T+72$).
5. **Conformal Uncertainty Quantification:** Empirical 80% (P10–P90) and 95% (P05–P95) prediction intervals.
6. **5-Year Long-Range Policy Scenario Outlook:** Aggregated multi-year scenario simulation comparing BAU, Accelerated NCAP, Climate Stagnation, and Clean Transition.

---

## 2. Verified Benchmark Performance (Untouched 2023 Test Set)

Evaluated on the **untouched 2023 Test Set** ($3,393$ sequence samples, $244,296$ hourly forecast timesteps):

| Model / Baseline | Parameters | Overall MAE | Overall RMSE | $R^2$ Score | Severe MAE ($\ge 345 \text{ µg/m}^3$) | Severe Mean Bias | Leaderboard Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ATMOSAIR v2 Corrected Ensemble** | `819k + XGB` | **`56.66`** | **`81.24`** | **`0.3768`** | **`198.74`** | **`-198.29`** | **Rank #1 (CHAMPION)** |
| **Proposed Multi-Branch Model** | `819,874` | **`56.66`** | **`82.56`** | **`0.3564`** | `219.78` | `-219.66` | **Rank #2** |
| **TCN Champion Baseline** | `570,625` | `61.19` | `91.74` | `0.2053` | `223.01` | `-222.80` | **Rank #3** |
| **LSTM Baseline** | `223,873` | `74.48` | `102.18` | `0.0142` | `223.50` | `+4.98` | **Rank #4** |
| **GRU Baseline** | `167,937` | `75.10` | `111.16` | `-0.1667` | `247.68` | `-12.85` | **Rank #5** |
| **Naive Persistence** | N/A | `75.42` | `107.00` | `-0.0810` | `199.79` | `-199.40` | **Rank #6** |

### Horizon-by-Horizon Comparison (Key Intervals):
* **+1h:** Persistence wins (`21.02 µg/m³`) due to strong short-term autocorrelation.
* **+6h:** Corrected Ensemble wins (**`53.12 µg/m³`** vs TCN `60.55`).
* **+12h:** Corrected Ensemble wins (**`55.20 µg/m³`** vs TCN `61.51`).
* **+24h:** Corrected Ensemble wins (**`54.40 µg/m³`** vs TCN `60.82`).
* **+48h:** Corrected Ensemble wins (**`55.90 µg/m³`** vs TCN `62.53`).
* **+72h:** Corrected Ensemble wins (**`55.85 µg/m³`** vs TCN `61.24`).

---

## 3. Strict Leakage Prevention & Audit
* **Issue Time Causality ($T$):** Only observations with $t_{obs} \le T$ are consumed as historical features.
* **Future Meteorology ($T+1 \dots T+72$):** Strictly derived from NWP / WRF forecast models, never from future ERA5 reanalysis or future ground station observations.
* **Automated Audit:** Verified through `pytest tests/test_leakage.py` (3/3 passing).

---

## 4. Multi-Source Environmental Data Ingestion
1. **CPCB / DPCC CAAQMS:** Ground criteria pollutant monitoring across 12 key Delhi-NCR stations.
2. **NASA FIRMS:** MODIS & VIIRS active fire detections and Fire Radiative Power (FRP) in 25km, 50km, and 100km radii.
3. **Copernicus CAMS:** Atmosphere Data Store background composition (PM2.5, PM10, AOD 550nm).
4. **Copernicus Sentinel-5P:** TROPOMI Level-2 tropospheric NO2, CO, and UV Aerosol Index columns.
5. **Copernicus CDS (ERA5):** Historical reanalysis used strictly for offline training and WRF boundary conditions.
6. **WRF Regional Meteorology:** Regional NWP forecast forcing with boundary layer height (PBLH) dynamics.

---

## 5. Quickstart & Verification

### A. Run Complete Automated Test Suite (36 Tests)
```bash
pytest -q tests/
```

### B. Launch FastAPI Backend & Interactive Dashboard
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
Open your browser to:
* **Interactive Dashboard:** `http://localhost:8000/dashboard/`
* **Swagger API Docs:** `http://localhost:8000/docs`

---

## 6. Key Documentation Links
* [System Architecture](ARCHITECTURE.md)
* [REST API Integration Guide](API_INTEGRATION.md)
* [WRF / WPS Regional Setup Guide](WRF_SETUP.md)
* [Official Model Card](MODEL_CARD.md)
* [Feature Data Dictionary](DATA_DICTIONARY.md)
* [Validation & Benchmark Report](VALIDATION_REPORT.md)
* [Production Deployment](DEPLOYMENT.md)
* [Security Policy](SECURITY.md)

---

## 7. Repository Structure

```
SIH2026_PersonB/
├── api/                         # FastAPI Production Server
│   └── app.py
├── configs/                     # Feature Groups & Training Configurations
│   ├── feature_groups.json
│   └── training_config.json
├── dashboard/                   # Interactive ATMOSAIR Demo Web Frontend
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── data/                        # Processed Data Arrays & Scaled Data
├── demo/                        # Sample Input Data
│   └── sample_input.json
├── docs/                        # Project Documentation & Reports
│   ├── FINAL_PROJECT_REPORT.md
│   ├── PRESENTATION_CONTENT.md
│   ├── ARCHITECTURE_DIAGRAM.md
│   └── FINAL_HANDOFF_CHECKLIST.md
├── models/                      # PyTorch Architectures & Checkpoints
│   ├── checkpoints/
│   │   └── proposed_best.pt     # Verified Best Model Weights (Epoch 1)
│   ├── scalers/                 # Preprocessing Scalers (Train-only)
│   └── proposed_model.py
├── pipeline/                    # Production Inference Pipeline
│   └── inference_pipeline.py
├── results/                     # Evaluation Outputs, Metrics & Plots
│   ├── plots/                   # 31 Generated Evaluation Visualizations
│   └── proposed_final_evaluation.json
├── tests/                       # Test Suites
│   ├── test_proposed_training_integrity.py
│   ├── test_final_evaluation.py
│   └── test_production_pipeline.py
└── README.md
```

---

## 8. Reproducibility Guarantee
All training, validation, and evaluation pipelines enforce random seed **`42`** (`torch.manual_seed(42)`, `np.random.seed(42)`). Preprocessing scalers are strictly fitted on training data to ensure zero data leakage.
