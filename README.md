# ATMOSAIR — 72-Hour PM2.5 Air Quality Forecasting System
> **Smart India Hackathon (SIH) 2026 — Person B Project**  
> *Deep Learning Multi-Branch Temporal Architecture for Long-Horizon Environmental Forecasting*

---

## 1. Project Overview & Problem Statement
Air pollution (specifically fine particulate matter $\text{PM}_{2.5}$) poses severe health risks across urban regions in India. Standard single-step autoregressive or shallow forecasting models degrade rapidly beyond 6–12 hours. 

This project delivers **ATMOSAIR: A 72-Hour Multi-Step Direct PM2.5 Air Quality Forecasting System** powered by a custom **Coupled Multi-Branch Neural Network (`CoupledMultiBranchForecastModel`, 819,874 parameters)**. The system ingests 72 consecutive hours of 49 multi-modal environmental variables (pollution, meteorology, satellite atmospheric data, and temporal cyclic signals) to predict the complete 72-hour future trajectory of $\text{PM}_{2.5}$ concentrations in physical units ($\mu g/m^3$).

---

## 2. Key Achievements & Verified Leaderboard

Evaluated on the **untouched 2023 Test Set** ($3,393$ sequence samples, $244,296$ forecast timesteps):

| Model / Baseline | Parameters | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) | WMAPE (%) | $R^2$ Score | Leaderboard Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Multi-Branch Model** | **`819,874`** | **`56.66`** | **`82.56`** | **`41.11%`** | **`0.3564`** | **Rank #1 (NEW CHAMPION)** |
| **TCN Champion Baseline** | `570,625` | `61.19` | `91.74` | `44.40%` | `0.2053` | **Rank #2** |
| **LSTM Baseline** | `223,873` | `74.48` | `102.18` | `54.05%` | `0.0142` | **Rank #3** |
| **GRU Baseline** | `167,937` | `75.10` | `111.16` | `54.49%` | `-0.1667` | **Rank #4** |
| **Naive Persistence** | N/A | `75.42` | `107.00` | `54.73%` | `-0.0810` | **Rank #5** |

### Verified Improvements over TCN Champion:
* **MAE Reduction:** **`-4.53 µg/m³`** (**`+7.40%` relative error reduction**)
* **RMSE Reduction:** **`-9.18 µg/m³`** (**`+10.01%` relative error reduction**)
* **$R^2$ Score Gain:** **`+0.1511` absolute gain** ($0.2053 \to 0.3564$, **`+73.6%` relative increase**)
* **Statistical Significance:** Paired Student's t-test $p < 10^{-15}$, Wilcoxon signed-rank test $p = 3.72 \times 10^{-16} < 0.001$.

---

## 3. Dataset & 49 Input Feature Groups

The dataset spans multi-station continuous hourly environmental monitoring:
* **Train Split:** 19,005 sequences ($1,368,360$ sequence timesteps)
* **Validation Split:** 4,493 sequences ($323,496$ sequence timesteps)
* **Test Split (2023):** 3,393 sequences ($244,296$ sequence timesteps)

### 49 Input Features Breakdown:
1. **Pollution (9):** `pm25`, `pm10`, `no`, `no2`, `nox`, `nh3`, `so2`, `co`, `o3`
2. **Meteorology (21):** `temperature_c`, `relative_humidity`, `wind_speed`, `wind_direction`, `rainfall`, `solar_radiation`, `pressure_mmhg`, `wind_dir_sin`, `wind_dir_cos`, `wind_u_local`, `wind_v_local`, `gee_wind_speed_change_1d`, `temp_change_24h`, `pressure_change_24h`, `gee_precipitation_m`, `gee_surface_pressure`, `gee_temperature_2m`, `gee_u_wind_10m`, `gee_v_wind_10m`, `gee_wind_speed_10m`, `gee_relative_humidity`
3. **Atmospheric External (9):** `aot_470`, `aot_550`, `angstrom_exp`, `gee_no2_slant`, `gee_so2_slant`, `gee_co_slant`, `gee_o3_slant`, `gee_hcho_slant`, `gee_ch4_slant`
4. **Temporal Cycles (6):** `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos`
5. **Data Availability Indicators (4):** `avail_station_pollution`, `avail_station_met`, `avail_aeronet_aod`, `avail_satellite_gas`

---

## 4. Model Architecture & Training Methodology

```
                   Input Historical Sequence Window [B, 72, 49]
                                        │
    ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
    ▼                   ▼                               ▼                   ▼
Pollution Branch   Meteorology Branch       Atmospheric External Branch  Temporal Branch
  [B, 72, 64]         [B, 72, 64]                  [B, 72, 64]             [B, 72, 32]
    │                   │                               │                   │
    └───────────────────┴───────────────┬───────────────┴───────────────────┘
                                        ▼
                         Concatenation [B, 72, 224]
                                        │
                         Gated Linear Unit (GLU, 256)
                                        │
                 4-Head Causal Multi-Head Self-Attention (256)
                                        │
               ┌────────────────────────┴────────────────────────┐
               ▼                                                 ▼
     Primary Forecast Head                            Auxiliary Spike Head
      PM2.5 Regression [B, 72, 1]                    Spike Probabilities [B, 72, 1]
```

* **Loss Function (`SpikeAwareForecastLoss`):** Combined base MSE + $3.0\times$ spike-weighted MSE ($\tau = 345.0 \mu g/m^3$) + $0.2\times$ auxiliary BCE classification loss.
* **Optimization:** AdamW ($\text{lr}=10^{-3}$, weight decay $=10^{-4}$), `ReduceLROnPlateau`, early stopping (patience = 7).

---

## 5. Multi-Step Horizon Performance

| Horizon | Persistence MAE | TCN MAE | Proposed MAE | Horizon Winner |
| :---: | :---: | :---: | :---: | :---: |
| **+1h** | **`21.02`** | `54.50` | `52.16` | **Persistence** |
| **+6h** | `65.20` | `60.55` | **`54.93`** | **Proposed Model** |
| **+12h** | `83.82` | `61.51` | **`56.85`** | **Proposed Model** |
| **+24h** | `56.43` | `60.82` | **`55.77`** | **Proposed Model** |
| **+48h** | `64.19` | `62.53` | **`57.38`** | **Proposed Model** |
| **+72h** | `67.12` | `61.24` | **`57.22`** | **Proposed Model** |

> **Honest Limitation Callout:** Naive Persistence wins at $+1$h due to immediate temporal autocorrelation. From $+6$h to $+72$h, the Proposed Model is the undisputed winner.

---

## 6. How to Run the System

### A. Run Automated Production Test Suite
```bash
python tests/test_production_pipeline.py
```

### B. Start FastAPI Backend API
```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

### C. Access Endpoints & Interactive Dashboard
* **Swagger API Docs:** `http://127.0.0.1:8000/docs`
* **Health Check:** `http://127.0.0.1:8000/health`
* **Model Info:** `http://127.0.0.1:8000/model-info`
* **Verified Metrics:** `http://127.0.0.1:8000/metrics`
* **Demo Prediction:** `http://127.0.0.1:8000/demo-predict?sample_id=1493`
* **Frontend Dashboard:** Open `dashboard/index.html` in any web browser.

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
