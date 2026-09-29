# ATMOSAIR — FINAL PROJECT REPORT
> **Project:** ATMOSAIR Deep Learning Air Quality Forecasting (72-Hour PM2.5 Trajectory Prediction)  
> **Model Architecture:** Coupled Multi-Branch Forecast Model (`CoupledMultiBranchForecastModel`)  
> **Parameter Budget:** 819,874 Parameters  
> **Evaluated Test Set:** Untouched 2023 Test Set (3,393 sequences, 244,296 timesteps)  

---

## 1. Abstract
Long-horizon air quality forecasting plays a vital role in public health advisories, urban environmental management, and emergency pollution mitigation. This project presents **ATMOSAIR: A 72-Hour Multi-Step Direct PM2.5 Air Quality Forecasting System** developed for the Smart India Hackathon (SIH) 2026. The proposed **Coupled Multi-Branch Neural Network** ingests 72 historical hours of 49 multi-modal environmental variables across 4 domain-specific causal convolutional streams, fused via a Gated Linear Unit (GLU) and refined by 4-Head Causal Self-Attention. Evaluated on an untouched test set of 3,393 sequences (244,296 hourly predictions), the proposed model achieved an **MAE of 56.66 µg/m³**, **RMSE of 82.56 µg/m³**, **$R^2$ of 0.3564**, and **WMAPE of 41.11%**, establishing a new project leaderboard record and significantly outperforming the TCN champion baseline ($p < 0.001$).

---

## 2. Problem Statement
Fine particulate matter ($\text{PM}_{2.5}$) is a primary air pollutant responsible for adverse cardiovascular and respiratory health outcomes in major Indian urban centers. Traditional single-step autoregressive models or numerical weather prediction methods suffer from error accumulation over extended horizons. The goal of this work is to build, train, and validate a direct multi-step deep learning model capable of predicting the exact 72-hour future PM2.5 trajectory ($t+1$ to $t+72$ hours) in physical concentration units ($\mu g/m^3$).

---

## 3. Motivation
Standard persistence or baseline RNN models exhibit severe degradation when forecasting beyond 6–12 hours. Furthermore, extreme pollution episodes ($\text{PM}_{2.5} \ge 345.0 \mu g/m^3$) are frequently underestimated due to variance smoothing inherent in standard mean squared error (MSE) loss minimization. A multi-branch temporal architecture coupled with spike-aware loss functions is required to capture long-term temporal dependencies while preserving extreme peak sensitivity.

---

## 4. Existing Approaches & Baselines
Four baseline models were implemented, trained, and frozen under identical dataset splits and evaluation protocols:
1. **Naive Persistence:** Assumes $y_{t+h} = y_t$ (Last observed PM2.5 value repeated for 72 hours).
2. **GRU Baseline (B6):** 2-Layer Gated Recurrent Unit (167,937 parameters).
3. **LSTM Baseline (B7):** 2-Layer Long Short-Term Memory Network (223,873 parameters).
4. **TCN Champion Baseline (B8):** Causal Dilated Temporal Convolutional Network (570,625 parameters).

---

## 5. Proposed Approach
The proposed **Coupled Multi-Branch Forecast Model** introduces:
- **Domain-Separated Convolutional Branches:** Parallel processing of Pollution, Meteorology, Atmospheric Satellite, and Temporal Cycle inputs.
- **Gated Linear Unit (GLU) Fusion:** Non-linear feature gating across branch channels (224 $\to$ 256 channels).
- **Causal Multi-Head Self-Attention:** 4-head attention mechanism capturing multi-day cross-timestep dependencies without temporal leakage.
- **Dual Output Heads:** Primary 72-hour forecast regression head + auxiliary binary spike classification head.
- **Spike-Aware Loss:** Asymmetric MSE weighting favoring high pollution spikes ($\ge 345.0 \mu g/m^3$).

---

## 6. Dataset Description
The dataset combines continuous hourly air quality monitoring, meteorological measurements, AERONET optical depth data, and satellite atmospheric observations:
- **Train Set:** 19,005 sequence samples
- **Validation Set:** 4,493 sequence samples
- **Test Set (2023):** 3,393 sequence samples

---

## 7. Data Preprocessing
- Missing value imputation using `SimpleImputer` (median strategy).
- Feature scaling using `StandardScaler` fitted **strictly on the training set**.
- Leakage-safe target scaling saved to `models/scalers/target_scaler.joblib`.

---

## 8. Feature Engineering
49 continuous features organized into 5 groups:
1. **Pollution (9):** `pm25`, `pm10`, `no`, `no2`, `nox`, `nh3`, `so2`, `co`, `o3`.
2. **Meteorology (21):** Temperature, relative humidity, wind speed/direction, local wind vectors ($u, v$), pressure, 24h delta features, and GEE weather variables.
3. **Atmospheric External (9):** Aerosol Optical Thickness (`aot_470`, `aot_550`), Angstrom exponent, and satellite trace gas slant columns ($\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$, $\text{HCHO}$, $\text{CH}_4$).
4. **Temporal Cycles (6):** Cyclical sine/cosine transformations for hour, day of week, and month.
5. **Data Availability Indicators (4):** Station, Aeronet, and satellite availability masks.

---

## 9. Sequence Construction
Sliding window temporal sequence generation:
- **Input Window:** $X \in \mathbb{R}^{B \times 72 \times 49}$ (72 historical hours).
- **Forecast Output:** $Y \in \mathbb{R}^{B \times 72 \times 1}$ (72 future hours).

---

## 10. Model Architecture Specification
- **Total Parameters:** `819,874` parameters (100% trainable).
- **Receptive Field:** 127 hours (causal dilated convolutions with dilations $[1, 2]$).
- **Attention Heads:** 4 heads, $d_{\text{model}} = 256$, dropout $= 0.1$.

---

## 11. Training Methodology
- **Optimizer:** AdamW ($\text{lr} = 0.001$, weight decay $= 10^{-4}$).
- **Scheduler:** `ReduceLROnPlateau` (factor $= 0.5$, patience $= 3$).
- **Early Stopping:** Triggered at Epoch 8 (patience $= 7$).
- **Best Epoch:** Epoch 1 saved to `models/checkpoints/proposed_best.pt` (Val Loss: `0.907336`).

---

## 12. Validation Protocol
Model selection was guided exclusively by Validation Set loss. The 2023 Test Set remained strictly isolated and untouched throughout training.

---

## 13. Testing Protocol
Evaluation was performed in physical PM2.5 units ($\mu g/m^3$) by inverse-transforming predictions using the training target scaler.

---

## 14. Empirical Results Summary

| Metric | Proposed Model Value | TCN Baseline Value | Improvement |
| :--- | :---: | :---: | :---: |
| **MAE** | **`56.66 µg/m³`** | `61.19 µg/m³` | **`-4.53 µg/m³` (+7.40%)** |
| **RMSE** | **`82.56 µg/m³`** | `91.74 µg/m³` | **`-9.18 µg/m³` (+10.01%)** |
| **$R^2$ Score** | **`0.3564`** | `0.2053` | **`+0.1511` (+73.6%)** |
| **WMAPE** | **`41.11%`** | `44.40%` | **`-3.29%` absolute** |

---

## 15. Baseline Comparison & Project Leaderboard

| Model | Parameters | MAE (µg/m³) | RMSE (µg/m³) | $R^2$ | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proposed Multi-Branch Model** | **`819,874`** | **`56.66`** | **`82.56`** | **`0.3564`** | **Rank #1** |
| **TCN Champion Baseline** | `570,625` | `61.19` | `91.74` | `0.2053` | **Rank #2** |
| **LSTM Baseline** | `223,873` | `74.48` | `102.18` | `0.0142` | **Rank #3** |
| **GRU Baseline** | `167,937` | `75.10` | `111.16` | `-0.1667` | **Rank #4** |
| **Naive Persistence** | N/A | `75.42` | `107.00` | `-0.0810` | **Rank #5** |

---

## 16. Multi-Horizon Analysis

The proposed model outperforms TCN at all multi-step horizons from $+6$h to $+72$h:
- **+1h:** Proposed `52.16` vs Persistence `21.02` (Persistence Wins)
- **+6h:** Proposed **`54.93`** vs TCN `60.55` (Proposed Wins)
- **+12h:** Proposed **`56.85`** vs TCN `61.51` (Proposed Wins)
- **+24h:** Proposed **`55.77`** vs TCN `60.82` (Proposed Wins)
- **+48h:** Proposed **`57.38`** vs TCN `62.53` (Proposed Wins)
- **+72h:** Proposed **`57.22`** vs TCN `61.24` (Proposed Wins)

---

## 17. Extreme Pollution Analysis ($\ge 345.0 \mu g/m^3$)
On 12,662 qualifying test timesteps, the Proposed Model achieved:
- **High-Pollution MAE:** **`219.78 µg/m³`** (vs TCN `223.01 µg/m³`)
- **High-Pollution RMSE:** **`235.45 µg/m³`** (vs TCN `247.76 µg/m³`)

---

## 18. Statistical Significance Verification
- **Paired Student's t-test:** $t = -56.72$, $p < 10^{-15}$
- **Wilcoxon Signed-Rank Test:** $p = 3.72 \times 10^{-16} < 0.001$

---

## 19. Error Analysis
- **Mean Residual (Bias):** `-14.98 µg/m³`
- **Median Residual:** `+1.71 µg/m³`
- **Residual Standard Deviation:** `81.19 µg/m³`

---

## 20. Failure Modes
1. **Severe Peak Underestimation:** For $\text{PM}_{2.5} \ge 600 \mu g/m^3$, mean predicted is `225.80 µg/m³` (Bias: `-429.05 µg/m³`).
2. **Prediction Variance Compression:** Variance ratio is `0.58`.

---

## 21. Limitations
- Short-term $+1$h predictions are weaker than Naive Persistence.
- Extreme spikes above $600 \mu g/m^3$ are smoothed due to standard target scaling.

---

## 22. Future Work
- Incorporate log-transformed target scaling ($\log(1 + y)$).
- Add direct $+1$h residual skip connections from last observed values.

---

## 23. Deployment Architecture
FastAPI backend (`api/app.py`) + Interactive JavaScript frontend (`dashboard/index.html`).

---

## 24. Conclusion
The Proposed Coupled Multi-Branch Model successfully achieves state-of-the-art 72-hour PM2.5 forecasting performance, delivering statistically significant error reductions over standard baselines.
