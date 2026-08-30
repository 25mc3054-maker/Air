# B8 — THREE-WAY BASELINE COMPARISON REPORT (GRU vs. LSTM vs. TCN)

## Overview

Objective comparative evaluation between the **Naive Persistence Baseline**, **GRU Baseline (B6)**, **LSTM Baseline (B7)**, and **TCN Baseline (B8)** on the untouched 2023 Test Set (`3,393` samples, `244,296` total sequence timesteps). All metrics are calculated in **physical PM2.5 units (µg/m³)**.

## Overall Test Performance Comparison

| Model / Baseline | Total Parameters | Overall MAE (µg/m³) | Overall RMSE (µg/m³) | Overall $R^2$ Score | Overall Winner |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Persistence Baseline** | `N/A` | `75.42` | `107.00` | `-0.0810` | Baseline |
| **GRU Baseline (B6)** | `167,937` | `75.10` | `111.16` | `-0.1667` | - |
| **LSTM Baseline (B7)** | `223,873` | `74.48` | `102.18` | `0.0142` | - |
| **TCN Baseline (B8)** | `570,625` | `61.19` | `91.74` | `0.2053` | **TCN (Overall Winner)** |

## Horizon-by-Horizon Performance Comparison

| Horizon | GRU MAE | LSTM MAE | TCN MAE | Pers MAE | MAE Winner | GRU RMSE | LSTM RMSE | TCN RMSE | Pers RMSE | RMSE Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | `60.28` | `62.27` | `54.50` | `21.02` | **Pers** | `77.06` | `79.15` | `73.19` | `33.20` | **Pers** |
| **+6h** | `64.59` | `68.36` | `60.55` | `65.20` | **TCN** | `94.65` | `96.31` | `87.95` | `93.93` | **TCN** |
| **+12h** | `70.96` | `75.05` | `61.51` | `83.82` | **TCN** | `104.92` | `103.53` | `91.38` | `115.52` | **TCN** |
| **+24h** | `76.70` | `75.43` | `60.82` | `56.43` | **Pers** | `113.12` | `102.94` | `90.79` | `83.19` | **Pers** |
| **+48h** | `78.51` | `76.48` | `62.53` | `64.19` | **TCN** | `116.40` | `104.21` | `94.59` | `92.44` | **Pers** |
| **+72h** | `76.62` | `74.30` | `61.24` | `67.12` | **TCN** | `112.29` | `101.29` | `90.77` | `96.33` | **TCN** |

## Extreme Pollution Comparison ($\ge 345.00$ µg/m³)

| Model / Baseline | Qualifying Timesteps | High-Pollution MAE (µg/m³) | High-Pollution RMSE (µg/m³) | Mean Predicted Spikes |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Ground Truth** | `12,662` | `0.00` | `0.00` | `417.94 µg/m³` |
| **Persistence Baseline** | `12,662` | `199.79` | `232.04` | `232.92 µg/m³` |
| **GRU Baseline (B6)** | `12,662` | `247.68` | `281.03` | `177.55 µg/m³` |
| **LSTM Baseline (B7)** | `12,662` | `223.50` | `252.36` | `197.13 µg/m³` |
| **TCN Baseline (B8)** | `12,662` | `223.01` | `247.76` | `198.32 µg/m³` |

## Distribution & Bias Audit

| Distribution Metric | Actual Ground Truth | GRU Prediction | LSTM Prediction | TCN Prediction | Persistence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean** | `137.81` | `124.96` | `142.79` | `126.01` | `136.06` |
| **Mean Bias (Pred - Act)** | `0.00` | `-12.85` | `4.98` | `-11.80` | `-1.75` |
| **Median** | `105.67` | `97.03` | `125.75` | `107.36` | `105.75` |
| **Std Dev (Variance)** | `102.91` | `86.45` | `76.29` | `74.12` | `100.45` |
| **Min / Max Range** | `[3.00, 818.25]` | `[11.76, 603.02]` | `[20.98, 557.62]` | `[23.05, 609.94]` | `[3.00, 818.25]` |

## Comprehensive Architectural Analysis & Conclusions

1. **Causal TCN Convolution vs Recurrent Baselines:** Dilated causal 1D convolutions provide a large receptive field (127 hours) and parallel training capability.
2. **Parameter Capacity:** TCN utilizes `570,625` parameters (2.55x LSTM and 3.40x GRU capacity).
3. **Persistence Dynamics:** Short-term predictions (+1h) remain dominated by persistence due to high auto-correlation, whereas deep models provide valuable non-linear multi-step forecasts.
4. **Spike Underprediction Across All Baselines:** Standard MSE training across all single-stream baselines (GRU, LSTM, TCN) exhibits variance compression, strongly motivating the coupled multi-branch atmospheric attention model in upcoming Person-B phases.

---

```
B8 STATUS: COMPLETED & VERIFIED
```

```
READY FOR B9 — BASELINE COMPARISON
```
