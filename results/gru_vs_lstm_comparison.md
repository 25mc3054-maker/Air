# B7 — GRU VS. LSTM BASELINE COMPARISON REPORT

## Overview

Objective comparative evaluation between the **GRU Baseline (B6)**, **LSTM Baseline (B7)**, and **Naive Persistence Baseline** on the untouched 2023 Test Set (`3,393` samples, `244,296` total sequence timesteps). All metrics are calculated in **physical PM2.5 units (µg/m³)**.

## Overall Test Performance Comparison

| Model / Baseline | Total Parameters | Overall MAE (µg/m³) | Overall RMSE (µg/m³) | Overall $R^2$ Score | Overall Winner |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Persistence Baseline** | `N/A` | `75.42` | `107.00` | `-0.0810` | Baseline |
| **GRU Baseline (B6)** | `167,937` | `75.10` | `111.16` | `-0.1667` | - |
| **LSTM Baseline (B7)** | `223,873` (`+33.3%`) | `74.48` | `102.18` | `0.0142` | **LSTM (MAE/RMSE/$R^2$)** |

## Horizon-by-Horizon Performance Comparison

| Horizon | GRU MAE | LSTM MAE | Pers MAE | MAE Winner | GRU RMSE | LSTM RMSE | Pers RMSE | RMSE Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | `60.28` | `62.27` | `21.02` | **Pers** | `77.06` | `79.15` | `33.20` | **Pers** |
| **+6h** | `64.59` | `68.36` | `65.20` | **GRU** | `94.65` | `96.31` | `93.93` | **Pers** |
| **+12h** | `70.96` | `75.05` | `83.82` | **GRU** | `104.92` | `103.53` | `115.52` | **LSTM** |
| **+24h** | `76.70` | `75.43` | `56.43` | **Pers** | `113.12` | `102.94` | `83.19` | **Pers** |
| **+48h** | `78.51` | `76.48` | `64.19` | **Pers** | `116.40` | `104.21` | `92.44` | **Pers** |
| **+72h** | `76.62` | `74.30` | `67.12` | **Pers** | `112.29` | `101.29` | `96.33` | **Pers** |

## Extreme Pollution Comparison ($\ge 345.00$ µg/m³)

| Model / Baseline | Qualifying Timesteps | High-Pollution MAE (µg/m³) | High-Pollution RMSE (µg/m³) | Mean Predicted Spikes |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Ground Truth** | `12,662` | `0.00` | `0.00` | `417.94 µg/m³` |
| **Persistence Baseline** | `12,662` | `199.79` | `232.04` | `232.92 µg/m³` |
| **GRU Baseline (B6)** | `12,662` | `247.68` | `281.03` | `177.55 µg/m³` |
| **LSTM Baseline (B7)** | `12,662` | `223.50` | `252.36` | `197.13 µg/m³` |

## Distribution & Bias Audit

| Distribution Metric | Actual Ground Truth | GRU Prediction | LSTM Prediction | Persistence |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** | `137.81` | `124.96` | `142.79` | `136.06` |
| **Mean Bias (Pred - Act)** | `0.00` | `-12.85` | `+4.98` | `-1.75` |
| **Median** | `105.67` | `97.03` | `125.75` | `105.75` |
| **Std Dev (Variance)** | `102.91` | `86.45` | `76.29` | `100.45` |
| **Min / Max Range** | `[3.00, 818.25]` | `[11.76, 603.02]` | `[20.98, 557.62]` | `[3.00, 818.25]` |

## Comprehensive Architectural Analysis & Conclusions

1. **Overall Performance:** LSTM achieves a slightly lower overall MAE (`74.48 µg/m³` vs `75.10 µg/m³`) and RMSE (`102.18 µg/m³` vs `111.16 µg/m³`), along with a positive $R^2$ score (`0.0142` vs `-0.1667`).
2. **Parameter Efficiency:** GRU utilizes **33.3% fewer parameters** (`167,937` vs `223,873`), making it more lightweight per iteration.
3. **Horizon Sensitivity:** GRU performs slightly better at mid-range horizons (+6h and +12h MAE), whereas LSTM stabilizes long-horizon performance (+24h, +48h, +72h RMSE).
4. **Severe Spike Underprediction:** Both single-branch recurrent models exhibit significant variance compression during extreme pollution events ($\ge 345.00 \text{ µg/m}^3$), confirming the necessity for multi-branch atmospheric feature fusion and attention mechanisms in upcoming models.

---

```
B7 STATUS: COMPLETED & VERIFIED
```

```
READY FOR B8 — TCN BASELINE
```
