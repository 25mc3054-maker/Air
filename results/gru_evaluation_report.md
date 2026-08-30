# B6 — GRU BASELINE EVALUATION REPORT

## Overview

Evaluation results for the direct sequence-to-sequence GRU baseline model on the validation and testing sets. All metrics are calculated in **physical PM2.5 units (µg/m³)** after inverse target transformation.

## Overall Performance Summary

| Split | Sample Count | Overall MAE (µg/m³) | Overall RMSE (µg/m³) | Overall $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Set** | 4,493 | `70.38` | `101.46` | `0.1825` |
| **Test Set** | 3,393 | **`75.10`** | **`111.16`** | **`-0.1667`** |

## Forecast Horizon Breakdown (Test Set)

| Forecast Horizon | MAE (µg/m³) | RMSE (µg/m³) |
| :---: | :---: | :---: |
| **+1 Hour (1h)** | `60.28` | `77.06` |
| **+6 Hour (6h)** | `64.59` | `94.65` |
| **+12 Hour (12h)** | `70.96` | `104.92` |
| **+24 Hour (24h)** | `76.70` | `113.12` |
| **+48 Hour (48h)** | `78.51` | `116.40` |
| **+72 Hour (72h)** | `76.62` | `112.29` |

## Extreme Pollution Analysis (Test Set)

- **High-Pollution Threshold (Training 90th Percentile):** `345.00 µg/m³`
- **High-Pollution Test Evaluation Sample Count:** `12,662` timesteps
- **High-Pollution MAE:** `247.68 µg/m³`
- **High-Pollution RMSE:** `281.03 µg/m³`

## Generated Visualizations

- **Representative Forecast Plot:** [`results/plots/gru_actual_vs_predicted.png`](file:///d:/My Projects/SIH2026_PersonB/results/plots/gru_actual_vs_predicted.png)
- **Horizon Error Curve Plot:** [`results/plots/gru_horizon_error.png`](file:///d:/My Projects/SIH2026_PersonB/results/plots/gru_horizon_error.png)

## Baseline Architecture Notice

> **Note:** The GRU model serves as the initial sequence-to-sequence **baseline model**. It establishes the benchmark against which subsequent architectures (LSTM, TCN, and the proposed coupled multi-branch attention model) will be evaluated.

---

```
B6 STATUS: COMPLETED & VERIFIED
```

```
READY FOR B7 — LSTM BASELINE
```
