# B7 — LSTM BASELINE EVALUATION REPORT

## Overview

Evaluation results for the direct sequence-to-sequence LSTM baseline model on the validation and testing sets. All metrics are calculated in **physical PM2.5 units (µg/m³)** after inverse target transformation.

## Overall Performance Summary

| Split | Sample Count | Overall MAE (µg/m³) | Overall RMSE (µg/m³) | Overall $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Set** | 4,493 | `66.98` | `97.82` | `0.2401` |
| **Test Set** | 3,393 | **`74.48`** | **`102.18`** | **`0.0142`** |

## Forecast Horizon Breakdown (Test Set)

| Forecast Horizon | MAE (µg/m³) | RMSE (µg/m³) |
| :---: | :---: | :---: |
| **+1 Hour (1h)** | `62.27` | `79.15` |
| **+6 Hour (6h)** | `68.36` | `96.31` |
| **+12 Hour (12h)** | `75.05` | `103.53` |
| **+24 Hour (24h)** | `75.43` | `102.94` |
| **+48 Hour (48h)** | `76.48` | `104.21` |
| **+72 Hour (72h)** | `74.30` | `101.29` |

## Extreme Pollution Analysis (Test Set)

- **High-Pollution Threshold (Training 90th Percentile):** `345.00 µg/m³`
- **High-Pollution Test Evaluation Sample Count:** `12,662` timesteps
- **Actual High-Pollution Mean:** `417.94 µg/m³`
- **Predicted High-Pollution Mean:** `197.13 µg/m³`
- **High-Pollution MAE:** `223.50 µg/m³`
- **High-Pollution RMSE:** `252.36 µg/m³`

## Bias & Variance Audit

- **Actual Mean:** `137.81 µg/m³` | **Predicted Mean:** `142.79 µg/m³` (Bias: `4.98 µg/m³`)
- **Actual Median:** `105.67 µg/m³` | **Predicted Median:** `125.75 µg/m³`
- **Actual Std:** `102.91 µg/m³` | **Predicted Std:** `76.29 µg/m³`
- **Min / Max Actual:** `[3.00, 818.25] µg/m³` | **Min / Max Predicted:** `[20.98, 557.62] µg/m³`

## Generated Visualizations

- **Representative Forecast Plot:** [`results/plots/lstm_actual_vs_predicted.png`](file:///d:/My Projects/SIH2026_PersonB/results/plots/lstm_actual_vs_predicted.png)
- **Horizon Error Curve Plot:** [`results/plots/lstm_horizon_error.png`](file:///d:/My Projects/SIH2026_PersonB/results/plots/lstm_horizon_error.png)
