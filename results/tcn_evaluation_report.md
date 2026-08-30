# B8 — TCN BASELINE EVALUATION REPORT

## Overview

Evaluation results for the direct sequence-to-sequence Causal Dilated TCN baseline model on the validation and testing sets. All metrics are calculated in **physical PM2.5 units (µg/m³)** after inverse target transformation.

## Overall Performance Summary

| Split | Sample Count | Overall MAE (µg/m³) | Overall RMSE (µg/m³) | Overall $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Set** | 4,493 | `64.52` | `95.43` | `0.2769` |
| **Test Set** | 3,393 | **`61.19`** | **`91.74`** | **`0.2053`** |

## Forecast Horizon Breakdown (Test Set)

| Forecast Horizon | MAE (µg/m³) | RMSE (µg/m³) |
| :---: | :---: | :---: |
| **+1 Hour (1h)** | `54.50` | `73.19` |
| **+6 Hour (6h)** | `60.55` | `87.95` |
| **+12 Hour (12h)** | `61.51` | `91.38` |
| **+24 Hour (24h)** | `60.82` | `90.79` |
| **+48 Hour (48h)** | `62.53` | `94.59` |
| **+72 Hour (72h)** | `61.24` | `90.77` |

## Extreme Pollution Analysis (Test Set)

- **High-Pollution Threshold (Training 90th Percentile):** `345.00 µg/m³`
- **High-Pollution Test Evaluation Sample Count:** `12,662` timesteps
- **Actual High-Pollution Mean:** `417.94 µg/m³`
- **Predicted High-Pollution Mean:** `198.32 µg/m³`
- **High-Pollution MAE:** `223.01 µg/m³`
- **High-Pollution RMSE:** `247.76 µg/m³`

## Bias & Variance Audit

- **Actual Mean:** `137.81 µg/m³` | **Predicted Mean:** `126.01 µg/m³` (Bias: `-11.80 µg/m³`)
- **Actual Median:** `105.67 µg/m³` | **Predicted Median:** `107.36 µg/m³`
- **Actual Std:** `102.91 µg/m³` | **Predicted Std:** `74.12 µg/m³`
- **Min / Max Actual:** `[3.00, 818.25] µg/m³` | **Min / Max Predicted:** `[23.05, 609.94] µg/m³`

## Generated Visualizations

- **Representative Forecast Plot:** [`results/plots/tcn_actual_vs_predicted.png`](file:///d:/My Projects/SIH2026_PersonB/results/plots/tcn_actual_vs_predicted.png)
- **Horizon Error Curve Plot:** [`results/plots/tcn_horizon_error.png`](file:///d:/My Projects/SIH2026_PersonB/results/plots/tcn_horizon_error.png)
