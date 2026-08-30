# B13 — PROPOSED MODEL FINAL TEST EVALUATION REPORT

## Executive Summary

Rigorous final evaluation of the **Coupled Multi-Branch Forecast Model** (`819,874` parameters) on the untouched 2023 Test Set (`3,393` sequence samples, `244,296` sequence timesteps). All metrics are presented in **physical PM2.5 units (µg/m³)**.

## Overall Performance Summary

| Metric / Property | Measured Value |
| :--- | :--- |
| **Model Class** | `CoupledMultiBranchForecastModel` |
| **Parameters** | `819,874` parameters |
| **Overall MAE** | **`56.66 µg/m³`** |
| **Overall RMSE** | **`82.56 µg/m³`** |
| **Overall WMAPE** | `41.11%` |
| **Overall $R^2$ Score** | **`0.3564`** |
| **Mean Bias** | `-14.98 µg/m³` |
| **Variance Ratio ($\sigma_p/\sigma_y$)** | `0.58` (`59.39` vs `102.91`) |
| **Predicted Range** | `[25.32, 490.19] µg/m³` |
| **Actual Range** | `[3.00, 818.25] µg/m³` |

## Forecast Horizon Breakdown (Key Horizons)

| Horizon | Proposed Model MAE | Proposed Model RMSE |
| :---: | :---: | :---: |
| **+1h** | `52.16 µg/m³` | `76.48 µg/m³` |
| **+6h** | `54.93 µg/m³` | `79.84 µg/m³` |
| **+12h** | `56.85 µg/m³` | `82.41 µg/m³` |
| **+24h** | `55.77 µg/m³` | `81.18 µg/m³` |
| **+48h** | `57.38 µg/m³` | `83.21 µg/m³` |
| **+72h** | `57.22 µg/m³` | `83.22 µg/m³` |

## Extreme Pollution Analysis ($\ge 345.00 	ext{ µg/m}^3$)

- **High-Pollution Threshold:** `345.00 µg/m³`
- **Qualifying Samples:** `12,662` timesteps (5.18% of test set)
- **High-Pollution MAE:** `219.78 µg/m³`
- **High-Pollution RMSE:** `235.45 µg/m³`
- **Spike Capture Ratio:** `0.47` (`198.27` vs `417.94`)

## Error by Pollution Regime

| Pollution Concentration Regime | Sample Count | Proposed MAE (µg/m³) | Proposed RMSE (µg/m³) | Mean Bias |
| :--- | :---: | :---: | :---: | :---: |
| **< 100 µg/m³** | `115,494` | `35.31` | `48.93` | `30.05` |
| **100–200 µg/m³** | `72,086` | `42.38` | `54.10` | `-9.52` |
| **200–345 µg/m³** | `44,054` | `89.13` | `103.93` | `-83.15` |
| **>= 345 µg/m³** | `12,662` | `219.78` | `235.45` | `-219.66` |