# B4 — LEAKAGE-SAFE SCALING REPORT

## Overview

Leakage-safe feature scaling and target normalization were successfully implemented for the 49 input features and primary PM2.5 target.

## Scaling Configuration

- **Scaling Method:** `StandardScaler` (z-score normalization)
- **Feature Scaler Type:** `sklearn.preprocessing.StandardScaler` (Fit strictly on `X_train`)
- **Imputation Strategy:** `sklearn.impute.SimpleImputer(strategy='mean')` (Fit strictly on `X_train`)
- **Target Scaler Type:** `sklearn.preprocessing.StandardScaler` (Fit strictly on `y_train`)
- **Training Observations Used for Fitting:** `1,368,360` timesteps ($19,005 \text{ windows} \times 72 \text{ hours}$)
- **Target Feature:** `pm25` (Raw Mean: `155.77 µg/m³`, Raw Std: `130.28 µg/m³`)

## Pre-Scaling Non-Finite Value Audit

| Array Name | Shape | NaN Count | +Inf Count | -Inf Count | Missing % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `X_train` | `(19005, 72, 49)` | 2,855,437 | 0 | 0 | 4.26% |
| `y_train` | `(19005, 72, 1)` | 0 | 0 | 0 | 0.00% |
| `X_val` | `(4493, 72, 49)` | 658,729 | 0 | 0 | 4.16% |
| `y_val` | `(4493, 72, 1)` | 0 | 0 | 0 | 0.00% |
| `X_test` | `(3393, 72, 49)` | 486,897 | 0 | 0 | 4.07% |
| `y_test` | `(3393, 72, 1)` | 0 | 0 | 0 | 0.00% |

> **Imputation Rationale:** Raw input features contain sparse missing values (e.g., station rainfall 99.22%, CAMS reanalysis 10.12%, NO 11.34%). SimpleImputer was fitted strictly on training data to replace missing entries with training-set feature means. Post-imputation and scaling, 0 NaNs remain across all training, validation, and testing tensors.

## Scaled Array Specifications

| Partition | Unscaled Array Path | Scaled Array Path | Scaled Shape | Data Type | Memory (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Train Features** | `data/processed/X_train.npy` | `data/processed/scaled/X_train_scaled.npy` | `(19005, 72, 49)` | `float32` | 255.77 MB |
| **Train Target** | `data/processed/y_train.npy` | `data/processed/scaled/y_train_scaled.npy` | `(19005, 72, 1)` | `float32` | 5.22 MB |
| **Val Features** | `data/processed/X_val.npy` | `data/processed/scaled/X_val_scaled.npy` | `(4493, 72, 49)` | `float32` | 60.47 MB |
| **Val Target** | `data/processed/y_val.npy` | `data/processed/scaled/y_val_scaled.npy` | `(4493, 72, 1)` | `float32` | 1.23 MB |
| **Test Features** | `data/processed/X_test.npy` | `data/processed/scaled/X_test_scaled.npy` | `(3393, 72, 49)` | `float32` | 45.66 MB |
| **Test Target** | `data/processed/y_test.npy` | `data/processed/scaled/y_test_scaled.npy` | `(3393, 72, 1)` | `float32` | 0.93 MB |

## Inverse Transformation & Reconstruction Audit

- **Test Formula:** $y_{\text{reconstructed}} = \text{target\_scaler.inverse\_transform}(y_{\text{scaled}})$

- **Max Absolute Error:** `0.0000610352`
- **Root Mean Squared Error (RMSE):** `0.0000035914`
- **Reconstruction Verdict:** `PASS` (Negligible numerical error $< 10^{-4}$)

## Data Leakage Audit

- `feature_scaler.n_samples_seen_`: `1,368,360` (Exact match for $19005 \times 72$)
- `target_scaler.n_samples_seen_`: `1,368,360` (Exact match for $19005 \times 72$)
- **Validation & Test Isolation:** Verified that neither `fit` nor `fit_transform` was ever called on `X_val`, `y_val`, `X_test`, or `y_test`.
```
SCALING LEAKAGE AUDIT: PASS
```

## Programmatic Scaling Verification Results

| # | Verification Check | Status | Details |
| :---: | :--- | :---: | :--- |
| 1 | Scaled array shapes match unscaled shapes exactly | **PASS** | Verified programmatically |
| 2 | Zero NaNs remain in scaled feature & target arrays | **PASS** | Verified programmatically |
| 3 | Target inverse transformation reconstruction error < 1e-4 | **PASS** | Verified programmatically |
| 4 | Scalers fitted strictly on training data (n_samples_seen_ == N_train * T) | **PASS** | Verified programmatically |
| 5 | Training feature means approximately equal 0 | **PASS** | Verified programmatically |

---

## Final Status

```
B4 STATUS: COMPLETED & VERIFIED
```

```
READY FOR B5 — PYTORCH DATASET AND DATALOADER
```
