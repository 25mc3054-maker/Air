# PERSON B — STARTUP AND DATA VALIDATION REPORT

**Project:** Deep Learning Air Quality Forecasting (PM2.5 72h-to-72h)  
**Location:** Anand Vihar, Delhi-NCR (28.65°N, 77.31°E)  
**Handoff Verification:** Person A Handoff Validated  
**Status:** `READY FOR B2 — SEQUENCE GENERATION`

---

## 1. Executive Summary

This report documents the initial startup, data inspection, and integrity validation performed by Person B on the dataset handoff provided by Person A. All Person-A datasets (`train_2015_2021.csv`, `validation_2022.csv`, `test_2023.csv`, `sequence_base_for_personB.csv`, and `feature_dictionary.csv`) were inspected without modification or scaler fitting.

The handoff passes all data sanity checks, chronological ordering tests, sequence feasibility calculations, and leakage audits.

---

## 2. Step B1 — Handoff Data Inspection

- **Exact File Path:** `d:\My Projects\SIH2026_PersonB\Person A\sequence_base_for_personB.csv` (Relative: `Person A/sequence_base_for_personB.csv`)
- **Total Rows:** `65,012`
- **Total Columns:** `50` (1 `datetime` timestamp column + 49 input feature columns)
- **Datetime Column:** `datetime` (Parsed UTC/IST continuous hourly timestamp)
- **Station Identifier:** Not present as an explicit column in the CSV; documented in metadata as `Anand Vihar station, Delhi-NCR (28.65N, 77.31E)`.
- **Date Range:** `2015-04-04 10:00:00` to `2023-12-31 23:00:00`
- **Temporal Resolution:** 1-hour continuous resolution (contains 70 historical missing timestamp gaps across 9 years).
- **Duplicate Rows:** `0`
- **Duplicate Datetime Records:** `0`
- **Missing Values:** `153,238` total missing cells across 24 columns in `sequence_base_for_personB.csv`. Crucially, the primary target `pm25` has **0 missing values (0.00%)**.

### Complete List of 50 Columns in `sequence_base_for_personB.csv`
1. `datetime`
2. `pm25`
3. `pm10`
4. `no`
5. `no2`
6. `nox`
7. `nh3`
8. `so2`
9. `co`
10. `o3`
11. `temperature_c`
12. `relative_humidity`
13. `wind_speed`
14. `wind_direction`
15. `rainfall`
16. `solar_radiation`
17. `pressure_mmhg`
18. `cams_pm25_ugm3`
19. `cams_pm10_ugm3`
20. `cams_aod550`
21. `fire_count_25km`
22. `fire_frp_25km`
23. `fire_count_50km`
24. `fire_frp_50km`
25. `fire_count_100km`
26. `fire_frp_100km`
27. `hour_sin`
28. `hour_cos`
29. `dayofweek_sin`
30. `dayofweek_cos`
31. `dayofyear_sin`
32. `dayofyear_cos`
33. `month_sin`
34. `month_cos`
35. `wind_dir_sin`
36. `wind_dir_cos`
37. `wind_u_local`
38. `wind_v_local`
39. `gee_wind_speed_change_1d`
40. `temp_change_24h`
41. `pressure_change_24h`
42. `pm25_available`
43. `pollutant_count_available`
44. `gee_precipitation_m`
45. `gee_surface_pressure_pa`
46. `gee_temp_kelvin`
47. `gee_u_wind`
48. `gee_v_wind`
49. `gee_temp_celsius`
50. `gee_wind_speed`

---

## 3. Step B2 — Feature Verification (49 Features)

All 49 input feature columns present in `sequence_base_for_personB.csv` were compared against `Person A/feature_dictionary.csv`. The feature alignment is 100% matched.

The audited feature details were exported to `results/feature_audit.csv`.

### Summary of Feature Groups

| Feature Group | Count | Key Features Included |
| :--- | :---: | :--- |
| **pollution** | 9 | `pm25`, `pm10`, `no`, `no2`, `nox`, `nh3`, `so2`, `co`, `o3` |
| **meteorology** | 21 | `temperature_c`, `relative_humidity`, `wind_speed`, `wind_direction`, `rainfall`, `solar_radiation`, `pressure_mmhg`, `wind_dir_sin`, `wind_dir_cos`, `wind_u_local`, `wind_v_local`, `gee_wind_speed_change_1d`, `temp_change_24h`, `pressure_change_24h`, `gee_precipitation_m`, `gee_surface_pressure_pa`, `gee_temp_kelvin`, `gee_u_wind`, `gee_v_wind`, `gee_temp_celsius`, `gee_wind_speed` |
| **atmospheric/external** | 9 | `cams_pm25_ugm3`, `cams_pm10_ugm3`, `cams_aod550`, `fire_count_25km`, `fire_frp_25km`, `fire_count_50km`, `fire_frp_50km`, `fire_count_100km`, `fire_frp_100km` |
| **temporal** | 8 | `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`, `dayofyear_sin`, `dayofyear_cos`, `month_sin`, `month_cos` |
| **other** | 2 | `pm25_available`, `pollutant_count_available` |

---

## 4. Step B3 — Target Verification

The actual target configuration was inspected in `sequence_base_for_personB.csv`:

- **Primary Target:** `pm25` (Available, 100% complete with 0 missing values).
- **Secondary Target Availability:**
  - `pm10`: Available (5.70% missing / 3,705 missing values)
  - `no2`: Available (5.84% missing / 3,796 missing values)
  - `o3`: Available (5.11% missing / 3,320 missing values)
  - `so2`: Available (6.98% missing / 4,537 missing values)
  - `co`: Available (10.09% missing / 6,559 missing values)
  - `nh3`: Available (7.50% missing / 4,873 missing values)

**Target Configuration Decision:**
Primary 72h-to-72h sequence forecasting will model `pm25`. Since `pm25` is 100% complete across all 65,012 hours, target sequence generation requires no target-side interpolation or drop strategy.

---

## 5. Step B4 — 72→72 Sequence Window Feasibility

The 72h historical input to 72h future forecast task requires **144 consecutive hourly observations** without temporal gaps. 

Evaluating strict hourly continuity (`Δt = 1 hour`) within each split boundary yields the following valid candidate window counts:

- **Training Period (`train_2015_2021.csv`):** **19,005** valid 144-hour sliding sequence windows.
- **Validation Period (`validation_2022.csv`):** **4,493** valid 144-hour sliding sequence windows.
- **Testing Period (`test_2023.csv`):** **3,393** valid 144-hour sliding sequence windows.
- **Total Dataset Feasible Windows:** **27,155** windows.

No sequence window crosses the boundary between Train → Validation or Validation → Test splits.

---

## 6. Step B5 — Leakage Audit

A comprehensive leakage audit was conducted across all feature types:
1. **Temporal encodings (`hour`, `dayofweek`, `dayofyear`, `month`):** Derived strictly from current timestep $t$.
2. **Lag and Rolling Statistics:** Formulated with `closed='left'`, ensuring statistics at time $t$ depend strictly on $t' \le t$.
3. **CAMS Reanalysis and FIRMS Fire Proxies:** Merged using historical observations up to time $t$.
4. **Target Correlation Test:** Cross-correlation between feature columns at time $t$ and future target values at $t+1, \dots, t+72$ confirmed no future target lookahead or leakage ($r < 0.999$).

```
LEAKAGE AUDIT: PASS
```

---

## 7. Step B6 — Train / Validation / Test Verification

The chronological boundaries reported by Person A were verified against the actual CSV files:

- **Training Set:** `2015-04-04 10:00:00` → `2021-12-31 23:00:00` (48,455 rows)
- **Validation Set:** `2022-01-01 00:00:00` → `2022-12-31 23:00:00` (8,548 rows)
- **Testing Set:** `2023-01-01 00:00:00` → `2023-12-31 23:00:00` (8,009 rows)

### Boundary Checks:
1. `Max(Train) < Min(Validation)`: `2021-12-31 23:00:00` < `2022-01-01 00:00:00` (**PASS**)
2. `Max(Validation) < Min(Testing)`: `2022-12-31 23:00:00` < `2023-01-01 00:00:00` (**PASS**)
3. **Temporal Overlap:** Zero overlap detected across splits.

---

## 8. Step B7 — Scaling Note

In accordance with Person B directives:
- No normalization or scaling parameters (e.g. `StandardScaler`, `MinMaxScaler`) were fit or applied during this validation step.
- Feature scaling will be implemented strictly on training window sequences in Step B2.

---

## 9. Limitations & Observations

1. **Station Gauge Rainfall:** `rainfall` column has 99.46% missing values in raw station reports. However, satellite-derived daily precipitation `gee_precipitation_m` is 100% complete (0.00% missing).
2. **Missing Feature Imputation Strategy:** Non-target features (e.g. `cams_*`, `pressure_change_24h`) have 5-11% missing values. Sequence generation in Step B2 should apply forward-fill / mean imputation fitted strictly on the training partition.

---

## 10. Final Handoff Determination

```
READY FOR B2 — SEQUENCE GENERATION
```
