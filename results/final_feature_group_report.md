# B2 — FINAL FEATURE GROUPING REPORT

**Total Input Features:** 49

## Summary of Feature Groups

| Feature Group | Count | Features |
| :--- | :---: | :--- |
| `pollution` | 9 | `pm25`, `pm10`, `no`, `no2`, `nox`, `nh3`, `so2`, `co`, `o3` |
| `meteorology` | 21 | `temperature_c`, `relative_humidity`, `wind_speed`, `wind_direction`, `rainfall`, `solar_radiation`, `pressure_mmhg`, `wind_dir_sin`, `wind_dir_cos`, `wind_u_local`, `wind_v_local`, `gee_wind_speed_change_1d`, `temp_change_24h`, `pressure_change_24h`, `gee_precipitation_m`, `gee_surface_pressure_pa`, `gee_temp_kelvin`, `gee_u_wind`, `gee_v_wind`, `gee_temp_celsius`, `gee_wind_speed` |
| `atmospheric_external` | 9 | `cams_pm25_ugm3`, `cams_pm10_ugm3`, `cams_aod550`, `fire_count_25km`, `fire_frp_25km`, `fire_count_50km`, `fire_frp_50km`, `fire_count_100km`, `fire_frp_100km` |
| `temporal` | 8 | `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`, `dayofyear_sin`, `dayofyear_cos`, `month_sin`, `month_cos` |
| `other` | 2 | `pm25_available`, `pollutant_count_available` |

---

## Detailed Feature Inventory

| # | Feature Name | Group | Data Type | Missing Count | Missing % |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `pm25` | `pollution` | `float64` | 0 | 0.00% |
| 2 | `pm10` | `pollution` | `float64` | 3705 | 5.70% |
| 3 | `no` | `pollution` | `float64` | 6765 | 10.41% |
| 4 | `no2` | `pollution` | `float64` | 3796 | 5.84% |
| 5 | `nox` | `pollution` | `float64` | 7078 | 10.89% |
| 6 | `nh3` | `pollution` | `float64` | 4873 | 7.50% |
| 7 | `so2` | `pollution` | `float64` | 4537 | 6.98% |
| 8 | `co` | `pollution` | `float64` | 6559 | 10.09% |
| 9 | `o3` | `pollution` | `float64` | 3320 | 5.11% |
| 10 | `temperature_c` | `meteorology` | `float64` | 1588 | 2.44% |
| 11 | `relative_humidity` | `meteorology` | `float64` | 240 | 0.37% |
| 12 | `wind_speed` | `meteorology` | `float64` | 884 | 1.36% |
| 13 | `wind_direction` | `meteorology` | `float64` | 1090 | 1.68% |
| 14 | `rainfall` | `meteorology` | `float64` | 64661 | 99.46% |
| 15 | `solar_radiation` | `meteorology` | `float64` | 1406 | 2.16% |
| 16 | `pressure_mmhg` | `meteorology` | `float64` | 4063 | 6.25% |
| 17 | `cams_pm25_ugm3` | `atmospheric_external` | `float64` | 7436 | 11.44% |
| 18 | `cams_pm10_ugm3` | `atmospheric_external` | `float64` | 7436 | 11.44% |
| 19 | `cams_aod550` | `atmospheric_external` | `float64` | 7436 | 11.44% |
| 20 | `fire_count_25km` | `atmospheric_external` | `float64` | 0 | 0.00% |
| 21 | `fire_frp_25km` | `atmospheric_external` | `float64` | 0 | 0.00% |
| 22 | `fire_count_50km` | `atmospheric_external` | `float64` | 0 | 0.00% |
| 23 | `fire_frp_50km` | `atmospheric_external` | `float64` | 0 | 0.00% |
| 24 | `fire_count_100km` | `atmospheric_external` | `float64` | 0 | 0.00% |
| 25 | `fire_frp_100km` | `atmospheric_external` | `float64` | 0 | 0.00% |
| 26 | `hour_sin` | `temporal` | `float64` | 0 | 0.00% |
| 27 | `hour_cos` | `temporal` | `float64` | 0 | 0.00% |
| 28 | `dayofweek_sin` | `temporal` | `float64` | 0 | 0.00% |
| 29 | `dayofweek_cos` | `temporal` | `float64` | 0 | 0.00% |
| 30 | `dayofyear_sin` | `temporal` | `float64` | 0 | 0.00% |
| 31 | `dayofyear_cos` | `temporal` | `float64` | 0 | 0.00% |
| 32 | `month_sin` | `temporal` | `float64` | 0 | 0.00% |
| 33 | `month_cos` | `temporal` | `float64` | 0 | 0.00% |
| 34 | `wind_dir_sin` | `meteorology` | `float64` | 1090 | 1.68% |
| 35 | `wind_dir_cos` | `meteorology` | `float64` | 1090 | 1.68% |
| 36 | `wind_u_local` | `meteorology` | `float64` | 1090 | 1.68% |
| 37 | `wind_v_local` | `meteorology` | `float64` | 1090 | 1.68% |
| 38 | `gee_wind_speed_change_1d` | `meteorology` | `float64` | 0 | 0.00% |
| 39 | `temp_change_24h` | `meteorology` | `float64` | 4856 | 7.47% |
| 40 | `pressure_change_24h` | `meteorology` | `float64` | 7149 | 11.00% |
| 41 | `pm25_available` | `other` | `int64` | 0 | 0.00% |
| 42 | `pollutant_count_available` | `other` | `int64` | 0 | 0.00% |
| 43 | `gee_precipitation_m` | `meteorology` | `float64` | 0 | 0.00% |
| 44 | `gee_surface_pressure_pa` | `meteorology` | `float64` | 0 | 0.00% |
| 45 | `gee_temp_kelvin` | `meteorology` | `float64` | 0 | 0.00% |
| 46 | `gee_u_wind` | `meteorology` | `float64` | 0 | 0.00% |
| 47 | `gee_v_wind` | `meteorology` | `float64` | 0 | 0.00% |
| 48 | `gee_temp_celsius` | `meteorology` | `float64` | 0 | 0.00% |
| 49 | `gee_wind_speed` | `meteorology` | `float64` | 0 | 0.00% |