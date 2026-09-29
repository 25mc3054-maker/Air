# ATMOSAIR v2 — Data Dictionary & Feature Schema
**Total Features:** 49 features across 4 domain branches + 2 availability flags.

---

## 1. Pollution Branch (9 Features)
| Feature Name | Description | Source | Native Units | Physical Range |
| :--- | :--- | :--- | :---: | :---: |
| `pm25` | Particulate Matter < 2.5 µm (Primary Target) | CPCB / DPCC | µg/m³ | 0.0 – 1500.0 |
| `pm10` | Particulate Matter < 10 µm | CPCB / DPCC | µg/m³ | 0.0 – 3000.0 |
| `no` | Nitric Oxide | CPCB / DPCC | µg/m³ | 0.0 – 500.0 |
| `no2` | Nitrogen Dioxide | CPCB / DPCC | µg/m³ | 0.0 – 500.0 |
| `nox` | Total Nitrogen Oxides | CPCB / DPCC | ppb | 0.0 – 800.0 |
| `nh3` | Ammonia | CPCB / DPCC | µg/m³ | 0.0 – 500.0 |
| `so2` | Sulfur Dioxide | CPCB / DPCC | µg/m³ | 0.0 – 400.0 |
| `co` | Carbon Monoxide | CPCB / DPCC | mg/m³ | 0.0 – 50.0 |
| `o3` | Surface Ozone | CPCB / DPCC | µg/m³ | 0.0 – 600.0 |

---

## 2. Meteorology Branch (21 Features)
| Feature Name | Description | Source | Native Units | Physical Range |
| :--- | :--- | :--- | :---: | :---: |
| `temperature_c` | Surface Air Temperature | CAAQMS / WRF | °C | -5.0 – 55.0 |
| `relative_humidity` | Relative Humidity | CAAQMS / WRF | % | 0.0 – 100.0 |
| `wind_speed` | Horizontal Wind Speed | CAAQMS / WRF | m/s | 0.0 – 40.0 |
| `wind_direction` | Meteorological Wind Direction | CAAQMS / WRF | degrees (0–360) | 0.0 – 360.0 |
| `rainfall` | Hourly Precipitation | CAAQMS / WRF | mm | 0.0 – 300.0 |
| `solar_radiation` | Downward Global Solar Flux | CAAQMS / WRF | W/m² | 0.0 – 1500.0 |
| `pressure_mmhg` | Atmospheric Surface Pressure | CAAQMS / WRF | mmHg | 650.0 – 850.0 |
| `wind_dir_sin` | $\sin(\text{wind\_direction})$ | Derived | dimensionless | -1.0 – 1.0 |
| `wind_dir_cos` | $\cos(\text{wind\_direction})$ | Derived | dimensionless | -1.0 – 1.0 |
| `wind_u_local` | Local U-wind (East-West Vector) | Derived | m/s | -40.0 – 40.0 |
| `wind_v_local` | Local V-wind (North-South Vector) | Derived | m/s | -40.0 – 40.0 |
| `gee_wind_speed_change_1d` | 24-Hour Wind Speed Trend | GEE / ERA5 | m/s | -20.0 – 20.0 |
| `temp_change_24h` | 24-Hour Temperature Difference | Derived | °C | -20.0 – 20.0 |
| `pressure_change_24h` | 24-Hour Pressure Difference | Derived | mmHg | -25.0 – 25.0 |
| `gee_precipitation_m` | Daily Precipitation | GEE / ERA5 | m | 0.0 – 0.5 |
| `gee_surface_pressure_pa` | Surface Pressure | GEE / ERA5 | Pa | 85000 – 105000 |
| `gee_temp_kelvin` | Ambient Temperature | GEE / ERA5 | K | 260.0 – 330.0 |
| `gee_u_wind` | Zonal Wind Speed | GEE / ERA5 | m/s | -30.0 – 30.0 |
| `gee_v_wind` | Meridional Wind Speed | GEE / ERA5 | m/s | -30.0 – 30.0 |
| `gee_temp_celsius` | Ambient Temperature | Derived | °C | -10.0 – 55.0 |
| `gee_wind_speed` | Scalar Wind Velocity | Derived | m/s | 0.0 – 40.0 |

---

## 3. Atmospheric External Branch (9 Features)
| Feature Name | Description | Source | Native Units | Physical Range |
| :--- | :--- | :--- | :---: | :---: |
| `cams_pm25_ugm3` | CAMS Reanalysis PM2.5 Context | Copernicus CAMS | µg/m³ | 0.0 – 1000.0 |
| `cams_pm10_ugm3` | CAMS Reanalysis PM10 Context | Copernicus CAMS | µg/m³ | 0.0 – 2000.0 |
| `cams_aod550` | Total Aerosol Optical Depth at 550nm | Copernicus CAMS | dimensionless | 0.0 – 5.0 |
| `fire_count_25km` | Active Fire Detections within 25 km | NASA FIRMS | count | 0 – 500 |
| `fire_frp_25km` | Fire Radiative Power within 25 km | NASA FIRMS | MW | 0.0 – 10000.0 |
| `fire_count_50km` | Active Fire Detections within 50 km | NASA FIRMS | count | 0 – 1500 |
| `fire_frp_50km` | Fire Radiative Power within 50 km | NASA FIRMS | MW | 0.0 – 25000.0 |
| `fire_count_100km` | Active Fire Detections within 100 km | NASA FIRMS | count | 0 – 5000 |
| `fire_frp_100km` | Fire Radiative Power within 100 km | NASA FIRMS | MW | 0.0 – 60000.0 |

---

## 4. Temporal & Availability Branch (10 Features)
| Feature Name | Description | Source | Range |
| :--- | :--- | :--- | :---: |
| `hour_sin` | Diurnal sinusoidal cyclic encoding: $\sin(2\pi \cdot \text{hour} / 24)$ | Astronomical | -1.0 – 1.0 |
| `hour_cos` | Diurnal cosinusoidal cyclic encoding: $\cos(2\pi \cdot \text{hour} / 24)$ | Astronomical | -1.0 – 1.0 |
| `dayofweek_sin` | Weekly cyclic encoding: $\sin(2\pi \cdot \text{dow} / 7)$ | Astronomical | -1.0 – 1.0 |
| `dayofweek_cos` | Weekly cyclic encoding: $\cos(2\pi \cdot \text{dow} / 7)$ | Astronomical | -1.0 – 1.0 |
| `dayofyear_sin` | Annual cyclic encoding: $\sin(2\pi \cdot \text{doy} / 365.25)$ | Astronomical | -1.0 – 1.0 |
| `dayofyear_cos` | Annual cyclic encoding: $\cos(2\pi \cdot \text{doy} / 365.25)$ | Astronomical | -1.0 – 1.0 |
| `month_sin` | Seasonal cyclic encoding: $\sin(2\pi \cdot \text{month} / 12)$ | Astronomical | -1.0 – 1.0 |
| `month_cos` | Seasonal cyclic encoding: $\cos(2\pi \cdot \text{month} / 12)$ | Astronomical | -1.0 – 1.0 |
| `pm25_available` | Binary observation presence flag | Quality Control | 0 or 1 |
| `pollutant_count_available` | Number of criteria pollutants actively reporting | Quality Control | 0 – 7 |
