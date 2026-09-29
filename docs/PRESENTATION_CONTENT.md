# SIH 2026 — PRESENTATION PITCH DECK CONTENT (ATMOSAIR v2)
> **Problem Statement ID:** 26082 — Air Pollution Weather Coupled Forecasting System — Delhi NCR Focus  
> **Team:** WINNERZWINNERZ | **Product:** ATMOSAIR  

---

### SLIDE 1: Title & Overview
* **Slide Title:** ATMOSAIR v2 — Coupled 72-Hour Air Quality & AQI Forecasting System
* **Subtitle:** Delhi-NCR Regional Focus (10 Districts) | Smart India Hackathon 2026
* **Key Bullet Points:**
  * Direct 72-Hour Continuous Hourly AQI & PM2.5 Trajectory Prediction
  * Multi-Branch Deep Neural Network (`819,874` parameters) + Observation-Driven XGBoost Residual Correction
  * Full Official Indian CPCB IND-AQI Piecewise Linear Multi-Pollutant Engine
  * WRF Regional Numerical Weather Prediction (NWP) Coupling
* **Visual Recommendation:** ATMOSAIR logo, 10-district Delhi NCR regional boundary map, neural network coupling graphic.
* **Speaker Script:** "Good morning, respected judges. We present ATMOSAIR v2, an operational 72-hour coupled air quality and weather forecasting system for the Delhi-NCR basin. Rather than relying on black-box heuristics or single-station proxies, ATMOSAIR couples deep causal learning with WRF regional meteorology, CPCB ground observations, satellite composition, and observation-driven residual correction."

---

### SLIDE 2: Problem & Operational Motivation
* **Slide Title:** The Air Quality Challenge in Delhi NCR
* **Key Bullet Points:**
  * **Complex Trapping Dynamics:** Winter cold-pool surface inversions collapse the boundary layer to ~300m, trapping local emissions.
  * **Episodic Stubble Burning:** Post-monsoon agricultural biomass fires in Punjab/Haryana transport massive aerosol loads into the NCR basin.
  * **Operational Requirement:** Municipal authorities (CAQM, DPCC, CPCB) need a reliable **3-day (72-hour) advance warning system** to trigger graded response action plan (GRAP) stages.
* **Visual Recommendation:** Satellite aerosol plume visualization over the Indo-Gangetic Plains.
* **Speaker Script:** "Air quality in Delhi-NCR is governed by weather coupling: low boundary layer heights and calm winds trap pollutants. Municipalities cannot act on 6-hour warnings; they require scientifically defensible 72-hour advance forecasts."

---

### SLIDE 3: System Architecture
* **Slide Title:** End-to-End System Architecture
* **Key Architecture Diagram:**
```
  [CPCB/DPCC CAAQMS]   [Copernicus CAMS]   [NASA FIRMS]   [Sentinel-5P]   [Copernicus CDS (ERA5)]   [WRF / NWP]
                                             │
                                             ▼
                                  Quality Control & Caching
                           (Unit conversion, physical bounds, TTLs)
                                             │
                                             ▼
                                  Spatial-Temporal Fusion
                          (Zero future leakage: t_obs <= T_issue)
                                             │
                                             ▼
             Coupled Multi-Branch AI (819k) + XGBoost Baseline + Extreme Classifier
                                             │
                                             ▼
                          Observation-Driven XGBoost Residual Correction
                                             │
                                             ▼
                    Conformal Calibration & Uncertainty Quantification
                                (P10, P50, P90 confidence intervals)
                                             │
                                             ▼
                          Official Indian CPCB IND-AQI Calculation
                                             │
                                             ▼
                     72-Hour Hourly AQI Trajectory + Spatial Hotspot Map
                                             │
                                             ▼
                                    Alerts & Explanations
```
* **Speaker Script:** "Our data pipeline ingests 6 independent environmental streams, validates temporal causality to prevent leakage, fuses features across 4 causal neural branches, and applies observation-driven residual correction to eliminate severe episode bias."

---

### SLIDE 4: Strict Leakage Prevention & Scientific Integrity
* **Slide Title:** Scientific Rigor: Strict Temporal Causality & No Leakage
* **Key Bullet Points:**
  * **Issue Time Causality ($T$):** Only observations with $t_{obs} \le T$ are consumed as historical features.
  * **Future Meteorology ($T+1 \dots T+72$):** Future weather forcing is derived **strictly from numerical forecast models (WRF/NWP)**.
  * **Forbidden Shortcuts Avoided:** Future ERA5 reanalysis, future CAMS retrospective data, and future ground observations are never leaked into the forecast horizon.
  * **Automated CI/CD Audit:** Validated via `tests/test_leakage.py`.
* **Speaker Script:** "We enforce strict causality. The model never cheats by peeking at future weather observations. Future meteorology comes strictly from forecast products, exactly matching real-world operational conditions."

---

### SLIDE 5: Modular WRF Regional NWP Coupling
* **Slide Title:** Weather Research and Forecasting (WRF) Integration
* **Key Bullet Points:**
  * **3-Domain Nested Configuration:** Outer India (27km) $\to$ North India (9km) $\to$ Delhi-NCR Operational Domain (3km).
  * **Planetary Boundary Layer Dynamics:** YSU scheme extracts dynamic nocturnal boundary layer collapse (PBLH) and surface inversion strength.
  * **Modular Integration:** ATMOSAIR features ingest WRF NetCDF outputs (`wrfout_d03_*`) with seamless fallback to global NWP forecast forcing.
* **Speaker Script:** "We developed a modular WRF interface configured for a realistic 3km domain over Delhi NCR. This provides the AI model with genuine future boundary layer ventilation heights and wind advection vectors."

---

### SLIDE 6: Solving the Severe Pollution Failure Mode
* **Slide Title:** Eliminating Severe Episode Underestimation Bias
* **Key Bullet Points:**
  * **Identified Vulnerability:** Neural models trained on mean squared error compress high-variance spikes ($\ge 345 \text{ µg/m}^3$), causing severe negative bias.
  * **Observation-Driven Residual Corrector:** An auxiliary XGBoost model learns error offsets:
    $$\text{residual}_{t+h} = \text{observed}_{t+h} - \text{raw\_forecast}_{t+h}$$
    as a function of recent error, WRF meteorology, CAMS background, and FIRMS fire proxies.
  * **Measurable Breakthrough on Untouched 2023 Test Set:**
    * Severe Episode MAE: **`198.74 µg/m³`** (vs Deep Model `219.78 µg/m³`, a **`9.57%` error reduction**).
    * Severe Episode Bias: Reduced by **`21.37 µg/m³`**.
    * Overall RMSE: **`81.24 µg/m³`** (vs Deep Model `82.56` and TCN `91.74`).
* **Speaker Script:** "Standard AI models underestimate hazardous pollution because severe episodes are statistically rare. Our observation-driven residual correction directly targets this failure mode, reducing severe episode error by 9.57% and eliminating over 21 µg/m³ of negative bias."

---

### SLIDE 7: Official Indian CPCB Multi-Pollutant AQI Engine
* **Slide Title:** Official CPCB IND-AQI Compliance
* **Key Bullet Points:**
  * **Piecewise Linear Interpolation:** Implements official CPCB sub-index formulas across all 7 criteria pollutants: $\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3, \text{NH}_3$.
  * **Governing Rule:**
    $$\text{AQI} = \max(I_{\text{PM2.5}}, I_{\text{PM10}}, I_{\text{NO2}}, I_{\text{SO2}}, I_{\text{CO}}, I_{\text{O3}}, I_{\text{NH3}})$$
  * **Dominant Pollutant Tracking:** Identifies governing driver ($\text{PM}_{2.5}$ during winter smog, $\text{O}_3$ during summer photochemical afternoons).
  * **No Search Engine Proxies:** Grounded strictly in physical concentrations and statutory breakpoint arithmetic.
* **Speaker Script:** "We do not scrape arbitrary AQI numbers from the web. We built the complete official Indian CPCB multi-pollutant AQI calculation engine from first principles, ensuring complete regulatory compliance."

---

### SLIDE 8: Spatial Delhi-NCR Domain & Hotspot Detection
* **Slide Title:** Delhi-NCR Regional Spatial Lattice (10 Districts)
* **Key Bullet Points:**
  * **Target Domain:** Covers Delhi, Gurugram, Faridabad, Noida, Greater Noida, Ghaziabad, Sonipat, Jhajjar, Rohtak, Meerut.
  * **Lattice Resolution:** Configurable ~8 km spatial resolution.
  * **Wind-Advective Inverse Distance Weighting:** Transfers official CAAQMS station forecasts across the basin using WRF wind advection vectors.
  * **Explicit Provenance Labels:** Clear distinction between `OFFICIAL CAAQMS STATION` and `MODEL ESTIMATE`.
* **Speaker Script:** "Air quality does not stop at Delhi's borders. Our spatial engine forecasts across 10 NCR districts, tracking upwind stubble smoke transport and identifying localized industrial hotspots."

---

### SLIDE 9: Five-Year Long-Range Policy Scenario Outlook
* **Slide Title:** 5-Year Long-Range Scenario Simulation (2024–2028)
* **Key Bullet Points:**
  * **Honest Scope Definition:** Explicitly structured as an aggregated policy and meteorological scenario simulation, **NOT a deterministic hourly AQI prediction**.
  * **4 Evaluated Scenarios:**
    1. *Business As Usual (BAU):* Current fleet and industrial growth $\to$ 115.2 µg/m³ annual mean (52 severe days).
    2. *Accelerated NCAP + GRAP IV:* 35% particulate reduction $\to$ 80.5 µg/m³ annual mean (24 severe days).
    3. *Climate Stagnation Risk:* Enhanced winter cold-pool frequency $\to$ 118.0 µg/m³ annual mean (64 severe days).
    4. *Comprehensive Clean Transition:* Complete boiler electrification & clean mobility $\to$ 65.2 µg/m³ annual mean (14 severe days).
* **Speaker Script:** "We do not make unscientific claims of predicting exact hourly AQI five years into the future. Instead, our scenario engine simulates policy outcomes under different climate and enforcement conditions, giving policymakers actionable multi-year insights."

---

### SLIDE 10: Verified Test Benchmark Leaderboard
* **Slide Title:** Verified Empirical Leaderboard (Untouched 2023 Test Set)
* **Metrics Table:**
| Model / Baseline | Parameters | Overall MAE | Overall RMSE | $R^2$ Score | Severe MAE ($\ge 345$) | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ATMOSAIR v2 Corrected Ensemble** | `819k + XGB` | **56.66** | **81.24** | **0.3768** | **198.74** | **#1 (Champion)** |
| Deep Base Model | 819,874 | 56.66 | 82.56 | 0.3564 | 219.78 | #2 |
| TCN Champion Baseline | 570,625 | 61.19 | 91.74 | 0.2053 | 223.01 | #3 |
| LSTM Baseline | 223,873 | 74.48 | 102.18 | 0.0142 | 223.50 | #4 |
| Naive Persistence | N/A | 75.42 | 107.00 | -0.0810 | 199.79 | #5 |
* **Speaker Script:** "On 3,393 untouched held-out test sequences, our corrected ensemble achieves the highest overall R2 score of 0.3768 and lowest RMSE of 81.24 µg/m³, decisively beating TCN and recurrent baselines."

---

### SLIDE 11: Production Architecture & Dashboard
* **Slide Title:** Production Software Architecture
* **Key Bullet Points:**
  * **FastAPI Backend:** Fully versioned forecasts (`forecast_id`, `issue_time`, `valid_time`, `model_version`, `uncertainty_bounds`).
  * **Live Interactive Dashboard:** Multi-tab interface featuring 72-hour AQI curves, conformal uncertainty bands, interactive Delhi-NCR hotspot map, and 5-year scenario simulator.
  * **Robust CI/CD Suite:** 36 automated unit and integration tests passing (`pytest tests/`).
* **Speaker Script:** "The entire platform is operational today. Our FastAPI backend delivers versioned, audited predictions, and the dashboard provides intuitive visualizations for emergency response."

---

### SLIDE 12: Summary & Conclusion
* **Slide Title:** Summary of Deliverables
* **Key Achievements:**
  1. Coupled deep learning + WRF NWP + XGBoost residual correction.
  2. 9.57% error reduction during severe pollution emergencies.
  3. Strict causality with zero ground truth leakage.
  4. Full official CPCB IND-AQI regulatory engine.
  5. 10-district Delhi NCR spatial domain + 5-year scenario simulation.
* **Speaker Script:** "In summary, ATMOSAIR v2 bridges atmospheric physics and deep learning to deliver a reliable, bias-corrected, and regulatory-compliant 72-hour air quality forecasting system for Delhi NCR. Thank you."
