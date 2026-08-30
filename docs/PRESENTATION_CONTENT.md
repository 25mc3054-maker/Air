# SIH 2026 — PRESENTATION PITCH DECK CONTENT (12 SLIDES)
> **Project:** 72-Hour PM2.5 Air Quality Forecasting System  
> **Team Role:** Person B (Deep Learning Neural Architecture & Evaluation Lead)  

---

### SLIDE 1: Title & Overview
* **Slide Title:** Deep Learning Multi-Branch 72-Hour PM2.5 Air Quality Forecasting
* **Subtitle:** Smart India Hackathon (SIH) 2026 Pitch Deck
* **Key Bullet Points:**
  * Direct 72-Hour Multi-Step Trajectory Prediction
  * Coupled Multi-Branch Deep Neural Network (`819,874` parameters)
  * Real-time CPCB Air Quality Categorization & Spike Detection
* **Visual Recommendation:** Team logo, project title, and high-level neural network icon.
* **Speaker Script:** "Good morning judges. Today we present our 72-Hour Multi-Step Direct PM2.5 Air Quality Forecasting System, built to overcome the rapid error accumulation of conventional single-step forecasting models."

---

### SLIDE 2: Problem & Motivation
* **Slide Title:** The Air Quality Challenge in Urban India
* **Key Bullet Points:**
  * $\text{PM}_{2.5}$ fine particulate matter poses major respiratory & cardiovascular health risks.
  * Current AQI advisories rely on 6–12 hour short-term estimates.
  * Municipalities need a reliable **3-day (72-hour) advance warning system** to institute traffic & industrial restrictions.
* **Visual Recommendation:** Map of urban air pollution with health impact icons.
* **Speaker Script:** "Current forecasting systems fail when projecting 3 days into the future. Municipalities cannot take proactive action without accurate long-horizon predictions."

---

### SLIDE 3: Why Conventional Approaches Fail
* **Slide Title:** Limitations of Existing Methods
* **Key Bullet Points:**
  * **Autoregressive Single-Step Models:** Error explodes exponentially over multi-step horizons.
  * **Standard Recurrent Networks (GRU/LSTM):** Suffer from gradient vanishing and variance compression.
  * **Variance Smoothing:** Standard MSE loss ignores rare, extreme hazardous spikes ($\ge 345 \mu g/m^3$).
* **Visual Recommendation:** Error growth comparison chart (Autoregressive vs Direct Multi-Step).
* **Speaker Script:** "Standard models smooth out high pollution spikes, treating toxic episodes as minor statistical noise. Our solution directly targets long horizons and extreme spikes."

---

### SLIDE 4: Multi-Modal Dataset & 49 Input Features
* **Slide Title:** Comprehensive Multi-Modal Feature Integration
* **Key Bullet Points:**
  * **Pollution (9):** $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$, etc.
  * **Meteorology (21):** Temperature, humidity, local wind vectors ($u, v$), surface pressure.
  * **Atmospheric Satellite (9):** Aerosol Optical Thickness (AOD) & trace gas slant columns.
  * **Temporal & Availability (10):** Cyclical sine/cosine temporal features + data quality masks.
* **Visual Recommendation:** 5-block diagram grouping the 49 input features.
* **Speaker Script:** "We ingest 49 environmental features every hour, combining ground station readings, meteorological vectors, and atmospheric satellite data."

---

### SLIDE 5: Sequence Construction & Leakage Safety
* **Slide Title:** Rigorous Temporal Sequence Pipeline
* **Key Bullet Points:**
  * **Sequence Window:** 72 historical hours ($t-71$ to $t$) $\to$ 72 future forecast hours ($t+1$ to $t+72$).
  * **Strict Data Isolation:** Preprocessing scalers fitted **TRAIN ONLY**.
  * **Untouched Test Set:** 3,393 test sequences (244,296 forecast timesteps) strictly isolated.
* **Visual Recommendation:** Sliding window sequence construction diagram.
* **Speaker Script:** "We maintain 100% temporal isolation. All scalers are fitted strictly on training data, preventing any data leakage into our test evaluations."

---

### SLIDE 6: Proposed Multi-Branch Architecture
* **Slide Title:** Coupled Multi-Branch Deep Neural Network
* **Key Bullet Points:**
  * **4 Domain-Specific Streams:** Causal dilated convolutions ($64$ channels each).
  * **Gated Linear Unit (GLU):** Non-linear feature gating (224 $\to$ 256 dimensions).
  * **4-Head Multi-Head Self-Attention:** Captures multi-day temporal dependencies.
  * **Dual Output Heads:** Regression Forecast Head + Auxiliary Spike Classification Head.
* **Visual Recommendation:** Full architectural block diagram ([`results/proposed_architecture_diagram.png`](file:///d:/My%20Projects/SIH2026_PersonB/results/proposed_architecture_diagram.png)).
* **Speaker Script:** "Our model processes environmental domains independently before fusing them through gated linear units and causal self-attention."

---

### SLIDE 7: Why GRU + LSTM + TCN Synthesis?
* **Slide Title:** Architectural Synergies & Branch Rationale
* **Key Bullet Points:**
  * **GRU & LSTM Baselines:** Taught us the necessity of multi-step direct projection.
  * **Temporal Convolutional Networks (TCN):** Provided 127-hour receptive field via dilated causal convolutions.
  * **Coupled Synthesis:** Combines multi-scale convolutions with self-attention for superior long-horizon stability.
* **Visual Recommendation:** Receptive field coverage comparison diagram.
* **Speaker Script:** "By synthesizing causal convolutions with attention mechanisms, our model achieves stable multi-step forecasting without autoregressive error accumulation."

---

### SLIDE 8: Training & Spike-Aware Loss
* **Slide Title:** Specialized Loss Function & Training Protocol
* **Key Bullet Points:**
  * **Asymmetric Spike MSE:** $3.0\times$ weight penalty for extreme spikes ($\ge 345 \mu g/m^3$).
  * **Auxiliary BCE Classification Loss:** Supervises spike probability detection.
  * **Optimizer:** AdamW with `ReduceLROnPlateau` and early stopping at Epoch 8.
* **Visual Recommendation:** Loss curve plot showing training vs validation convergence.
* **Speaker Script:** "Our spike-aware loss penalizes underpredictions during hazardous episodes, forcing the neural network to prioritize public safety."

---

### SLIDE 9: Results & Verified Leaderboard
* **Slide Title:** Project Benchmark Leaderboard
* **Key Bullet Points:**
  * **MAE:** **`56.66 µg/m³`** (TCN `61.19` / **`+7.40%` improvement**)
  * **RMSE:** **`82.56 µg/m³`** (TCN `91.74` / **`+10.01%` improvement**)
  * **$R^2$ Score:** **`0.3564`** (TCN `0.2053` / **`+0.1511` gain**)
  * **Statistical Significance:** $p < 0.001$ across paired Student's t-test and Wilcoxon tests.
* **Visual Recommendation:** Leaderboard table with rank #1 highlighted in green.
* **Speaker Script:** "Our model is the undisputed project champion, outperforming all baseline models across MAE, RMSE, and R² scores with statistical significance."

---

### SLIDE 10: Multi-Horizon & Extreme Pollution
* **Slide Title:** Superior Multi-Step & High-Pollution Accuracy
* **Key Bullet Points:**
  * Outperforms TCN across all multi-step horizons: **+6h, +12h, +24h, +48h, and +72h**.
  * **High-Pollution MAE ($\ge 345 \mu g/m^3$):** `219.78 µg/m³` (Best among neural models).
  * Flat horizon error curve demonstrates long-term stability up to Day 3.
* **Visual Recommendation:** 72h MAE Horizon Curve plot ([`results/plots/final_72h_mae_curve.png`](file:///d:/My%20Projects/SIH2026_PersonB/results/plots/final_72h_mae_curve.png)).
* **Speaker Script:** "Notice how our error curve remains flat all the way to 72 hours, whereas standard models degrade rapidly."

---

### SLIDE 11: Limitations & Future Roadmap
* **Slide Title:** Honest Limitations & Research Roadmap
* **Key Bullet Points:**
  * **+1h Persistence Strength:** Persistence wins at $+1$h (`21.02` vs `52.16 µg/m³`) due to immediate autocorrelation.
  * **Extreme Spike Compression ($> 600 \mu g/m^3$):** Variance remains compressed under linear MSE.
  * **Future Work:** Log-transformed target scaling ($\log(1+y)$) and direct $+1$h residual skip connections.
* **Visual Recommendation:** Residual distribution plot and future enhancement roadmap.
* **Speaker Script:** "We remain scientifically honest: persistence is stronger at hour 1, but our model takes over from hour 6 onward."

---

### SLIDE 12: Conclusion & SIH Impact
* **Slide Title:** Deployment-Ready Solution for SIH 2026
* **Key Bullet Points:**
  * **Production Backend:** FastAPI server (`api/app.py`) providing `/predict` and `/metrics`.
  * **Interactive Frontend:** Real-time JavaScript dashboard (`dashboard/index.html`).
  * **Public Health Impact:** Empowers urban authorities with actionable 3-day PM2.5 advisories.
* **Visual Recommendation:** Live dashboard preview screenshot ([`results/plots/final_model_dashboard.png`](file:///d:/My%20Projects/SIH2026_PersonB/results/plots/final_model_dashboard.png)).
* **Speaker Script:** "Our end-to-end system is fully built, tested, and ready for deployment. Thank you!"
