# B14 — FINAL MODEL FAILURE MODES & BOTTLENECK AUDIT

## Executive Summary

Detailed diagnostic analysis of remaining failure modes in the **Coupled Multi-Branch Forecast Model**.

## Ranked Failure Modes

### 1. Extreme Peak Underprediction (Severity: HIGH)

- **Evidence:** For actual PM2.5 $\ge 600 \mu g/m^3$, predicted average is `228.14 µg/m³` (Bias: `-428.51 µg/m³`).

- **Likely Cause:** Standard scalar target normalization compresses peak gradients during MSE loss minimization.

- **Potential Future Solution:** Extreme value loss scaling or log-transformed target space.


### 2. Variance Compression (Severity: MEDIUM)

- **Evidence:** Prediction std is `59.39 µg/m³` vs ground truth std `102.91 µg/m³` (Variance Ratio: `0.58`).

- **Likely Cause:** MSE loss penalizes point variance over predictions.

- **Potential Future Solution:** Generative/probabilistic forecasting heads.


### 3. +1h Short-Term Autocorrelation Gap (Severity: LOW)

- **Evidence:** Persistence wins at +1h (`21.02 µg/m³` vs Proposed `52.16 µg/m³`).

- **Likely Cause:** Deep neural representations require multi-step context.

- **Potential Future Solution:** Direct persistence-residual skip connection at $t+1$.


```
B14 STATUS: COMPLETED & VERIFIED
```
