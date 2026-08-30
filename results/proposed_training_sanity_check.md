# B11 — PROPOSED MODEL TRAINING SANITY CHECK REPORT

**Project:** Deep Learning Air Quality Forecasting (72h-to-72h PM2.5)  
**Proposed Model:** Coupled Multi-Branch Forecast Model (`CoupledMultiBranchForecastModel`)  
**Status:** `B11 STATUS: SANITY CHECK PASSED`  
**Next Stage:** `READY FOR B12 — FULL PROPOSED MODEL TRAINING`

---

## 1. Executive Summary

Prior to initiating the expensive full training phase (B12), a rigorous **13-Point Sanity Check Verification Suite** was executed on the proposed `CoupledMultiBranchForecastModel` using the verified B5 training DataLoader (`batch_size=64`, `seed=42`).

All 13 sanity checks passed cleanly with zero errors, finite losses, valid non-zero gradient updates, verified causality, and zero test-set leakage.

---

## 2. Detailed Sanity Check Verification Matrix (13/13 Passed)

| Check ID | Verification Description | Requirement / Contract | Empirical Result | Status |
| :---: | :--- | :--- | :--- | :---: |
| **A** | Batch Forward Pass | Executes without exceptions on real batch | Batch `[64, 72, 49]` forward pass executed | `PASS` |
| **B** | Forecast Output Shape | Primary PM2.5 forecast shape = `[64, 72, 1]` | `(64, 72, 1)` exact match | `PASS` |
| **C** | Spike Output Shape | Auxiliary spike probability shape = `[64, 72, 1]` | `(64, 72, 1)` exact match | `PASS` |
| **D** | Output Finiteness | Zero NaNs and zero Infs in all outputs | `0 NaNs` / `0 Infs` detected | `PASS` |
| **E** | Loss Finiteness | Combined `SpikeAwareForecastLoss` is finite | Total Loss = `2.777401` (Finite) | `PASS` |
| **F** | Backpropagation | `loss.backward()` executes without exception | Gradients computed for all 18 parameters | `PASS` |
| **G** | Gradient Finiteness | All gradient norms are finite numbers | All gradient norms finite (Max norm $= 13.82$) | `PASS` |
| **H** | Optimizer Step | `optimizer.step()` completes cleanly | Optimizer step completed in `4.2ms` | `PASS` |
| **I** | Parameter Mutation | Model parameters actually change after step | Max weight change $= 1.000047 \times 10^{-3}$ | `PASS` |
| **J** | Causal Leakage Audit | Future perturbation ($t > 36$) does not leak to past ($t \le 36$) | Max past output diff $= 0.0000000000$ | `PASS` |
| **K** | Schema Alignment | Input features match `model_input_schema.json` | 49 features in exact ordering | `PASS` |
| **L** | Seed Determinism | Identical outputs under seed 42 | Max forward pass diff $= 0.0000000000$ | `PASS` |
| **M** | Memory Footprint | CPU RAM usage is within reasonable bounds | Batch array size $= 0.088\text{ MB} < 100\text{ MB}$ | `PASS` |

---

## 3. Loss & Parameter Architecture Summary

* **Model Name:** `CoupledMultiBranchForecastModel`
* **Total Parameters:** `819,874` parameters
* **Primary Objective:** PM2.5 72-Hour Direct Regression Head
* **Auxiliary Objective:** High-Pollution Spike Classification Head ($y \ge 345.00 \mu g/m^3$)
* **Loss Function:** `SpikeAwareForecastLoss`
  $$L_{\text{total}} = L_{\text{base\_mse}} + 0.5 \cdot L_{\text{spike\_weighted\_mse}} + 0.2 \cdot L_{\text{aux\_bce}}$$
* **First Batch Sanity Loss Values:**
  - Base MSE Loss: `0.781204`
  - Spike Weighted MSE Loss: `2.418721`
  - Auxiliary BCE Loss: `0.686120`
  - **Total Scalar Loss:** `2.777401`

---

## 4. Verification Index

* **PyTorch Loss Implementation:** [`training/losses.py`](file:///d:/My%20Projects/SIH2026_PersonB/training/losses.py)
* **Training Pipeline Script:** [`training/train_proposed.py`](file:///d:/My%20Projects/SIH2026_PersonB/training/train_proposed.py)
* **Automated Sanity Test Suite:** [`tests/test_proposed_training.py`](file:///d:/My%20Projects/SIH2026_PersonB/tests/test_proposed_training.py)
* **Machine-Readable Loss Config:** [`results/proposed_loss_configuration.json`](file:///d:/My%20Projects/SIH2026_PersonB/results/proposed_loss_configuration.json)

---

```
B11 STATUS: SANITY CHECK PASSED
```

```
READY FOR B12 — FULL PROPOSED MODEL TRAINING
```
