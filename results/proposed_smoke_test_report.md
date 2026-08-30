# B11 — PROPOSED MODEL 2-EPOCH SMOKE TEST REPORT

**Project:** Deep Learning Air Quality Forecasting (72h-to-72h PM2.5)  
**Proposed Model:** Coupled Multi-Branch Forecast Model (`CoupledMultiBranchForecastModel`)  
**Status:** `B11 STATUS: SANITY CHECK PASSED`  
**Next Stage:** `READY FOR B12 — FULL PROPOSED MODEL TRAINING`

---

## 1. Executive Summary

Following the completion of the 13-Point Sanity Check Verification, a **2-Epoch Smoke Test** was conducted on the `CoupledMultiBranchForecastModel` using the training and validation DataLoaders.

* **Dataset Scope:** **TRAIN (19,005 sequences)** and **VALIDATION (4,493 sequences)** ONLY.
* **Test Isolation:** The untouched 2023 TEST set (`3,393` sequence samples) was **STRICTLY ISOLATED** and zero test evaluation was performed during B11.
* **Stability:** Training loss decreased smoothly from **`0.696688`** (Epoch 1) to **`0.368215`** (Epoch 2) with zero NaNs, zero Infs, and finite validation losses.
* **Smoke Checkpoint:** Saved temporary checkpoint to [`models/checkpoints/proposed_smoke_test.pt`](file:///d:/My%20Projects/SIH2026_PersonB/models/checkpoints/proposed_smoke_test.pt).

---

## 2. Smoke Test Execution History

| Epoch | Total Train Batches | Training Loss | Validation Loss | Batch Loss Progression | Checkpoint Status |
| :---: | :---: | :---: | :---: | :--- | :---: |
| **Epoch 1/2** | `297` | `0.696688` | `0.907336` | `1.365556` (B0) $\to$ `0.713062` (B100) $\to$ `0.576193` (B200) | Valid |
| **Epoch 2/2** | `297` | **`0.368215`** | `1.040696` | `0.407046` (B0) $\to$ `0.418359` (B100) $\to$ `0.354540` (B200) | Saved (`proposed_smoke_test.pt`) |

---

## 3. Verification & Compliance Audit

- [x] **No Test-Set Leakage:** Untouched Test Set (`3,393` samples) was never loaded or evaluated.
- [x] **Stable Gradient Dynamics:** Gradients remained finite with gradient norm clipping at `1.0`.
- [x] **Loss Convergence:** Training loss decreased by **`-47.15%`** over 2 epochs.
- [x] **Multi-Branch & Dual Head Compatibility:** Primary forecast head and auxiliary spike head operated in parallel without dimension mismatches.
- [x] **Checkpoint Integrity:** Checkpoint `proposed_smoke_test.pt` successfully written and verified.

---

## 4. Deliverable Verification Index

* **PyTorch Loss Implementation:** [`training/losses.py`](file:///d:/My%20Projects/SIH2026_PersonB/training/losses.py)
* **Training Pipeline Script:** [`training/train_proposed.py`](file:///d:/My%20Projects/SIH2026_PersonB/training/train_proposed.py)
* **Automated Sanity Test Suite:** [`tests/test_proposed_training.py`](file:///d:/My%20Projects/SIH2026_PersonB/tests/test_proposed_training.py)
* **Sanity Check Markdown Report:** [`results/proposed_training_sanity_check.md`](file:///d:/My%20Projects/SIH2026_PersonB/results/proposed_training_sanity_check.md)
* **Smoke Test Markdown Report:** [`results/proposed_smoke_test_report.md`](file:///d:/My%20Projects/SIH2026_PersonB/results/proposed_smoke_test_report.md)
* **Machine-Readable Loss Config:** [`results/proposed_loss_configuration.json`](file:///d:/My%20Projects/SIH2026_PersonB/results/proposed_loss_configuration.json)

---

```
B11 STATUS: SANITY CHECK PASSED
```

```
READY FOR B12 — FULL PROPOSED MODEL TRAINING
```
