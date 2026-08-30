# B12 — PROPOSED MODEL FULL TRAINING REPORT

## Executive Summary

Full training run of the **Coupled Multi-Branch Forecast Model** (`819,874` parameters) completed over `8` epochs. The best model checkpoint was saved at **Epoch 1** with a Best Validation Loss of **`0.907336`**.

## Training Performance Summary

| Metric / Property | Measured Value |
| :--- | :--- |
| **Model Class** | `CoupledMultiBranchForecastModel` |
| **Total Parameters** | `819,874` parameters |
| **Total Epochs Executed** | `8` epochs |
| **Best Checkpoint Epoch** | **Epoch 1** |
| **Best Validation Loss** | **`0.907336`** |
| **Final Training Loss** | `0.111479` |
| **Final Validation Loss** | `0.993220` |
| **Total Training Duration** | `1495.38` seconds (`24.92` minutes) |
| **Saved Checkpoint Path** | [`models/checkpoints/proposed_best.pt`](file:///d:/My Projects/SIH2026_PersonB/models/checkpoints/proposed_best.pt) |
| **Untouched Test Set Status** | **100% ISOLATED (Zero test data used during B12)** |

## Epoch-by-Epoch History Table

| Epoch | Train Total Loss | Val Total Loss | Val Base MSE | Val Spike MSE | Val Aux BCE | Learning Rate | Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | `0.696688` | `0.907336` | `0.469452` | `0.799202` | `0.191416` | `0.001000` | `204.87s` |
| Epoch 2 | `0.368215` | `1.040696` | `0.568205` | `0.850277` | `0.236764` | `0.001000` | `153.30s` |
| Epoch 3 | `0.252884` | `1.023257` | `0.539493` | `0.857748` | `0.274451` | `0.001000` | `210.05s` |
| Epoch 4 | `0.198882` | `0.966641` | `0.512618` | `0.796780` | `0.278164` | `0.001000` | `279.12s` |
| Epoch 5 | `0.167684` | `1.053113` | `0.549244` | `0.878717` | `0.322551` | `0.001000` | `118.79s` |
| Epoch 6 | `0.128327` | `0.984484` | `0.512413` | `0.812920` | `0.328056` | `0.000500` | `279.08s` |
| Epoch 7 | `0.117597` | `0.993243` | `0.519453` | `0.803638` | `0.359852` | `0.000500` | `137.58s` |
| Epoch 8 | `0.111479` | `0.993220` | `0.512099` | `0.811176` | `0.377665` | `0.000500` | `112.47s` |

## Generated Training Curves

- **Total Loss Curve:** [`results/plots/proposed_training_history.png`](file:///d:/My Projects/SIH2026_PersonB/results/plots/proposed_training_history.png)
- **Validation Component Curve:** [`results/plots/proposed_validation_curve.png`](file:///d:/My Projects/SIH2026_PersonB/results/plots/proposed_validation_curve.png)

---

```
B12 STATUS: FULL TRAINING COMPLETED & VERIFIED
```

```
READY FOR B13 — FINAL TEST EVALUATION
```
