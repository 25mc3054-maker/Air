# B6 — GRU TRAINING REPORT

## Executive Summary

Direct sequence-to-sequence GRU baseline model trained for 72-hour continuous PM2.5 forecasting.

## Model Architecture & Parameter Count

- **Architecture:** `GRUForecastModel` (2-Layer GRU + Linear Projection)
- **Input Size:** `49` features
- **Sequence Length:** `72` hours
- **Hidden Dimension:** `128` units
- **GRU Layers:** `2` layers (Dropout = `0.2`)
- **Output Size:** `1` (PM2.5 prediction)
- **Total Parameters:** `167,937`
- **Trainable Parameters:** `167,937`

## Training Hyperparameters

- **Optimizer:** `Adam` (Initial LR = `0.001`, Weight Decay = `1e-05`)
- **LR Scheduler:** `ReduceLROnPlateau` (Factor = `0.5`, Patience = `3`, Min LR = `1e-06`)
- **Gradient Clipping:** `max_norm = 1.0` (`torch.nn.utils.clip_grad_norm_`)
- **Loss Function:** `Mean Squared Error (MSE)` (in z-score scaled space)
- **Batch Size:** `64`
- **Early Stopping:** Patience = `7` epochs monitoring validation MSE

## Execution & Convergence Summary

- **Hardware Device:** `CPU` (`cpu`)
- **Total Epochs Completed:** `9` / `50`
- **Best Epoch:** `2`
- **Best Validation Loss (Scaled MSE):** `0.605181`
- **Total Training Time:** `572.68 seconds` (9.54 minutes)
- **Best Checkpoint Path:** `models/checkpoints/gru_best.pt`

## Training Progression (First 5 & Best Epochs)

| Epoch | Train Loss (MSE) | Val Loss (MSE) | Train MAE (Scaled) | Val MAE (Scaled) | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.385537 | 0.664802 | 0.420820 | 0.558208 | 0.001000 |
| 2 | 0.191737 | 0.605181 | 0.294251 | 0.539620 | 0.001000 |
| 3 | 0.124769 | 0.651025 | 0.235971 | 0.560068 | 0.001000 |
| 4 | 0.096637 | 0.659660 | 0.205927 | 0.560718 | 0.001000 |
| 5 | 0.081062 | 0.621156 | 0.187610 | 0.546791 | 0.001000 |

---

```
GRU TRAINING RUN: COMPLETED SUCCESSFULLY
```