# B8 — TCN TRAINING REPORT

## Executive Summary

Direct sequence-to-sequence Causal Dilated TCN baseline model trained for 72-hour continuous PM2.5 forecasting.

## Model Architecture & Parameter Count Comparison

- **Architecture:** `TCNForecastModel` (6 Causal Dilated Residual Blocks + 1x1 Conv Projection)
- **Input Channels:** `49` features
- **Hidden Channels:** `128` channels across 6 residual levels
- **Dilation Levels:** `[1, 2, 4, 8, 16, 32]` (Receptive Field = `127` hours > 72h sequence length)
- **Kernel Size:** `3` (Dropout = `0.2`)
- **Output Size:** `1` (PM2.5 prediction)
- **TCN Total Parameters:** `570,625`
- **TCN Trainable Parameters:** `570,625`
- **GRU Total Parameters (B6 Baseline):** `167,937`
- **LSTM Total Parameters (B7 Baseline):** `223,873`

## Training Hyperparameters (Fair Comparison Framework)

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
- **Best Validation Loss (Scaled MSE):** `0.535314`
- **Total Training Time:** `6016.91 seconds` (100.28 minutes)
- **Best Checkpoint Path:** `models/checkpoints/tcn_best.pt`

## Training Progression

| Epoch | Train Loss (MSE) | Val Loss (MSE) | Train MAE (Scaled) | Val MAE (Scaled) | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.501769 | 0.832256 | 0.479839 | 0.623895 | 0.001000 |
| 2 | 0.222102 | 0.535314 | 0.328003 | 0.494708 | 0.001000 |
| 3 | 0.169000 | 0.690241 | 0.286348 | 0.582811 | 0.001000 |
| 4 | 0.135870 | 0.594026 | 0.257028 | 0.549734 | 0.001000 |
| 5 | 0.111171 | 0.552108 | 0.232642 | 0.517194 | 0.001000 |

---

```
TCN TRAINING RUN: COMPLETED SUCCESSFULLY
```