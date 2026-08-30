# B7 — LSTM TRAINING REPORT

## Executive Summary

Direct sequence-to-sequence LSTM baseline model trained for 72-hour continuous PM2.5 forecasting.

## Model Architecture & Parameter Count Comparison

- **Architecture:** `LSTMForecastModel` (2-Layer LSTM + Linear Projection)
- **Input Size:** `49` features
- **Sequence Length:** `72` hours
- **Hidden Dimension:** `128` units
- **LSTM Layers:** `2` layers (Dropout = `0.2`)
- **Output Size:** `1` (PM2.5 prediction)
- **LSTM Total Parameters:** `223,873`
- **LSTM Trainable Parameters:** `223,873`
- **GRU Total Parameters (B6 Baseline):** `167,937`
- **Parameter Difference:** LSTM has `+55,936` parameters (+33.3% capacity due to 4 gates vs 3 gates)

## Training Hyperparameters (Fair Comparison Framework)

- **Optimizer:** `Adam` (Initial LR = `0.001`, Weight Decay = `1e-05`)
- **LR Scheduler:** `ReduceLROnPlateau` (Factor = `0.5`, Patience = `3`, Min LR = `1e-06`)
- **Gradient Clipping:** `max_norm = 1.0` (`torch.nn.utils.clip_grad_norm_`)
- **Loss Function:** `Mean Squared Error (MSE)` (in z-score scaled space)
- **Batch Size:** `64`
- **Early Stopping:** Patience = `7` epochs monitoring validation MSE

## Execution & Convergence Summary

- **Hardware Device:** `CPU` (`cpu`)
- **Total Epochs Completed:** `10` / `50`
- **Best Epoch:** `3`
- **Best Validation Loss (Scaled MSE):** `0.562519`
- **Total Training Time:** `406.43 seconds` (6.77 minutes)
- **Best Checkpoint Path:** `models/checkpoints/lstm_best.pt`

## Training Progression

| Epoch | Train Loss (MSE) | Val Loss (MSE) | Train MAE (Scaled) | Val MAE (Scaled) | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.326314 | 0.598489 | 0.383467 | 0.522406 | 0.001000 |
| 2 | 0.142434 | 0.588896 | 0.252931 | 0.522655 | 0.001000 |
| 3 | 0.098690 | 0.562519 | 0.208029 | 0.513522 | 0.001000 |
| 4 | 0.077362 | 0.605148 | 0.181175 | 0.534731 | 0.001000 |
| 5 | 0.065026 | 0.585155 | 0.164150 | 0.526648 | 0.001000 |

---

```
LSTM TRAINING RUN: COMPLETED SUCCESSFULLY
```