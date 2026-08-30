import numpy as np

def calculate_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

def calculate_overall_metrics(y_true, y_pred):
    """
    Computes MAE, RMSE, and R2 across all samples and timesteps.
    y_true, y_pred shape: [N, 72, 1] or [N, 72]
    """
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    
    mae = calculate_mae(y_true_flat, y_pred_flat)
    rmse = calculate_rmse(y_true_flat, y_pred_flat)
    r2 = calculate_r2(y_true_flat, y_pred_flat)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

def calculate_horizon_metrics(y_true, y_pred, horizons=[1, 6, 12, 24, 48, 72]):
    """
    Computes MAE and RMSE at specific forecast horizons (1-indexed).
    y_true, y_pred shape: [N, 72, 1]
    """
    horizon_results = {}
    for h in horizons:
        idx = h - 1 # 0-indexed timestep
        y_t = y_true[:, idx, 0] if y_true.ndim == 3 else y_true[:, idx]
        y_p = y_pred[:, idx, 0] if y_pred.ndim == 3 else y_pred[:, idx]
        
        h_mae = calculate_mae(y_t, y_p)
        h_rmse = calculate_rmse(y_t, y_p)
        horizon_results[f"{h}h"] = {
            "mae": h_mae,
            "rmse": h_rmse
        }
    return horizon_results

def calculate_high_pollution_metrics(y_true, y_pred, threshold_90th):
    """
    Computes MAE and RMSE exclusively on samples/timesteps where ground truth target >= threshold_90th.
    Threshold derived strictly from training target distribution.
    """
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    
    mask = (y_true_flat >= threshold_90th)
    if mask.sum() == 0:
        return {"high_pollution_mae": 0.0, "high_pollution_rmse": 0.0, "sample_count": 0}
        
    hp_mae = calculate_mae(y_true_flat[mask], y_pred_flat[mask])
    hp_rmse = calculate_rmse(y_true_flat[mask], y_pred_flat[mask])
    
    return {
        "threshold_90th": float(threshold_90th),
        "sample_count": int(mask.sum()),
        "mae": hp_mae,
        "rmse": hp_rmse
    }
