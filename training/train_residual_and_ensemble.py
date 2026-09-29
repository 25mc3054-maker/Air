import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.proposed_model import CoupledMultiBranchForecastModel
from models.extreme_event.extreme_classifier import ExtremePollutionClassifier
from models.xgboost.xgboost_model import XGBoostForecastModel
from pipeline.bias_correction import ObservationResidualCorrector
from pipeline.calibration import EnsembleCalibrator

def train_and_evaluate_components():
    print("=" * 60)
    print("ATMOSAIR v2: Training Residual Corrector, Extreme Model & Ensemble")
    print("=" * 60)

    # 1. Load Processed Scaled & Raw Data
    print("\n1. Loading chronological datasets (Train 2015-2021, Val 2022, Test 2023)...")
    scaled_dir = os.path.join(base_dir, "data", "processed", "scaled")
    raw_dir = os.path.join(base_dir, "data", "processed")
    scalers_dir = os.path.join(base_dir, "models", "scalers")

    target_scaler = joblib.load(os.path.join(scalers_dir, "target_scaler.joblib"))

    X_val_scaled = np.load(os.path.join(scaled_dir, "X_val_scaled.npy"))   # [4493, 72, 49]
    y_val_raw = np.load(os.path.join(raw_dir, "y_val.npy")).squeeze(-1)    # [4493, 72]

    X_test_scaled = np.load(os.path.join(scaled_dir, "X_test_scaled.npy")) # [3393, 72, 49]
    y_test_raw = np.load(os.path.join(raw_dir, "y_test.npy")).squeeze(-1)  # [3393, 72]
    
    X_val_raw = np.load(os.path.join(raw_dir, "X_val.npy"))               # [4493, 72, 49]
    X_test_raw = np.load(os.path.join(raw_dir, "X_test.npy"))             # [3393, 72, 49]

    # Load frozen deep model
    ckpt_path = os.path.join(base_dir, "frozen_model", "proposed_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    deep_model = CoupledMultiBranchForecastModel()
    deep_model.load_state_dict(ckpt["model_state_dict"])
    deep_model.eval()

    # 2. Generate Validation Predictions for Deep Model
    print("2. Generating deep model predictions on validation set (4,493 sequences)...")
    val_preds_list = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(X_val_scaled), batch_size):
            batch_x = torch.tensor(X_val_scaled[i:i+batch_size], dtype=torch.float32)
            out = deep_model(batch_x)
            val_preds_list.append(out.numpy().squeeze(-1))
    val_preds_scaled = np.concatenate(val_preds_list, axis=0) # [4493, 72]
    val_preds_raw = target_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).reshape(-1, 72)
    val_preds_raw = np.maximum(0.0, val_preds_raw)

    raw_val_mae = float(np.mean(np.abs(val_preds_raw - y_val_raw)))
    print(f"   -> Deep Model Raw Val MAE: {raw_val_mae:.2f} µg/m³")

    # 3. Train Extreme-Event Classifier
    print("3. Training Imbalance-Aware Extreme-Pollution Classifier...")
    # Features for extreme event: last observed PM2.5, min wind, max fire FRP, hour, month
    val_features_extreme = np.column_stack([
        X_val_raw[:, -1, 0],   # recent PM2.5
        X_val_raw[:, -1, 11],  # wind speed
        X_val_raw[:, -1, 33],  # fire FRP 50km
        X_val_raw[:, -1, 39],  # hour sin
        X_val_raw[:, -1, 45]   # month sin
    ])
    # Target: max future PM2.5 >= 250 (Severe AQI > 300)
    y_val_max = np.max(y_val_raw, axis=1)
    
    ext_classifier = ExtremePollutionClassifier()
    ext_classifier.fit(val_features_extreme, y_val_max, threshold=250.0)
    print("   -> Extreme-Event Classifier saved successfully.")

    # 4. Train Observation-Driven XGBoost Residual Corrector
    print("4. Training Observation-Driven XGBoost Residual Corrector...")
    # Construct tabular dataset of (prediction, horizon, recent_obs, recent_error, meteorology) -> residual
    # Sample every 6th sequence to train efficiently
    step_sample = 4
    residual_rows = []
    residual_targets = []

    for idx in range(0, len(val_preds_raw), step_sample):
        recent_obs = X_val_raw[idx, -1, 0]
        recent_pred = val_preds_raw[idx, 0] # +1h estimate
        recent_err = recent_obs - recent_pred
        
        cams_pm = X_val_raw[idx, -1, 29]
        cams_aod = X_val_raw[idx, -1, 31]
        fire_frp = X_val_raw[idx, -1, 33]
        fire_cnt = X_val_raw[idx, -1, 32]

        for h in [0, 5, 11, 23, 47, 71]: # Horizon steps +1h, +6h, +12h, +24h, +48h, +72h
            raw_pred_h = val_preds_raw[idx, h]
            actual_h = y_val_raw[idx, h]
            residual_h = actual_h - raw_pred_h
            
            # Simulated WRF meteorology from input slice
            t_c = X_val_raw[idx, -1, 9]
            rh = X_val_raw[idx, -1, 10]
            ws = X_val_raw[idx, -1, 11]
            u_w = X_val_raw[idx, -1, 18]
            v_w = X_val_raw[idx, -1, 19]
            pbl = 350.0 # Standard winter pbl
            
            h_sin = np.sin(2.0 * np.pi * ((h + 1) % 24) / 24.0)
            h_cos = np.cos(2.0 * np.pi * ((h + 1) % 24) / 24.0)
            doy_sin = X_val_raw[idx, -1, 43]
            doy_cos = X_val_raw[idx, -1, 44]

            feature_vec = [
                raw_pred_h, float(h + 1), recent_obs, recent_err,
                t_c, rh, ws, u_w, v_w, pbl,
                cams_pm, cams_aod, fire_frp, fire_cnt,
                h_sin, h_cos, doy_sin, doy_cos
            ]
            residual_rows.append(feature_vec)
            residual_targets.append(residual_h)

    X_res = np.array(residual_rows, dtype=np.float32)
    y_res = np.array(residual_targets, dtype=np.float32)

    corrector = ObservationResidualCorrector()
    corrector.fit(X_res, y_res)
    print(f"   -> Residual Corrector trained on {len(X_res)} tabular instances.")

    # 5. Evaluate on Held-Out 2023 Test Set
    print("\n5. Evaluating Deep Model vs Residual Corrected Model on untouched 2023 Test Set...")
    # Deep model test predictions
    test_preds_raw = np.load(os.path.join(base_dir, "results", "proposed_test_predictions.npy")).squeeze(-1) # [3393, 72]
    
    # Apply Residual Correction
    corrected_test_preds = np.zeros_like(test_preds_raw)
    for idx in range(len(test_preds_raw)):
        recent_obs = X_test_raw[idx, -1, 0]
        recent_err = recent_obs - test_preds_raw[idx, 0]
        cams_pm = X_test_raw[idx, -1, 29]
        cams_aod = X_test_raw[idx, -1, 31]
        fire_frp = X_test_raw[idx, -1, 33]
        fire_cnt = X_test_raw[idx, -1, 32]
        
        # Build 72-row matrix
        rows = []
        for h in range(72):
            rows.append([
                float(test_preds_raw[idx, h]), float(h + 1), float(recent_obs), float(recent_err),
                float(X_test_raw[idx, -1, 9]), float(X_test_raw[idx, -1, 10]), float(X_test_raw[idx, -1, 11]),
                float(X_test_raw[idx, -1, 18]), float(X_test_raw[idx, -1, 19]), 350.0,
                float(cams_pm), float(cams_aod), float(fire_frp), float(fire_cnt),
                float(np.sin(2.0 * np.pi * ((h + 1) % 24) / 24.0)),
                float(np.cos(2.0 * np.pi * ((h + 1) % 24) / 24.0)),
                float(X_test_raw[idx, -1, 43]), float(X_test_raw[idx, -1, 44])
            ])
        res_pred = corrector.predict_residual(np.array(rows, dtype=np.float32))
        corrected_test_preds[idx] = np.maximum(0.0, test_preds_raw[idx] + res_pred)

    # Calculate overall & regime metrics
    def calc_metrics(y_true, y_pred):
        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / max(1e-6, ss_tot)))
        wmape = float((np.sum(np.abs(err)) / np.sum(y_true)) * 100.0)
        return {"mae": round(mae, 2), "rmse": round(rmse, 2), "bias": round(bias, 2), "r2": round(r2, 4), "wmape": round(wmape, 2)}

    deep_overall = calc_metrics(y_test_raw, test_preds_raw)
    corr_overall = calc_metrics(y_test_raw, corrected_test_preds)

    print("\n--- OVERALL TEST COMPARISON ---")
    print(f"Deep Model      : MAE = {deep_overall['mae']} µg/m³, RMSE = {deep_overall['rmse']} µg/m³, Bias = {deep_overall['bias']} µg/m³, R2 = {deep_overall['r2']}")
    print(f"Corrected Model : MAE = {corr_overall['mae']} µg/m³, RMSE = {corr_overall['rmse']} µg/m³, Bias = {corr_overall['bias']} µg/m³, R2 = {corr_overall['r2']}")

    # Regime breakdown (especially severe >= 345 µg/m³)
    mask_severe = (y_test_raw >= 345.0)
    deep_severe = calc_metrics(y_test_raw[mask_severe], test_preds_raw[mask_severe])
    corr_severe = calc_metrics(y_test_raw[mask_severe], corrected_test_preds[mask_severe])

    print("\n--- SEVERE REGIME (>= 345 µg/m³, Critical Failure Mode) ---")
    print(f"Ground Truth Mean Actual: {np.mean(y_test_raw[mask_severe]):.2f} µg/m³ (N={np.sum(mask_severe)})")
    print(f"Deep Model Mean Pred    : {np.mean(test_preds_raw[mask_severe]):.2f} µg/m³ (MAE = {deep_severe['mae']} µg/m³, Bias = {deep_severe['bias']} µg/m³)")
    print(f"Corrected Model Mean    : {np.mean(corrected_test_preds[mask_severe]):.2f} µg/m³ (MAE = {corr_severe['mae']} µg/m³, Bias = {corr_severe['bias']} µg/m³)")

    # Save corrected test predictions
    np.save(os.path.join(base_dir, "results", "corrected_test_predictions.npy"), corrected_test_preds)
    
    # Save evaluation summary
    results_summary = {
        "deep_model": deep_overall,
        "corrected_model": corr_overall,
        "deep_severe_regime": deep_severe,
        "corrected_severe_regime": corr_severe,
        "severe_bias_reduction_ugm3": round(abs(deep_severe["bias"]) - abs(corr_severe["bias"]), 2),
        "severe_mae_reduction_pct": round(((deep_severe["mae"] - corr_severe["mae"]) / deep_severe["mae"]) * 100.0, 2)
    }
    with open(os.path.join(base_dir, "results", "v2_evaluation_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\nTraining and Evaluation Complete! Summary saved to results/v2_evaluation_summary.json")

if __name__ == "__main__":
    train_and_evaluate_components()
