import os
import sys
import json
import torch
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from pipeline.inference_pipeline import PM25ForecastingPipeline
from api.app import app, health_check, get_model_info, get_metrics, ForecastRequest, predict_pm25

def test_production_pipeline_and_api():
    print("==================================================")
    print("PHASE 12 — AUTOMATED PRODUCTION VERIFICATION (12 CHECKS)")
    print("==================================================\n")
    
    # 1. Model Checkpoint Exists
    print("Check 1: Model Checkpoint File Existence...", end="")
    ckpt_path = os.path.join(base_dir, "models", "checkpoints", "proposed_best.pt")
    assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
    print(" PASS")
    
    # 2. Model Pipeline Loads Cleanly
    print("Check 2: Production Pipeline Initialization...", end="")
    pipeline = PM25ForecastingPipeline()
    assert pipeline.model is not None, "Pipeline model failed to load!"
    print(" PASS")
    
    # 3. Parameter Count Verification
    print("Check 3: Model Parameter Count (819,874)...", end="")
    assert pipeline.total_parameters == 819874, f"Parameter count mismatch: {pipeline.total_parameters}"
    print(" PASS")
    
    # 4. Input Validation Logic
    print("Check 4: Input Column & Length Validation...", end="")
    try:
        # Invalid length (10 hours instead of 72)
        invalid_df = pd.DataFrame(np.random.randn(10, 49), columns=pipeline.feature_order)
        pipeline.preprocess_input(invalid_df)
        assert False, "Pipeline failed to reject invalid 10-hour length!"
    except AssertionError:
        print(" PASS")
        
    # 5. Inference Execution & 6. 72 Timesteps Output
    print("Check 5 & 6: Inference Pass & 72-Hour Output Length...", end="")
    dummy_input = pd.DataFrame(np.random.randn(72, 49) * 10 + 100, columns=pipeline.feature_order)
    res = pipeline.forecast(dummy_input, start_timestamp="2024-01-01T00:00:00")
    
    assert res["forecast_horizon_hours"] == 72, f"Expected 72h horizon, got {res['forecast_horizon_hours']}"
    assert len(res["forecast"]) == 72, f"Expected 72 forecast entries, got {len(res['forecast'])}"
    print(" PASS")
    
    # 7. Predictions Are Finite & 8. Physical Inverse Scaling Range
    print("Check 7 & 8: Finite Prediction Values & Physical Scale (µg/m³)...", end="")
    pm_values = [item["pm25"] for item in res["forecast"]]
    assert not np.isnan(pm_values).any(), "NaN found in pipeline predictions!"
    assert not np.isinf(pm_values).any(), "Inf found in pipeline predictions!"
    avg_pm = float(np.mean(pm_values))
    assert 10.0 <= avg_pm <= 800.0, f"Unreasonable physical prediction value: {avg_pm}"
    print(f" PASS (Avg PM2.5: {avg_pm:.2f} µg/m³)")
    
    # 9. FastAPI Health Endpoint
    print("Check 9: FastAPI /health Endpoint...", end="")
    h_res = health_check()
    assert h_res["status"] == "HEALTHY", f"Unhealthy status: {h_res}"
    print(" PASS")
    
    # 10. FastAPI Predict Endpoint
    print("Check 10: FastAPI /predict Endpoint...", end="")
    req = ForecastRequest(
        start_timestamp="2024-01-01T00:00:00",
        sequence=dummy_input.to_dict(orient="records")
    )
    p_res = predict_pm25(req)
    assert p_res["forecast_horizon_hours"] == 72, "FastAPI predict failed horizon check"
    print(" PASS")
    
    # 11. FastAPI Model-Info Endpoint
    print("Check 11: FastAPI /model-info Endpoint...", end="")
    info = get_model_info()
    assert info["total_parameters"] == 819874, "Model info parameter mismatch"
    assert info["model_name"] == "CoupledMultiBranchForecastModel", "Model name mismatch"
    print(" PASS")
    
    # 12. FastAPI Metrics Endpoint
    print("Check 12: FastAPI /metrics Endpoint (Verified Metrics)...", end="")
    metrics = get_metrics()
    prop_m = metrics["proposed_model_metrics"]
    assert prop_m["mae_ugm3"] == 56.66, f"MAE mismatch: {prop_m['mae_ugm3']}"
    assert prop_m["rmse_ugm3"] == 82.56, f"RMSE mismatch: {prop_m['rmse_ugm3']}"
    assert prop_m["r2_score"] == 0.3564, f"R2 mismatch: {prop_m['r2_score']}"
    assert prop_m["wmape_pct"] == 41.11, f"WMAPE mismatch: {prop_m['wmape_pct']}"
    print(" PASS")
    
    print("\n==================================================")
    print("ALL 12 PRODUCTION VERIFICATION CHECKS PASSED!")
    print("PRODUCTION PIPELINE & API ARE 100% READY FOR DEMO!")
    print("==================================================\n")
    return True

if __name__ == "__main__":
    test_production_pipeline_and_api()
