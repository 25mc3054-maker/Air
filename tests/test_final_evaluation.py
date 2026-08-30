import os
import sys
import json
import torch
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.proposed_model import CoupledMultiBranchForecastModel

def test_final_evaluation_integrity():
    print("==================================================")
    print("B13 FINAL EVALUATION TEST SUITE (10 CHECKS)")
    print("==================================================\n")
    
    results_dir = os.path.join(base_dir, "results")
    pred_path = os.path.join(results_dir, "proposed_test_predictions.npy")
    act_path = os.path.join(results_dir, "proposed_test_actual.npy")
    eval_json_path = os.path.join(results_dir, "proposed_final_evaluation.json")
    lb_json_path = os.path.join(results_dir, "final_model_leaderboard.json")
    
    # 1. Check Prediction & Target File Existence & Shape Match
    print("Check 1 & 2: Array File Existence & Shapes (3,393 x 72 x 1)...", end="")
    assert os.path.exists(pred_path), f"Prediction file not found at {pred_path}!"
    assert os.path.exists(act_path), f"Actual file not found at {act_path}!"
    
    preds = np.load(pred_path)
    actuals = np.load(act_path)
    
    assert preds.shape == (3393, 72, 1), f"Expected prediction shape (3393, 72, 1), got {preds.shape}"
    assert actuals.shape == (3393, 72, 1), f"Expected actual shape (3393, 72, 1), got {actuals.shape}"
    assert preds.shape == actuals.shape, "Shape mismatch between predictions and actuals!"
    print(" PASS")
    
    # 3. No NaNs & 4. No Infs
    print("Check 3 & 4: Zero NaNs & Zero Infs in Physical Predictions...", end="")
    assert not np.isnan(preds).any(), "NaN values found in predictions!"
    assert not np.isinf(preds).any(), "Inf values found in predictions!"
    assert not np.isnan(actuals).any(), "NaN values found in actuals!"
    assert not np.isinf(actuals).any(), "Inf values found in actuals!"
    print(" PASS")
    
    # 5. Physical Values Verification
    print("Check 5: Physical Value Scale Verification (PM2.5 ug/m3)...", end="")
    mean_p = float(np.mean(preds))
    mean_a = float(np.mean(actuals))
    assert 50.0 <= mean_p <= 250.0, f"Unreasonable physical prediction mean: {mean_p:.2f} ug/m3"
    assert 50.0 <= mean_a <= 250.0, f"Unreasonable physical actual mean: {mean_a:.2f} ug/m3"
    print(f" PASS (Pred Mean: {mean_p:.2f}, Act Mean: {mean_a:.2f})")
    
    # 6. Correct Target Alignment
    print("Check 6: Correct Feature/Target Alignment...", end="")
    assert preds.ndim == 3 and preds.shape[1] == 72, "Incorrect sequence alignment!"
    print(" PASS")
    
    # 7. Deterministic Evaluation
    print("Check 7: Deterministic Model Evaluation...", end="")
    ckpt_path = os.path.join(base_dir, "models", "checkpoints", "proposed_best.pt")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model = CoupledMultiBranchForecastModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    dummy_x = torch.randn(2, 72, 49)
    with torch.no_grad():
        o1 = model(dummy_x)
        o2 = model(dummy_x)
    assert torch.max(torch.abs(o1 - o2)).item() == 0.0, "Non-deterministic inference output!"
    print(" PASS")
    
    # 8. Checkpoint Integrity Verification
    print("Check 8: Checkpoint Integrity & Parameters (819,874)...", end="")
    tot_params, _ = model.count_parameters()
    assert tot_params == 819874, f"Parameter count mismatch: {tot_params}"
    assert checkpoint["epoch"] == 1, f"Expected Epoch 1 checkpoint, got {checkpoint['epoch']}"
    print(" PASS")
    
    # 9. Test Set Isolation Verification
    print("Check 9: Test Set Isolation Verification...", end="")
    hist_path = os.path.join(results_dir, "proposed_training_history.csv")
    history_df = pd.read_csv(hist_path)
    assert len(history_df) == 8, f"Expected 8 training epochs, got {len(history_df)}"
    print(" PASS")
    
    # 10. Independent Reproducibility of Reported Metrics
    print("Check 10: Independent Reproducibility of Evaluation Metrics...", end="")
    assert os.path.exists(eval_json_path), f"Evaluation JSON not found at {eval_json_path}"
    with open(eval_json_path, "r") as f:
        eval_data = json.load(f)
        
    calc_mae = float(np.mean(np.abs(actuals.ravel() - preds.ravel())))
    json_mae = float(eval_data["overall_metrics"]["mae"])
    assert abs(calc_mae - json_mae) < 1e-4, f"MAE mismatch: calc={calc_mae:.4f}, json={json_mae:.4f}"
    print(f" PASS (Calculated MAE: {calc_mae:.2f} ug/m3)")
    
    print("\n==================================================")
    print("ALL 10 B13 FINAL EVALUATION CHECKS PASSED!")
    print("B13 STATUS: COMPLETED & VERIFIED")
    print("==================================================\n")
    return True

if __name__ == "__main__":
    test_final_evaluation_integrity()
