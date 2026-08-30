import os
import sys
import json
import torch
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.proposed_model import CoupledMultiBranchForecastModel

def test_proposed_training_integrity():
    print("==================================================")
    print("B12 TRAINING INTEGRITY AUDIT SUITE (7 CHECKS)")
    print("==================================================\n")
    
    ckpt_path = os.path.join(base_dir, "models", "checkpoints", "proposed_best.pt")
    hist_path = os.path.join(base_dir, "results", "proposed_training_history.csv")
    
    # 1. Checkpoint Loads Successfully
    print("Check 1: Best Checkpoint File Existence & Loading...", end="")
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}!"
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    assert "model_state_dict" in checkpoint, "Missing model_state_dict in checkpoint!"
    print(" PASS")
    
    # 2. No NaNs/Infs in Model Parameters
    print("Check 2: Zero NaNs & Zero Infs in Checkpoint Weights...", end="")
    state_dict = checkpoint["model_state_dict"]
    for param_name, param_tensor in state_dict.items():
        assert not torch.isnan(param_tensor).any(), f"NaN found in parameter {param_name}"
        assert not torch.isinf(param_tensor).any(), f"Inf found in parameter {param_name}"
    print(" PASS")
    
    # 3. No NaNs/Infs in Training History CSV
    print("Check 3: Zero NaNs & Zero Infs in Training History CSV...", end="")
    assert os.path.exists(hist_path), f"Training history CSV not found at {hist_path}!"
    history_df = pd.read_csv(hist_path)
    assert not history_df.isna().any().any(), "NaN values found in training history CSV!"
    assert np.isfinite(history_df.values).all(), "Inf values found in training history CSV!"
    print(" PASS")
    
    # 4. Best Epoch Corresponds to Minimum Validation Loss
    print("Check 4: Best Epoch Corresponds to Minimum Validation Loss...", end="")
    min_val_loss_hist = float(history_df["val_loss"].min())
    best_epoch_hist = int(history_df.loc[history_df["val_loss"].idxmin(), "epoch"])
    ckpt_best_epoch = int(checkpoint["epoch"])
    ckpt_best_val_loss = float(checkpoint["best_val_loss"])
    
    assert best_epoch_hist == ckpt_best_epoch, f"Mismatch in best epoch: history={best_epoch_hist}, ckpt={ckpt_best_epoch}"
    assert abs(min_val_loss_hist - ckpt_best_val_loss) < 1e-5, f"Mismatch in best val loss: history={min_val_loss_hist}, ckpt={ckpt_best_val_loss}"
    print(f" PASS (Best Epoch: {best_epoch_hist}, Val Loss: {ckpt_best_val_loss:.6f})")
    
    # 5. Architecture Matches B10 Exactly & 6. Parameter Count Verification
    print("Check 5 & 6: Architecture Class & Parameter Count (819,874)...", end="")
    model = CoupledMultiBranchForecastModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    tot_params, _ = model.count_parameters()
    assert tot_params == 819874, f"Expected 819,874 parameters, got {tot_params:,}"
    print(" PASS")
    
    # 7. Test Set Prediction Deliverable Verification
    print("Check 7: Verified B13 Test Predictions Deliverable...", end="")
    test_pred_file = os.path.join(base_dir, "results", "proposed_test_predictions.npy")
    assert os.path.exists(test_pred_file), "B13 test predictions array missing!"
    preds = np.load(test_pred_file)
    assert preds.shape == (3393, 72, 1), f"Unexpected test predictions shape: {preds.shape}"
    print(" PASS (B13 Deliverable Array Verified: 3,393 x 72 x 1)")
    
    print("\n==================================================")
    print("ALL 7 TRAINING INTEGRITY AUDIT CHECKS PASSED!")
    print("B12 STATUS: FULL TRAINING COMPLETED & VERIFIED")
    print("==================================================\n")
    return True

if __name__ == "__main__":
    test_proposed_training_integrity()
