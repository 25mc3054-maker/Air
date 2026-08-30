import os
import sys
import json
import torch
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.proposed_model import CoupledMultiBranchForecastModel
from training.dataloader import get_dataloaders

def test_proposed_architecture():
    print("==================================================")
    print("B10 ARCHITECTURE VALIDATION SUITE (12/12 CHECKS)")
    print("==================================================\n")
    
    # 1. Model Import & 2. Model Instantiation
    print("Check 1 & 2: Model Import & Instantiation...", end="")
    config_path = os.path.join(base_dir, "configs", "proposed_architecture.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    model = CoupledMultiBranchForecastModel(
        input_size=config["input_size"],
        forecast_horizon=config["forecast_horizon"],
        dropout=config["dropout"]
    )
    print(" PASS")
    
    # 3. Input Shape & 4. Output Shape & 6. Float32 Compatibility
    print("Check 3, 4, 6: Input [B, 72, 49] -> Output [B, 72, 1] (Float32)...", end="")
    dummy_input = torch.randn(64, 72, 49, dtype=torch.float32)
    output = model(dummy_input)
    assert output.shape == (64, 72, 1), f"Expected shape (64, 72, 1), got {output.shape}"
    assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
    print(" PASS")
    
    # 5. Batch-Size Flexibility
    print("Check 5: Batch-Size Flexibility (B=1, B=16, B=64)...", end="")
    model.eval()
    for b_size in [1, 16, 64]:
        b_in = torch.randn(b_size, 72, 49)
        b_out = model(b_in)
        assert b_out.shape == (b_size, 72, 1), f"Failed for batch size {b_size}"
    print(" PASS")
    
    # 7. No NaN Output & 8. No Inf Output
    print("Check 7 & 8: Zero NaNs & Zero Infs...", end="")
    assert not torch.isnan(output).any(), "NaN values detected in model output!"
    assert not torch.isinf(output).any(), "Inf values detected in model output!"
    print(" PASS")
    
    # 9. Parameter Count Verification
    print("Check 9: Parameter Count Verification (Target: 0.8M - 2.0M)...", end="")
    tot_params, trn_params = model.count_parameters()
    budget_min = config["parameter_budget"]["target_min"]
    budget_max = config["parameter_budget"]["target_max"]
    assert budget_min <= tot_params <= budget_max, f"Parameter count {tot_params:,} outside budget [{budget_min:,}, {budget_max:,}]!"
    print(f" PASS ({tot_params:,} parameters)")
    
    # 10. Strict Causal Behavior Test
    print("Check 10: Programmatic Causal Leakage Test...", end="")
    model.eval()
    x1 = torch.randn(1, 72, 49)
    with torch.no_grad():
        out1 = model(x1)
        
    t_cutoff = 35
    x2 = x1.clone()
    x2[:, t_cutoff + 1:, :] += torch.randn_like(x2[:, t_cutoff + 1:, :]) * 100.0 + 50.0
    
    with torch.no_grad():
        out2 = model(x2)
        
    past_diff = torch.max(torch.abs(out1[:, :t_cutoff + 1, :] - out2[:, :t_cutoff + 1, :])).item()
    assert past_diff < 1e-5, f"Causality violation! Past output changed by {past_diff}"
    print(f" PASS (Max past output diff: {past_diff:.10f})")
    
    # 11. Deterministic Forward Pass Under Seed 42
    print("Check 11: Deterministic Forward Pass (Seed 42)...", end="")
    torch.manual_seed(42)
    in_det = torch.randn(2, 72, 49)
    
    torch.manual_seed(42)
    m1 = CoupledMultiBranchForecastModel()
    m1.eval()
    with torch.no_grad():
        out_det1 = m1(in_det)
    
    torch.manual_seed(42)
    m2 = CoupledMultiBranchForecastModel()
    m2.eval()
    with torch.no_grad():
        out_det2 = m2(in_det)
    
    det_diff = torch.max(torch.abs(out_det1 - out_det2)).item()
    assert det_diff == 0.0, f"Non-deterministic outputs! Max diff: {det_diff}"
    print(" PASS")
    
    # 12. Compatibility with Existing B5 DataLoader
    print("Check 12: Compatibility with Verified B5 DataLoaders...", end="")
    train_loader, _, _, _ = get_dataloaders(
        config_path=os.path.join(base_dir, "configs", "training_config.json"),
        data_dir=os.path.join(base_dir, "data", "processed", "scaled")
    )
    real_x, real_y = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        real_out = model(real_x)
    assert real_out.shape == (real_x.size(0), 72, 1)
    print(" PASS")
    
    print("\n==================================================")
    print("ALL 12 ARCHITECTURE VALIDATION CHECKS PASSED!")
    print("B10 STATUS: COMPLETED & VERIFIED")
    print("==================================================\n")
    return True

if __name__ == "__main__":
    test_proposed_architecture()
