import os
import sys
import torch
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.tcn_model import TCNForecastModel

def verify_causality():
    print("--- RUNNING PROGRAMMATIC CAUSALITY TEST ---")
    
    torch.manual_seed(42)
    model = TCNForecastModel(
        input_size=49,
        num_channels=[128, 128, 128, 128, 128, 128],
        dilations=[1, 2, 4, 8, 16, 32],
        kernel_size=3,
        dropout=0.2,
        target_size=1
    )
    model.eval()
    
    # 1. Generate base input: shape [1, 72, 49]
    x1 = torch.randn(1, 72, 49)
    
    # 2. Get baseline model predictions
    with torch.no_grad():
        out1 = model(x1) # [1, 72, 1]
        
    # 3. Test causality at multiple cutoff timesteps (e.g. t_cutoff = 20, 40, 60)
    cutoffs = [10, 20, 35, 50, 65]
    all_passed = True
    
    for t_cutoff in cutoffs:
        # Create perturbed input where future timesteps (> t_cutoff) are heavily altered
        x2 = x1.clone()
        x2[:, t_cutoff + 1:, :] += torch.randn_like(x2[:, t_cutoff + 1:, :]) * 100.0 + 50.0
        
        with torch.no_grad():
            out2 = model(x2)
            
        # Check outputs for all timesteps <= t_cutoff
        past_diff = torch.max(torch.abs(out1[:, :t_cutoff + 1, :] - out2[:, :t_cutoff + 1, :])).item()
        future_diff = torch.max(torch.abs(out1[:, t_cutoff + 1:, :] - out2[:, t_cutoff + 1:, :])).item()
        
        print(f"  Cutoff t = {t_cutoff:2d} | Max Past Output Difference (<= t): {past_diff:.10f} | Max Future Diff (> t): {future_diff:.4f}")
        
        if past_diff > 1e-6:
            all_passed = False
            print(f"  [ERROR] Causality Violated at t = {t_cutoff}! Past output changed by {past_diff}")
            
    if all_passed:
        print("\n```\nCAUSALITY TEST: PASS (Zero future leakage detected)\n```\n")
        return True
    else:
        print("\n```\nB8 STATUS: NOT READY — CAUSALITY FAILURE\n```\n")
        sys.exit(1)

if __name__ == "__main__":
    verify_causality()
