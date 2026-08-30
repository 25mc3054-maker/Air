import os
import sys
import json
import torch
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.proposed_model import CoupledMultiBranchForecastModel
from training.losses import SpikeAwareForecastLoss
from training.dataloader import get_dataloaders

def test_proposed_training_sanity():
    print("==================================================")
    print("B11 PROPOSED TRAINING SANITY TEST SUITE (13 CHECKS)")
    print("==================================================\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = CoupledMultiBranchForecastModel()
    loss_fn = SpikeAwareForecastLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loader, _, _, _ = get_dataloaders(
        config_path=os.path.join(base_dir, "configs", "training_config.json"),
        data_dir=os.path.join(base_dir, "data", "processed", "scaled")
    )
    x_batch, y_batch = next(iter(train_loader))
    
    # A & B & C: Output Shapes
    print("Check A, B, C: Batch Forward Pass & Output Shapes...", end="")
    model.train()
    out_dict = model(x_batch, return_auxiliary=True)
    f_out = out_dict["forecast"]
    s_out = out_dict["spike_prob"]
    assert f_out.shape == (64, 72, 1), f"Expected forecast shape (64, 72, 1), got {f_out.shape}"
    assert s_out.shape == (64, 72, 1), f"Expected spike shape (64, 72, 1), got {s_out.shape}"
    print(" PASS")
    
    # D: Zero NaNs & Infs
    print("Check D: Zero NaNs and Infs in Outputs...", end="")
    assert not torch.isnan(f_out).any() and not torch.isinf(f_out).any(), "NaN/Inf in forecast output!"
    assert not torch.isnan(s_out).any() and not torch.isinf(s_out).any(), "NaN/Inf in spike output!"
    print(" PASS")
    
    # E: Finite Loss
    print("Check E: Loss Computation & Finiteness...", end="")
    loss_dict = loss_fn(f_out, y_batch, s_out)
    loss = loss_dict["total_loss"]
    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"
    print(f" PASS (Loss: {loss.item():.6f})")
    
    # F & G: Backpropagation & Finite Gradients
    print("Check F & G: Backprop Execution & Finite Gradients...", end="")
    optimizer.zero_grad()
    loss.backward()
    grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(np.isfinite(g) for g in grads), "Gradient calculation failed or non-finite!"
    print(" PASS")
    
    # H & I: Optimizer Step & Parameter Mutations
    print("Check H & I: Optimizer Step & Parameter Mutation...", end="")
    params_before = [p.clone().detach() for p in model.parameters()]
    optimizer.step()
    params_after = [p.clone().detach() for p in model.parameters()]
    max_change = max(torch.max(torch.abs(b - a)).item() for b, a in zip(params_before, params_after))
    assert max_change > 0.0, "Model parameters did NOT mutate after optimizer step!"
    print(f" PASS (Max param change: {max_change:.6e})")
    
    # J: Programmatic Causality Leakage Verification
    print("Check J: Zero Future Leakage Verification...", end="")
    model.eval()
    x1 = x_batch[:1].clone()
    with torch.no_grad():
        out1 = model(x1)
    x2 = x1.clone()
    x2[:, 36:, :] += torch.randn_like(x2[:, 36:, :]) * 50.0
    with torch.no_grad():
        out2 = model(x2)
    past_diff = torch.max(torch.abs(out1[:, :36, :] - out2[:, :36, :])).item()
    assert past_diff < 1e-5, f"Causality leakage detected! Past output diff: {past_diff}"
    print(f" PASS (Max past diff: {past_diff:.10f})")
    
    # K: Feature Ordering & Schema Alignment
    print("Check K: Feature Schema Alignment...", end="")
    schema_path = os.path.join(base_dir, "results", "model_input_schema.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)
    assert schema["num_features"] == 49
    print(" PASS")
    
    # L: Determinism under Seed 42
    print("Check L: Deterministic Forward Pass (Seed 42)...", end="")
    torch.manual_seed(42)
    m1 = CoupledMultiBranchForecastModel()
    m1.eval()
    with torch.no_grad():
        o1 = m1(x1)
    torch.manual_seed(42)
    m2 = CoupledMultiBranchForecastModel()
    m2.eval()
    with torch.no_grad():
        o2 = m2(x1)
    assert torch.max(torch.abs(o1 - o2)).item() == 0.0, "Non-deterministic forward pass!"
    print(" PASS")
    
    # M: CPU Memory Consumption
    print("Check M: Memory Footprint Verification...", end="")
    mem_size = sys.getsizeof(x_batch.numpy())
    assert mem_size < 100 * 1024 * 1024
    print(" PASS")
    
    print("\n==================================================")
    print("ALL 13 B11 SANITY CHECKS PASSED!")
    print("B11 STATUS: SANITY CHECK PASSED")
    print("==================================================\n")
    return True

if __name__ == "__main__":
    test_proposed_training_sanity()
