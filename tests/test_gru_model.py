import os
import sys
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.gru_model import GRUForecastModel

def test_gru_architecture():
    print("--- TESTING GRU ARCHITECTURE ---")
    model = GRUForecastModel(input_size=49, hidden_size=128, num_layers=2, dropout=0.2, output_size=1)
    tot, trn = model.count_parameters()
    print(f"Total Parameters    : {tot:,}")
    print(f"Trainable Parameters: {trn:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 72, 49)
    output = model(dummy_input)
    
    print(f"Input Shape : {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    assert output.shape == (4, 72, 1), f"Expected shape (4, 72, 1), got {output.shape}"
    assert torch.isnan(output).sum() == 0, "Output contains NaNs!"
    assert torch.isinf(output).sum() == 0, "Output contains Infs!"
    
    print("GRU Model Architecture Test: PASS")

if __name__ == "__main__":
    test_gru_architecture()
