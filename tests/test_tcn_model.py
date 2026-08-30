import os
import sys
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.tcn_model import TCNForecastModel
from tests.test_causality import verify_causality

def test_tcn_model():
    print("--- RUNNING TCN MODEL UNIT TEST ---")
    batch_size = 4
    seq_len = 72
    input_size = 49
    target_size = 1
    
    # 1. Model Construction
    model = TCNForecastModel(
        input_size=input_size,
        num_channels=[128, 128, 128, 128, 128, 128],
        dilations=[1, 2, 4, 8, 16, 32],
        kernel_size=3,
        dropout=0.2,
        target_size=target_size
    )
    
    # 2. Causality Test
    causality_pass = verify_causality()
    assert causality_pass, "Causality test failed!"
    
    # 3. Forward Pass & Output Shape Test
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    output = model(dummy_input)
    
    assert output.shape == (batch_size, seq_len, target_size), f"Expected shape {(batch_size, seq_len, target_size)}, got {output.shape}"
    assert not torch.isnan(output).any(), "NaN values found in TCN model output!"
    assert not torch.isinf(output).any(), "Inf values found in TCN model output!"
    
    tot_params, trn_params = model.count_parameters()
    print(f"Dummy Input Shape  : {dummy_input.shape}")
    print(f"Output Shape       : {output.shape}")
    print(f"Total Parameters   : {tot_params:,}")
    print(f"Trainable Params   : {trn_params:,}")
    print("TCN Model Unit Test: PASS\n")

if __name__ == "__main__":
    test_tcn_model()
