import os
import sys
import torch
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.lstm_model import LSTMForecastModel

def test_lstm_model():
    print("--- RUNNING LSTM MODEL UNIT TEST ---")
    batch_size = 4
    seq_len = 72
    input_size = 49
    target_size = 1
    
    model = LSTMForecastModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        output_size=target_size
    )
    
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    output = model(dummy_input)
    
    assert output.shape == (batch_size, seq_len, target_size), f"Expected shape {(batch_size, seq_len, target_size)}, got {output.shape}"
    assert not torch.isnan(output).any(), "NaN values found in LSTM model output!"
    assert not torch.isinf(output).any(), "Inf values found in LSTM model output!"
    
    tot_params, trn_params = model.count_parameters()
    print(f"Dummy Input Shape  : {dummy_input.shape}")
    print(f"Output Shape       : {output.shape}")
    print(f"Total Parameters   : {tot_params:,}")
    print(f"Trainable Params   : {trn_params:,}")
    print("LSTM Model Unit Test: PASS\n")

if __name__ == "__main__":
    test_lstm_model()
