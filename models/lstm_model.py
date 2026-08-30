import torch
import torch.nn as nn

class LSTMForecastModel(nn.Module):
    """
    Direct Sequence-to-Sequence LSTM Baseline Model for Air Quality PM2.5 Forecasting.
    
    Input shape:  [batch_size, sequence_length=72, input_size=49]
    Output shape: [batch_size, forecast_length=72, target_size=1]
    """
    def __init__(self, input_size=49, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMForecastModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        # x shape: [B, 72, 49]
        lstm_out, _ = self.lstm(x) # lstm_out shape: [B, 72, 128]
        out = self.fc(lstm_out)    # out shape: [B, 72, 1]
        return out

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

if __name__ == "__main__":
    model = LSTMForecastModel()
    dummy_input = torch.randn(4, 72, 49)
    output = model(dummy_input)
    tot, trn = model.count_parameters()
    print("LSTM Forecast Model Initialized Successfully")
    print(f"Input Shape  : {dummy_input.shape}")
    print(f"Output Shape : {output.shape}")
    print(f"Total Params : {tot:,}")
    print(f"Trainable    : {trn:,}")
