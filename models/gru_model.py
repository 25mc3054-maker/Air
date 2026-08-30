import torch
import torch.nn as nn

class GRUForecastModel(nn.Module):
    """
    Direct Sequence-to-Sequence GRU model for 72-hour PM2.5 forecasting.
    Input shape:  [batch_size, 72, 49]
    Output shape: [batch_size, 72, 1]
    """
    def __init__(self, input_size=49, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super(GRUForecastModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.output_size = output_size
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: [B, 72, 49]
        gru_out, _ = self.gru(x) # [B, 72, 128]
        prediction = self.fc(gru_out) # [B, 72, 1]
        return prediction

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

if __name__ == "__main__":
    model = GRUForecastModel()
    tot, trn = model.count_parameters()
    dummy_x = torch.randn(4, 72, 49)
    dummy_y = model(dummy_x)
    print(f"GRU Model instantiated cleanly.")
    print(f"Dummy input shape : {dummy_x.shape}")
    print(f"Dummy output shape: {dummy_y.shape}")
    print(f"Total Parameters  : {tot:,}")
    print(f"Trainable Params  : {trn:,}")
