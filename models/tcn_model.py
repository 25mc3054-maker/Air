import torch
import torch.nn as nn
import torch.nn.utils as utils

class Chomp1d(nn.Module):
    """
    Slices off the trailing padding to enforce strict causal sequence alignment.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    Causal Dilated Residual TCN Block.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNForecastModel(nn.Module):
    """
    Direct Sequence-to-Sequence Causal Dilated TCN Model for 72h-to-72h PM2.5 Forecasting.
    
    Input shape:  [batch_size, sequence_length=72, input_size=49]
    Output shape: [batch_size, forecast_length=72, target_size=1]
    """
    def __init__(self, input_size=49, num_channels=[128, 128, 128, 128, 128, 128],
                 dilations=[1, 2, 4, 8, 16, 32], kernel_size=3, dropout=0.2, target_size=1):
        super(TCNForecastModel, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = dilations[i]
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout
                )
            )
            
        self.tcn = nn.Sequential(*layers)
        self.projection = nn.Conv1d(num_channels[-1], target_size, 1)
        
    def forward(self, x):
        # Input shape: [B, 72, 49]
        # Permute to [B, 49, 72] for Conv1D
        x_perm = x.permute(0, 2, 1)
        
        # TCN Residual Stack: [B, 128, 72]
        tcn_out = self.tcn(x_perm)
        
        # 1x1 Projection: [B, 1, 72]
        out_proj = self.projection(tcn_out)
        
        # Permute back to [B, 72, 1]
        out = out_proj.permute(0, 2, 1)
        return out

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

if __name__ == "__main__":
    model = TCNForecastModel()
    dummy_input = torch.randn(4, 72, 49)
    output = model(dummy_input)
    tot, trn = model.count_parameters()
    print("TCN Forecast Model Initialized Successfully")
    print(f"Input Shape  : {dummy_input.shape}")
    print(f"Output Shape : {output.shape}")
    print(f"Total Params : {tot:,}")
    print(f"Trainable    : {trn:,}")
