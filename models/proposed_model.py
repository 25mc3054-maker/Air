import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """
    Slices off trailing padding to guarantee strict causality in 1D convolutions.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class CausalConvBlock(nn.Module):
    """
    Causal Dilated Convolution Block with BatchNorm and Dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(CausalConvBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.shortcut is None else self.shortcut(x)
        return self.relu(self.net(x) + res)

class CausalMultiHeadAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention over sequence length T=72.
    """
    def __init__(self, d_model=256, n_heads=4, dropout=0.2):
        super(CausalMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, T=72, D=256]
        B, T, D = x.shape
        # Create lower-triangular causal mask to prevent future attention leakage
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        attn_out, _ = self.mha(x, x, x, attn_mask=causal_mask)
        out = self.norm(x + self.dropout(attn_out))
        return out

class CoupledMultiBranchForecastModel(nn.Module):
    """
    Proposed Architecture for SIH2026 Person B Air Quality Forecasting.
    
    Multi-Branch Domain Encoding -> Cross-Feature Gated Fusion -> Causal Attention -> Dual Forecast/Spike Heads
    Input:  [B, 72, 49]
    Output: [B, 72, 1] (Primary 72h PM2.5 Forecast)
    """
    def __init__(self, input_size=49, forecast_horizon=72, dropout=0.2):
        super(CoupledMultiBranchForecastModel, self).__init__()
        
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        
        # Domain Feature Indices (Fixed 49 features)
        self.pollution_indices = list(range(0, 9))          # 9 features
        self.meteorology_indices = list(range(9, 30))       # 21 features
        self.atmospheric_indices = list(range(30, 39))     # 9 features
        self.temporal_indices = list(range(39, 49))        # 10 features (8 cyclic + 2 availability)
        
        # 1. Domain-Specific Causal Branch Encoders
        # Branch 1: Pollution (9 features -> 64 channels)
        self.pollution_branch = nn.Sequential(
            CausalConvBlock(9, 64, kernel_size=3, dilation=1, dropout=dropout),
            CausalConvBlock(64, 64, kernel_size=3, dilation=2, dropout=dropout)
        )
        
        # Branch 2: Meteorology (21 features -> 64 channels)
        self.meteorology_branch = nn.Sequential(
            CausalConvBlock(21, 64, kernel_size=3, dilation=1, dropout=dropout),
            CausalConvBlock(64, 64, kernel_size=3, dilation=2, dropout=dropout)
        )
        
        # Branch 3: Atmospheric & Fire External (9 features -> 64 channels)
        self.atmospheric_branch = nn.Sequential(
            CausalConvBlock(9, 64, kernel_size=3, dilation=1, dropout=dropout),
            CausalConvBlock(64, 64, kernel_size=3, dilation=2, dropout=dropout)
        )
        
        # Branch 4: Temporal & Availability (10 features -> 32 channels)
        self.temporal_branch = nn.Sequential(
            CausalConvBlock(10, 32, kernel_size=3, dilation=1, dropout=dropout)
        )
        
        # Total concatenated channels = 64 + 64 + 64 + 32 = 224
        concat_dim = 224
        self.fusion_dim = 256
        
        # 2. Cross-Branch Gated Feature Fusion
        self.gate_value = nn.Conv1d(concat_dim, self.fusion_dim, 1)
        self.gate_signal = nn.Conv1d(concat_dim, self.fusion_dim, 1)
        
        # 3. Causal Multi-Head Temporal Self-Attention
        self.causal_attn = CausalMultiHeadAttention(d_model=self.fusion_dim, n_heads=4, dropout=dropout)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.fusion_dim)
        )
        self.ffn_norm = nn.LayerNorm(self.fusion_dim)
        
        # 4. Dual Output Heads
        # Primary Forecast Head: [B, 256, 72] -> [B, 1, 72] -> [B, 72, 1]
        self.forecast_head = nn.Sequential(
            nn.Conv1d(self.fusion_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 1, 1)
        )
        
        # Auxiliary High-Pollution Spike Classification Head: [B, 256, 72] -> [B, 1, 72] -> Sigmoid
        self.spike_head = nn.Sequential(
            nn.Conv1d(self.fusion_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, return_auxiliary=False):
        # Input x shape: [B, 72, 49]
        B, T, C = x.shape
        assert C == 49, f"Expected 49 input features, got {C}"
        
        # Slice Feature Groups
        x_pol = x[:, :, self.pollution_indices].permute(0, 2, 1)    # [B, 9, 72]
        x_met = x[:, :, self.meteorology_indices].permute(0, 2, 1)  # [B, 21, 72]
        x_atm = x[:, :, self.atmospheric_indices].permute(0, 2, 1)  # [B, 9, 72]
        x_tmp = x[:, :, self.temporal_indices].permute(0, 2, 1)     # [B, 10, 72]
        
        # Encode Domain Branches
        h_pol = self.pollution_branch(x_pol)    # [B, 64, 72]
        h_met = self.meteorology_branch(x_met)  # [B, 64, 72]
        h_atm = self.atmospheric_branch(x_atm)  # [B, 64, 72]
        h_tmp = self.temporal_branch(x_tmp)     # [B, 32, 72]
        
        # Concatenate Cross-Branch Representations: [B, 224, 72]
        h_concat = torch.cat([h_pol, h_met, h_atm, h_tmp], dim=1)
        
        # Gated Feature Fusion: [B, 256, 72]
        val = self.gate_value(h_concat)
        sig = torch.sigmoid(self.gate_signal(h_concat))
        h_fused = val * sig
        
        # Temporal Causal Self-Attention: [B, 72, 256]
        h_perm = h_fused.permute(0, 2, 1)
        h_attn = self.causal_attn(h_perm)
        
        # FFN with Residual Connection
        h_ffn = self.ffn_norm(h_attn + self.ffn(h_attn)) # [B, 72, 256]
        
        # Permute back for Conv1D Output Heads: [B, 256, 72]
        h_out = h_ffn.permute(0, 2, 1)
        
        # Primary Forecast Prediction: [B, 72, 1]
        forecast_out = self.forecast_head(h_out).permute(0, 2, 1)
        
        if return_auxiliary:
            spike_prob = self.spike_head(h_out).permute(0, 2, 1)
            return {
                "forecast": forecast_out,
                "spike_prob": spike_prob
            }
            
        return forecast_out

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

if __name__ == "__main__":
    model = CoupledMultiBranchForecastModel()
    dummy_input = torch.randn(4, 72, 49)
    out_forecast = model(dummy_input)
    out_dict = model(dummy_input, return_auxiliary=True)
    tot, trn = model.count_parameters()
    
    print("Coupled Multi-Branch Forecast Model Initialized Successfully")
    print(f"Input Shape            : {dummy_input.shape}")
    print(f"Primary Output Shape   : {out_forecast.shape}")
    print(f"Spike Head Output Shape: {out_dict['spike_prob'].shape}")
    print(f"Total Parameter Count  : {tot:,}")
    print(f"Trainable Parameters   : {trn:,}")
