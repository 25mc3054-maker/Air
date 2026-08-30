import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikeAwareForecastLoss(nn.Module):
    """
    Spike-Aware Multi-Task Loss for Coupled Multi-Branch PM2.5 Forecast Model.
    
    Combines:
    1. Standard Mean Squared Error (MSE) on scaled target
    2. Asymmetric Spike-Weighted MSE for extreme pollution events (y_scaled >= scaled_threshold)
    3. Auxiliary Binary Cross Entropy (BCE) Loss on the Spike Classification Head
    """
    def __init__(self, scaled_threshold=1.4410, spike_weight_multiplier=3.0, alpha=0.5, beta=0.2):
        super(SpikeAwareForecastLoss, self).__init__()
        self.scaled_threshold = float(scaled_threshold)
        self.spike_weight_multiplier = float(spike_weight_multiplier)
        self.alpha = float(alpha)
        self.beta = float(beta)
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred_forecast, y_true, y_pred_spike_prob=None):
        """
        Args:
            y_pred_forecast: Primary forecast output tensor [B, 72, 1]
            y_true: Scaled target ground truth tensor [B, 72, 1]
            y_pred_spike_prob: Optional auxiliary spike probability tensor [B, 72, 1]
        
        Returns:
            dict containing:
                "total_loss": Combined scalar loss for backprop
                "mse_loss": Base MSE loss
                "spike_weighted_loss": Asymmetric spike MSE component
                "aux_bce_loss": Auxiliary BCE classification loss
        """
        # 1. Base MSE Loss
        base_mse = torch.mean(self.mse_loss(y_pred_forecast, y_true))
        
        # 2. Asymmetric Spike-Weighted MSE Component
        # Binary mask for high pollution timesteps (y_true >= scaled_threshold)
        spike_mask = (y_true >= self.scaled_threshold).float()
        weights = 1.0 + (self.spike_weight_multiplier - 1.0) * spike_mask
        
        squared_errors = (y_pred_forecast - y_true) ** 2
        spike_weighted_mse = torch.mean(weights * squared_errors)
        
        # 3. Auxiliary BCE Classification Loss (if spike head active)
        if y_pred_spike_prob is not None:
            aux_bce = self.bce_loss(y_pred_spike_prob, spike_mask)
        else:
            aux_bce = torch.tensor(0.0, device=y_true.device)
            
        # Total Combined Loss
        total_loss = base_mse + self.alpha * spike_weighted_mse + self.beta * aux_bce
        
        return {
            "total_loss": total_loss,
            "base_mse_loss": base_mse,
            "spike_weighted_loss": spike_weighted_mse,
            "aux_bce_loss": aux_bce
        }

if __name__ == "__main__":
    loss_fn = SpikeAwareForecastLoss()
    y_pred = torch.randn(4, 72, 1, requires_grad=True)
    y_true = torch.randn(4, 72, 1)
    p_spike = torch.sigmoid(torch.randn(4, 72, 1))
    
    res = loss_fn(y_pred, y_true, p_spike)
    res["total_loss"].backward()
    
    print("SpikeAwareForecastLoss Initialized Successfully")
    print(f"Total Loss         : {res['total_loss'].item():.6f}")
    print(f"Base MSE Loss      : {res['base_mse_loss'].item():.6f}")
    print(f"Spike Weighted Loss: {res['spike_weighted_loss'].item():.6f}")
    print(f"Aux BCE Loss       : {res['aux_bce_loss'].item():.6f}")
    print(f"Gradient Check     : {y_pred.grad is not None and not torch.isnan(y_pred.grad).any()}")
