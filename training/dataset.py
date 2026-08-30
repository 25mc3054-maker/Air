import torch
from torch.utils.data import Dataset
import numpy as np

class AirQualitySequenceDataset(Dataset):
    """
    PyTorch Dataset for 72-hour historical input to 72-hour future PM2.5 forecasting.
    Pre-converts arrays to contiguous float32 PyTorch tensors for zero-overhead batch slicing.
    """
    def __init__(self, X, y):
        assert len(X) == len(y), f"X length ({len(X)}) does not match y length ({len(y)})"
        
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(np.array(X, copy=True)).float()
        else:
            self.X = X.float()
            
        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(np.array(y, copy=True)).float()
        else:
            self.y = y.float()
            
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
