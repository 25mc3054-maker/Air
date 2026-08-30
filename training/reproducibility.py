import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Sets deterministic seeds across Python random, NumPy, and PyTorch (CPU & CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"Reproducibility seed set to: {seed}")

if __name__ == "__main__":
    set_seed(42)
