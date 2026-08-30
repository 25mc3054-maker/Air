import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.dataset import AirQualitySequenceDataset

def get_dataloaders(config_path=r"d:\My Projects\SIH2026_PersonB\configs\training_config.json",
                    data_dir=r"d:\My Projects\SIH2026_PersonB\data\processed\scaled"):
    """
    Constructs PyTorch DataLoaders for train, validation, and test splits.
    Loads scaled arrays into RAM for sub-second epoch iteration performance.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 0)
    pin_memory = config.get("pin_memory", False)
    
    if torch.cuda.is_available() and not pin_memory:
        pin_memory = True
        
    # Load into memory for fast CPU slicing
    X_train = np.load(os.path.join(data_dir, "X_train_scaled.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train_scaled.npy"))
    
    X_val = np.load(os.path.join(data_dir, "X_val_scaled.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val_scaled.npy"))
    
    X_test = np.load(os.path.join(data_dir, "X_test_scaled.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test_scaled.npy"))
    
    train_dataset = AirQualitySequenceDataset(X_train, y_train)
    val_dataset = AirQualitySequenceDataset(X_val, y_val)
    test_dataset = AirQualitySequenceDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, config
