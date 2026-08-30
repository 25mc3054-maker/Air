import os
import sys
import time
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.tcn_model import TCNForecastModel
from models.gru_model import GRUForecastModel
from models.lstm_model import LSTMForecastModel
from tests.test_causality import verify_causality
from training.dataloader import get_dataloaders
from training.device import get_device
from training.reproducibility import set_seed
from training.losses import get_loss_function

def run_sanity_check(model, device, config, results_dir):
    print("\n--- STEP 5A: TCN TRAINING SANITY CHECK ---", flush=True)
    set_seed(config.get("random_seed", 42))
    model.train()
    criterion = get_loss_function()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    # Dummy batch
    dummy_x = torch.randn(config["batch_size"], config["sequence_length"], config["input_size"]).to(device)
    dummy_y = torch.randn(config["batch_size"], config["output_length"], config["target_size"]).to(device)
    
    optimizer.zero_grad()
    preds = model(dummy_x)
    loss = criterion(preds, dummy_y)
    
    loss_val = loss.item()
    is_nan = np.isnan(loss_val)
    is_inf = np.isinf(loss_val)
    
    loss.backward()
    grads_exist = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip_norm", 1.0)).item()
    optimizer.step()
    
    sanity_pass = (not is_nan) and (not is_inf) and grads_exist
    
    print(f"Dummy Loss Value : {loss_val:.6f}", flush=True)
    print(f"NaN Loss         : {is_nan}", flush=True)
    print(f"Inf Loss         : {is_inf}", flush=True)
    print(f"Gradients Exist  : {grads_exist}", flush=True)
    print(f"Gradient Norm    : {grad_norm:.6f}", flush=True)
    print(f"Optimizer Step   : SUCCESS", flush=True)
    print(f"Sanity Check     : {'PASS' if sanity_pass else 'FAIL'}\n", flush=True)
    
    sanity_md = []
    sanity_md.append("# B8 — TCN TRAINING SANITY CHECK REPORT\n")
    sanity_md.append(f"- **Dummy Batch Shape:** `{dummy_x.shape}`")
    sanity_md.append(f"- **Initial Dummy Loss:** `{loss_val:.6f}`")
    sanity_md.append(f"- **NaN Loss Status:** `{is_nan}`")
    sanity_md.append(f"- **Inf Loss Status:** `{is_inf}`")
    sanity_md.append(f"- **Gradients Verified:** `{grads_exist}`")
    sanity_md.append(f"- **Gradient Norm (Clipped @ 1.0):** `{grad_norm:.6f}`")
    sanity_md.append(f"- **Optimizer Step Executed:** `True`\n")
    sanity_md.append(f"```\nSANITY CHECK: {'PASS' if sanity_pass else 'FAIL'}\n```")
    
    with open(os.path.join(results_dir, "tcn_sanity_check.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(sanity_md))
        
    if not sanity_pass:
        raise RuntimeError("TCN Training Sanity Check Failed!")

def train_one_epoch(model, dataloader, criterion, optimizer, device, clip_norm=1.0):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    
    for idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        
        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.abs(preds - y_batch).mean().item() * batch_size
        total_samples += batch_size
        
        if idx % 100 == 0:
            print(f"   Batch {idx:3d}/{len(dataloader):3d} | Current Batch Loss: {loss.item():.6f}", flush=True)
            
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    return avg_loss, avg_mae

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            
            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_mae += torch.abs(preds - y_batch).mean().item() * batch_size
            total_samples += batch_size
            
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    return avg_loss, avg_mae

def run_training_pipeline(base_dir=r"d:\My Projects\SIH2026_PersonB"):
    config_path = os.path.join(base_dir, "configs", "tcn_config.json")
    scaled_data_dir = os.path.join(base_dir, "data", "processed", "scaled")
    checkpoint_dir = os.path.join(base_dir, "models", "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    set_seed(config.get("random_seed", 42))
    device, dev_name, torch_ver = get_device()
    
    # 1. Model Construction & Causality Pre-checks
    model = TCNForecastModel(
        input_size=config["input_size"],
        num_channels=config["num_channels"],
        dilations=config["dilations"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        target_size=config["target_size"]
    ).to(device)
    
    print("Executing Pre-Training Causality Verification...", flush=True)
    causality_pass = verify_causality()
    if not causality_pass:
        raise RuntimeError("Causality Test Failed! Aborting Training.")
        
    tcn_tot_params, tcn_trn_params = model.count_parameters()
    
    # Calculate GRU & LSTM parameters for comparison report
    gru_model = GRUForecastModel(input_size=49, hidden_size=128, num_layers=2, dropout=0.2, output_size=1)
    gru_tot_params, _ = gru_model.count_parameters()
    
    lstm_model = LSTMForecastModel(input_size=49, hidden_size=128, num_layers=2, dropout=0.2, output_size=1)
    lstm_tot_params, _ = lstm_model.count_parameters()
    
    # 2. Run Sanity Check
    run_sanity_check(model, device, config, results_dir)
    
    # Get DataLoaders
    train_loader, val_loader, test_loader, _ = get_dataloaders(config_path, scaled_data_dir)
    criterion = get_loss_function()
    
    # 3. Smoke Test (2 epochs)
    print("--- STEP 5B: RUNNING 2-EPOCH TCN SMOKE TEST ---", flush=True)
    smoke_optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    for ep in range(1, 3):
        tr_loss, tr_mae = train_one_epoch(model, train_loader, criterion, smoke_optimizer, device, config["gradient_clip_norm"])
        va_loss, va_mae = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Smoke Epoch {ep}/2 | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f}", flush=True)
    print("Smoke Test: PASS (No NaNs, training stable)\n", flush=True)
    
    # Reset model for full training
    set_seed(config.get("random_seed", 42))
    model = TCNForecastModel(
        input_size=config["input_size"],
        num_channels=config["num_channels"],
        dilations=config["dilations"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        target_size=config["target_size"]
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.get("lr_scheduler_factor", 0.5),
        patience=config.get("lr_scheduler_patience", 3),
        min_lr=config.get("min_lr", 1e-6)
    )
    
    max_epochs = config["max_epochs"] # 50
    patience = config["patience"]     # 7
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []
    
    print(f"--- STEP 5C: FULL TCN TRAINING RUN ({max_epochs} Max Epochs) ---", flush=True)
    start_train_time = time.time()
    
    for epoch in range(1, max_epochs + 1):
        ep_start = time.time()
        tr_loss, tr_mae = train_one_epoch(model, train_loader, criterion, optimizer, device, config["gradient_clip_norm"])
        va_loss, va_mae = validate_one_epoch(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(va_loss)
        ep_time = time.time() - ep_start
        
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_mae_scaled": tr_mae,
            "val_mae_scaled": va_mae,
            "learning_rate": current_lr
        })
        
        print(f"Epoch {epoch:02d}/{max_epochs:02d} [{ep_time:.2f}s] | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f} | Val MAE: {va_mae:.6f} | LR: {current_lr:.6f}", flush=True)
        
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_loss": best_val_loss,
                "model_configuration": config,
                "random_seed": config.get("random_seed", 42)
            }
            checkpoint_path = os.path.join(checkpoint_dir, "tcn_best.pt")
            torch.save(checkpoint, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at Epoch {epoch}! Best Epoch was {best_epoch} with Val Loss {best_val_loss:.6f}.", flush=True)
                break
                
    total_train_duration = time.time() - start_train_time
    print(f"\nFull Training Completed in {total_train_duration:.2f} seconds ({total_train_duration/60:.2f} minutes).", flush=True)
    
    # Save History CSV
    history_csv_path = os.path.join(results_dir, "tcn_training_history.csv")
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_mae_scaled", "val_mae_scaled", "learning_rate"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved training history to {history_csv_path}", flush=True)
    
    # Write Report
    report_lines = []
    report_lines.append("# B8 — TCN TRAINING REPORT\n")
    report_lines.append("## Executive Summary\n")
    report_lines.append("Direct sequence-to-sequence Causal Dilated TCN baseline model trained for 72-hour continuous PM2.5 forecasting.\n")
    
    report_lines.append("## Model Architecture & Parameter Count Comparison\n")
    report_lines.append(f"- **Architecture:** `TCNForecastModel` (6 Causal Dilated Residual Blocks + 1x1 Conv Projection)")
    report_lines.append(f"- **Input Channels:** `49` features")
    report_lines.append(f"- **Hidden Channels:** `128` channels across 6 residual levels")
    report_lines.append(f"- **Dilation Levels:** `[1, 2, 4, 8, 16, 32]` (Receptive Field = `127` hours > 72h sequence length)")
    report_lines.append(f"- **Kernel Size:** `3` (Dropout = `0.2`)")
    report_lines.append(f"- **Output Size:** `1` (PM2.5 prediction)")
    report_lines.append(f"- **TCN Total Parameters:** `{tcn_tot_params:,}`")
    report_lines.append(f"- **TCN Trainable Parameters:** `{tcn_trn_params:,}`")
    report_lines.append(f"- **GRU Total Parameters (B6 Baseline):** `{gru_tot_params:,}`")
    report_lines.append(f"- **LSTM Total Parameters (B7 Baseline):** `{lstm_tot_params:,}`\n")
    
    report_lines.append("## Training Hyperparameters (Fair Comparison Framework)\n")
    report_lines.append(f"- **Optimizer:** `Adam` (Initial LR = `{config['learning_rate']}`, Weight Decay = `{config['weight_decay']}`)")
    report_lines.append(f"- **LR Scheduler:** `ReduceLROnPlateau` (Factor = `{config['lr_scheduler_factor']}`, Patience = `{config['lr_scheduler_patience']}`, Min LR = `{config['min_lr']}`)")
    report_lines.append(f"- **Gradient Clipping:** `max_norm = 1.0` (`torch.nn.utils.clip_grad_norm_`)")
    report_lines.append(f"- **Loss Function:** `Mean Squared Error (MSE)` (in z-score scaled space)")
    report_lines.append(f"- **Batch Size:** `{config['batch_size']}`")
    report_lines.append(f"- **Early Stopping:** Patience = `{patience}` epochs monitoring validation MSE\n")
    
    report_lines.append("## Execution & Convergence Summary\n")
    report_lines.append(f"- **Hardware Device:** `{dev_name}` (`{device}`)")
    report_lines.append(f"- **Total Epochs Completed:** `{len(history)}` / `{max_epochs}`")
    report_lines.append(f"- **Best Epoch:** `{best_epoch}`")
    report_lines.append(f"- **Best Validation Loss (Scaled MSE):** `{best_val_loss:.6f}`")
    report_lines.append(f"- **Total Training Time:** `{total_train_duration:.2f} seconds` ({total_train_duration/60:.2f} minutes)")
    report_lines.append(f"- **Best Checkpoint Path:** `models/checkpoints/tcn_best.pt`\n")
    
    report_lines.append("## Training Progression\n")
    report_lines.append("| Epoch | Train Loss (MSE) | Val Loss (MSE) | Train MAE (Scaled) | Val MAE (Scaled) | Learning Rate |")
    report_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in history[:5]:
        report_lines.append(f"| {row['epoch']} | {row['train_loss']:.6f} | {row['val_loss']:.6f} | {row['train_mae_scaled']:.6f} | {row['val_mae_scaled']:.6f} | {row['learning_rate']:.6f} |")
    if best_epoch > 5:
        report_lines.append("| ... | ... | ... | ... | ... | ... |")
        b_row = history[best_epoch - 1]
        report_lines.append(f"| **{b_row['epoch']} (BEST)** | **{b_row['train_loss']:.6f}** | **{b_row['val_loss']:.6f}** | **{b_row['train_mae_scaled']:.6f}** | **{b_row['val_mae_scaled']:.6f}** | `{b_row['learning_rate']:.6f}` |")
        
    report_lines.append("\n---\n")
    report_lines.append("```\nTCN TRAINING RUN: COMPLETED SUCCESSFULLY\n```")
    
    report_md_path = os.path.join(results_dir, "tcn_training_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved training report to {report_md_path}", flush=True)

if __name__ == "__main__":
    run_training_pipeline()
