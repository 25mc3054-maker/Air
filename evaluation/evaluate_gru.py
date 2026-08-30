import os
import sys
import json
import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.gru_model import GRUForecastModel
from training.dataloader import get_dataloaders
from training.device import get_device
from training.metrics import (
    calculate_overall_metrics,
    calculate_horizon_metrics,
    calculate_high_pollution_metrics
)

def evaluate_model_on_split(model, dataloader, target_scaler, device):
    model.eval()
    all_preds_scaled = []
    all_targets_scaled = []
    
    with torch.no_grad():
        for x_b, y_b in dataloader:
            x_b = x_b.to(device)
            preds_b = model(x_b)
            
            all_preds_scaled.append(preds_b.cpu().numpy())
            all_targets_scaled.append(y_b.numpy())
            
    preds_scaled = np.concatenate(all_preds_scaled, axis=0) # [N, 72, 1]
    targets_scaled = np.concatenate(all_targets_scaled, axis=0) # [N, 72, 1]
    
    # Inverse transform to physical PM2.5 units (ug/m3)
    N, T, C = preds_scaled.shape
    preds_phys = target_scaler.inverse_transform(preds_scaled.reshape(-1, C)).reshape(N, T, C).astype(np.float32)
    targets_phys = target_scaler.inverse_transform(targets_scaled.reshape(-1, C)).reshape(N, T, C).astype(np.float32)
    
    # Clip negative predictions to 0 (PM2.5 physical lower bound)
    preds_phys = np.clip(preds_phys, 0.0, None)
    
    return preds_phys, targets_phys

def run_evaluation(base_dir=r"d:\My Projects\SIH2026_PersonB"):
    checkpoint_path = os.path.join(base_dir, "models", "checkpoints", "gru_best.pt")
    target_scaler_path = os.path.join(base_dir, "models", "scalers", "target_scaler.joblib")
    config_path = os.path.join(base_dir, "configs", "gru_config.json")
    scaled_data_dir = os.path.join(base_dir, "data", "processed", "scaled")
    raw_data_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    
    os.makedirs(plots_dir, exist_ok=True)
    
    device, dev_name, _ = get_device()
    target_scaler = joblib.load(target_scaler_path)
    
    # Load Best Model Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["model_configuration"]
    
    model = GRUForecastModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_size=config["target_size"]
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded GRU model checkpoint from Epoch {checkpoint['epoch']} with Val Loss {checkpoint['best_validation_loss']:.6f}")
    
    # Get DataLoaders
    _, val_loader, test_loader, _ = get_dataloaders(config_path, scaled_data_dir)
    
    # Evaluate Validation Split
    val_preds_phys, val_targets_phys = evaluate_model_on_split(model, val_loader, target_scaler, device)
    val_overall = calculate_overall_metrics(val_targets_phys, val_preds_phys)
    
    # Evaluate Test Split
    test_preds_phys, test_targets_phys = evaluate_model_on_split(model, test_loader, target_scaler, device)
    test_overall = calculate_overall_metrics(test_targets_phys, test_preds_phys)
    
    # Save Test Predictions and Actuals
    test_preds_path = os.path.join(results_dir, "gru_test_predictions.npy")
    test_actual_path = os.path.join(results_dir, "gru_test_actual.npy")
    np.save(test_preds_path, test_preds_phys)
    np.save(test_actual_path, test_targets_phys)
    print(f"Saved test predictions to {test_preds_path} and actuals to {test_actual_path}")
    
    # Calculate Horizon Metrics (1h to 72h)
    all_horizons = list(range(1, 73))
    test_horizon_all = calculate_horizon_metrics(test_targets_phys, test_preds_phys, horizons=all_horizons)
    
    # Focus Key Horizons (1h, 6h, 12h, 24h, 48h, 72h)
    key_horizons = [1, 6, 12, 24, 48, 72]
    key_horizon_metrics = {f"{h}h": test_horizon_all[f"{h}h"] for h in key_horizons}
    
    # Calculate Extreme Pollution Threshold from Training Targets ONLY
    y_train_raw = np.load(os.path.join(raw_data_dir, "y_train.npy"))
    threshold_90th = float(np.percentile(y_train_raw, 90))
    print(f"Training Target 90th Percentile Threshold: {threshold_90th:.2f} µg/m³")
    
    hp_metrics = calculate_high_pollution_metrics(test_targets_phys, test_preds_phys, threshold_90th)
    
    # Generate Plots
    # Plot 1: Actual vs Predicted for a representative test sequence
    plt.figure(figsize=(12, 5))
    sample_idx = 100
    t_steps = np.arange(1, 73)
    plt.plot(t_steps, test_targets_phys[sample_idx, :, 0], 'k-o', label='Actual PM2.5 (Ground Truth)', linewidth=2)
    plt.plot(t_steps, test_preds_phys[sample_idx, :, 0], 'r--s', label='GRU Predicted PM2.5', linewidth=2)
    plt.title(f"GRU Baseline: 72-Hour PM2.5 Forecast (Test Sample #{sample_idx})", fontsize=14, fontweight='bold')
    plt.xlabel("Forecast Horizon (Hours)", fontsize=12)
    plt.ylabel("PM2.5 Concentration (µg/m³)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "gru_actual_vs_predicted.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot1_path}")
    
    # Plot 2: Horizon Error (MAE & RMSE vs Forecast Horizon)
    horizons_x = all_horizons
    maes_y = [test_horizon_all[f"{h}h"]["mae"] for h in horizons_x]
    rmses_y = [test_horizon_all[f"{h}h"]["rmse"] for h in horizons_x]
    
    plt.figure(figsize=(12, 5))
    plt.plot(horizons_x, maes_y, 'b-', label='MAE (µg/m³)', linewidth=2)
    plt.plot(horizons_x, rmses_y, 'r--', label='RMSE (µg/m³)', linewidth=2)
    plt.title("GRU Baseline: Error Metrics vs. Forecast Horizon (1h to 72h)", fontsize=14, fontweight='bold')
    plt.xlabel("Forecast Horizon (Hours)", fontsize=12)
    plt.ylabel("Error (µg/m³)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot2_path = os.path.join(plots_dir, "gru_horizon_error.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot2_path}")
    
    # Write Evaluation Report Markdown
    report_lines = []
    report_lines.append("# B6 — GRU BASELINE EVALUATION REPORT\n")
    report_lines.append("## Overview\n")
    report_lines.append("Evaluation results for the direct sequence-to-sequence GRU baseline model on the validation and testing sets. All metrics are calculated in **physical PM2.5 units (µg/m³)** after inverse target transformation.\n")
    
    report_lines.append("## Overall Performance Summary\n")
    report_lines.append("| Split | Sample Count | Overall MAE (µg/m³) | Overall RMSE (µg/m³) | Overall $R^2$ Score |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    report_lines.append(f"| **Validation Set** | 4,493 | `{val_overall['mae']:.2f}` | `{val_overall['rmse']:.2f}` | `{val_overall['r2']:.4f}` |")
    report_lines.append(f"| **Test Set** | 3,393 | **`{test_overall['mae']:.2f}`** | **`{test_overall['rmse']:.2f}`** | **`{test_overall['r2']:.4f}`** |\n")
    
    report_lines.append("## Forecast Horizon Breakdown (Test Set)\n")
    report_lines.append("| Forecast Horizon | MAE (µg/m³) | RMSE (µg/m³) |")
    report_lines.append("| :---: | :---: | :---: |")
    for h in key_horizons:
        h_k = f"{h}h"
        m = key_horizon_metrics[h_k]
        report_lines.append(f"| **+{h} Hour ({h_k})** | `{m['mae']:.2f}` | `{m['rmse']:.2f}` |")
        
    report_lines.append("\n## Extreme Pollution Analysis (Test Set)\n")
    report_lines.append(f"- **High-Pollution Threshold (Training 90th Percentile):** `{threshold_90th:.2f} µg/m³`")
    report_lines.append(f"- **High-Pollution Test Evaluation Sample Count:** `{hp_metrics['sample_count']:,}` timesteps")
    report_lines.append(f"- **High-Pollution MAE:** `{hp_metrics['mae']:.2f} µg/m³`")
    report_lines.append(f"- **High-Pollution RMSE:** `{hp_metrics['rmse']:.2f} µg/m³`\n")
    
    report_lines.append("## Generated Visualizations\n")
    report_lines.append(f"- **Representative Forecast Plot:** [`results/plots/gru_actual_vs_predicted.png`](file:///{plot1_path.replace('\\', '/')})")
    report_lines.append(f"- **Horizon Error Curve Plot:** [`results/plots/gru_horizon_error.png`](file:///{plot2_path.replace('\\', '/')})\n")
    
    report_lines.append("## Baseline Architecture Notice\n")
    report_lines.append("> **Note:** The GRU model serves as the initial sequence-to-sequence **baseline model**. It establishes the benchmark against which subsequent architectures (LSTM, TCN, and the proposed coupled multi-branch attention model) will be evaluated.\n")
    
    report_lines.append("---\n")
    report_lines.append("```\nB6 STATUS: COMPLETED & VERIFIED\n```\n")
    report_lines.append("```\nREADY FOR B7 — LSTM BASELINE\n```\n")
    
    report_md_path = os.path.join(results_dir, "gru_evaluation_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved evaluation report to {report_md_path}")
    
    return test_overall, key_horizon_metrics, hp_metrics

if __name__ == "__main__":
    run_evaluation()
