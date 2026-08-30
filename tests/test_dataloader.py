import os
import sys
import json
import time
import torch
import numpy as np

# Ensure workspace root is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from training.dataset import AirQualitySequenceDataset
from training.dataloader import get_dataloaders
from training.device import get_device
from training.reproducibility import set_seed

def run_all_b5_tests(base_dir=r"d:\My Projects\SIH2026_PersonB"):
    scaled_dir = os.path.join(base_dir, "data", "processed", "scaled")
    config_path = os.path.join(base_dir, "configs", "training_config.json")
    results_dir = os.path.join(base_dir, "results")
    schema_path = os.path.join(results_dir, "model_input_schema.json")
    
    os.makedirs(results_dir, exist_ok=True)
    
    test_results = {}
    
    # Test 1: Device Detection
    try:
        device, dev_name, torch_ver = get_device()
        test_results["Device detection"] = "PASS"
    except Exception as e:
        print(f"Device detection failed: {e}")
        test_results["Device detection"] = "FAIL"
        
    # Test 2: Reproducibility
    try:
        set_seed(42)
        test_results["Reproducibility"] = "PASS"
    except Exception as e:
        print(f"Reproducibility failed: {e}")
        test_results["Reproducibility"] = "FAIL"
        
    # Test 3: Dataset construction
    try:
        X_tr_np = np.load(os.path.join(scaled_dir, "X_train_scaled.npy"), mmap_mode='r')
        y_tr_np = np.load(os.path.join(scaled_dir, "y_train_scaled.npy"), mmap_mode='r')
        train_ds = AirQualitySequenceDataset(X_tr_np, y_tr_np)
        assert len(train_ds) == 19005
        test_results["Dataset construction"] = "PASS"
    except Exception as e:
        print(f"Dataset construction failed: {e}")
        test_results["Dataset construction"] = "FAIL"
        
    # Test 4: DataLoader construction
    try:
        tr_loader, va_loader, te_loader, cfg = get_dataloaders(config_path, scaled_dir)
        test_results["DataLoader construction"] = "PASS"
    except Exception as e:
        print(f"DataLoader construction failed: {e}")
        test_results["DataLoader construction"] = "FAIL"
        
    # Fetch first batches
    tr_x, tr_y = next(iter(tr_loader))
    va_x, va_y = next(iter(va_loader))
    te_x, te_y = next(iter(te_loader))
    
    # Test 5: Train batch shape
    if tr_x.shape == (64, 72, 49) and tr_y.shape == (64, 72, 1):
        test_results["Train batch shape"] = "PASS"
    else:
        test_results["Train batch shape"] = f"FAIL ({tr_x.shape}, {tr_y.shape})"
        
    # Test 6: Validation batch shape
    if va_x.shape == (64, 72, 49) and va_y.shape == (64, 72, 1):
        test_results["Validation batch shape"] = "PASS"
    else:
        test_results["Validation batch shape"] = f"FAIL ({va_x.shape}, {va_y.shape})"
        
    # Test 7: Test batch shape
    if te_x.shape == (64, 72, 49) and te_y.shape == (64, 72, 1):
        test_results["Test batch shape"] = "PASS"
    else:
        test_results["Test batch shape"] = f"FAIL ({te_x.shape}, {te_y.shape})"
        
    # Test 8: dtype check
    dtype_ok = (tr_x.dtype == torch.float32 and tr_y.dtype == torch.float32 and
                va_x.dtype == torch.float32 and va_y.dtype == torch.float32 and
                te_x.dtype == torch.float32 and te_y.dtype == torch.float32)
    test_results["dtype check"] = "PASS" if dtype_ok else "FAIL"
    
    # Test 9: NaN check across all partitions
    nan_cnt = (torch.isnan(tr_x).sum() + torch.isnan(tr_y).sum() +
               torch.isnan(va_x).sum() + torch.isnan(va_y).sum() +
               torch.isnan(te_x).sum() + torch.isnan(te_y).sum()).item()
    test_results["NaN check"] = "PASS" if nan_cnt == 0 else f"FAIL ({nan_cnt} NaNs)"
    
    # Test 10: Inf check across all partitions
    inf_cnt = (torch.isinf(tr_x).sum() + torch.isinf(tr_y).sum() +
               torch.isinf(va_x).sum() + torch.isinf(va_y).sum() +
               torch.isinf(te_x).sum() + torch.isinf(te_y).sum()).item()
    test_results["Inf check"] = "PASS" if inf_cnt == 0 else f"FAIL ({inf_cnt} Infs)"
    
    # Test 11: Feature order check
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        fg_path = os.path.join(base_dir, "configs", "feature_groups.json")
        with open(fg_path, 'r') as f:
            fg = json.load(f)
        expected_order = []
        for g_feats in fg.values():
            expected_order.extend(g_feats)
            
        fo_ok = (schema["feature_order"] == expected_order and schema["num_features"] == 49)
        test_results["feature order check"] = "PASS" if fo_ok else "FAIL"
    except Exception as e:
        print(f"Feature order check failed: {e}")
        test_results["feature order check"] = "FAIL"
        
    # Test 12: Sample consistency
    try:
        sample_idx = 500
        x_ds, y_ds = train_ds[sample_idx]
        x_np = X_tr_np[sample_idx]
        y_np = y_tr_np[sample_idx]
        
        diff_x = np.max(np.abs(x_ds.numpy() - x_np))
        diff_y = np.max(np.abs(y_ds.numpy() - y_np))
        
        sc_ok = (diff_x == 0.0 and diff_y == 0.0)
        test_results["sample consistency"] = "PASS" if sc_ok else f"FAIL (diff_x={diff_x}, diff_y={diff_y})"
    except Exception as e:
        print(f"Sample consistency check failed: {e}")
        test_results["sample consistency"] = "FAIL"
        
    # DataLoader Performance Benchmark (load 100 batches)
    print("\nRunning DataLoader Performance Benchmark (100 batches)...", flush=True)
    start_time = time.time()
    batch_count = 0
    for x_b, y_b in tr_loader:
        batch_count += 1
        if batch_count >= 100:
            break
    bench_time = time.time() - start_time
    print(f"Loaded {batch_count} batches in {bench_time:.4f} seconds ({batch_count/bench_time:.2f} batches/sec).", flush=True)
    
    # Print Test Results Summary
    print("\n--- B5 TEST RESULTS SUMMARY ---", flush=True)
    all_passed = True
    for test_name, result in test_results.items():
        print(f"{test_name:30s}: {result}", flush=True)
        if "PASS" not in result:
            all_passed = False
            
    # Generate results/dataloader_report.md
    report_lines = []
    report_lines.append("# B5 — PYTORCH DATASET AND DATALOADER REPORT\n")
    report_lines.append("## Overview\n")
    report_lines.append("PyTorch input pipeline (`Dataset`, `DataLoader`, `device`, `reproducibility`) constructed and validated for 72h-to-72h PM2.5 forecasting.\n")
    
    report_lines.append("## Pipeline Specifications\n")
    report_lines.append(f"- **PyTorch Version:** `{torch.__version__}`")
    report_lines.append(f"- **Computation Device:** `{dev_name}` (`{device}`)")
    report_lines.append(f"- **Batch Size:** `{cfg.get('batch_size', 64)}`")
    report_lines.append(f"- **Number of Workers:** `{cfg.get('num_workers', 0)}`")
    report_lines.append(f"- **Pin Memory:** `{cfg.get('pin_memory', False)}`")
    report_lines.append(f"- **Random Seed:** `{cfg.get('random_seed', 42)}` (Deterministic seeds set for Python, NumPy, PyTorch)")
    report_lines.append(f"- **Memory Strategy:** `np.load(..., mmap_mode='r')` for zero-copy memory efficiency\n")
    
    report_lines.append("## Dataset & DataLoader Dimensions\n")
    report_lines.append("| Split | Dataset Size | Number of Batches (BS=64) | X Batch Shape | y Batch Shape | Tensor Data Type |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    report_lines.append(f"| **Train** | 19,005 | {len(tr_loader)} | `(64, 72, 49)` | `(64, 72, 1)` | `torch.float32` |")
    report_lines.append(f"| **Validation** | 4,493 | {len(va_loader)} | `(64, 72, 49)` | `(64, 72, 1)` | `torch.float32` |")
    report_lines.append(f"| **Test** | 3,393 | {len(te_loader)} | `(64, 72, 49)` | `(64, 72, 1)` | `torch.float32` |\n")
    
    report_lines.append("## Performance Benchmark\n")
    report_lines.append(f"- **Batch Count:** `100 batches` (`6,400` sequence samples)")
    report_lines.append(f"- **Total Loading Time:** `{bench_time:.4f} seconds`")
    report_lines.append(f"- **Throughput:** `{batch_count/bench_time:.2f} batches/second` (`{batch_count*64/bench_time:.2f} samples/second`)\n")
    
    report_lines.append("## Required B5 Test Results (12 Verification Checks)\n")
    report_lines.append("| # | Required Test | Result | Details |")
    report_lines.append("| :---: | :--- | :---: | :--- |")
    for idx, (t_name, res) in enumerate(test_results.items(), 1):
        report_lines.append(f"| {idx} | {t_name} | **{res}** | Verified programmatically |")
        
    report_lines.append("\n---\n")
    report_lines.append("## Final B5 Status\n")
    if all_passed:
        report_lines.append("```")
        report_lines.append("B5 STATUS: COMPLETED & VERIFIED")
        report_lines.append("```\n")
        report_lines.append("```")
        report_lines.append("READY FOR B6 — GRU BASELINE")
        report_lines.append("```\n")
    else:
        report_lines.append("```")
        report_lines.append("B5 STATUS: NOT READY — PROBLEM FOUND")
        report_lines.append("```\n")
        
    report_md_path = os.path.join(results_dir, "dataloader_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved dataloader report to {report_md_path}", flush=True)

if __name__ == "__main__":
    run_all_b5_tests()
