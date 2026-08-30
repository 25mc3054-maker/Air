import torch
import os

def get_device():
    """
    Detects and returns the available PyTorch computation device (CUDA or CPU).
    Configures multi-threading for CPU acceleration.
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        # Optimize CPU multi-threading
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(min(cpu_count, 8))
        
    print("--- PYTORCH DEVICE DETECTION ---")
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {cuda_available}")
    print(f"Selected Device : {device}")
    print(f"Device Name     : {device_name}")
    print(f"CPU Threads     : {torch.get_num_threads()}")
    print("--------------------------------")
    
    return device, device_name, torch.__version__

if __name__ == "__main__":
    get_device()
