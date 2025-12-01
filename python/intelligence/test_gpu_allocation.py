#!/usr/bin/env python3
"""
Diagnostic script to understand GPU memory allocation issues on Windows
"""

import torch
import os
import subprocess
import psutil

def check_gpu_mode():
    """Check if GPU is in WDDM or TCC mode"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-d", "COMPUTE"],
            capture_output=True, text=True, timeout=5
        )
        if "WDDM" in result.stdout:
            print("WARNING: GPU is in WDDM mode (Windows Display Driver Model)")
            print("  - This mode shares GPU with Windows desktop")
            print("  - Limits contiguous memory allocations")
            print("  - Maximum contiguous allocation is typically 50-75% of VRAM")
        elif "TCC" in result.stdout:
            print("[OK] GPU is in TCC mode (Tesla Compute Cluster)")
            print("  - Dedicated compute mode")
            print("  - Better for large allocations")
        else:
            print("? Could not determine GPU mode")
    except Exception as e:
        print(f"Error checking GPU mode: {e}")

def test_allocations():
    """Test various allocation sizes to find the limit"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    # Get initial state
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_mem:.2f} GB")
    print(f"Free VRAM: {free_mem:.2f} GB")
    
    # Test allocations
    print("\n--- Testing Contiguous Allocations ---")
    test_sizes_gb = [1, 2, 4, 8, 10, 12, 14, 14.43, 16, 18, 20, 22]
    
    for size_gb in test_sizes_gb:
        if size_gb > free_mem:
            print(f"Skipping {size_gb:.2f}GB (exceeds free memory)")
            continue
            
        try:
            # Try to allocate
            num_elements = int(size_gb * 1024**3 / 2)  # float16 = 2 bytes
            tensor = torch.empty(num_elements, dtype=torch.float16, device='cuda')
            actual_size = tensor.element_size() * tensor.numel() / 1024**3
            print(f"[OK] {size_gb:.2f}GB allocation succeeded (actual: {actual_size:.2f}GB)")
            del tensor
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"[FAIL] {size_gb:.2f}GB allocation failed: OOM")
        except Exception as e:
            print(f"[FAIL] {size_gb:.2f}GB allocation failed: {e}")

def check_fragmentation():
    """Check for memory fragmentation"""
    if not torch.cuda.is_available():
        return
    
    print("\n--- Memory Fragmentation Check ---")
    
    # Allocate and deallocate to create fragmentation
    tensors = []
    try:
        # Allocate 10 x 1GB tensors
        for i in range(10):
            t = torch.empty(int(1024**3/2), dtype=torch.float16, device='cuda')
            tensors.append(t)
        print(f"Allocated {len(tensors)} x 1GB tensors")
        
        # Delete every other tensor to create gaps
        for i in range(0, len(tensors), 2):
            del tensors[i]
            tensors[i] = None
        torch.cuda.empty_cache()
        print("Deleted every other tensor (creating fragmentation)")
        
        # Now try to allocate a large contiguous block
        try:
            large = torch.empty(int(5*1024**3/2), dtype=torch.float16, device='cuda')
            print("[OK] 5GB contiguous allocation succeeded despite fragmentation")
            del large
        except:
            print("[FAIL] 5GB contiguous allocation failed due to fragmentation")
            
    finally:
        # Cleanup
        tensors = []
        torch.cuda.empty_cache()

def check_system_resources():
    """Check system RAM and other resources"""
    print("\n--- System Resources ---")
    
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / 1024**3:.2f} GB")
    print(f"Available RAM: {mem.available / 1024**3:.2f} GB")
    print(f"Used RAM: {mem.used / 1024**3:.2f} GB ({mem.percent:.1f}%)")
    
    # Check page file
    swap = psutil.swap_memory()
    print(f"Total Swap: {swap.total / 1024**3:.2f} GB")
    print(f"Available Swap: {swap.free / 1024**3:.2f} GB")

def main():
    print("="*60)
    print("GPU Memory Allocation Diagnostic")
    print("="*60)
    
    # Check environment variables
    print("\n--- Environment Variables ---")
    env_vars = [
        'PYTORCH_CUDA_ALLOC_CONF',
        'PYTORCH_NO_CUDA_MEMORY_CACHING',
        'CUDA_LAUNCH_BLOCKING'
    ]
    for var in env_vars:
        val = os.environ.get(var, "Not set")
        print(f"{var}: {val}")
    
    check_system_resources()
    check_gpu_mode()
    test_allocations()
    check_fragmentation()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("1. If in WDDM mode, consider switching to TCC mode:")
    print("   nvidia-smi -g 0 -dm 1  (requires admin, disables display)")
    print("2. Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
    print("3. Ensure sufficient system RAM (16GB+ available)")
    print("4. Close other GPU applications to reduce fragmentation")
    print("5. Consider using model sharding or quantization if needed")

if __name__ == "__main__":
    main()
