#!/usr/bin/env python3
"""
Debug script to understand the CONTEXT of the allocation failure
"""

import torch
import os
import gc

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def test_allocation_context():
    """Test if the context matters for the allocation"""
    
    print("Testing allocation in different contexts...")
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    # Clear everything first
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    print("\n1. Testing clean allocation (14.43 GB)...")
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"   Free VRAM: {free_mem:.2f} GB")
    
    try:
        tensor = torch.empty(int(14.43 * 1024**3 / 2), dtype=torch.float16, device='cuda')
        print(f"   SUCCESS! Clean allocation works")
        del tensor
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"   FAILED! Clean allocation failed")
    
    print("\n2. Testing with PYTORCH_CUDA_ALLOC_CONF set...")
    # The env var is already set, let's verify
    print(f"   PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    
    try:
        tensor = torch.empty(int(14.43 * 1024**3 / 2), dtype=torch.float16, device='cuda')
        print(f"   SUCCESS! Works with max_split_size_mb")
        del tensor
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"   FAILED! Even with max_split_size_mb")
    
    print("\n3. Testing with prior small allocations (fragmentation)...")
    # Create some fragmentation
    small_tensors = []
    for i in range(10):
        small_tensors.append(torch.empty(int(0.1 * 1024**3 / 2), dtype=torch.float16, device='cuda'))
    print(f"   Created 10 x 0.1GB allocations")
    
    try:
        tensor = torch.empty(int(14.43 * 1024**3 / 2), dtype=torch.float16, device='cuda')
        print(f"   SUCCESS! Works even with fragmentation")
        del tensor
    except torch.cuda.OutOfMemoryError:
        print(f"   FAILED! Fragmentation causes failure")
    
    # Clean up
    small_tensors = []
    torch.cuda.empty_cache()
    
    print("\n4. Testing allocation in a specific order (like model loading)...")
    # What if we allocate in the order that model loading does?
    # First some smaller allocations (like loading config, tokenizer)
    
    config_tensor = torch.empty(int(0.01 * 1024**3 / 2), dtype=torch.float16, device='cuda')
    print(f"   Allocated 0.01GB (config)")
    
    tokenizer_tensor = torch.empty(int(0.05 * 1024**3 / 2), dtype=torch.float16, device='cuda')
    print(f"   Allocated 0.05GB (tokenizer)")
    
    # Now the big allocation
    try:
        tensor = torch.empty(int(14.43 * 1024**3 / 2), dtype=torch.float16, device='cuda')
        print(f"   SUCCESS! Works even with prior allocations")
        del tensor
    except torch.cuda.OutOfMemoryError:
        print(f"   FAILED! Prior allocations cause failure")
    
    del config_tensor, tokenizer_tensor
    torch.cuda.empty_cache()
    
    print("\n5. CRITICAL TEST: What if CLIP is already loaded?")
    # This is KEY - CLIP might be loaded first!
    
    # Simulate CLIP being loaded (1.6GB)
    clip_tensor = torch.empty(int(1.6 * 1024**3 / 2), dtype=torch.float16, device='cuda')
    print(f"   Allocated 1.6GB (simulating CLIP)")
    
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"   Free VRAM after CLIP: {free_mem:.2f} GB")
    
    # Now try the 14.43GB allocation
    try:
        tensor = torch.empty(int(14.43 * 1024**3 / 2), dtype=torch.float16, device='cuda')
        print(f"   SUCCESS! Works even with CLIP loaded")
        del tensor
    except torch.cuda.OutOfMemoryError:
        print(f"   FAILED! CLIP being loaded causes failure")
        print(f"   THIS MIGHT BE THE ISSUE!")
    
    del clip_tensor
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_allocation_context()
