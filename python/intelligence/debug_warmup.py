#!/usr/bin/env python3
"""
Debug script to understand EXACTLY why caching_allocator_warmup fails
"""

import torch
import os

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def test_exact_warmup_allocation():
    """Replicate EXACTLY what caching_allocator_warmup does"""
    
    print("Replicating caching_allocator_warmup behavior...")
    
    # The warmup function does this:
    # 1. Calculates total model size in bytes
    # 2. Divides by factor (2 for non-quantized)
    # 3. Tries to allocate that as float16
    
    # Qwen3-VL-8B is ~16GB = 16 * 1024^3 bytes
    model_size_bytes = 14 * 1024**3  # 14GB in bytes
    
    # Factor is 2 for non-quantized models
    factor = 2
    
    # This is what warmup tries to allocate
    allocation_size = model_size_bytes // factor  # 7GB
    
    # Convert to number of float16 elements
    num_elements = allocation_size // 2  # float16 = 2 bytes
    
    print(f"Model size: {model_size_bytes / 1024**3:.2f} GB")
    print(f"Factor: {factor}")
    print(f"Warmup tries to allocate: {allocation_size / 1024**3:.2f} GB")
    print(f"Number of float16 elements: {num_elements:,}")
    
    # Get current GPU state
    if torch.cuda.is_available():
        free_before = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"\nFree VRAM before: {free_before:.2f} GB")
        
        # Try the allocation
        try:
            print(f"\nAttempting to allocate {allocation_size / 1024**3:.2f} GB...")
            tensor = torch.empty(num_elements, dtype=torch.float16, device='cuda', requires_grad=False)
            actual_size = tensor.element_size() * tensor.numel() / 1024**3
            print(f"SUCCESS! Allocated {actual_size:.2f} GB")
            del tensor
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"FAILED with OOM: {e}")
            
        # Now test what happens with the ACTUAL warmup size (14.43GB)
        print("\n" + "="*60)
        print("Testing with ACTUAL error size (14.43 GB)...")
        
        # This is what the error message says it's trying to allocate
        actual_error_size_gb = 14.43
        actual_bytes = actual_error_size_gb * 1024**3
        actual_elements = int(actual_bytes / 2)  # float16
        
        print(f"Error says it tries to allocate: {actual_error_size_gb} GB")
        print(f"That's {actual_elements:,} float16 elements")
        
        # Wait, let's check the math...
        # If model is 14GB and factor is 2, it should allocate 7GB
        # But error says 14.43GB... that means factor is 1!
        
        print("\nANALYSIS:")
        print(f"If allocation is 14.43GB and model is ~14GB...")
        print(f"Then factor must be: {14 / 14.43:.2f}")
        print("Wait, that's ~1, not 2!")
        
        # Let's check what happens with factor=1
        factor_1_size = model_size_bytes // 1
        print(f"\nWith factor=1, allocation would be: {factor_1_size / 1024**3:.2f} GB")
        print("That matches the error!")
        
        # So the real question is: why is factor 1 instead of 2?
        print("\nHYPOTHESIS:")
        print("The factor is 1, not 2, which means either:")
        print("1. hf_quantizer is not None (but we're not quantizing)")
        print("2. The model size calculation is wrong")
        print("3. Something else is doubling the allocation")
        
        # Let's test the ACTUAL allocation that's failing
        print("\n" + "="*60)
        print("Testing the EXACT failing allocation (14.43 GB)...")
        
        try:
            print(f"Attempting to allocate {actual_error_size_gb} GB...")
            tensor = torch.empty(actual_elements, dtype=torch.float16, device='cuda', requires_grad=False)
            actual_size = tensor.element_size() * tensor.numel() / 1024**3
            print(f"SUCCESS! Allocated {actual_size:.2f} GB")
            print("WAIT WHAT?! The allocation that fails in warmup WORKS here!")
            del tensor
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"FAILED with OOM: {e}")

if __name__ == "__main__":
    test_exact_warmup_allocation()
