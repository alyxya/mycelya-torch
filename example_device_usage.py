#!/usr/bin/env python3
"""
Example script demonstrating the new device backend functionality.

This script shows how to use different GPU types with torch_remote.
"""

import torch
import torch_remote

def main():
    print("🚀 PyTorch Remote Device Backend Example")
    print("="*50)
    
    # Remote execution is now always enabled for remote tensors
    
    # Create different GPU devices
    print("\n1. Creating different GPU devices:")
    
    # Create T4 and L4 devices (cheaper options)
    t4_device = torch_remote.create_modal_device("T4")
    l4_device = torch_remote.create_modal_device("L4")
    
    # Create A10G device (mid-range option)
    a10g_device = torch_remote.create_modal_device("A10G")
    
    print(f"   T4:   {t4_device}")
    print(f"   L4:   {l4_device}")
    print(f"   A10G: {a10g_device}")
    
    # Create tensors on different devices
    print("\n2. Creating tensors on different devices:")
    
    # Use larger tensors to definitely trigger remote execution
    # Method 1: Using torch.randn with device parameter
    x_t4 = torch.randn(64, 64, device=t4_device)  # 4096 elements
    y_t4 = torch.randn(64, 64, device=t4_device)  # 4096 elements
    
    # Method 2: Using .remote() method
    x_l4 = torch.randn(50, 50).remote(l4_device)  # 2500 elements
    y_l4 = torch.randn(50, 50).remote(l4_device)  # 2500 elements
    
    print(f"   T4 tensors: {x_t4.shape} ({x_t4.numel()} elements) on {x_t4.device}")
    print(f"   L4 tensors: {x_l4.shape} ({x_l4.numel()} elements) on {x_l4.device}")
    print(f"   Total T4 elements: {x_t4.numel() + y_t4.numel()} (>1000 → should use REMOTE execution)")
    print(f"   Total L4 elements: {x_l4.numel() + y_l4.numel()} (>1000 → should use REMOTE execution)")
    
    # Check if device IDs are attached
    print(f"   T4 device ID: {getattr(x_t4, '_device_id', 'NOT SET')}")
    print(f"   L4 device ID: {getattr(x_l4, '_device_id', 'NOT SET')}")
    
    # Operations within the same device work
    print("\n3. Operations within the same device:")
    
    print("   Note: Watch for '🚀 Creating remote job' messages below...")
    
    try:
        # This should work - same device and trigger remote execution
        print("   Executing T4 addition (should see '🚀 Creating remote job' message)...")
        z_t4 = x_t4 + y_t4
        print(f"   ✅ T4: {x_t4.shape} + {y_t4.shape} = {z_t4.shape}")
    except Exception as e:
        print(f"   ❌ T4 operation failed: {e}")
    
    try:
        # This should work - same device and trigger remote execution
        print("   Executing L4 addition (should see '🚀 Creating remote job' message)...")
        z_l4 = x_l4 + y_l4
        print(f"   ✅ L4: {x_l4.shape} + {y_l4.shape} = {z_l4.shape}")
    except Exception as e:
        print(f"   ❌ L4 operation failed: {e}")
    
    # Add a matrix multiplication which is definitely compute-intensive
    print("\n3.5. Testing matrix multiplication (guaranteed remote execution):")
    try:
        print("   Executing T4 matrix multiplication (large compute)...")
        mm_result = torch.mm(x_t4, y_t4)
        print(f"   ✅ T4 matmul: {x_t4.shape} @ {y_t4.shape} = {mm_result.shape}")
    except Exception as e:
        print(f"   ❌ T4 matmul failed: {e}")
    
    # Skip matrix multiplication for now to avoid potential issues
    
    # Operations between different devices should fail
    print("\n4. Operations between different devices (should fail):")
    
    try:
        # This should fail - different devices
        z_mixed = x_t4 + x_l4
        print(f"   ❌ This should not have worked: {z_mixed.shape}")
    except RuntimeError as e:
        print(f"   ✅ Correctly prevented cross-device operation: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Show device registry
    print("\n5. Device registry:")
    registry = torch_remote.get_device_registry()
    print(f"   Registered devices: {len(registry._devices)}")
    for device_id, device in registry._devices.items():
        print(f"     {device_id}: {device}")
    
    # Show available GPU types
    print("\n6. Available GPU types:")
    for gpu_type in torch_remote.GPUType:
        print(f"   - {gpu_type.value}")
    
    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    main()