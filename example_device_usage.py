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
    
    # Method 1: Using torch.randn with device parameter
    x_t4 = torch.randn(3, 3, device=t4_device)
    y_t4 = torch.randn(3, 3, device=t4_device)
    
    # Method 2: Using .remote() method
    x_l4 = torch.randn(3, 3).remote(l4_device)
    y_l4 = torch.randn(3, 3).remote(l4_device)
    
    print(f"   T4 tensors: {x_t4.shape} on {x_t4.device}")
    print(f"   L4 tensors: {x_l4.shape} on {x_l4.device}")
    
    # Operations within the same device work
    print("\n3. Operations within the same device:")
    
    try:
        # This should work - same device
        z_t4 = x_t4 + y_t4
        print(f"   ✅ T4: {x_t4.shape} + {y_t4.shape} = {z_t4.shape}")
    except Exception as e:
        print(f"   ❌ T4 operation failed: {e}")
    
    try:
        # This should work - same device
        z_l4 = x_l4 + y_l4
        print(f"   ✅ L4: {x_l4.shape} + {y_l4.shape} = {z_l4.shape}")
    except Exception as e:
        print(f"   ❌ L4 operation failed: {e}")
    
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