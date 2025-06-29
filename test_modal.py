#!/usr/bin/env python3
import torch
import torch_modal

def test_modal_device():
    print("Testing torch-modal package...")
    
    # Test device availability
    print(f"Modal device available: {torch_modal.modal.is_available()}")
    print(f"Modal device count: {torch_modal.modal.device_count()}")
    print(f"Modal device name: {torch_modal.modal.get_device_name()}")
    
    # Test tensor creation and movement
    print("\nTesting tensor operations...")
    
    # Create CPU tensors
    x_cpu = torch.randn(2, 2)
    y_cpu = torch.randn(2, 2)
    print(f"CPU tensor x device: {x_cpu.device}")
    print(f"CPU tensor y device: {y_cpu.device}")
    
    # Move to modal device
    x_modal = x_cpu.modal()
    y_modal = y_cpu.modal()
    print(f"Modal tensor x device: {x_modal.device}")
    print(f"Modal tensor y device: {y_modal.device}")
    
    # Test operation between modal tensors (should work)
    try:
        z_modal = x_modal.mm(y_modal)
        print(f"Modal tensor operation successful: {z_modal.device}")
        print(f"Result shape: {z_modal.shape}")
    except Exception as e:
        print(f"Modal tensor operation failed: {e}")
    
    # Test operation between modal and CPU tensors (should fail)
    try:
        z_mixed = x_modal.mm(y_cpu)
        print("ERROR: Mixed device operation should have failed!")
    except Exception as e:
        print(f"Mixed device operation correctly failed: {e}")
    
    # Test direct modal tensor creation
    try:
        direct_modal = torch.randn(3, 3, device='modal')
        print(f"Direct modal tensor creation: {direct_modal.device}")
    except Exception as e:
        print(f"Direct modal tensor creation failed: {e}")

if __name__ == "__main__":
    test_modal_device()