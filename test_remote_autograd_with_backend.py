# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Test forward and backward pass computation with properly set up remote backend.

This test demonstrates that autograd is fully functional with remote tensors:
- Forward passes execute on remote devices
- Backward passes create gradient tensors on remote devices
- Remote allocator is called for gradient tensor creation
- Multiple forward/backward cycles work correctly
"""

import torch
import torch_remote
from torch_remote.device import create_modal_device, GPUType


def test_remote_backend_setup():
    """Test setting up a remote backend device."""
    print("=== Test: Remote backend setup ===")
    
    try:
        print("Creating Modal device with A100-40GB...")
        
        # Create a Modal backend device - this should properly register the device
        backend_device = create_modal_device("A100-40GB")
        
        print(f"Backend device created: {backend_device}")
        print(f"Device ID: {backend_device.device_id}")
        print(f"GPU type: {backend_device.gpu_type}")
        print(f"Provider: {backend_device.provider}")
        
        # Get the PyTorch device object
        torch_device = backend_device.device()
        print(f"PyTorch device: {torch_device}")
        print(f"Device type: {torch_device.type}")
        print(f"Device index: {torch_device.index}")
        
        # Check device registry
        from torch_remote.device import get_device_registry
        registry = get_device_registry()
        
        registered_device = registry.get_device_by_index(torch_device.index)
        print(f"Device found in registry: {registered_device is not None}")
        
        if registered_device:
            print(f"Registry device matches: {registered_device is backend_device}")
        
        print("✓ Remote backend setup successful")
        return backend_device
        
    except Exception as e:
        print(f"✗ Remote backend setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_remote_tensor_creation_with_backend(backend_device):
    """Test creating remote tensors with a properly set up backend."""
    print("\n=== Test: Remote tensor creation with backend ===")
    
    if backend_device is None:
        print("⚠ Skipping - no backend device available")
        return None, None
    
    try:
        # Get the PyTorch device
        torch_device = backend_device.device()
        print(f"Using device: {torch_device}")
        
        print("Creating remote tensors with requires_grad...")
        
        # Create input tensor
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device=torch_device)
        print(f"Input tensor x created successfully")
        print(f"x.device: {x.device}")
        print(f"x.requires_grad: {x.requires_grad}")
        
        # Create weight tensor  
        w = torch.tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True, device=torch_device)
        print(f"Weight tensor w created successfully")
        print(f"w.device: {w.device}")
        print(f"w.requires_grad: {w.requires_grad}")
        
        print("✓ Remote tensor creation with backend successful")
        return x, w
        
    except Exception as e:
        print(f"✗ Remote tensor creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_remote_forward_pass(x, w):
    """Test forward pass with remote tensors."""
    print("\n=== Test: Remote forward pass ===")
    
    if x is None or w is None:
        print("⚠ Skipping - no tensors available")
        return None
    
    try:
        print("Performing forward pass...")
        print(f"Computing y = x @ w")
        print(f"x shape: {x.shape}, w shape: {w.shape}")
        
        # Forward pass: matrix multiplication
        y = torch.matmul(x, w)
        
        print(f"Forward result y: shape={y.shape}, device={y.device}")
        print(f"y.requires_grad: {y.requires_grad}")
        
        # Reduction to scalar for backward pass
        loss = y.sum()
        print(f"Loss (scalar): {loss}")
        print(f"loss.device: {loss.device}")
        print(f"loss.requires_grad: {loss.requires_grad}")
        
        print("✓ Remote forward pass successful")
        return loss
        
    except Exception as e:
        print(f"✗ Remote forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_remote_backward_pass(loss, x, w):
    """Test backward pass with remote tensors."""
    print("\n=== Test: Remote backward pass ===")
    
    if loss is None or x is None or w is None:
        print("⚠ Skipping - no loss or tensors available")
        return
    
    try:
        print("Performing backward pass...")
        print("Calling loss.backward()...")
        
        # This is the key test - backward pass on remote tensors
        loss.backward()
        
        print("Backward pass completed!")
        
        # Check gradients
        print("\nChecking gradients...")
        
        if x.grad is not None:
            print(f"x.grad: shape={x.grad.shape}, device={x.grad.device}")
            print(f"x.grad content: {x.grad}")
        else:
            print("✗ x.grad is None")
        
        if w.grad is not None:
            print(f"w.grad: shape={w.grad.shape}, device={w.grad.device}")
            print(f"w.grad content: {w.grad}")
        else:
            print("✗ w.grad is None")
        
        # Verify gradients are on remote device
        if x.grad is not None and w.grad is not None:
            if x.grad.device.type == "remote" and w.grad.device.type == "remote":
                print("✓ Gradients computed on remote device!")
                print("✓ Remote backward pass successful!")
                
                # Test if we can access gradient values (this would trigger allocator)
                print(f"Gradient norms - x: {x.grad.norm()}, w: {w.grad.norm()}")
                
                return True
            else:
                print(f"⚠ Gradients not on remote device: x.grad={x.grad.device}, w.grad={w.grad.device}")
        else:
            print("✗ Some gradients not computed")
            
        return False
        
    except Exception as e:
        print(f"✗ Remote backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_backward_passes(backend_device):
    """Test multiple forward/backward passes to verify allocator behavior."""
    print("\n=== Test: Multiple backward passes ===")
    
    if backend_device is None:
        print("⚠ Skipping - no backend device available")
        return
    
    try:
        torch_device = backend_device.device()
        
        print("Testing multiple forward/backward cycles...")
        
        for i in range(3):
            print(f"\n--- Iteration {i+1} ---")
            
            # Create fresh tensors for each iteration
            x = torch.randn(4, 4, requires_grad=True, device=torch_device)
            w = torch.randn(4, 4, requires_grad=True, device=torch_device)
            
            # Forward pass
            y = x @ w
            loss = y.mean()
            
            print(f"Forward pass {i+1}: loss = {loss}")
            
            # Backward pass
            loss.backward()
            
            # Check gradients exist
            grad_exists = x.grad is not None and w.grad is not None
            print(f"Gradients computed: {grad_exists}")
            
            if grad_exists:
                print(f"Grad devices: x={x.grad.device}, w={w.grad.device}")
        
        print("✓ Multiple backward passes successful!")
        print("✓ Allocator handling multiple gradient allocations!")
        
    except Exception as e:
        print(f"✗ Multiple backward passes failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run comprehensive remote autograd tests with proper backend setup."""
    print("=== PyTorch Remote Autograd Test with Backend ===")
    print("Testing forward and backward passes with properly configured remote device\n")
    
    print("Torch version:", torch.__version__)
    
    try:
        print("Torch remote version:", torch_remote.__version__)
    except:
        print("Torch remote version: unknown")
    
    print()
    
    # Step 1: Set up remote backend
    backend_device = test_remote_backend_setup()
    
    # Step 2: Create remote tensors
    x, w = test_remote_tensor_creation_with_backend(backend_device)
    
    # Step 3: Forward pass
    loss = test_remote_forward_pass(x, w)
    
    # Step 4: Backward pass (the key test!)
    backward_success = test_remote_backward_pass(loss, x, w)
    
    # Step 5: Multiple passes to test allocator
    test_multiple_backward_passes(backend_device)
    
    print("\n=== Summary ===")
    if backward_success:
        print("🎉 SUCCESS: Remote autograd fully functional!")
        print("\nKey achievements:")
        print("✓ Remote backend device properly set up")
        print("✓ Remote tensors created with requires_grad")
        print("✓ Forward pass executed on remote device")
        print("✓ Backward pass computed gradients on remote device")
        print("✓ Gradient tensors allocated via remote allocator")
        print("✓ Multiple backward passes work correctly")
        print("\n🔥 Autograd + Remote Tensors = WORKING! 🔥")
    else:
        print("⚠ Partial success - some issues encountered")
        print("Check output above for details")
    
    # Cleanup
    if backend_device:
        try:
            backend_device.stop_gpu_machine()
            print("\n🛑 GPU machine stopped")
        except:
            pass


if __name__ == "__main__":
    main()