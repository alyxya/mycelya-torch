# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Simple demonstration of MSE loss computation on CPU vs Remote Device

This script creates random predictions and targets, computes MSE loss on both
CPU and a remote Modal device, and compares the results.
"""

import torch
import torch.nn.functional as F
import torch_remote


def main():
    print("MSE Loss Demo: CPU vs Remote Device")
    print("=" * 50)
    
    # Set random seed for reproducible results
    torch.manual_seed(42)
    
    # Create sample data on CPU
    batch_size, num_features = 32, 10
    predictions_cpu = torch.randn(batch_size, num_features, requires_grad=True)
    targets_cpu = torch.randn(batch_size, num_features)
    
    print(f"Data shape: {predictions_cpu.shape}")
    print(f"Predictions (first 3): {predictions_cpu[0, :3].data}")
    print(f"Targets (first 3): {targets_cpu[0, :3].data}")
    print()
    
    # Compute MSE loss on CPU
    print("Computing MSE loss on CPU...")
    mse_loss_cpu = F.mse_loss(predictions_cpu, targets_cpu)
    mse_loss_value = mse_loss_cpu.item()  # Save original CPU value for comparison
    print(f"CPU MSE Loss: {mse_loss_value:.6f}")
    
    # Backward pass on CPU
    mse_loss_cpu.backward()
    cpu_grad_norm = predictions_cpu.grad.norm().item()
    print(f"CPU Gradient Norm: {cpu_grad_norm:.6f}")
    print()
    
    # Create remote device (using T4 for this demo)
    print("Creating remote Modal device (T4)...")
    try:
        remote_device = torch_remote.create_modal_machine("T4")
        torch_device = remote_device.device()
        print(f"Remote device created: {torch_device}")
        print()
        
        # Transfer data to remote device
        print("Transferring data to remote device...")
        # Reset gradients first
        predictions_cpu.grad = None
        
        # Transfer to remote device
        predictions_remote = predictions_cpu.to(torch_device)
        targets_remote = targets_cpu.to(torch_device)
        
        print("Data transferred successfully")
        print()
        
        # Compute MSE loss on remote device
        print("Computing MSE loss on remote device...")
        mse_loss_remote = F.mse_loss(predictions_remote, targets_remote)
        
        # Transfer result back to CPU for display (avoid .item() to prevent scalar conversion error)
        mse_loss_cpu = mse_loss_remote.cpu()
        print(f"Remote MSE Loss: {mse_loss_cpu} (shape: {mse_loss_cpu.shape})")
        
        # Backward pass on remote device
        mse_loss_remote.backward()
        
        # Check gradients (they should be on the original CPU tensor)
        if predictions_cpu.grad is not None:
            remote_grad_norm = predictions_cpu.grad.norm().item()
            print(f"Remote Gradient Norm: {remote_grad_norm:.6f}")
        else:
            print("Warning: No gradients found on CPU tensor after remote backward pass")
        print()
        
        # Compare results
        print("Comparison:")
        print(f"MSE Loss difference: {abs(mse_loss_cpu.item() - mse_loss_value):.10f}")
        if predictions_cpu.grad is not None:
            # Note: gradients accumulate, so we need to compare with the accumulated gradients
            print(f"Gradient Norm difference: {abs(cpu_grad_norm - remote_grad_norm):.10f}")
        print()
        
        if abs(mse_loss_cpu.item() - mse_loss_value) < 1e-6:
            print("✅ Results match! Remote execution is working correctly.")
        else:
            print("❌ Results differ. There may be an issue with remote execution.")
            
    except Exception as e:
        print(f"❌ Error with remote device: {e}")
        print("Make sure Modal is properly configured and accessible.")


if __name__ == "__main__":
    main()