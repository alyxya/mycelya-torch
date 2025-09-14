#!/usr/bin/env python3
"""
Simple test for remote execution without decorator source code issues.
"""

import torch
import mycelya_torch
import mycelya_torch._utils

# Create a remote machine
machine = mycelya_torch.RemoteMachine("mock")
device = machine.device()

print(f"Created machine: {machine.machine_id}")
print(f"Device: {device}")

# Create test tensors
a_cpu = torch.randn(3, 3)
b_cpu = torch.randn(3, 3)

a = a_cpu.to(device)
b = b_cpu.to(device)

print(f"Tensor a: shape={a.shape}, device={a.device}, storage_id={mycelya_torch._utils.get_storage_id(a)}")
print(f"Tensor b: shape={b.shape}, device={b.device}, storage_id={mycelya_torch._utils.get_storage_id(b)}")

# Define a simple function without decorator first
def simple_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Add two tensors."""
    print("Executing simple_add remotely")
    result = x + y
    print(f"Addition complete, result shape: {result.shape}")
    return result

# Apply the decorator manually to avoid source code issues
simple_add_remote = mycelya_torch.remote(simple_add)

print("\n" + "="*50)
print("TESTING SIMPLE REMOTE ADDITION")
print("="*50)

try:
    result = simple_add_remote(a, b)
    print(f"✅ Remote execution successful!")
    print(f"   Result shape: {result.shape}")
    print(f"   Result device: {result.device}")
    print(f"   Result sample: {result.flatten()[:3]}")
except Exception as e:
    print(f"❌ Remote execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")