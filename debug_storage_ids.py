# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3

import torch
import torch_remote

# Create remote device
machine = torch_remote.create_modal_machine("A100")
device = machine.device()

print("=== Creating tensors and checking storage IDs ===")

# Create a few tensors and check their storage IDs
x = torch.tensor([1.0, 2.0, 3.0], device=device)
print(f"Vector tensor x: storage_id={x.untyped_storage().data_ptr()}")

y = torch.tensor([4.0, 5.0, 6.0], device=device)
print(f"Vector tensor y: storage_id={y.untyped_storage().data_ptr()}")

# Try to create a scalar
scalar = torch.tensor(42.0, device=device)
print(f"Scalar tensor: storage_id={scalar.untyped_storage().data_ptr()}")

# Try empty tensor
empty = torch.empty([], device=device)
print(f"Empty tensor: storage_id={empty.untyped_storage().data_ptr()}")

# Try zero-dimensional tensor
zero_dim = torch.zeros((), device=device)
print(f"Zero-dim tensor: storage_id={zero_dim.untyped_storage().data_ptr()}")

print("\n=== Testing MSE loss manually ===")
try:
    # Test MSE loss
    pred = torch.tensor([1.0, 2.0, 3.0], device=device)
    target = torch.tensor([1.5, 2.5, 3.5], device=device)
    
    print(f"pred storage_id: {pred.untyped_storage().data_ptr()}")
    print(f"target storage_id: {target.untyped_storage().data_ptr()}")
    
    loss = torch.nn.functional.mse_loss(pred, target, reduction='mean')
    print(f"MSE loss successful: {loss}")
except Exception as e:
    print(f"MSE loss failed: {e}")
    import traceback
    traceback.print_exc()