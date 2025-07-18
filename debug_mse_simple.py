# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3

import torch
import torch_remote
import logging

# Set up logging to see what's happening - only our package
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("torch_remote")
logger.setLevel(logging.DEBUG)

# Create remote device
machine = torch_remote.create_modal_machine("A100")
device = machine.device()

print("=== Testing MSE loss step by step ===")

# Create simple test tensors
pred = torch.tensor([1.0, 2.0, 3.0], device=device)
target = torch.tensor([1.5, 2.5, 3.5], device=device)

print(f"pred shape: {pred.shape}, storage_id: {pred.untyped_storage().data_ptr()}")
print(f"target shape: {target.shape}, storage_id: {target.untyped_storage().data_ptr()}")

# Test the individual steps of MSE loss
print("\n=== Step 1: Computing difference ===")
try:
    diff = pred - target
    print(f"diff successful: shape={diff.shape}, storage_id={diff.untyped_storage().data_ptr()}")
    
    print("\n=== Step 2: Squaring difference ===")
    squared_diff = diff * diff
    print(f"squared_diff successful: shape={squared_diff.shape}, storage_id={squared_diff.untyped_storage().data_ptr()}")
    
    print("\n=== Step 3: Mean reduction ===")
    mean_loss = squared_diff.mean()
    print(f"mean_loss successful: shape={mean_loss.shape}, storage_id={mean_loss.untyped_storage().data_ptr()}")
    
    print("\n=== Testing MSE loss function directly ===")
    mse_loss = torch.nn.functional.mse_loss(pred, target, reduction='mean')
    print(f"mse_loss successful: shape={mse_loss.shape}, storage_id={mse_loss.untyped_storage().data_ptr()}")
    
except Exception as e:
    print(f"Operation failed: {e}")
    import traceback
    traceback.print_exc()