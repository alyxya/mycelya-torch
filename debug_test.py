#!/usr/bin/env python3
"""
Simple debug test for torch-modal package.

This is a minimal test script to verify basic functionality:
- Import torch_modal
- Create tensor and convert to modal
- Perform basic operations

Use this for quick debugging during development.
"""

import torch
print("Importing torch_modal...")
import torch_modal
print("Import successful")

print("Creating tensor...")
x = torch.randn(2, 2)
print(f"Tensor created: {x.device}")

print("Trying modal conversion...")
try:
    y = x.modal()
    print(f"Modal conversion success: {y.device}")
    print(f"Modal tensor data: {y}")
    
    # Try a simple operation
    print("Trying tensor addition...")
    z = y + y
    print(f"Addition successful: {z.device}")
    print(f"Result: {z}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()