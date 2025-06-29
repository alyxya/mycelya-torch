#!/usr/bin/env python3
"""
Simple functionality test for torch-modal.

Tests basic modal method availability and usage.
This is the simplest possible test to verify the modal() method works.
"""

import torch
import torch_modal

print("Creating CPU tensor...")
x = torch.randn(2, 2)
print("CPU tensor device:", x.device)

print("Checking modal method...")
print("Has modal method:", hasattr(x, 'modal'))

if hasattr(x, 'modal'):
    print("Calling modal method...")
    try:
        y = x.modal()
        print("Success! Modal tensor device:", y.device)
    except Exception as e:
        print("Modal method failed:", str(e))
        import traceback
        traceback.print_exc()
else:
    print("Modal method not found on tensor")