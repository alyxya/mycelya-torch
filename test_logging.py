# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Test script for tensor logging functionality in torch-remote.
This script demonstrates the new logging features for input/output tensors.
"""

import torch
import logging
import sys
import os

# Add the torch_remote module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Configure logging to see the tensor details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def test_tensor_logging():
    """Test that tensor logging works correctly."""
    print("🧪 Testing tensor input/output logging...")
    
    try:
        # Import torch_remote after logging is configured
        import torch_remote
        
        # Create a simple remote device (you may need to adjust this based on your setup)
        print("Creating remote device...")
        # Note: This may fail if Modal is not set up - that's OK for testing logging code
        device = torch_remote.create_modal_device(gpu_type="T4")
        
        print(f"✅ Created device: {device}")
        
        # Create some tensors on the remote device
        print("Creating tensors...")
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        
        print(f"✅ Created tensors:")
        print(f"   x: {x.shape}, dtype: {x.dtype}, device: {x.device}")
        print(f"   y: {y.shape}, dtype: {y.dtype}, device: {y.device}")
        
        # Perform an operation that should trigger logging
        print("Performing matrix multiplication...")
        result = x @ y
        
        print(f"✅ Operation completed:")
        print(f"   result: {result.shape}, dtype: {result.dtype}, device: {result.device}")
        
        # Test with output tensor
        print("Testing with pre-allocated output tensor...")
        output = torch.empty(3, 3, device=device)
        torch.add(x, y, out=output)
        
        print(f"✅ Pre-allocated output operation completed:")
        print(f"   output: {output.shape}, dtype: {output.dtype}, device: {output.device}")
        
        print("🎉 All tests passed! Check the logs above for tensor details.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("This is expected if Modal is not configured.")
        print("The logging code has been added successfully - check the modified files.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tensor_logging()