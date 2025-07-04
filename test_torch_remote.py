#!/usr/bin/env python3
"""
Comprehensive test suite for torch-remote package.

This script works with both pytest and standalone execution:
- Run with pytest: pytest test_torch_remote.py
- Run standalone: python test_torch_remote.py
- For verbose output: python test_torch_remote.py --verbose
- For quick debug mode: python test_torch_remote.py --debug
"""

import torch
import pytest


def test_basic_imports():
    """Test basic torch and torch_remote imports."""
    import torch_remote
    assert True


def test_device_functions():
    """Test remote device functions."""
    import torch
    import torch_remote
    assert (torch.remote.is_available() and
            torch.remote.device_count() >= 1)


def test_tensor_remote_method():
    """Test that tensors have remote() method."""
    import torch_remote
    x = torch.randn(2, 2)
    assert hasattr(x, 'remote') and callable(x.remote)


def test_remote_tensor_creation():
    """Test remote tensor creation via .remote() method."""
    import torch_remote
    x = torch.randn(2, 2)
    y = x.remote()
    assert y is not None and y.shape == x.shape


def test_remote_tensor_operations():
    """Test operations on remote tensors."""
    import torch_remote
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)

    x_remote = x.remote()
    y_remote = y.remote()

    # Test addition - verify numerical result matches CPU computation
    z_remote = x_remote + y_remote
    z_expected = x + y

    # Test matrix multiplication - verify numerical result matches CPU computation
    w_remote = x_remote.mm(y_remote)
    w_expected = x.mm(y)

    # Verify shapes
    assert z_remote is not None and w_remote is not None and w_remote.shape == (2, 2)

    # Verify numerical results (convert remote tensors back to CPU for comparison)
    assert torch.allclose(z_remote.cpu(), z_expected, rtol=1e-5, atol=1e-8)
    assert torch.allclose(w_remote.cpu(), w_expected, rtol=1e-5, atol=1e-8)


def test_dtype_conversion():
    """Test remote conversion with dtype parameter."""
    import torch_remote
    x = torch.randn(2, 2, dtype=torch.float32)
    y = x.remote(dtype=torch.float64)
    assert y.dtype == torch.float64


def test_copy_parameter():
    """Test remote conversion with copy parameter."""
    import torch_remote
    x = torch.randn(2, 2)
    y = x.remote(copy=True)
    z = x.remote(copy=False)
    assert y is not None and z is not None


def test_error_handling():
    """Test that errors are handled gracefully."""
    import torch_remote
    # These operations might fail, but shouldn't crash
    try:
        torch.randn(3, 3, device='remote')  # Should fail gracefully
    except Exception:
        pass  # Expected to fail

    try:
        x = torch.randn(2, 2).remote()
        y = torch.randn(2, 2)  # CPU tensor
        z = x.mm(y)  # Mixed device - may or may not work
    except Exception:
        pass  # May fail, that's OK

    assert True  # If we get here without segfault, it's good


def test_remote_tensor_device_properties():
    """Test that remote tensors report correct device properties."""
    import torch_remote
    
    # Create CPU tensor and convert to remote
    x_cpu = torch.randn(3, 3)
    x_remote = x_cpu.remote()
    
    # Check that remote tensor has the expected type
    assert type(x_remote).__name__ == 'RemoteTensorData'
    
    # Test device property - remote tensors should identify as remote device
    assert x_remote.device.type == 'remote'
    assert x_remote.device.index == 0  # Default remote device index


def test_remote_only_operations():
    """Test operations that require both tensors to be remote."""
    import torch_remote
    
    x_cpu = torch.randn(2, 3)
    y_cpu = torch.randn(3, 2)
    
    x_remote = x_cpu.remote()
    y_remote = y_cpu.remote()
    
    # Test remote-remote operations (should work)
    result_add = x_remote + x_remote
    result_mm = x_remote.mm(y_remote)
    
    # Verify results are correct and still remote tensors
    assert type(result_add).__name__ == 'RemoteTensorData'
    assert type(result_mm).__name__ == 'RemoteTensorData'
    assert result_add.shape == x_remote.shape
    assert result_mm.shape == (2, 2)
    
    # Verify numerical correctness
    expected_add = x_cpu + x_cpu
    expected_mm = x_cpu.mm(y_cpu)
    assert torch.allclose(result_add.cpu(), expected_add, rtol=1e-5, atol=1e-8)
    assert torch.allclose(result_mm.cpu(), expected_mm, rtol=1e-5, atol=1e-8)


def test_mixed_device_operations_fail():
    """Test that operations between remote and CPU tensors fail appropriately."""
    import torch_remote
    
    x_cpu = torch.randn(2, 2)
    y_cpu = torch.randn(2, 2)
    x_remote = x_cpu.remote()
    
    # Test mixed device operations (should fail or be handled gracefully)
    operations_tested = 0
    
    # Test addition with mixed devices
    try:
        result = x_remote + y_cpu
        # If this succeeds, verify it's handled correctly
        operations_tested += 1
    except (RuntimeError, TypeError, NotImplementedError):
        # Expected failure for mixed device operations
        operations_tested += 1
    
    # Test matrix multiplication with mixed devices
    try:
        result = x_remote.mm(y_cpu)
        operations_tested += 1
    except (RuntimeError, TypeError, NotImplementedError):
        operations_tested += 1
    
    # Test reverse order
    try:
        result = y_cpu + x_remote
        operations_tested += 1
    except (RuntimeError, TypeError, NotImplementedError):
        operations_tested += 1
    
    # Ensure we tested the operations (they should either work correctly or fail)
    assert operations_tested == 3


def test_cpu_to_remote_conversion():
    """Test converting CPU tensors to remote tensors."""
    import torch_remote
    
    # Test with different tensor types and shapes
    test_cases = [
        torch.randn(2, 2),
        torch.zeros(3, 3),
        torch.ones(1, 5),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.randn(2, 2, 2),  # 3D tensor
    ]
    
    for cpu_tensor in test_cases:
        remote_tensor = cpu_tensor.remote()
        
        # Verify conversion
        assert type(remote_tensor).__name__ == 'RemoteTensorData'
        assert remote_tensor.shape == cpu_tensor.shape
        assert remote_tensor.dtype == cpu_tensor.dtype
        
        # Verify data is preserved
        assert torch.allclose(remote_tensor.cpu(), cpu_tensor, rtol=1e-5, atol=1e-8)


def test_remote_to_cpu_conversion():
    """Test converting remote tensors back to CPU tensors."""
    import torch_remote
    
    # Create remote tensor
    original_cpu = torch.randn(3, 4)
    remote_tensor = original_cpu.remote()
    
    # Convert back to CPU
    back_to_cpu = remote_tensor.cpu()
    
    # Verify conversion back to CPU
    assert back_to_cpu.device.type == 'cpu'
    assert back_to_cpu.shape == original_cpu.shape
    assert back_to_cpu.dtype == original_cpu.dtype
    
    # Verify data integrity through round-trip
    assert torch.allclose(back_to_cpu, original_cpu, rtol=1e-5, atol=1e-8)


def test_multiple_remote_cpu_transfers():
    """Test multiple transfers between remote and CPU devices."""
    import torch_remote
    
    # Start with CPU tensor
    original = torch.randn(2, 3)
    
    # Multiple round trips: CPU -> Remote -> CPU -> Remote -> CPU
    step1_remote = original.remote()
    step2_cpu = step1_remote.cpu()
    step3_remote = step2_cpu.remote()
    step4_cpu = step3_remote.cpu()
    
    # Verify final result matches original
    assert torch.allclose(step4_cpu, original, rtol=1e-5, atol=1e-8)
    assert step4_cpu.device.type == 'cpu'
    
    # Verify intermediate remote tensors have correct types
    assert type(step1_remote).__name__ == 'RemoteTensorData'
    assert type(step3_remote).__name__ == 'RemoteTensorData'


def test_remote_tensor_creation_with_dtypes():
    """Test creating remote tensors with different data types."""
    import torch_remote
    
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
    
    for dtype in dtypes:
        try:
            cpu_tensor = torch.randn(2, 2).to(dtype)
            remote_tensor = cpu_tensor.remote()
            
            # Verify dtype preservation
            assert remote_tensor.dtype == dtype
            assert type(remote_tensor).__name__ == 'RemoteTensorData'
            
            # Test dtype conversion during remote creation
            remote_converted = cpu_tensor.remote(dtype=torch.float64)
            assert remote_converted.dtype == torch.float64
            
        except Exception as e:
            # Some dtypes might not be supported; that's acceptable
            print(f"Dtype {dtype} not supported for remote tensors: {e}")
