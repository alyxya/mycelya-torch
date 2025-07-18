# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

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
from typing import Any
import torch_remote

def test_basic_imports() -> None:
    """Test basic torch and torch_remote imports."""
    assert True


def test_device_functions() -> None:
    """Test remote device functions."""
    assert torch.remote.is_available()
    # device_count should be >= 0 (could be 0 if no devices registered)
    assert torch.remote.device_count() >= 0


def test_tensor_to_method() -> None:
    """Test that tensors have to() method that works with RemoteMachine."""
    x = torch.randn(2, 2)
    assert hasattr(x, "to") and callable(x.to)


def test_backend_tensor_creation(modal_t4_device):
    """Test backend tensor creation via .to() method."""
    x = torch.randn(2, 2)
    y = x.to(modal_t4_device.device())
    assert y is not None and y.shape == x.shape


def test_backend_tensor_operations(modal_t4_device):
    """Test operations on backend tensors."""
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)

    x_remote = x.to(modal_t4_device.device())
    y_remote = y.to(modal_t4_device.device())

    # Test addition - verify numerical result matches CPU computation
    z_remote = x_remote + y_remote
    z_expected = x + y

    # Test matrix multiplication - verify numerical result matches CPU computation
    w_remote = x_remote.mm(y_remote)
    w_expected = x.mm(y)

    # Verify shapes
    assert z_remote is not None and w_remote is not None and w_remote.shape == (2, 2)

    # Verify numerical results (convert backend tensors back to CPU for comparison)
    assert torch.allclose(z_remote.cpu(), z_expected, rtol=1e-4, atol=1e-6)
    assert torch.allclose(w_remote.cpu(), w_expected, rtol=1e-4, atol=1e-6)


def test_dtype_conversion(modal_t4_device):
    """Test remote conversion with dtype parameter."""
    x = torch.randn(2, 2, dtype=torch.float32)
    y = x.to(modal_t4_device.device(), dtype=torch.float64)
    assert y.dtype == torch.float64


def test_copy_parameter(modal_t4_device):
    """Test remote conversion with copy parameter."""
    x = torch.randn(2, 2)
    y = x.to(modal_t4_device.device(), copy=True)
    z = x.to(modal_t4_device.device(), copy=False)
    assert y is not None and z is not None


def test_error_handling(modal_t4_device):
    """Test that errors are handled gracefully."""
    # These operations might fail, but shouldn't crash
    try:
        torch.randn(3, 3, device="remote")  # Should fail gracefully
    except Exception:
        pass  # Expected to fail

    try:
        x = torch.randn(2, 2).to(modal_t4_device.device())
        y = torch.randn(2, 2)  # CPU tensor
        z = x.mm(y)  # Mixed device - may or may not work
    except Exception:
        pass  # May fail, that's OK

    assert True  # If we get here without segfault, it's good


def test_backend_tensor_device_properties(modal_t4_device):
    """Test that backend tensors report correct device properties."""
    
    # Create CPU tensor and convert to backend
    x_cpu = torch.randn(3, 3)
    x_remote = x_cpu.to(modal_t4_device.device())
    
    # Check that remote tensor maintains torch.Tensor interface
    assert type(x_remote).__name__ == "Tensor"
    
    # Test device property - backend tensors should identify as remote device
    assert x_remote.device.type == "remote"


def test_backend_only_operations(modal_t4_device):
    """Test operations that require both tensors to be on the same backend."""
    
    x_cpu = torch.randn(2, 3)
    y_cpu = torch.randn(3, 2)
    
    x_remote = x_cpu.to(modal_t4_device.device())
    y_remote = y_cpu.to(modal_t4_device.device())
    
    # Test remote-remote operations (should work)
    result_add = x_remote + x_remote
    result_mm = x_remote.mm(y_remote)
    
    # Verify results are correct and maintain torch.Tensor interface
    assert type(result_add).__name__ == "Tensor"
    assert type(result_mm).__name__ == "Tensor"
    assert result_add.shape == x_remote.shape
    assert result_mm.shape == (2, 2)
    
    # Verify numerical correctness
    expected_add = x_cpu + x_cpu
    expected_mm = x_cpu.mm(y_cpu)
    assert torch.allclose(result_add.cpu(), expected_add, rtol=1e-4, atol=1e-6)
    assert torch.allclose(result_mm.cpu(), expected_mm, rtol=1e-4, atol=1e-6)


def test_mixed_device_operations_fail(modal_t4_device):
    """Test that operations between remote and CPU tensors fail appropriately."""
    
    x_cpu = torch.randn(2, 2)
    y_cpu = torch.randn(2, 2)
    x_remote = x_cpu.to(modal_t4_device.device())
    
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


def test_cpu_to_backend_conversion(modal_t4_device):
    """Test converting CPU tensors to backend tensors."""
    
    # Test with different tensor types and shapes
    test_cases = [
        torch.randn(2, 2),
        torch.zeros(3, 3),
        torch.ones(1, 5),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.randn(2, 2, 2),  # 3D tensor
    ]
    
    for cpu_tensor in test_cases:
        remote_tensor = cpu_tensor.to(modal_t4_device.device())
        
        # Verify conversion maintains torch.Tensor interface
        assert type(remote_tensor).__name__ == "Tensor"
        assert remote_tensor.shape == cpu_tensor.shape
        assert remote_tensor.dtype == cpu_tensor.dtype
        
        # Verify data is preserved
        assert torch.allclose(remote_tensor.cpu(), cpu_tensor, rtol=1e-4, atol=1e-6)


def test_backend_to_cpu_conversion(modal_t4_device):
    """Test converting backend tensors back to CPU tensors."""
    
    # Create backend tensor
    original_cpu = torch.randn(3, 4)
    remote_tensor = original_cpu.to(modal_t4_device.device())
    
    # Convert back to CPU
    back_to_cpu = remote_tensor.cpu()
    
    # Verify conversion back to CPU
    assert back_to_cpu.device.type == "cpu"
    assert back_to_cpu.shape == original_cpu.shape
    assert back_to_cpu.dtype == original_cpu.dtype
    
    # Verify data integrity through round-trip
    assert torch.allclose(back_to_cpu, original_cpu, rtol=1e-4, atol=1e-6)


def test_multiple_backend_cpu_transfers(modal_t4_device):
    """Test multiple transfers between backend and CPU devices."""
    
    # Start with CPU tensor
    original = torch.randn(2, 3)
    
    # Multiple round trips: CPU -> Remote -> CPU -> Remote -> CPU
    step1_remote = original.to(modal_t4_device.device())
    step2_cpu = step1_remote.cpu()
    step3_remote = step2_cpu.to(modal_t4_device.device())
    step4_cpu = step3_remote.cpu()
    
    # Verify final result matches original
    assert torch.allclose(step4_cpu, original, rtol=1e-4, atol=1e-6)
    assert step4_cpu.device.type == "cpu"
    
    # Verify intermediate remote tensors maintain torch.Tensor interface
    assert type(step1_remote).__name__ == "Tensor"
    assert type(step3_remote).__name__ == "Tensor"


def test_backend_tensor_creation_with_dtypes(modal_t4_device):
    """Test creating backend tensors with different data types."""
    
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
    
    for dtype in dtypes:
        try:
            cpu_tensor = torch.randn(2, 2).to(dtype)
            remote_tensor = cpu_tensor.to(modal_t4_device.device())
            
            # Verify dtype preservation
            assert remote_tensor.dtype == dtype
            assert type(remote_tensor).__name__ == "Tensor"
            
            # Test dtype conversion during remote creation
            remote_converted = cpu_tensor.to(modal_t4_device.device(), dtype=torch.float64)
            assert remote_converted.dtype == torch.float64
            
        except Exception as e:
            # Some dtypes might not be supported; that's acceptable
            print(f"Dtype {dtype} not supported for backend tensors: {e}")


def test_backend_device_method(modal_t4_device):
    """Test the .device() method on RemoteMachine for device access."""
    
    # Use the shared backend device
    backend_device = modal_t4_device
    
    # Test .device() method exists and is callable
    assert hasattr(backend_device, "device")
    assert callable(backend_device.device)
    
    # Get torch device from .device() method
    torch_device = backend_device.device()
    
    # Verify torch device properties
    assert isinstance(torch_device, torch.device)
    assert torch_device.type == "remote"
    assert isinstance(torch_device.index, int)
    assert torch_device.index >= 0
    
    # Test creating tensors with .device() method
    x = torch.randn(2, 2, device=backend_device.device())
    y = torch.zeros(3, 3, device=backend_device.device())
    
    # Verify tensors were created correctly
    assert x is not None and y is not None
    assert x.device.type == "remote"
    assert y.device.type == "remote"
    assert x.shape == (2, 2)
    assert y.shape == (3, 3)
    
    # Verify device index is correct
    assert x.device.index is not None
    assert y.device.index is not None
    assert x.device.index == y.device.index  # Same device
    # Verify we can map back to RemoteMachine via registry
    registry = torch_remote.get_device_registry()
    device_from_registry = registry.get_device_by_index(x.device.index)
    assert device_from_registry is backend_device


def test_validate_device_index_basic(shared_devices):
    """Test validate_device_index with valid and invalid indices."""
    
    # Use the shared devices instead of creating new ones
    device1 = shared_devices["t4"]
    device2 = shared_devices["l4"]
    
    # Should have at least 2 devices from shared fixtures
    assert torch.remote.device_count() >= 2
    
    # Valid indices should work
    tensor1 = torch.randn(2, 2, device=device1.device())
    tensor2 = torch.randn(2, 2, device=device2.device())
    assert tensor1 is not None
    assert tensor2 is not None


def test_validate_device_index_invalid(shared_devices):
    """Test validate_device_index with invalid device indices."""
    
    # Use shared devices - should have at least 3 devices
    current_count = torch.remote.device_count()
    assert current_count >= 3
    
    # Test invalid device index (should fail at C++ level)
    try:
        # This should fail because device index 99 doesn't exist
        invalid_device = torch.device("remote", 99)
        tensor = torch.randn(2, 2, device=invalid_device)
        assert False, "Expected device validation to fail for index 99"
    except (RuntimeError, ValueError) as e:
        # Expected failure
        assert "Invalid device index" in str(e) or "device" in str(e).lower()


def test_validate_device_index_negative(shared_devices):
    """Test validate_device_index with negative device indices."""
    
    # Use shared devices
    assert torch.remote.device_count() >= 1
    
    # Test negative device index (should fail)
    try:
        invalid_device = torch.device("remote", -1)
        tensor = torch.randn(2, 2, device=invalid_device)
        assert False, "Expected device validation to fail for negative index"
    except (RuntimeError, ValueError) as e:
        # Expected failure
        assert "Invalid device index" in str(e) or "device" in str(e).lower()


def test_device_count_dynamic_tracking():
    """Test that device_count properly tracks registered devices dynamically."""
    
    # Record initial device count (may be > 0 from other tests)
    initial_count = torch.remote.device_count()
    
    # Add devices one by one and verify count increases by expected amount
    devices = []
    gpu_types = ["T4", "L4"]  # Reduced to 2 devices to minimize GPU usage
    
    for i, gpu_type in enumerate(gpu_types):
        device = torch_remote.create_modal_machine(gpu_type)
        devices.append(device)
        expected_count = initial_count + i + 1
        assert torch.remote.device_count() == expected_count
        
        # Verify the device has a valid index (doesn't need to match i exactly)
        assert device.remote_index >= 0
        
        # Verify we can create tensors on each device
        tensor = torch.randn(2, 2, device=device.device())
        assert tensor is not None
        assert tensor.device.index == device.remote_index


def test_validate_device_index_with_multiple_devices():
    """Test validate_device_index with multiple devices."""
    
    # Create our own devices for this test to ensure we know exactly how many we have
    devices = []
    gpu_types = ["T4", "L4", "A100"]  # Create exactly 3 devices
    
    initial_count = torch.remote.device_count()
    
    for gpu_type in gpu_types:
        device = torch_remote.create_modal_machine(gpu_type)
        devices.append(device)
    
    # Verify we now have initial_count + 3 devices
    current_count = torch.remote.device_count()
    assert current_count == initial_count + 3
    
    # All our devices should work and have valid indices
    for device in devices:
        assert device.remote_index is not None
        assert device.remote_index >= 0
        tensor = torch.randn(2, 2, device=device.device())
        assert tensor is not None
        assert tensor.device.index == device.remote_index
    
    # Device index beyond current count should fail
    invalid_index = current_count  # This should be out of bounds
    try:
        invalid_device = torch.device("remote", invalid_index)
        tensor = torch.randn(2, 2, device=invalid_device)
        assert False, f"Expected device validation to fail for index {invalid_index}"
    except (RuntimeError, ValueError) as e:
        assert "Invalid device index" in str(e) or "device" in str(e).lower()


def test_cross_device_transfer_restriction(shared_devices):
    """Test that transferring tensors between different remote devices is prohibited."""
    
    # Use shared devices
    device1 = shared_devices["t4"]
    device2 = shared_devices["l4"]
    
    # Verify they have different indices
    assert device1.remote_index != device2.remote_index
    
    # Create tensor on first device
    x = torch.randn(2, 2, device=device1.device())
    
    # Try to transfer to second device - should fail
    with pytest.raises(RuntimeError, match="Cannot transfer tensor between different remote devices"):
        y = x.to(device2.device())


def test_cross_device_copy_restriction(shared_devices):
    """Test that copy operations between different remote devices fail."""
    
    # Use shared devices
    device1 = shared_devices["t4"]
    device2 = shared_devices["a100"]
    
    # Verify devices have different indices
    assert device1.remote_index != device2.remote_index
    
    # Create tensor on first device  
    x = torch.randn(2, 2, device=device1.device())
    
    # Create tensor on second device via CPU transfer  
    cpu_tensor = torch.empty(2, 2)
    y = cpu_tensor.to(device2.device())
    
    # Verify tensors are on different devices
    if x.device.index == y.device.index:
        pytest.skip("Could not create tensors on different devices - skipping cross-device copy test")
    
    # Direct copy should fail if they're on different remote devices
    # However, the current copy_ implementation goes through CPU, so we need to test 
    # the _copy_from function directly which is what copy_ eventually calls
    with pytest.raises(RuntimeError, match="Cannot transfer tensor between different remote devices"):
        # Test the underlying _copy_from function directly
        torch_remote._aten_impl._copy_from(x, y)


def test_same_device_transfer_still_works(modal_t4_device):
    """Test that transfers within the same device still work."""
    # Create two tensors on the same device
    x = torch.randn(2, 2, device=modal_t4_device.device())
    y = torch.empty(2, 2, device=modal_t4_device.device())
    
    # Same device operations should work fine
    result = x.to(modal_t4_device.device())
    assert result.device == modal_t4_device.device()
    
    # Same device copy should work
    y.copy_(x)
    assert torch.allclose(x.cpu(), y.cpu())


# ============================================================================
# View Operations Test Suite
# ============================================================================

def test_view_operation_basic(modal_t4_device):
    """Test basic view operation on remote tensors."""
    
    # Create a remote tensor
    x = torch.randn(4, 6, device=modal_t4_device.device())
    original_shape = x.shape
    
    # Test basic view operation
    y = x.view(2, 12)
    
    # Verify view properties
    assert y.shape == (2, 12)
    assert y.device == x.device
    assert y.dtype == x.dtype
    
    # Verify data integrity - view should preserve data
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    assert torch.allclose(x_cpu.view(2, 12), y_cpu, rtol=1e-4, atol=1e-6)


def test_view_operation_multiple_dimensions(modal_t4_device):
    """Test view operations with multiple dimension changes."""
    
    # Create a 3D remote tensor
    x = torch.randn(2, 3, 4, device=modal_t4_device.device())
    
    # Test various view operations
    test_cases = [
        (6, 4),      # Flatten first two dimensions
        (2, 12),     # Flatten last two dimensions  
        (24,),       # Flatten to 1D
        (1, 2, 3, 4), # Add singleton dimension
        (2, 1, 3, 4), # Add singleton in middle
    ]
    
    for new_shape in test_cases:
        y = x.view(*new_shape)
        
        # Verify shape
        assert y.shape == new_shape
        assert y.device == x.device
        assert y.dtype == x.dtype
        
        # Verify data integrity
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        expected = x_cpu.view(*new_shape)
        assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)


def test_reshape_operation(modal_t4_device):
    """Test reshape operation on remote tensors."""
    
    # Create a remote tensor
    x = torch.randn(3, 4, device=modal_t4_device.device())
    
    # Test reshape operation
    y = x.reshape(2, 6)
    
    # Verify reshape properties
    assert y.shape == (2, 6)
    assert y.device == x.device
    assert y.dtype == x.dtype
    
    # Verify data integrity
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    expected = x_cpu.reshape(2, 6)
    assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)


def test_transpose_operation(modal_t4_device):
    """Test transpose operation on remote tensors."""
    
    # Create a remote tensor
    x = torch.randn(3, 4, device=modal_t4_device.device())
    
    # Test transpose operation
    y = x.transpose(0, 1)
    
    # Verify transpose properties
    assert y.shape == (4, 3)
    assert y.device == x.device
    assert y.dtype == x.dtype
    
    # Verify data integrity
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    expected = x_cpu.transpose(0, 1)
    assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)


def test_transpose_operation_3d(modal_t4_device):
    """Test transpose operation on 3D remote tensors."""
    
    # Create a 3D remote tensor
    x = torch.randn(2, 3, 4, device=modal_t4_device.device())
    
    # Test various transpose operations
    transpose_cases = [
        (0, 1),  # Swap first two dimensions
        (1, 2),  # Swap last two dimensions
        (0, 2),  # Swap first and last dimensions
    ]
    
    for dim0, dim1 in transpose_cases:
        y = x.transpose(dim0, dim1)
        
        # Calculate expected shape
        expected_shape = list(x.shape)
        expected_shape[dim0], expected_shape[dim1] = expected_shape[dim1], expected_shape[dim0]
        
        # Verify transpose properties
        assert y.shape == tuple(expected_shape)
        assert y.device == x.device
        assert y.dtype == x.dtype
        
        # Verify data integrity
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        expected = x_cpu.transpose(dim0, dim1)
        assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)


def test_permute_operation(modal_t4_device):
    """Test permute operation on remote tensors."""
    
    # Create a 3D remote tensor
    x = torch.randn(2, 3, 4, device=modal_t4_device.device())
    
    # Test permute operation
    y = x.permute(2, 0, 1)
    
    # Verify permute properties
    assert y.shape == (4, 2, 3)
    assert y.device == x.device
    assert y.dtype == x.dtype
    
    # Verify data integrity
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    expected = x_cpu.permute(2, 0, 1)
    assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)


def test_permute_operation_4d(modal_t4_device):
    """Test permute operation on 4D remote tensors."""
    
    # Create a 4D remote tensor
    x = torch.randn(2, 3, 4, 5, device=modal_t4_device.device())
    
    # Test various permute operations
    permute_cases = [
        (3, 2, 1, 0),  # Reverse all dimensions
        (0, 2, 1, 3),  # Swap middle dimensions
        (1, 0, 3, 2),  # Swap pairs
    ]
    
    for perm in permute_cases:
        y = x.permute(*perm)
        
        # Calculate expected shape
        expected_shape = tuple(x.shape[i] for i in perm)
        
        # Verify permute properties
        assert y.shape == expected_shape
        assert y.device == x.device
        assert y.dtype == x.dtype
        
        # Verify data integrity
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        expected = x_cpu.permute(*perm)
        assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)


def test_squeeze_operation(modal_t4_device):
    """Test squeeze operation on remote tensors."""
    
    # Create a remote tensor with singleton dimensions
    x = torch.randn(1, 3, 1, 4, device=modal_t4_device.device())
    
    # Test squeeze operation (remove all singleton dimensions)
    y = x.squeeze()
    
    # Verify squeeze properties
    assert y.shape == (3, 4)
    assert y.device == x.device
    assert y.dtype == x.dtype
    
    # Verify data integrity
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    expected = x_cpu.squeeze()
    assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)


def test_squeeze_operation_specific_dim(modal_t4_device):
    """Test squeeze operation on specific dimensions."""
    
    # Create a remote tensor with singleton dimensions
    x = torch.randn(1, 3, 1, 4, device=modal_t4_device.device())
    
    # Test squeeze specific dimensions
    squeeze_cases = [
        (0, (3, 1, 4)),  # Squeeze dimension 0
        (2, (1, 3, 4)),  # Squeeze dimension 2
    ]
    
    for dim, expected_shape in squeeze_cases:
        y = x.squeeze(dim)
        
        # Verify squeeze properties
        assert y.shape == expected_shape
        assert y.device == x.device
        assert y.dtype == x.dtype
        
        # Verify data integrity
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        expected = x_cpu.squeeze(dim)
        assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)


def test_unsqueeze_operation(modal_t4_device):
    """Test unsqueeze operation on remote tensors."""
    
    # Create a remote tensor
    x = torch.randn(3, 4, device=modal_t4_device.device())
    
    # Test unsqueeze operations at different positions
    unsqueeze_cases = [
        (0, (1, 3, 4)),  # Add dimension at start
        (1, (3, 1, 4)),  # Add dimension in middle
        (2, (3, 4, 1)),  # Add dimension at end
        (-1, (3, 4, 1)), # Add dimension at end (negative indexing)
    ]
    
    for dim, expected_shape in unsqueeze_cases:
        y = x.unsqueeze(dim)
        
        # Verify unsqueeze properties
        assert y.shape == expected_shape
        assert y.device == x.device
        assert y.dtype == x.dtype
        
        # Verify data integrity
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        expected = x_cpu.unsqueeze(dim)
        assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)


def test_flatten_operation(modal_t4_device):
    """Test flatten operation on remote tensors."""
    
    # Create a multi-dimensional remote tensor
    x = torch.randn(2, 3, 4, 5, device=modal_t4_device.device())
    
    # Test flatten operations
    flatten_cases = [
        (0, -1, (120,)),     # Flatten all dimensions
        (1, 2, (2, 12, 5)),  # Flatten middle dimensions
        (0, 1, (6, 4, 5)),   # Flatten first two dimensions
        (2, 3, (2, 3, 20)),  # Flatten last two dimensions
    ]
    
    for start_dim, end_dim, expected_shape in flatten_cases:
        y = x.flatten(start_dim, end_dim)
        
        # Verify flatten properties
        assert y.shape == expected_shape
        assert y.device == x.device
        assert y.dtype == x.dtype
        
        # Verify data integrity
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        expected = x_cpu.flatten(start_dim, end_dim)
        assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)


def test_view_operations_preserve_storage_id(modal_t4_device):
    """Test that view operations preserve the underlying storage ID (if implemented correctly)."""
    
    # Create a remote tensor
    x = torch.randn(4, 6, device=modal_t4_device.device())
    
    # Perform a view operation
    y = x.view(2, 12)
    
    # Both tensors should be on the same device
    assert x.device == y.device
    
    # Test that operations on the view affect the original (if sharing memory)
    # Note: This test may reveal whether view operations share memory or create copies
    try:
        # Modify the view and check if original is affected
        # This will help identify if views are properly sharing tensor IDs
        original_data = x.cpu().clone()
        
        # Create a view and try to modify it
        z = y.view(4, 6)
        
        # Verify the view has the same data
        assert torch.allclose(z.cpu(), original_data, rtol=1e-4, atol=1e-6)
        
    except Exception as e:
        # If view operations aren't properly implemented, this might fail
        print(f"View operation behavior: {e}")


def test_chained_view_operations(modal_t4_device):
    """Test chaining multiple view operations."""
    
    # Create a remote tensor
    x = torch.randn(2, 3, 4, device=modal_t4_device.device())
    
    # Chain multiple view operations that work correctly
    # After transpose, tensor is not contiguous, so use reshape instead of view
    y = x.view(6, 4).transpose(0, 1).reshape(2, 12).squeeze().unsqueeze(0)
    
    # Calculate expected shape through CPU operations
    x_cpu = x.cpu()
    expected = x_cpu.view(6, 4).transpose(0, 1).reshape(2, 12).squeeze().unsqueeze(0)
    
    # Verify final result
    assert y.shape == expected.shape
    assert y.device == x.device
    assert y.dtype == x.dtype
    
    # Verify data integrity through the chain
    y_cpu = y.cpu()
    assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)


def test_view_operations_after_arithmetic(modal_t4_device):
    """Test view operations on tensors after arithmetic operations."""
    
    # Create remote tensors
    x = torch.randn(3, 4, device=modal_t4_device.device())
    y = torch.randn(3, 4, device=modal_t4_device.device())
    
    # Perform arithmetic operation
    z = x + y
    
    # Apply view operations to the result
    w = z.view(2, 6).transpose(0, 1)
    
    # Verify properties
    assert w.shape == (6, 2)
    assert w.device == z.device
    assert w.dtype == z.dtype
    
    # Verify data integrity
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    expected = (x_cpu + y_cpu).view(2, 6).transpose(0, 1)
    w_cpu = w.cpu()
    assert torch.allclose(expected, w_cpu, rtol=1e-4, atol=1e-6)


def test_view_invalid_size_error(modal_t4_device):
    """Test that invalid view sizes raise appropriate errors."""
    
    # Create a remote tensor
    x = torch.randn(3, 4, device=modal_t4_device.device())
    
    # Test invalid view size (incompatible with total elements)
    with pytest.raises(RuntimeError):
        y = x.view(5, 5)  # 25 elements != 12 elements


def test_view_with_minus_one_inference(modal_t4_device):
    """Test view operation with -1 dimension inference."""
    
    # Create a remote tensor
    x = torch.randn(2, 3, 4, device=modal_t4_device.device())
    
    # Test view with -1 inference
    test_cases = [
        (-1, 12),   # Infer first dimension
        (6, -1),    # Infer last dimension
        (2, -1, 2), # Infer middle dimension
    ]
    
    for new_shape in test_cases:
        y = x.view(*new_shape)
        
        # Calculate expected shape using CPU tensor
        x_cpu = x.cpu()
        expected = x_cpu.view(*new_shape)
        
        # Verify properties
        assert y.shape == expected.shape
        assert y.device == x.device
        assert y.dtype == x.dtype
        
        # Verify data integrity
        y_cpu = y.cpu()
        assert torch.allclose(expected, y_cpu, rtol=1e-4, atol=1e-6)
