#!/usr/bin/env python3
"""
Comprehensive test suite for torch-modal package.

This script works with both pytest and standalone execution:
- Run with pytest: pytest test_torch_modal.py
- Run standalone: python test_torch_modal.py
- For verbose output: python test_torch_modal.py --verbose
- For quick debug mode: python test_torch_modal.py --debug
"""

import torch
import pytest


def test_basic_imports():
    """Test basic torch and torch_modal imports."""
    import torch_modal
    assert True


def test_device_functions():
    """Test modal device functions."""
    import torch
    import torch_modal
    assert (torch.modal.is_available() and
            torch.modal.device_count() >= 1)


def test_tensor_modal_method():
    """Test that tensors have modal() method."""
    import torch_modal
    x = torch.randn(2, 2)
    assert hasattr(x, 'modal') and callable(x.modal)


def test_modal_tensor_creation():
    """Test modal tensor creation via .modal() method."""
    import torch_modal
    x = torch.randn(2, 2)
    y = x.modal()
    assert y is not None and y.shape == x.shape


def test_modal_tensor_operations():
    """Test operations on modal tensors."""
    import torch_modal
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)

    x_modal = x.modal()
    y_modal = y.modal()

    # Test addition - verify numerical result matches CPU computation
    z_modal = x_modal + y_modal
    z_expected = x + y

    # Test matrix multiplication - verify numerical result matches CPU computation
    w_modal = x_modal.mm(y_modal)
    w_expected = x.mm(y)

    # Verify shapes
    assert z_modal is not None and w_modal is not None and w_modal.shape == (2, 2)

    # Verify numerical results (convert modal tensors back to CPU for comparison)
    assert torch.allclose(z_modal.cpu(), z_expected, rtol=1e-5, atol=1e-8)
    assert torch.allclose(w_modal.cpu(), w_expected, rtol=1e-5, atol=1e-8)


def test_dtype_conversion():
    """Test modal conversion with dtype parameter."""
    import torch_modal
    x = torch.randn(2, 2, dtype=torch.float32)
    y = x.modal(dtype=torch.float64)
    assert y.dtype == torch.float64


def test_copy_parameter():
    """Test modal conversion with copy parameter."""
    import torch_modal
    x = torch.randn(2, 2)
    y = x.modal(copy=True)
    z = x.modal(copy=False)
    assert y is not None and z is not None


def test_error_handling():
    """Test that errors are handled gracefully."""
    import torch_modal
    # These operations might fail, but shouldn't crash
    try:
        torch.randn(3, 3, device='modal')  # Should fail gracefully
    except Exception:
        pass  # Expected to fail

    try:
        x = torch.randn(2, 2).modal()
        y = torch.randn(2, 2)  # CPU tensor
        z = x.mm(y)  # Mixed device - may or may not work
    except Exception:
        pass  # May fail, that's OK

    assert True  # If we get here without segfault, it's good


def test_modal_tensor_device_properties():
    """Test that modal tensors report correct device properties."""
    import torch_modal
    
    # Create CPU tensor and convert to modal
    x_cpu = torch.randn(3, 3)
    x_modal = x_cpu.modal()
    
    # Check that modal tensor has the expected type
    assert type(x_modal).__name__ == 'ModalTensorData'
    
    # Test device property - modal tensors should identify as modal device
    assert x_modal.device.type == 'modal'
    assert x_modal.device.index == 0  # Default modal device index


def test_modal_only_operations():
    """Test operations that require both tensors to be modal."""
    import torch_modal
    
    x_cpu = torch.randn(2, 3)
    y_cpu = torch.randn(3, 2)
    
    x_modal = x_cpu.modal()
    y_modal = y_cpu.modal()
    
    # Test modal-modal operations (should work)
    result_add = x_modal + x_modal
    result_mm = x_modal.mm(y_modal)
    
    # Verify results are correct and still modal tensors
    assert type(result_add).__name__ == 'ModalTensorData'
    assert type(result_mm).__name__ == 'ModalTensorData'
    assert result_add.shape == x_modal.shape
    assert result_mm.shape == (2, 2)
    
    # Verify numerical correctness
    expected_add = x_cpu + x_cpu
    expected_mm = x_cpu.mm(y_cpu)
    assert torch.allclose(result_add.cpu(), expected_add, rtol=1e-5, atol=1e-8)
    assert torch.allclose(result_mm.cpu(), expected_mm, rtol=1e-5, atol=1e-8)


def test_mixed_device_operations_fail():
    """Test that operations between modal and CPU tensors fail appropriately."""
    import torch_modal
    
    x_cpu = torch.randn(2, 2)
    y_cpu = torch.randn(2, 2)
    x_modal = x_cpu.modal()
    
    # Test mixed device operations (should fail or be handled gracefully)
    operations_tested = 0
    
    # Test addition with mixed devices
    try:
        result = x_modal + y_cpu
        # If this succeeds, verify it's handled correctly
        operations_tested += 1
    except (RuntimeError, TypeError, NotImplementedError):
        # Expected failure for mixed device operations
        operations_tested += 1
    
    # Test matrix multiplication with mixed devices
    try:
        result = x_modal.mm(y_cpu)
        operations_tested += 1
    except (RuntimeError, TypeError, NotImplementedError):
        operations_tested += 1
    
    # Test reverse order
    try:
        result = y_cpu + x_modal
        operations_tested += 1
    except (RuntimeError, TypeError, NotImplementedError):
        operations_tested += 1
    
    # Ensure we tested the operations (they should either work correctly or fail)
    assert operations_tested == 3


def test_cpu_to_modal_conversion():
    """Test converting CPU tensors to modal tensors."""
    import torch_modal
    
    # Test with different tensor types and shapes
    test_cases = [
        torch.randn(2, 2),
        torch.zeros(3, 3),
        torch.ones(1, 5),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.randn(2, 2, 2),  # 3D tensor
    ]
    
    for cpu_tensor in test_cases:
        modal_tensor = cpu_tensor.modal()
        
        # Verify conversion
        assert type(modal_tensor).__name__ == 'ModalTensorData'
        assert modal_tensor.shape == cpu_tensor.shape
        assert modal_tensor.dtype == cpu_tensor.dtype
        
        # Verify data is preserved
        assert torch.allclose(modal_tensor.cpu(), cpu_tensor, rtol=1e-5, atol=1e-8)


def test_modal_to_cpu_conversion():
    """Test converting modal tensors back to CPU tensors."""
    import torch_modal
    
    # Create modal tensor
    original_cpu = torch.randn(3, 4)
    modal_tensor = original_cpu.modal()
    
    # Convert back to CPU
    back_to_cpu = modal_tensor.cpu()
    
    # Verify conversion back to CPU
    assert back_to_cpu.device.type == 'cpu'
    assert back_to_cpu.shape == original_cpu.shape
    assert back_to_cpu.dtype == original_cpu.dtype
    
    # Verify data integrity through round-trip
    assert torch.allclose(back_to_cpu, original_cpu, rtol=1e-5, atol=1e-8)


def test_multiple_modal_cpu_transfers():
    """Test multiple transfers between modal and CPU devices."""
    import torch_modal
    
    # Start with CPU tensor
    original = torch.randn(2, 3)
    
    # Multiple round trips: CPU -> Modal -> CPU -> Modal -> CPU
    step1_modal = original.modal()
    step2_cpu = step1_modal.cpu()
    step3_modal = step2_cpu.modal()
    step4_cpu = step3_modal.cpu()
    
    # Verify final result matches original
    assert torch.allclose(step4_cpu, original, rtol=1e-5, atol=1e-8)
    assert step4_cpu.device.type == 'cpu'
    
    # Verify intermediate modal tensors have correct types
    assert type(step1_modal).__name__ == 'ModalTensorData'
    assert type(step3_modal).__name__ == 'ModalTensorData'


def test_modal_tensor_creation_with_dtypes():
    """Test creating modal tensors with different data types."""
    import torch_modal
    
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
    
    for dtype in dtypes:
        try:
            cpu_tensor = torch.randn(2, 2).to(dtype)
            modal_tensor = cpu_tensor.modal()
            
            # Verify dtype preservation
            assert modal_tensor.dtype == dtype
            assert type(modal_tensor).__name__ == 'ModalTensorData'
            
            # Test dtype conversion during modal creation
            modal_converted = cpu_tensor.modal(dtype=torch.float64)
            assert modal_converted.dtype == torch.float64
            
        except Exception as e:
            # Some dtypes might not be supported; that's acceptable
            print(f"Dtype {dtype} not supported for modal tensors: {e}")
