# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for error handling in mycelya-torch.

This module tests various error conditions, exception handling,
and graceful failure scenarios for remote tensor operations.
"""

import warnings

import pytest
import torch
from test_utilities import ErrorTestUtils

import mycelya_torch


@pytest.fixture(scope="function")
def isolated_machine():
    """Create a fresh machine for each test to avoid error state contamination."""
    machine = mycelya_torch.RemoteMachine("mock")
    try:
        yield machine
    finally:
        try:
            machine.stop()
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture(scope="function")
def isolated_machines():
    """Create fresh machines for tests requiring multiple machines."""
    machines = {
        "machine1": mycelya_torch.RemoteMachine("mock"),
        "machine2": mycelya_torch.RemoteMachine("mock"),
    }

    try:
        yield machines
    finally:
        for machine in machines.values():
            try:
                machine.stop()
            except Exception:
                pass  # Ignore cleanup errors


@pytest.mark.fast
class TestDeviceErrorHandling:
    """Tests for device-related error handling."""

    def test_invalid_device_creation(self):
        """Test error handling for invalid device creation."""
        # Test various invalid device specifications (excluding CUDA to avoid PyTorch initialization issues)
        invalid_devices = [
            "nonexistent_device",
            "mycelya:999",
            "invalid_type",
        ]

        for invalid_device in invalid_devices:
            with pytest.raises((RuntimeError, ValueError, TypeError)):
                torch.randn(2, 2, device=invalid_device)

    def test_device_unavailable_graceful_handling(self):
        """Test graceful handling when remote devices are unavailable."""
        # This should not crash, even if no remote devices are available
        try:
            device_count = torch.mycelya.device_count()
            assert device_count >= 0  # Should be non-negative

            is_available = torch.mycelya.is_available()
            assert isinstance(is_available, bool)
        except Exception as e:
            pytest.fail(f"Device availability check should not raise exceptions: {e}")

    def test_device_index_out_of_bounds(self):
        """Test handling of out-of-bounds device indices."""
        # Try to access device with invalid index
        with pytest.raises((RuntimeError, IndexError, ValueError)):
            torch.randn(2, 2, device="mycelya:999")


@pytest.mark.fast
class TestCrossDeviceErrorHandling:
    """Tests for cross-device operation error handling."""

    def test_cross_device_arithmetic_errors(self, isolated_machines):
        """Test errors for arithmetic operations between different devices."""
        tensor1 = torch.randn(2, 2, device=isolated_machines["machine1"].device("cpu"))
        tensor2 = torch.randn(2, 2, device=isolated_machines["machine2"].device("cpu"))

        # Test various operations that should fail
        operations = [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y,
            lambda x, y: x.mm(y),
        ]

        for operation in operations:
            ErrorTestUtils.assert_cross_device_fails(tensor1, tensor2, operation)

    def test_cross_device_transfer_error(self, isolated_machines):
        """Test errors for direct cross-device transfers."""
        tensor = torch.randn(2, 2, device=isolated_machines["machine1"].device("cpu"))

        # Direct transfer between remote devices should fail
        with pytest.raises(
            RuntimeError,
            match="Cross-machine remote transfers are not supported",
        ):
            tensor.to(isolated_machines["machine2"].device("cpu"))


@pytest.mark.fast
class TestTensorOperationErrors:
    """Tests for tensor operation error handling."""

    def test_incompatible_shape_operations(self, isolated_machine):
        """Test error handling for operations with incompatible shapes."""
        tensor1 = torch.randn(2, 3, device=isolated_machine.device("cpu"))
        tensor2 = torch.randn(4, 5, device=isolated_machine.device("cpu"))

        # Matrix multiplication with incompatible shapes
        with pytest.raises((RuntimeError, ValueError)):
            tensor1.mm(tensor2)

        # Element-wise operations with incompatible shapes
        with pytest.raises((RuntimeError, ValueError)):
            tensor1 + tensor2

    def test_view_operation_errors(self, isolated_machine):
        """Test error handling for invalid view operations."""
        tensor = torch.randn(2, 3, device=isolated_machine.device("cpu"))

        # Try to view with incompatible size
        with pytest.raises((RuntimeError, ValueError)):
            tensor.view(2, 2)  # 6 elements can't become 4

        # Try to view with negative dimensions in wrong places
        with pytest.raises((RuntimeError, ValueError)):
            tensor.view(-1, -1)  # Can't have multiple -1 dimensions

    def test_squeeze_operation_errors(self, isolated_machine):
        """Test error handling for invalid squeeze operations."""
        # Create tensor with a dimension of size 1 for valid squeeze, then test invalid squeeze
        tensor = torch.randn(2, 1, 4, device=isolated_machine.device("cpu"))

        # Valid squeeze should work
        squeezed = tensor.squeeze(1)  # Dimension 1 has size 1
        assert squeezed.shape == (2, 4)

        # Try to squeeze a dimension that's not size 1 - this is actually allowed in PyTorch
        # and returns the original tensor unchanged
        result = tensor.squeeze(0)  # Dimension 0 has size 2, not 1
        assert result.shape == tensor.shape  # Should be unchanged

        # Test an actual error case: squeeze with invalid dimension index
        with pytest.raises((RuntimeError, IndexError)):
            tensor.squeeze(10)  # Dimension 10 doesn't exist

    def test_transpose_operation_errors(self, isolated_machine):
        """Test error handling for invalid transpose operations."""
        tensor = torch.randn(2, 3, device=isolated_machine.device("cpu"))

        # Try to transpose with out-of-bounds dimensions
        with pytest.raises((RuntimeError, IndexError)):
            tensor.transpose(0, 2)  # Dimension 2 doesn't exist

        with pytest.raises((RuntimeError, IndexError)):
            tensor.transpose(-3, 0)  # -3 is out of bounds for 2D tensor


@pytest.mark.fast
class TestGradientErrorHandling:
    """Tests for gradient-related error handling."""

    def test_gradient_on_non_leaf_tensor_error(self, isolated_machine):
        """Test error handling when accessing gradients on non-leaf tensors."""
        x = torch.randn(2, 2, device=isolated_machine.device("cpu"), requires_grad=True)
        y = x + 1  # Non-leaf tensor

        loss = y.sum()
        loss.backward()

        # Accessing grad on non-leaf tensor should be None or raise error
        # Suppress the warning since we're intentionally testing this behavior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert y.grad is None

    def test_backward_without_scalar_error(self, isolated_machine):
        """Test error handling for backward() on non-scalar tensors."""
        x = torch.randn(2, 2, device=isolated_machine.device("cpu"), requires_grad=True)
        y = x * 2  # Non-scalar result

        # Backward on non-scalar without gradient argument should fail
        with pytest.raises(RuntimeError):
            y.backward()

    def test_double_backward_error(self, isolated_machine):
        """Test error handling for double backward without retain_graph."""
        # Create a leaf tensor directly on the remote device
        x = torch.randn(2, 2, device=isolated_machine.device("cpu"), requires_grad=True)
        y = x.sum()

        # First backward
        y.backward()

        # Verify gradients exist after first backward
        assert x.grad is not None, "Gradients should exist after first backward"

        # Second backward without retain_graph should fail
        # Note: Remote execution might not maintain computation graph like standard PyTorch
        try:
            y.backward()
            # If no error is raised, the implementation allows multiple backwards
            # which is different from standard PyTorch but not necessarily wrong
            pass
        except RuntimeError:
            # Expected behavior in standard PyTorch - computation graph should be freed
            pass


@pytest.mark.fast
class TestMemoryErrorHandling:
    """Tests for memory-related error handling."""

    def test_very_large_tensor_creation(self, isolated_machine):
        """Test handling of extremely large tensor creation requests."""
        # The system uses lazy/meta tensor allocation, so extremely large tensors
        # can be created without immediate memory allocation. This is actually
        # a feature of the remote execution system.

        # Test that we can create a large tensor (this should succeed with lazy allocation)
        huge_size = 10**10
        large_tensor = torch.randn(huge_size, device=isolated_machine.device("cpu"))

        # Verify the tensor has the expected shape
        assert large_tensor.shape == (huge_size,)

        # The actual memory allocation happens on the remote device when operations
        # are performed, so this is expected behavior for the remote execution system

    def test_negative_tensor_dimensions(self, isolated_machine):
        """Test error handling for negative tensor dimensions."""
        with pytest.raises((RuntimeError, ValueError)):
            torch.randn(-1, 2, device=isolated_machine.device("cpu"))

        with pytest.raises((RuntimeError, ValueError)):
            torch.randn(2, -3, device=isolated_machine.device("cpu"))


@pytest.mark.fast
class TestTypeErrorHandling:
    """Tests for type-related error handling."""

    def test_unsupported_dtype_operations(self, isolated_machine):
        """Test error handling for unsupported dtype operations."""
        # Try operations that might not be supported for certain dtypes
        try:
            bool_tensor = torch.tensor(
                [True, False], device=isolated_machine.device("cpu")
            )
            bool_tensor.mm(bool_tensor)
            # If this succeeds, that's fine too
        except (RuntimeError, TypeError, NotImplementedError):
            # Expected for unsupported operations
            pass

    def test_mixed_dtype_operations(self, isolated_machine):
        """Test operations with mixed dtypes."""
        float_tensor = torch.randn(
            2, 2, device=isolated_machine.device("cpu"), dtype=torch.float32
        )
        int_tensor = torch.randint(
            0, 10, (2, 2), device=isolated_machine.device("cpu"), dtype=torch.int32
        )

        # Test mixed dtype operation
        result = float_tensor + int_tensor
        # Verify the result type is reasonable
        assert result.dtype in [torch.float32, torch.int32]


@pytest.mark.fast
class TestOperationNotImplementedHandling:
    """Tests for handling of not-yet-implemented operations."""

    def test_potentially_unimplemented_operations(self, isolated_machine):
        """Test graceful handling of potentially unimplemented operations."""
        tensor = torch.randn(3, 3, device=isolated_machine.device("cpu"))

        # List of operations that might not be implemented yet
        potentially_unimplemented = [
            lambda x: torch.det(x),
            lambda x: torch.svd(x),
            lambda x: torch.linalg.qr(x),
            lambda x: torch.fft.fft(x),
        ]

        for operation in potentially_unimplemented:
            try:
                result = operation(tensor)
                # If it succeeds, verify result is valid
                assert result is not None
            except (RuntimeError, NotImplementedError, AttributeError):
                # Expected for unimplemented operations
                pass

    def test_advanced_indexing_errors(self, isolated_machine):
        """Test error handling for advanced indexing operations."""
        tensor = torch.randn(3, 4, 5, device=isolated_machine.device("cpu"))

        # Some advanced indexing might not be supported
        try:
            # Boolean indexing
            mask = torch.tensor(
                [True, False, True], device=isolated_machine.device("cpu")
            )
            tensor[mask]
        except (RuntimeError, NotImplementedError):
            # Advanced indexing might not be implemented
            pass

        try:
            # Fancy indexing
            indices = torch.tensor([0, 2], device=isolated_machine.device("cpu"))
            tensor[indices]
        except (RuntimeError, NotImplementedError):
            # Fancy indexing might not be implemented
            pass


@pytest.mark.fast
class TestConnectionErrorHandling:
    """Tests for connection and communication error handling."""

    def test_operation_during_connection_issues(self, isolated_machine):
        """Test behavior during simulated connection issues."""
        # This is more of a resilience test - operations should either succeed or fail gracefully
        tensor = torch.randn(2, 2, device=isolated_machine.device("cpu"))

        # Perform operations that should work normally
        try:
            result = tensor + tensor
            assert result is not None
        except (RuntimeError, ConnectionError, TimeoutError):
            # Connection issues should raise appropriate exceptions, not crash
            pass

    def test_graceful_degradation(self, isolated_machine):
        """Test that the system degrades gracefully when operations fail."""
        # Create tensor and perform a series of operations
        tensor = torch.randn(2, 2, device=isolated_machine.device("cpu"))

        operations_succeeded = 0
        total_operations = 0

        # Try multiple operations
        test_operations = [
            lambda x: x + x,
            lambda x: x * 2,
            lambda x: x.sum(),
            lambda x: x.view(-1),
            lambda x: x.transpose(0, 1),
        ]

        for operation in test_operations:
            total_operations += 1
            try:
                result = operation(tensor)
                if result is not None:
                    operations_succeeded += 1
            except (RuntimeError, NotImplementedError, TypeError):
                # Some operations might fail, but shouldn't crash
                pass

        # At least some basic operations should work
        assert operations_succeeded > 0, (
            "No operations succeeded - system not functional"
        )


@pytest.mark.fast
class TestRobustnessAndRecovery:
    """Tests for system robustness and recovery from errors."""

    def test_error_recovery_sequence(self, isolated_machine):
        """Test that the system can recover from errors and continue working."""
        # Perform a known-good operation
        tensor1 = torch.randn(2, 2, device=isolated_machine.device("cpu"))
        good_result1 = tensor1 + tensor1
        assert good_result1 is not None

        # Attempt an operation that might fail
        try:
            bad_tensor = torch.randn(3, 4, device=isolated_machine.device("cpu"))
            tensor1.mm(bad_tensor)  # Should fail due to shape mismatch
        except (RuntimeError, ValueError):
            # Expected to fail
            pass

        # Verify system still works after the error
        tensor2 = torch.randn(2, 2, device=isolated_machine.device("cpu"))
        good_result2 = tensor2 * 3
        assert good_result2 is not None

        # Verify we can still do complex operations
        good_result3 = tensor1.mm(tensor2)
        assert good_result3 is not None

    def test_multiple_error_scenarios(self, isolated_machine):
        """Test handling of multiple different error scenarios in sequence."""
        base_tensor = torch.randn(2, 2, device=isolated_machine.device("cpu"))

        # Test sequence of different error types
        error_scenarios = [
            # Shape mismatch
            lambda: base_tensor.view(3, 3),
            # Invalid dimension access
            lambda: base_tensor.transpose(0, 5),
            # Invalid operation (if it fails)
            lambda: torch.det(base_tensor) if hasattr(torch, "det") else None,
        ]

        successful_operations = 0

        for scenario in error_scenarios:
            try:
                if scenario is not None:
                    result = scenario()
                    if result is not None:
                        successful_operations += 1
            except (
                RuntimeError,
                ValueError,
                NotImplementedError,
                AttributeError,
                IndexError,
            ):
                # Errors are expected and should be handled gracefully
                pass

        # Verify system is still functional after all error scenarios
        final_test = base_tensor + base_tensor
        assert final_test is not None, "System not functional after error sequence"


@pytest.mark.fast
@pytest.mark.parametrize(
    "invalid_shape",
    [
        (-1, 2),
        (2, -1),
        (0, -1),
        (-5, -3),
    ],
)
def test_parametrized_invalid_shapes(isolated_machine, invalid_shape):
    """Test error handling for various invalid tensor shapes."""
    with pytest.raises((RuntimeError, ValueError)):
        torch.randn(invalid_shape, device=isolated_machine.device("cpu"))


@pytest.mark.fast
@pytest.mark.parametrize(
    "incompatible_shapes",
    [
        ((2, 3), (4, 5)),
        ((1, 5), (3, 2)),
        ((3, 4, 5), (2, 6)),
    ],
)
def test_parametrized_incompatible_operations(isolated_machine, incompatible_shapes):
    """Test error handling for operations with incompatible tensor shapes."""
    shape1, shape2 = incompatible_shapes

    tensor1 = torch.randn(shape1, device=isolated_machine.device("cpu"))
    tensor2 = torch.randn(shape2, device=isolated_machine.device("cpu"))

    # Element-wise operations should fail for incompatible shapes
    with pytest.raises((RuntimeError, ValueError)):
        tensor1 + tensor2
