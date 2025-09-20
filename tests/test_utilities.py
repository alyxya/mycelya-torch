# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test utilities for mycelya-torch package.

This module provides common utility functions for test setup, verification,
and data generation to reduce code duplication across test files.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mycelya_torch import RemoteMachine

import pytest
import torch


class TestConstants:
    """Common constants used across tests."""

    # Standard tolerance values for numerical comparisons
    DEFAULT_RTOL = 1e-4
    DEFAULT_ATOL = 1e-6

    # Common tensor shapes
    SMALL_SHAPES = [(2, 2), (3, 3), (1, 5)]
    MEDIUM_SHAPES = [(3, 4), (5, 6), (4, 8)]
    TENSOR_3D_SHAPES = [(2, 3, 4), (1, 4, 5), (3, 2, 6)]
    TENSOR_4D_SHAPES = [(2, 3, 4, 5), (1, 2, 3, 4)]

    # Common device keys
    DEVICE_KEYS = ["T4", "L4"]

    # Classification test parameters
    DEFAULT_BATCH_SIZE = 3
    DEFAULT_NUM_CLASSES = 3


class DeviceTestUtils:
    """Utilities for device-related testing."""

    @staticmethod
    def create_remote_tensor(
        shape: tuple[int, ...],
        shared_machines: dict[str, "RemoteMachine"],
        machine_key: str = "T4",
        requires_grad: bool = False,
        dtype: torch.dtype = torch.float32,
        device_type: str = "cuda",
    ) -> torch.Tensor:
        """Create a tensor on a remote machine."""
        # Use appropriate tensor creation method based on dtype
        if dtype.is_floating_point or dtype.is_complex:
            # For floating point and complex types, use randn
            x_cpu = torch.randn(shape, dtype=dtype, requires_grad=requires_grad)
        else:
            # For integer types, use randint with reasonable range
            x_cpu = torch.randint(
                0, 10, shape, dtype=dtype, requires_grad=requires_grad
            )

        return x_cpu.to(shared_machines[machine_key].device(device_type))

    @staticmethod
    def create_test_tensors(
        shapes: list[tuple[int, ...]],
        shared_machines: dict[str, "RemoteMachine"],
        machine_key: str = "T4",
        device_type: str = "cuda",
    ) -> list[torch.Tensor]:
        """Create multiple test tensors on the same remote machine."""
        return [
            DeviceTestUtils.create_remote_tensor(
                shape, shared_machines, machine_key, device_type=device_type
            )
            for shape in shapes
        ]

    @staticmethod
    def verify_machine_properties(
        tensor: torch.Tensor,
        expected_machine: "RemoteMachine",
        device_type: str = "cuda",
    ) -> None:
        """Verify that a tensor has expected machine properties."""
        assert tensor.device.type == "mycelya"
        assert tensor.device.index == expected_machine.device(device_type).index

    @staticmethod
    def create_cpu_and_remote_pair(
        shape: tuple[int, ...],
        shared_machines: dict[str, "RemoteMachine"],
        machine_key: str = "T4",
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        device_type: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a CPU tensor and its remote counterpart."""
        cpu_tensor = torch.randn(shape, dtype=dtype, requires_grad=requires_grad)
        remote_tensor = cpu_tensor.to(shared_machines[machine_key].device(device_type))
        return cpu_tensor, remote_tensor


class NumericalTestUtils:
    """Utilities for numerical verification in tests."""

    @staticmethod
    def assert_tensors_close(
        actual: torch.Tensor,
        expected: torch.Tensor,
        rtol: float = TestConstants.DEFAULT_RTOL,
        atol: float = TestConstants.DEFAULT_ATOL,
        msg: str | None = None,
    ) -> None:
        """Assert that two tensors are numerically close."""
        # Handle NaN cases: if both tensors have NaN in the same positions, they're equal
        if torch.isnan(actual).any() or torch.isnan(expected).any():
            nan_match = torch.isnan(actual) == torch.isnan(expected)
            if not nan_match.all():
                raise AssertionError("NaN patterns don't match between tensors")
            # Check non-NaN values
            non_nan_mask = ~torch.isnan(actual)
            if non_nan_mask.any():
                actual_clean = actual[non_nan_mask]
                expected_clean = expected[non_nan_mask]
                assert torch.allclose(
                    actual_clean, expected_clean, rtol=rtol, atol=atol
                ), (
                    f"Non-NaN values not close: max diff = {torch.max(torch.abs(actual_clean - expected_clean)).item()}"
                )
            return

        if msg is None:
            msg = f"Tensors not close: max diff = {torch.max(torch.abs(actual - expected)).item()}"
        assert torch.allclose(actual, expected, rtol=rtol, atol=atol), msg

    @staticmethod
    def assert_remote_cpu_match(
        remote_tensor: torch.Tensor,
        cpu_tensor: torch.Tensor,
        rtol: float = TestConstants.DEFAULT_RTOL,
        atol: float = TestConstants.DEFAULT_ATOL,
    ) -> None:
        """Assert that a remote tensor matches its CPU counterpart."""
        NumericalTestUtils.assert_tensors_close(
            remote_tensor.cpu(),
            cpu_tensor,
            rtol=rtol,
            atol=atol,
            msg="Remote tensor result doesn't match CPU computation",
        )

    @staticmethod
    def verify_gradient_flow(
        leaf_tensor: torch.Tensor,
        expected_grad: torch.Tensor | None = None,
        rtol: float = TestConstants.DEFAULT_RTOL,
        atol: float = TestConstants.DEFAULT_ATOL,
    ) -> None:
        """Verify that gradients flow correctly through a tensor."""
        assert leaf_tensor.grad is not None, "Expected gradient but found None"
        if expected_grad is not None:
            NumericalTestUtils.assert_tensors_close(
                leaf_tensor.grad.cpu(),
                expected_grad,
                rtol=rtol,
                atol=atol,
                msg="Gradient doesn't match expected value",
            )


class ErrorTestUtils:
    """Utilities for testing error conditions."""

    @staticmethod
    def assert_cross_device_fails(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        operation_fn: callable,
        expected_error_message: str = "(Cross-device remote transfers are not supported|Cross-machine remote transfers are not supported|Expected all tensors to be on the same device|Cannot perform operation.*between different devices)",
    ) -> None:
        """Assert that an operation fails when using tensors from different devices."""
        with pytest.raises(RuntimeError, match=expected_error_message):
            operation_fn(tensor1, tensor2)

    @staticmethod
    def test_operation_gracefully(
        operation_fn: callable,
        expected_exceptions: tuple = (RuntimeError, TypeError, NotImplementedError),
    ) -> bool:
        """Test that an operation either succeeds or fails gracefully with expected exceptions."""
        try:
            operation_fn()
            return True
        except expected_exceptions:
            return True
        except Exception as e:
            pytest.fail(
                f"Operation failed with unexpected exception: {type(e).__name__}: {e}"
            )


class TestDataGenerator:
    """Utilities for generating test data."""

    @staticmethod
    def generate_classification_data(
        batch_size: int = TestConstants.DEFAULT_BATCH_SIZE,
        num_classes: int = TestConstants.DEFAULT_NUM_CLASSES,
        shared_machines: dict[str, "RemoteMachine"] | None = None,
        machine_key: str = "T4",
        device_type: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate data for classification tests."""
        # Create input data
        inputs = torch.randn(batch_size, num_classes, requires_grad=True)

        # Create targets
        targets = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

        # Move to remote device if requested
        if shared_machines is not None:
            inputs = inputs.to(shared_machines[machine_key].device(device_type))
            targets = targets.to(shared_machines[machine_key].device(device_type))

        return inputs, targets

    @staticmethod
    def generate_tensor_test_cases(
        base_shapes: list[tuple[int, ...]] | None = None,
    ) -> list[torch.Tensor]:
        """Generate a variety of tensor test cases."""
        if base_shapes is None:
            base_shapes = TestConstants.SMALL_SHAPES

        test_cases = []
        for shape in base_shapes:
            # Random tensor
            test_cases.append(torch.randn(shape))
            # Zero tensor
            test_cases.append(torch.zeros(shape))
            # Ones tensor
            test_cases.append(torch.ones(shape))

        # Add some special cases
        test_cases.append(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        return test_cases

    @staticmethod
    def create_gradient_test_setup(
        shape: tuple[int, ...],
        shared_machines: dict[str, Any],
        machine_key: str = "T4",
        device_type: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a standard setup for gradient testing."""
        # Create a leaf tensor that requires gradients
        x = torch.randn(shape, requires_grad=True)
        x_remote = x.to(shared_machines[machine_key].device(device_type))

        return x, x_remote


class ViewOperationTestUtils:
    """Utilities specific to view operation testing."""

    @staticmethod
    def test_view_operation(
        tensor: torch.Tensor,
        operation_fn: callable,
        cpu_reference: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        """Test a view operation against a CPU reference."""
        # Apply operation to remote tensor
        result_remote = operation_fn(tensor, *args, **kwargs)

        # Apply same operation to CPU reference
        result_cpu = operation_fn(cpu_reference, *args, **kwargs)

        # Verify results match
        NumericalTestUtils.assert_remote_cpu_match(result_remote, result_cpu)

        # Verify shape matches
        assert result_remote.shape == result_cpu.shape

    @staticmethod
    def generate_view_test_cases() -> list[tuple[str, callable, tuple, dict]]:
        """Generate test cases for view operations."""
        return [
            ("view", lambda x, *args: x.view(*args), (4, 1), {}),
            ("reshape", lambda x, *args: x.reshape(*args), (-1,), {}),
            ("transpose", lambda x, *args: x.transpose(*args), (0, 1), {}),
            ("permute", lambda x, *args: x.permute(*args), (1, 0), {}),
            ("squeeze", lambda x, *args: x.squeeze(*args), (), {}),
            ("unsqueeze", lambda x, *args: x.unsqueeze(*args), (0,), {}),
            ("flatten", lambda x, *args, **kwargs: x.flatten(*args, **kwargs), (), {}),
        ]


class IntegrationTestUtils:
    """Utilities for integration testing with PyTorch modules."""

    @staticmethod
    def create_simple_linear_model(
        input_size: int,
        output_size: int,
        shared_machines: dict[str, "RemoteMachine"],
        machine_key: str = "T4",
        device_type: str = "cuda",
    ) -> torch.nn.Module:
        """Create a simple linear model on a remote device."""
        model = torch.nn.Linear(input_size, output_size)
        return model.to(shared_machines[machine_key].device(device_type))

    @staticmethod
    def verify_model_forward_backward(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: callable = torch.nn.functional.mse_loss,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Verify a model's forward and backward pass."""
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass
        loss.backward()

        # Verify gradients exist
        for param in model.parameters():
            assert param.grad is not None, "Expected gradient but found None"

        return outputs, loss
