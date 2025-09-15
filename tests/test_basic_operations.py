# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for basic tensor operations in mycelya-torch.

This module tests arithmetic operations, tensor creation, conversions,
and fundamental tensor manipulations on remote devices.
"""

import pytest
import torch
from test_utilities import (
    DeviceTestUtils,
    NumericalTestUtils,
    TestConstants,
)


class TestBasicTensorCreation:
    """Tests for basic tensor creation on remote devices."""

    def test_backend_tensor_creation(self, shared_machines, provider):
        """Test backend tensor creation via .to() method."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 2)
        y = x.to(shared_machines["T4"].device(device_type))
        assert y is not None and y.shape == x.shape

    def test_tensor_creation_various_shapes(self, shared_machines, provider):
        """Test tensor creation with various shapes."""
        device_type = "cpu" if provider == "mock" else "cuda"
        test_shapes = TestConstants.SMALL_SHAPES + TestConstants.TENSOR_3D_SHAPES

        for shape in test_shapes:
            cpu_tensor, remote_tensor = DeviceTestUtils.create_cpu_and_remote_pair(
                shape, shared_machines, device_type=device_type
            )
            assert remote_tensor.shape == shape
            assert cpu_tensor.shape == remote_tensor.shape

    def test_tensor_creation_with_gradients(self, shared_machines, provider):
        """Test creating tensors with gradients enabled."""
        device_type = "cpu" if provider == "mock" else "cuda"
        cpu_tensor, remote_tensor = DeviceTestUtils.create_cpu_and_remote_pair(
            (3, 3), shared_machines, requires_grad=True, device_type=device_type
        )
        assert cpu_tensor.requires_grad
        assert remote_tensor.requires_grad

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_tensor_creation_different_dtypes(self, shared_machines, provider, dtype):
        """Test tensor creation with different data types."""
        device_type = "cpu" if provider == "mock" else "cuda"
        try:
            cpu_tensor, remote_tensor = DeviceTestUtils.create_cpu_and_remote_pair(
                (2, 2), shared_machines, dtype=dtype, device_type=device_type
            )
            assert cpu_tensor.dtype == dtype
            assert remote_tensor.dtype == dtype
        except (RuntimeError, NotImplementedError):
            pytest.skip(f"dtype {dtype} not supported on remote device")


class TestBasicArithmeticOperations:
    """Tests for basic arithmetic operations on remote tensors."""

    def test_tensor_addition(self, shared_machines, provider):
        """Test tensor addition on remote devices."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))

        # Test addition
        z_remote = x_remote + y_remote
        z_expected = x + y

        NumericalTestUtils.assert_remote_cpu_match(z_remote, z_expected)

    def test_tensor_subtraction(self, shared_machines, provider):
        """Test tensor subtraction on remote devices."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))

        z_remote = x_remote - y_remote
        z_expected = x - y

        NumericalTestUtils.assert_remote_cpu_match(z_remote, z_expected)

    def test_tensor_multiplication(self, shared_machines, provider):
        """Test element-wise tensor multiplication."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))

        z_remote = x_remote * y_remote
        z_expected = x * y

        NumericalTestUtils.assert_remote_cpu_match(z_remote, z_expected)

    def test_tensor_division(self, shared_machines, provider):
        """Test tensor division on remote devices."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 2) + 1.0  # Add 1 to avoid division by zero
        y = torch.randn(2, 2) + 1.0

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))

        z_remote = x_remote / y_remote
        z_expected = x / y

        NumericalTestUtils.assert_remote_cpu_match(z_remote, z_expected)

    def test_scalar_operations(self, shared_machines, provider):
        """Test operations with scalars."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 2)
        x_remote = x.to(shared_machines["T4"].device(device_type))
        scalar = 2.5

        # Addition with scalar
        result_add = x_remote + scalar
        expected_add = x + scalar
        NumericalTestUtils.assert_remote_cpu_match(result_add, expected_add)

        # Multiplication with scalar
        result_mul = x_remote * scalar
        expected_mul = x * scalar
        NumericalTestUtils.assert_remote_cpu_match(result_mul, expected_mul)


class TestMatrixOperations:
    """Tests for matrix operations on remote tensors."""

    def test_matrix_multiplication(self, shared_machines, provider):
        """Test matrix multiplication on remote devices."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))

        w_remote = x_remote.mm(y_remote)
        w_expected = x.mm(y)

        assert w_remote.shape == (2, 2)
        NumericalTestUtils.assert_remote_cpu_match(w_remote, w_expected)

    def test_matrix_multiplication_rectangular(self, shared_machines, provider):
        """Test matrix multiplication with rectangular matrices."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))

        result_remote = x_remote.mm(y_remote)
        result_expected = x.mm(y)

        assert result_remote.shape == (3, 5)
        NumericalTestUtils.assert_remote_cpu_match(result_remote, result_expected)

    def test_batch_matrix_multiplication(self, shared_machines, provider):
        """Test batch matrix multiplication on remote devices."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))

        try:
            result_remote = torch.bmm(x_remote, y_remote)
            result_expected = torch.bmm(x, y)

            assert result_remote.shape == (2, 3, 5)
            NumericalTestUtils.assert_remote_cpu_match(result_remote, result_expected)
        except (RuntimeError, NotImplementedError):
            pytest.skip("Batch matrix multiplication not implemented")


class TestTensorConversions:
    """Tests for tensor type and device conversions."""

    def test_dtype_conversion(self, shared_machines, provider):
        """Test remote conversion with dtype parameter."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 2, dtype=torch.float32)
        y = x.to(shared_machines["T4"].device(device_type), dtype=torch.float64)
        assert y.dtype == torch.float64

    def test_copy_parameter(self, shared_machines, provider):
        """Test remote conversion with copy parameter."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 2)
        y = x.to(shared_machines["T4"].device(device_type), copy=True)
        z = x.to(shared_machines["T4"].device(device_type), copy=False)
        assert y is not None and z is not None

    def test_cpu_to_remote_conversion(self, shared_machines, provider):
        """Test converting CPU tensors to remote devices."""
        device_type = "cpu" if provider == "mock" else "cuda"
        cpu_tensor = torch.randn(3, 3)
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        DeviceTestUtils.verify_machine_properties(
            remote_tensor, shared_machines["T4"], device_type=device_type
        )
        assert cpu_tensor.shape == remote_tensor.shape
        assert cpu_tensor.dtype == remote_tensor.dtype

    def test_remote_to_cpu_conversion(self, shared_machines, provider):
        """Test converting remote tensors back to CPU."""
        device_type = "cpu" if provider == "mock" else "cuda"
        original_cpu = torch.randn(2, 2)
        remote_tensor = original_cpu.to(shared_machines["T4"].device(device_type))
        back_to_cpu = remote_tensor.cpu()

        assert back_to_cpu.device.type == "cpu"
        NumericalTestUtils.assert_tensors_close(back_to_cpu, original_cpu)


class TestTensorProperties:
    """Tests for tensor property access and verification."""

    def test_tensor_shape_access(self, shared_machines, provider):
        """Test accessing tensor shape on remote devices."""
        device_type = "cpu" if provider == "mock" else "cuda"
        shapes_to_test = [(2, 2), (3, 4), (1, 5, 6)]

        for shape in shapes_to_test:
            remote_tensor = DeviceTestUtils.create_remote_tensor(
                shape, shared_machines, device_type=device_type
            )
            assert remote_tensor.shape == shape

    def test_tensor_dtype_access(self, shared_machines, provider):
        """Test accessing tensor dtype on remote devices."""
        device_type = "cpu" if provider == "mock" else "cuda"
        dtypes_to_test = [torch.float32, torch.float64]

        for dtype in dtypes_to_test:
            try:
                remote_tensor = DeviceTestUtils.create_remote_tensor(
                    (2, 2), shared_machines, dtype=dtype, device_type=device_type
                )
                assert remote_tensor.dtype == dtype
            except (RuntimeError, NotImplementedError):
                continue  # Skip unsupported dtypes

    def test_tensor_device_access(self, shared_machines, provider):
        """Test accessing tensor device information."""
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = DeviceTestUtils.create_remote_tensor(
            (2, 2), shared_machines, device_type=device_type
        )

        assert remote_tensor.device.type == "mycelya"
        assert isinstance(remote_tensor.device.index, int)
        assert remote_tensor.device.index >= 0

    def test_tensor_requires_grad_access(self, shared_machines, provider):
        """Test accessing requires_grad property."""
        device_type = "cpu" if provider == "mock" else "cuda"
        # Tensor without gradients
        tensor_no_grad = DeviceTestUtils.create_remote_tensor(
            (2, 2), shared_machines, device_type=device_type
        )
        assert not tensor_no_grad.requires_grad

        # Tensor with gradients
        tensor_with_grad = DeviceTestUtils.create_remote_tensor(
            (2, 2), shared_machines, requires_grad=True, device_type=device_type
        )
        assert tensor_with_grad.requires_grad


class TestTensorComparisons:
    """Tests for tensor comparison operations."""

    def test_tensor_equality(self, shared_machines, provider):
        """Test tensor equality operations."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        z = torch.tensor([[1.0, 2.0], [3.0, 5.0]])

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))
        z_remote = z.to(shared_machines["T4"].device(device_type))

        try:
            # Test equality
            eq_result = torch.eq(x_remote, y_remote)
            eq_expected = torch.eq(x, y)
            NumericalTestUtils.assert_remote_cpu_match(eq_result, eq_expected)

            # Test inequality
            neq_result = torch.eq(x_remote, z_remote)
            neq_expected = torch.eq(x, z)
            NumericalTestUtils.assert_remote_cpu_match(neq_result, neq_expected)
        except (RuntimeError, NotImplementedError):
            pytest.skip("Tensor comparison operations not implemented")


class TestTensorConcatenation:
    """Tests for tensor concatenation (aten::cat) operations."""

    def test_cat_dim0_basic(self, shared_machines, provider):
        """Test basic concatenation along dimension 0."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 3)
        y = torch.randn(3, 3)

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))

        # Concatenate along dim 0
        result_remote = torch.cat([x_remote, y_remote], dim=0)
        result_expected = torch.cat([x, y], dim=0)

        # Verify shape and values
        assert result_remote.shape == (5, 3)
        NumericalTestUtils.assert_remote_cpu_match(result_remote, result_expected)

    def test_cat_dim1_basic(self, shared_machines, provider):
        """Test basic concatenation along dimension 1."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 2)
        y = torch.randn(2, 3)

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))

        # Concatenate along dim 1
        result_remote = torch.cat([x_remote, y_remote], dim=1)
        result_expected = torch.cat([x, y], dim=1)

        # Verify shape and values
        assert result_remote.shape == (2, 5)
        NumericalTestUtils.assert_remote_cpu_match(result_remote, result_expected)

    def test_cat_multiple_tensors(self, shared_machines, provider):
        """Test concatenation with multiple tensors."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        z = torch.randn(2, 2)

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))
        z_remote = z.to(shared_machines["T4"].device(device_type))

        # Concatenate 3 tensors along dim 0
        result_remote = torch.cat([x_remote, y_remote, z_remote], dim=0)
        result_expected = torch.cat([x, y, z], dim=0)

        # Verify shape and values
        assert result_remote.shape == (6, 2)
        NumericalTestUtils.assert_remote_cpu_match(result_remote, result_expected)

    def test_cat_with_gradients(self, shared_machines, provider):
        """Test concatenation with gradient flow."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(3, 2, requires_grad=True)

        x_remote = (
            x.to(shared_machines["T4"].device(device_type)).detach().requires_grad_()
        )
        y_remote = (
            y.to(shared_machines["T4"].device(device_type)).detach().requires_grad_()
        )

        # Concatenate and compute loss
        cat_remote = torch.cat([x_remote, y_remote], dim=0)
        loss_remote = cat_remote.sum()

        # Backward pass
        loss_remote.backward()

        # Verify gradients exist and are correct
        NumericalTestUtils.verify_gradient_flow(x_remote)
        NumericalTestUtils.verify_gradient_flow(y_remote)

        # Gradients should be all ones (derivative of sum is 1)
        expected_x_grad = torch.ones_like(x)
        expected_y_grad = torch.ones_like(y)

        NumericalTestUtils.assert_tensors_close(
            x_remote.grad.cpu(), expected_x_grad, msg="x gradient should be ones"
        )
        NumericalTestUtils.assert_tensors_close(
            y_remote.grad.cpu(), expected_y_grad, msg="y gradient should be ones"
        )

    def test_cat_different_dtypes(self, shared_machines, provider):
        """Test concatenation behavior with different dtypes."""
        device_type = "cpu" if provider == "mock" else "cuda"
        # Create tensors with same dtype
        x = torch.randn(2, 2, dtype=torch.float32)
        y = torch.randn(3, 2, dtype=torch.float32)

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))

        result_remote = torch.cat([x_remote, y_remote], dim=0)
        result_expected = torch.cat([x, y], dim=0)

        # Verify dtype is preserved
        assert result_remote.dtype == torch.float32
        NumericalTestUtils.assert_remote_cpu_match(result_remote, result_expected)

    def test_cat_single_tensor(self, shared_machines, provider):
        """Test concatenation with a single tensor."""
        device_type = "cpu" if provider == "mock" else "cuda"
        x = torch.randn(2, 3)
        x_remote = x.to(shared_machines["T4"].device(device_type))

        # Cat with single tensor should return the same tensor
        result_remote = torch.cat([x_remote], dim=0)
        result_expected = torch.cat([x], dim=0)

        assert result_remote.shape == (2, 3)
        NumericalTestUtils.assert_remote_cpu_match(result_remote, result_expected)

    @pytest.mark.parametrize("dim", [0, 1])
    def test_cat_parametrized_dimensions(self, shared_machines, provider, dim):
        """Test concatenation along different dimensions."""
        device_type = "cpu" if provider == "mock" else "cuda"
        if dim == 0:
            x = torch.randn(2, 3)
            y = torch.randn(4, 3)
            expected_shape = (6, 3)
        else:  # dim == 1
            x = torch.randn(2, 3)
            y = torch.randn(2, 4)
            expected_shape = (2, 7)

        x_remote = x.to(shared_machines["T4"].device(device_type))
        y_remote = y.to(shared_machines["T4"].device(device_type))

        result_remote = torch.cat([x_remote, y_remote], dim=dim)
        result_expected = torch.cat([x, y], dim=dim)

        assert result_remote.shape == expected_shape
        NumericalTestUtils.assert_remote_cpu_match(result_remote, result_expected)


@pytest.mark.parametrize(
    "operation,expected_shape",
    [
        (lambda x, y: x + y, (2, 2)),
        (lambda x, y: x - y, (2, 2)),
        (lambda x, y: x * y, (2, 2)),
        (lambda x, y: x.mm(y), (2, 2)),
    ],
)
def test_parametrized_operations(shared_machines, provider, operation, expected_shape):
    """Test various operations with parametrized inputs."""
    device_type = "cpu" if provider == "mock" else "cuda"
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)

    x_remote = x.to(shared_machines["T4"].device(device_type))
    y_remote = y.to(shared_machines["T4"].device(device_type))

    try:
        result_remote = operation(x_remote, y_remote)
        result_expected = operation(x, y)

        assert result_remote.shape == expected_shape
        NumericalTestUtils.assert_remote_cpu_match(result_remote, result_expected)
    except (RuntimeError, NotImplementedError):
        pytest.skip(
            f"Operation {operation.__name__ if hasattr(operation, '__name__') else 'lambda'} not implemented"
        )
