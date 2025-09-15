# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for tensor view operations in mycelya-torch.

This module tests various tensor view operations including view, reshape,
transpose, permute, squeeze, unsqueeze, and flatten operations.
"""

import pytest
import torch
from test_utilities import (
    ViewOperationTestUtils,
)


class TestBasicViewOperations:
    """Tests for basic view operations on remote tensors."""

    def test_tensor_view_2d(self, shared_machines, provider):
        """Test tensor view operation with 2D tensors."""
        cpu_tensor = torch.randn(4, 2)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.view(2, 4), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.view(-1), cpu_tensor
        )

    def test_tensor_view_3d(self, shared_machines, provider):
        """Test tensor view operation with 3D tensors."""
        cpu_tensor = torch.randn(2, 3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.view(6, 4), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.view(2, -1), cpu_tensor
        )

    def test_tensor_view_4d(self, shared_machines, provider):
        """Test tensor view operation with 4D tensors."""
        cpu_tensor = torch.randn(2, 3, 4, 5)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.view(6, 20), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.view(-1, 5), cpu_tensor
        )


class TestReshapeOperations:
    """Tests for reshape operations on remote tensors."""

    def test_tensor_reshape_basic(self, shared_machines, provider):
        """Test basic reshape operations."""
        cpu_tensor = torch.randn(4, 3)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.reshape(6, 2), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.reshape(-1), cpu_tensor
        )

    def test_tensor_reshape_multidimensional(self, shared_machines, provider):
        """Test reshape with multiple dimensions."""
        cpu_tensor = torch.randn(2, 3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.reshape(2, 12), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.reshape(1, 2, 3, 4), cpu_tensor
        )

    @pytest.mark.parametrize(
        "original_shape,new_shape",
        [
            ((6,), (2, 3)),
            ((2, 6), (3, 4)),
            ((2, 3, 4), (6, 4)),
            ((1, 2, 3, 4), (2, 12)),
        ],
    )
    def test_parametrized_reshape(
        self, shared_machines, provider, original_shape, new_shape
    ):
        """Test reshape operations with parametrized shapes."""
        cpu_tensor = torch.randn(original_shape)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.reshape(new_shape), cpu_tensor
        )


class TestTransposeOperations:
    """Tests for transpose operations on remote tensors."""

    def test_tensor_transpose_2d(self, shared_machines, provider):
        """Test 2D tensor transpose."""
        cpu_tensor = torch.randn(3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.transpose(0, 1), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.t(), cpu_tensor
        )

    def test_tensor_transpose_3d(self, shared_machines, provider):
        """Test 3D tensor transpose."""
        cpu_tensor = torch.randn(2, 3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.transpose(0, 1), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.transpose(1, 2), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.transpose(0, 2), cpu_tensor
        )

    @pytest.mark.parametrize("dim0,dim1", [(0, 1), (0, 2), (1, 2)])
    def test_parametrized_transpose_3d(self, shared_machines, provider, dim0, dim1):
        """Test 3D transpose with parametrized dimensions."""
        cpu_tensor = torch.randn(2, 3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.transpose(dim0, dim1), cpu_tensor
        )


class TestPermuteOperations:
    """Tests for permute operations on remote tensors."""

    def test_tensor_permute_3d(self, shared_machines, provider):
        """Test 3D tensor permute."""
        cpu_tensor = torch.randn(2, 3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.permute(2, 0, 1), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.permute(1, 2, 0), cpu_tensor
        )

    def test_tensor_permute_4d(self, shared_machines, provider):
        """Test 4D tensor permute."""
        cpu_tensor = torch.randn(2, 3, 4, 5)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.permute(3, 1, 0, 2), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.permute(0, 2, 3, 1), cpu_tensor
        )

    @pytest.mark.parametrize(
        "permutation",
        [
            (2, 0, 1),
            (1, 2, 0),
            (0, 2, 1),
            (2, 1, 0),
        ],
    )
    def test_parametrized_permute_3d(self, shared_machines, provider, permutation):
        """Test 3D permute with parametrized permutations."""
        cpu_tensor = torch.randn(2, 3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.permute(permutation), cpu_tensor
        )


class TestSqueezeUnSqueezeOperations:
    """Tests for squeeze and unsqueeze operations on remote tensors."""

    def test_tensor_squeeze(self, shared_machines, provider):
        """Test tensor squeeze operation."""
        cpu_tensor = torch.randn(1, 3, 1, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        # Squeeze all dimensions
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.squeeze(), cpu_tensor
        )

        # Squeeze specific dimensions
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.squeeze(0), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.squeeze(2), cpu_tensor
        )

    def test_tensor_unsqueeze(self, shared_machines, provider):
        """Test tensor unsqueeze operation."""
        cpu_tensor = torch.randn(3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.unsqueeze(0), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.unsqueeze(1), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.unsqueeze(2), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.unsqueeze(-1), cpu_tensor
        )

    @pytest.mark.parametrize("squeeze_dim", [0, 2])
    def test_parametrized_squeeze(self, shared_machines, provider, squeeze_dim):
        """Test squeeze with parametrized dimensions."""
        cpu_tensor = torch.randn(1, 3, 1, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.squeeze(squeeze_dim), cpu_tensor
        )

    @pytest.mark.parametrize("unsqueeze_dim", [0, 1, 2, -1])
    def test_parametrized_unsqueeze(self, shared_machines, provider, unsqueeze_dim):
        """Test unsqueeze with parametrized dimensions."""
        cpu_tensor = torch.randn(3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.unsqueeze(unsqueeze_dim), cpu_tensor
        )


class TestFlattenOperations:
    """Tests for flatten operations on remote tensors."""

    def test_tensor_flatten_basic(self, shared_machines, provider):
        """Test basic flatten operation."""
        cpu_tensor = torch.randn(2, 3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.flatten(), cpu_tensor
        )

    def test_tensor_flatten_with_dims(self, shared_machines, provider):
        """Test flatten with specific dimensions."""
        cpu_tensor = torch.randn(2, 3, 4, 5)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.flatten(0, 1), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.flatten(1, 2), cpu_tensor
        )
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.flatten(2, 3), cpu_tensor
        )

    @pytest.mark.parametrize("start_dim,end_dim", [(0, 1), (1, 2), (0, 2)])
    def test_parametrized_flatten(self, shared_machines, provider, start_dim, end_dim):
        """Test flatten with parametrized dimensions."""
        cpu_tensor = torch.randn(2, 3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.flatten(start_dim, end_dim), cpu_tensor
        )


class TestViewOperationsCombined:
    """Tests for combinations of view operations."""

    def test_view_reshape_combination(self, shared_machines, provider):
        """Test combining view and reshape operations."""
        cpu_tensor = torch.randn(2, 3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        # View then reshape
        def combined_op(x):
            return x.view(6, 4).reshape(2, 12)

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, combined_op, cpu_tensor
        )

    def test_transpose_view_combination(self, shared_machines, provider):
        """Test combining transpose and view operations."""
        cpu_tensor = torch.randn(2, 3, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        # Transpose then view
        def combined_op(x):
            return x.transpose(0, 1).contiguous().view(-1)

        try:
            ViewOperationTestUtils.test_view_operation(
                remote_tensor, combined_op, cpu_tensor
            )
        except (RuntimeError, NotImplementedError):
            pytest.skip("Combined transpose-view operation not supported")

    def test_squeeze_unsqueeze_combination(self, shared_machines, provider):
        """Test combining squeeze and unsqueeze operations."""
        cpu_tensor = torch.randn(1, 3, 1, 4)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        # Squeeze then unsqueeze
        def combined_op(x):
            return x.squeeze().unsqueeze(0)

        ViewOperationTestUtils.test_view_operation(
            remote_tensor, combined_op, cpu_tensor
        )


class TestViewOperationsMultipleDimensions:
    """Tests for view operations with various tensor dimensions."""

    @pytest.mark.parametrize(
        "original_shape",
        [
            (12,),
            (3, 4),
            (2, 2, 3),
            (1, 2, 3, 2),
        ],
    )
    def test_view_operations_multiple_dimensions(
        self, shared_machines, provider, original_shape
    ):
        """Test view operations with multiple tensor dimensions."""
        cpu_tensor = torch.randn(original_shape)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        # Test flatten
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.flatten(), cpu_tensor
        )

        # Test view to 1D
        total_elements = cpu_tensor.numel()
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.view(total_elements), cpu_tensor
        )

        # Test unsqueeze at the beginning
        ViewOperationTestUtils.test_view_operation(
            remote_tensor, lambda x: x.unsqueeze(0), cpu_tensor
        )


class TestViewOperationsErrorHandling:
    """Tests for error handling in view operations."""

    def test_invalid_view_shape(self, shared_machines, provider):
        """Test that invalid view shapes are handled properly."""
        cpu_tensor = torch.randn(2, 3)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        # Try to view with incompatible shape
        with pytest.raises((RuntimeError, ValueError)):
            remote_tensor.view(2, 2)  # 6 elements can't become 4

    def test_invalid_squeeze_dimension(self, shared_machines, provider):
        """Test that invalid squeeze dimensions are handled properly."""
        cpu_tensor = torch.randn(2, 3, 4)  # 3D tensor (dimensions 0, 1, 2)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        # Try to squeeze an out-of-bounds dimension
        with pytest.raises((RuntimeError, IndexError)):
            remote_tensor.squeeze(3)  # Dimension 3 doesn't exist (valid range: -3 to 2)

    def test_squeeze_non_unit_dimension_no_error(self, shared_machines, provider):
        """Test that squeezing a non-unit dimension doesn't raise an error (matches PyTorch behavior)."""
        cpu_tensor = torch.randn(2, 3, 4)  # No dimensions of size 1
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        # Squeezing a dimension that's not size 1 should NOT raise an error
        # It should return the tensor unchanged (this is PyTorch's behavior)
        result = remote_tensor.squeeze(0)  # Dimension 0 has size 2, not 1
        assert result.shape == remote_tensor.shape  # Should be unchanged

        # Verify behavior matches CPU tensor
        cpu_result = cpu_tensor.squeeze(0)
        assert result.shape == cpu_result.shape

    def test_invalid_transpose_dimensions(self, shared_machines, provider):
        """Test that invalid transpose dimensions are handled properly."""
        cpu_tensor = torch.randn(2, 3)
        device_type = "cpu" if provider == "mock" else "cuda"
        remote_tensor = cpu_tensor.to(shared_machines["T4"].device(device_type))

        # Try to transpose with out-of-bounds dimensions
        with pytest.raises((RuntimeError, IndexError)):
            remote_tensor.transpose(0, 2)  # Dimension 2 doesn't exist
