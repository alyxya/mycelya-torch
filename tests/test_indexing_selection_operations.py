# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch
from test_utilities import NumericalTestUtils


class TestBasicIndexing:
    """Test basic tensor indexing operations."""

    @pytest.mark.parametrize("shape", [(10,), (5, 5), (3, 4, 5)])
    @pytest.mark.fast
    def test_basic_slice_indexing(self, shared_machines, provider, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test various slice patterns
        slice_patterns = [
            slice(None),  # [:]
            slice(1, None),  # [1:]
            slice(None, -1),  # [:-1]
            slice(1, -1),  # [1:-1]
        ]

        for pattern in slice_patterns:
            if len(shape) == 1:
                cpu_result = cpu_tensor[pattern]
                remote_result = remote_tensor[pattern]
            elif len(shape) == 2:
                cpu_result = cpu_tensor[pattern, :]
                remote_result = remote_tensor[pattern, :]
            else:  # 3D
                cpu_result = cpu_tensor[pattern, :, :]
                remote_result = remote_tensor[pattern, :, :]

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )

    @pytest.mark.fast
    def test_integer_indexing(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(5, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test single integer indexing
        for i in range(5):
            cpu_result = cpu_tensor[i]
            remote_result = remote_tensor[i]

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )

    @pytest.mark.fast
    def test_multidimensional_indexing(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(3, 4, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test various indexing patterns
        indexing_patterns = [
            (1, 2, 3),  # Specific indices
            (slice(None), 2, slice(1, 4)),  # [:, 2, 1:4]
            (1, slice(None), slice(None, 3)),  # [1, :, :3]
        ]

        for pattern in indexing_patterns:
            cpu_result = cpu_tensor[pattern]
            remote_result = remote_tensor[pattern]

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )


class TestAdvancedIndexing:
    """Test advanced indexing with tensor indices."""

    @pytest.mark.fast
    def test_tensor_indexing(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(5, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Create index tensors
        cpu_indices = torch.tensor([0, 2, 4])
        remote_indices = cpu_indices.to(machine.device(device_type))

        # Test tensor indexing
        cpu_result = cpu_tensor[cpu_indices]
        remote_result = remote_tensor[remote_indices]

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_boolean_indexing(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Create boolean mask
        cpu_mask = cpu_tensor > 0
        remote_mask = remote_tensor > 0

        # Test boolean indexing
        cpu_result = cpu_tensor[cpu_mask]
        remote_result = remote_tensor[remote_mask]

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_multidimensional_tensor_indexing(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(3, 4, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Create 2D index tensor
        cpu_indices = torch.tensor([[0, 1], [2, 0]])
        remote_indices = cpu_indices.to(machine.device(device_type))

        # Test multidimensional tensor indexing
        cpu_result = cpu_tensor[cpu_indices]
        remote_result = remote_tensor[remote_indices]

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )


class TestSelectionOperations:
    """Test tensor selection operations."""

    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.fast
    def test_index_select(self, shared_machines, provider, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 5, 6)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Skip invalid dimensions
        if abs(dim) >= len(cpu_tensor.shape):
            pytest.skip(f"Dimension {dim} invalid for tensor shape")

        # Create index tensor
        size = cpu_tensor.shape[dim]
        cpu_indices = torch.randint(0, size, (3,))
        remote_indices = cpu_indices.to(machine.device(device_type))

        cpu_result = torch.index_select(cpu_tensor, dim, cpu_indices)
        remote_result = torch.index_select(remote_tensor, dim, remote_indices)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.fast
    def test_gather_operations(self, shared_machines, provider, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(3, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Create index tensor for gather
        if dim == 0:
            cpu_indices = torch.randint(0, 3, (2, 4))
        else:
            cpu_indices = torch.randint(0, 4, (3, 2))

        remote_indices = cpu_indices.to(machine.device(device_type))

        cpu_result = torch.gather(cpu_tensor, dim, cpu_indices)
        remote_result = torch.gather(remote_tensor, dim, remote_indices)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.fast
    def test_scatter_operations(self, shared_machines, provider, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create separate base tensors for CPU and remote operations
        # This avoids in-place modification affecting both tests
        cpu_tensor = torch.zeros(3, 4)
        remote_tensor = torch.zeros(3, 4).to(machine.device(device_type))

        # Create index tensor and source tensor for scatter
        # Use deterministic indices to avoid undefined behavior with duplicates
        if dim == 0:
            # No duplicate indices - each position gets one value
            cpu_indices = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])  # 2x4
            cpu_src = torch.randn(2, 4)
        else:
            # No duplicate indices for dimension 1
            cpu_indices = torch.tensor([[0, 1], [2, 3], [1, 0]])  # 3x2
            cpu_src = torch.randn(3, 2)

        # Transfer the same data to remote device
        remote_indices = cpu_indices.to(machine.device(device_type))
        remote_src = cpu_src.to(machine.device(device_type))

        cpu_result = cpu_tensor.scatter(dim, cpu_indices, cpu_src)
        remote_result = remote_tensor.scatter(dim, remote_indices, remote_src)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_take_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(3, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Create flat indices for take operation
        cpu_indices = torch.tensor([0, 5, 10, 11])
        remote_indices = cpu_indices.to(machine.device(device_type))

        cpu_result = torch.take(cpu_tensor, cpu_indices)
        remote_result = torch.take(remote_tensor, remote_indices)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.fast
    def test_take_along_dim(self, shared_machines, provider, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Create indices for take_along_dim
        if dim == 0:
            cpu_indices = torch.randint(0, 4, (2, 5))
        else:
            cpu_indices = torch.randint(0, 5, (4, 3))

        remote_indices = cpu_indices.to(machine.device(device_type))

        cpu_result = torch.take_along_dim(cpu_tensor, cpu_indices, dim)
        remote_result = torch.take_along_dim(remote_tensor, remote_indices, dim)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )


class TestMaskedOperations:
    """Test masked selection and filling operations."""

    @pytest.mark.fast
    def test_masked_select(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Create mask
        cpu_mask = cpu_tensor > 0
        remote_mask = remote_tensor > 0

        cpu_result = torch.masked_select(cpu_tensor, cpu_mask)
        remote_result = torch.masked_select(remote_tensor, remote_mask)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("fill_value", [0.0, 1.5, -2.0])
    @pytest.mark.fast
    def test_masked_fill(self, shared_machines, provider, fill_value):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Create mask
        cpu_mask = cpu_tensor > 0
        remote_mask = remote_tensor > 0

        cpu_result = cpu_tensor.masked_fill(cpu_mask, fill_value)
        remote_result = remote_tensor.masked_fill(remote_mask, fill_value)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_masked_scatter(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Create mask and source tensor
        cpu_mask = cpu_tensor > 0
        remote_mask = remote_tensor > 0

        num_true = cpu_mask.sum().item()
        cpu_source = torch.randn(num_true)
        remote_source = cpu_source.to(machine.device(device_type))

        cpu_result = cpu_tensor.masked_scatter(cpu_mask, cpu_source)
        remote_result = remote_tensor.masked_scatter(remote_mask, remote_source)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )


class TestConditionalOperations:
    """Test conditional selection operations."""

    @pytest.mark.fast
    def test_where_three_args(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_condition = torch.rand(4, 4) > 0.5
        cpu_x = torch.randn(4, 4)
        cpu_y = torch.randn(4, 4)

        remote_condition = cpu_condition.to(machine.device(device_type))
        remote_x = cpu_x.to(machine.device(device_type))
        remote_y = cpu_y.to(machine.device(device_type))

        cpu_result = torch.where(cpu_condition, cpu_x, cpu_y)
        remote_result = torch.where(remote_condition, remote_x, remote_y)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_where_one_arg(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_condition = cpu_tensor > 0
        remote_condition = remote_tensor > 0

        cpu_indices = torch.where(cpu_condition)
        remote_indices = torch.where(remote_condition)

        # Compare the tuple of index tensors
        assert len(cpu_indices) == len(remote_indices)
        for cpu_idx, remote_idx in zip(cpu_indices, remote_indices):
            assert torch.equal(remote_idx.cpu(), cpu_idx)

    @pytest.mark.fast
    def test_where_broadcasting(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_condition = torch.rand(4, 1) > 0.5
        cpu_x = torch.randn(1, 5)
        cpu_y = torch.randn(4, 5)

        remote_condition = cpu_condition.to(machine.device(device_type))
        remote_x = cpu_x.to(machine.device(device_type))
        remote_y = cpu_y.to(machine.device(device_type))

        cpu_result = torch.where(cpu_condition, cpu_x, cpu_y)
        remote_result = torch.where(remote_condition, remote_x, remote_y)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_where_scalar_values(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_condition = torch.rand(4, 4) > 0.5
        remote_condition = cpu_condition.to(machine.device(device_type))

        # Test with scalar values
        cpu_result = torch.where(cpu_condition, 1.0, 0.0)
        remote_result = torch.where(remote_condition, 1.0, 0.0)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )


class TestIndexingWithGradients:
    """Test indexing operations maintain gradients properly."""

    @pytest.mark.fast
    def test_basic_indexing_with_gradients(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create independent leaf tensors for proper gradient comparison
        tensor_data = torch.randn(5, 5)
        cpu_tensor = tensor_data.clone().requires_grad_(True)
        remote_tensor = tensor_data.to(machine.device(device_type)).requires_grad_(True)

        # Test slicing with gradients
        cpu_result = cpu_tensor[1:4, 2:5]
        remote_result = remote_tensor[1:4, 2:5]

        cpu_loss = cpu_result.sum()
        remote_loss = remote_result.sum()

        cpu_loss.backward()
        remote_loss.backward()

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )
        NumericalTestUtils.assert_tensors_close(
            remote_tensor.grad.cpu(), cpu_tensor.grad, rtol=1e-6, atol=1e-7
        )

    @pytest.mark.fast
    def test_index_select_with_gradients(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create independent leaf tensors for proper gradient comparison
        tensor_data = torch.randn(5, 4)
        cpu_tensor = tensor_data.clone().requires_grad_(True)
        remote_tensor = tensor_data.to(machine.device(device_type)).requires_grad_(True)

        cpu_indices = torch.tensor([0, 2, 4])
        remote_indices = cpu_indices.to(machine.device(device_type))

        cpu_result = torch.index_select(cpu_tensor, 0, cpu_indices)
        remote_result = torch.index_select(remote_tensor, 0, remote_indices)

        cpu_loss = cpu_result.sum()
        remote_loss = remote_result.sum()

        cpu_loss.backward()
        remote_loss.backward()

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )
        NumericalTestUtils.assert_tensors_close(
            remote_tensor.grad.cpu(), cpu_tensor.grad, rtol=1e-6, atol=1e-7
        )

    @pytest.mark.fast
    def test_where_with_gradients(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create independent leaf tensors for proper gradient comparison
        x_data = torch.randn(4, 4)
        y_data = torch.randn(4, 4)
        cpu_condition = torch.rand(4, 4) > 0.5

        cpu_x = x_data.clone().requires_grad_(True)
        cpu_y = y_data.clone().requires_grad_(True)

        remote_x = x_data.to(machine.device(device_type)).requires_grad_(True)
        remote_y = y_data.to(machine.device(device_type)).requires_grad_(True)
        remote_condition = cpu_condition.to(machine.device(device_type))

        cpu_result = torch.where(cpu_condition, cpu_x, cpu_y)
        remote_result = torch.where(remote_condition, remote_x, remote_y)

        cpu_loss = cpu_result.sum()
        remote_loss = remote_result.sum()

        cpu_loss.backward()
        remote_loss.backward()

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )
        NumericalTestUtils.assert_tensors_close(
            remote_x.grad.cpu(), cpu_x.grad, rtol=1e-6, atol=1e-7
        )
        NumericalTestUtils.assert_tensors_close(
            remote_y.grad.cpu(), cpu_y.grad, rtol=1e-6, atol=1e-7
        )


class TestIndexingEdgeCases:
    """Test edge cases for indexing operations."""

    @pytest.mark.fast
    def test_empty_tensor_indexing(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_empty = torch.empty(0, 5)
        remote_empty = cpu_empty.to(machine.device(device_type))

        # Test indexing empty tensor
        cpu_result = cpu_empty[:, 1:3]
        remote_result = remote_empty[:, 1:3]

        assert cpu_result.shape == remote_result.shape
        assert cpu_result.numel() == 0

    @pytest.mark.fast
    def test_negative_indexing(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(5, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test negative indexing
        cpu_result = cpu_tensor[-1, -2]
        remote_result = remote_tensor[-1, -2]

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_step_indexing(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(10, 10)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test step indexing
        cpu_result = cpu_tensor[::2, 1::3]
        remote_result = remote_tensor[::2, 1::3]

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_single_element_indexing(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(3, 3)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test single element indexing
        cpu_result = cpu_tensor[1, 2]
        remote_result = remote_tensor[1, 2]

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_ellipsis_indexing(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(2, 3, 4, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test ellipsis indexing
        indexing_patterns = [
            (..., 0),  # [..., 0]
            (0, ...),  # [0, ...]
            (0, ..., 2),  # [0, ..., 2]
        ]

        for pattern in indexing_patterns:
            cpu_result = cpu_tensor[pattern]
            remote_result = remote_tensor[pattern]

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )

    @pytest.mark.fast
    def test_none_indexing(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(3, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test None indexing (adds dimension)
        cpu_result = cpu_tensor[None, :, :]
        remote_result = remote_tensor[None, :, :]

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )
        assert remote_result.shape == (1, 3, 4)
