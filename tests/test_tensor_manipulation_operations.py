# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch
from test_utilities import NumericalTestUtils


class TestSplittingOperations:
    """Test tensor splitting operations."""

    @pytest.mark.parametrize("split_size", [1, 2, 3])
    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.fast
    def test_split_operations(self, shared_machines, provider, split_size, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(6, 6)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Skip invalid dimensions
        if abs(dim) >= len(cpu_tensor.shape):
            pytest.skip(f"Dimension {dim} invalid for tensor shape")

        cpu_splits = torch.split(cpu_tensor, split_size, dim=dim)
        remote_splits = torch.split(remote_tensor, split_size, dim=dim)

        assert len(cpu_splits) == len(remote_splits)

        for cpu_split, remote_split in zip(cpu_splits, remote_splits):
            NumericalTestUtils.assert_tensors_close(
                remote_split.cpu(), cpu_split, rtol=1e-8, atol=1e-8
            )

    @pytest.mark.parametrize("num_chunks", [2, 3, 4])
    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.fast
    def test_chunk_operations(self, shared_machines, provider, num_chunks, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(8, 6)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_chunks = torch.chunk(cpu_tensor, num_chunks, dim=dim)
        remote_chunks = torch.chunk(remote_tensor, num_chunks, dim=dim)

        assert len(cpu_chunks) == len(remote_chunks)

        for cpu_chunk, remote_chunk in zip(cpu_chunks, remote_chunks):
            NumericalTestUtils.assert_tensors_close(
                remote_chunk.cpu(), cpu_chunk, rtol=1e-8, atol=1e-8
            )

    @pytest.mark.fast
    def test_split_with_list_sizes(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(10, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        split_sizes = [2, 3, 5]  # Must sum to 10

        cpu_splits = torch.split(cpu_tensor, split_sizes, dim=0)
        remote_splits = torch.split(remote_tensor, split_sizes, dim=0)

        assert len(cpu_splits) == len(remote_splits) == 3

        for cpu_split, remote_split in zip(cpu_splits, remote_splits):
            NumericalTestUtils.assert_tensors_close(
                remote_split.cpu(), cpu_split, rtol=1e-8, atol=1e-8
            )


class TestStackingOperations:
    """Test tensor stacking and concatenation operations."""

    @pytest.mark.parametrize("dim", [0, 1, 2, -1])
    @pytest.mark.fast
    def test_stack_operations(self, shared_machines, provider, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensors = [torch.randn(3, 4) for _ in range(3)]
        remote_tensors = [t.to(machine.device(device_type)) for t in cpu_tensors]

        # Skip invalid dimensions
        max_dim = len(cpu_tensors[0].shape)
        if abs(dim) > max_dim:
            pytest.skip(f"Dimension {dim} invalid for stacking")

        cpu_result = torch.stack(cpu_tensors, dim=dim)
        remote_result = torch.stack(remote_tensors, dim=dim)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_hstack_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensors = [torch.randn(3, 2) for _ in range(3)]
        remote_tensors = [t.to(machine.device(device_type)) for t in cpu_tensors]

        cpu_result = torch.hstack(cpu_tensors)
        remote_result = torch.hstack(remote_tensors)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_vstack_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensors = [torch.randn(2, 4) for _ in range(3)]
        remote_tensors = [t.to(machine.device(device_type)) for t in cpu_tensors]

        cpu_result = torch.vstack(cpu_tensors)
        remote_result = torch.vstack(remote_tensors)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_dstack_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensors = [torch.randn(3, 4) for _ in range(3)]
        remote_tensors = [t.to(machine.device(device_type)) for t in cpu_tensors]

        cpu_result = torch.dstack(cpu_tensors)
        remote_result = torch.dstack(remote_tensors)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )


class TestRepeatAndTileOperations:
    """Test tensor repetition and tiling operations."""

    @pytest.mark.parametrize("repeats", [(2, 3), (1, 4), (3, 1, 2)])
    @pytest.mark.fast
    def test_repeat_operations(self, shared_machines, provider, repeats):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(2, 3)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = cpu_tensor.repeat(*repeats)
        remote_result = remote_tensor.repeat(*repeats)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("repeats", [3, 5])
    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.fast
    def test_repeat_interleave_operations(
        self, shared_machines, provider, repeats, dim
    ):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.repeat_interleave(cpu_tensor, repeats, dim=dim)
        remote_result = torch.repeat_interleave(remote_tensor, repeats, dim=dim)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    @pytest.mark.skip(reason="repeat_interleave with tensor repeats not yet supported")
    def test_repeat_interleave_with_tensor_repeats(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 3)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_repeats = torch.tensor([1, 2, 3, 1])
        remote_repeats = cpu_repeats.to(machine.device(device_type))

        cpu_result = torch.repeat_interleave(cpu_tensor, cpu_repeats, dim=0)
        remote_result = torch.repeat_interleave(remote_tensor, remote_repeats, dim=0)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("tiles", [(2, 3), (1, 4), (3, 1)])
    @pytest.mark.fast
    def test_tile_operations(self, shared_machines, provider, tiles):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(2, 3)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.tile(cpu_tensor, tiles)
        remote_result = torch.tile(remote_tensor, tiles)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )


class TestSortingOperations:
    """Test tensor sorting operations."""

    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("descending", [False, True])
    @pytest.mark.fast
    def test_sort_operations(self, shared_machines, provider, dim, descending):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(5, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Skip invalid dimensions
        if abs(dim) >= len(cpu_tensor.shape):
            pytest.skip(f"Dimension {dim} invalid for tensor shape")

        cpu_values, cpu_indices = torch.sort(cpu_tensor, dim=dim, descending=descending)
        remote_values, remote_indices = torch.sort(
            remote_tensor, dim=dim, descending=descending
        )

        NumericalTestUtils.assert_tensors_close(
            remote_values.cpu(), cpu_values, rtol=1e-8, atol=1e-8
        )
        assert torch.equal(remote_indices.cpu(), cpu_indices)

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("descending", [False, True])
    @pytest.mark.fast
    def test_argsort_operations(self, shared_machines, provider, dim, descending):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_indices = torch.argsort(cpu_tensor, dim=dim, descending=descending)
        remote_indices = torch.argsort(remote_tensor, dim=dim, descending=descending)

        assert torch.equal(remote_indices.cpu(), cpu_indices)

    @pytest.mark.parametrize("k", [1, 3, 5])
    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.fast
    def test_topk_operations(self, shared_machines, provider, k, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(6, 8)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Skip invalid k values
        if k > cpu_tensor.shape[dim]:
            pytest.skip(f"k={k} too large for dimension {dim}")

        cpu_values, cpu_indices = torch.topk(cpu_tensor, k, dim=dim)
        remote_values, remote_indices = torch.topk(remote_tensor, k, dim=dim)

        NumericalTestUtils.assert_tensors_close(
            remote_values.cpu(), cpu_values, rtol=1e-8, atol=1e-8
        )
        assert torch.equal(remote_indices.cpu(), cpu_indices)

    @pytest.mark.parametrize("k", [2, 4])
    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.fast
    def test_kthvalue_operations(self, shared_machines, provider, k, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(6, 8)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Skip invalid k values
        if k > cpu_tensor.shape[dim]:
            pytest.skip(f"k={k} too large for dimension {dim}")

        cpu_values, cpu_indices = torch.kthvalue(cpu_tensor, k, dim=dim)
        remote_values, remote_indices = torch.kthvalue(remote_tensor, k, dim=dim)

        NumericalTestUtils.assert_tensors_close(
            remote_values.cpu(), cpu_values, rtol=1e-8, atol=1e-8
        )
        assert torch.equal(remote_indices.cpu(), cpu_indices)


class TestPaddingOperations:
    """Test tensor padding operations."""

    @pytest.mark.parametrize("pad", [(1, 1), (2, 3), (1, 1, 2, 2)])
    @pytest.mark.parametrize("mode", ["constant", "reflect", "replicate"])
    @pytest.mark.fast
    def test_pad_operations(self, shared_machines, provider, pad, mode):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Choose tensor size based on padding requirements
        if len(pad) == 2:
            cpu_tensor = torch.randn(4, 6)
        else:  # len(pad) == 4
            cpu_tensor = torch.randn(3, 4, 5)

        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # For reflect and replicate modes, ensure tensor is large enough
        if mode in ["reflect", "replicate"]:
            min_size = max(pad) + 1
            if any(s <= min_size for s in cpu_tensor.shape[-len(pad) // 2 :]):
                pytest.skip(f"Tensor too small for {mode} padding with pad={pad}")

        cpu_result = torch.nn.functional.pad(cpu_tensor, pad, mode=mode)
        remote_result = torch.nn.functional.pad(remote_tensor, pad, mode=mode)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("pad", [(1, 1), (2, 3)])
    @pytest.mark.parametrize("value", [0.0, 1.5, -2.0])
    @pytest.mark.fast
    def test_constant_pad_with_value(self, shared_machines, provider, pad, value):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.nn.functional.pad(
            cpu_tensor, pad, mode="constant", value=value
        )
        remote_result = torch.nn.functional.pad(
            remote_tensor, pad, mode="constant", value=value
        )

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )


class TestReshapingOperations:
    """Test advanced tensor reshaping operations."""

    @pytest.mark.parametrize("size", [(3, 4), (2, 6), (12,), (1, 12)])
    @pytest.mark.fast
    def test_expand_operations(self, shared_machines, provider, size):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(1, 4)  # Expandable tensor
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Check if expansion is valid
        try:
            cpu_result = cpu_tensor.expand(*size)
            remote_result = remote_tensor.expand(*size)

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )
        except RuntimeError:
            # If CPU expansion fails, remote should also fail
            with pytest.raises(RuntimeError):
                remote_tensor.expand(*size)

    @pytest.mark.fast
    def test_expand_as_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(1, 4)
        cpu_other = torch.randn(3, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))
        remote_other = cpu_other.to(machine.device(device_type))

        cpu_result = cpu_tensor.expand_as(cpu_other)
        remote_result = remote_tensor.expand_as(remote_other)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("start", [0, 1, 2])
    @pytest.mark.parametrize("length", [2, 3])
    @pytest.mark.fast
    def test_narrow_operations(self, shared_machines, provider, start, length):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(5, 6)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Skip invalid combinations
        if start + length > cpu_tensor.shape[0]:
            pytest.skip(
                f"Invalid narrow: start={start}, length={length} for size {cpu_tensor.shape[0]}"
            )

        cpu_result = torch.narrow(cpu_tensor, 0, start, length)
        remote_result = torch.narrow(remote_tensor, 0, start, length)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("dimension", [0, 1])
    @pytest.mark.parametrize("size", [2, 3])
    @pytest.mark.parametrize("step", [1, 2])
    @pytest.mark.fast
    def test_unfold_operations(self, shared_machines, provider, dimension, size, step):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(8, 6)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Skip invalid combinations
        if size > cpu_tensor.shape[dimension]:
            pytest.skip(f"Unfold size {size} too large for dimension {dimension}")

        cpu_result = cpu_tensor.unfold(dimension, size, step)
        remote_result = remote_tensor.unfold(dimension, size, step)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )


class TestTensorManipulationWithGradients:
    """Test tensor manipulation operations maintain gradients properly."""

    @pytest.mark.fast
    def test_split_with_gradients(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(6, 4, requires_grad=True)
        remote_tensor = (
            cpu_tensor.to(machine.device(device_type)).detach().requires_grad_()
        )

        cpu_splits = torch.split(cpu_tensor, 2, dim=0)
        remote_splits = torch.split(remote_tensor, 2, dim=0)

        # Sum all splits for gradient computation
        cpu_loss = sum(split.sum() for split in cpu_splits)
        remote_loss = sum(split.sum() for split in remote_splits)

        cpu_loss.backward()
        remote_loss.backward()

        NumericalTestUtils.assert_tensors_close(
            remote_tensor.grad.cpu(), cpu_tensor.grad, rtol=1e-6, atol=1e-7
        )

    @pytest.mark.fast
    def test_stack_with_gradients(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensors = [torch.randn(3, 4, requires_grad=True) for _ in range(3)]
        remote_tensors = [
            t.to(machine.device(device_type)).detach().requires_grad_()
            for t in cpu_tensors
        ]

        cpu_result = torch.stack(cpu_tensors, dim=0)
        remote_result = torch.stack(remote_tensors, dim=0)

        cpu_loss = cpu_result.sum()
        remote_loss = remote_result.sum()

        cpu_loss.backward()
        remote_loss.backward()

        for cpu_t, remote_t in zip(cpu_tensors, remote_tensors):
            NumericalTestUtils.assert_tensors_close(
                remote_t.grad.cpu(), cpu_t.grad, rtol=1e-6, atol=1e-7
            )

    @pytest.mark.fast
    def test_repeat_with_gradients(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(2, 3, requires_grad=True)
        remote_tensor = (
            cpu_tensor.to(machine.device(device_type)).detach().requires_grad_()
        )

        cpu_result = cpu_tensor.repeat(2, 1)
        remote_result = remote_tensor.repeat(2, 1)

        cpu_loss = cpu_result.sum()
        remote_loss = remote_result.sum()

        cpu_loss.backward()
        remote_loss.backward()

        NumericalTestUtils.assert_tensors_close(
            remote_tensor.grad.cpu(), cpu_tensor.grad, rtol=1e-6, atol=1e-7
        )


class TestTensorManipulationEdgeCases:
    """Test edge cases for tensor manipulation operations."""

    @pytest.mark.fast
    def test_empty_tensor_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_empty = torch.empty(0, 5)
        remote_empty = cpu_empty.to(machine.device(device_type))

        # Test operations on empty tensors
        cpu_result = torch.split(cpu_empty, 1, dim=1)
        remote_result = torch.split(remote_empty, 1, dim=1)

        assert len(cpu_result) == len(remote_result)
        for cpu_split, remote_split in zip(cpu_result, remote_result):
            assert cpu_split.shape == remote_split.shape

    @pytest.mark.fast
    def test_single_element_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.tensor([[1.5]])
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test repeat on single element
        cpu_result = cpu_tensor.repeat(3, 4)
        remote_result = remote_tensor.repeat(3, 4)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_large_dimension_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Test with larger tensors
        cpu_tensor = torch.randn(100, 50)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test chunk operation
        cpu_chunks = torch.chunk(cpu_tensor, 10, dim=0)
        remote_chunks = torch.chunk(remote_tensor, 10, dim=0)

        assert len(cpu_chunks) == len(remote_chunks)
        for cpu_chunk, remote_chunk in zip(cpu_chunks, remote_chunks):
            NumericalTestUtils.assert_tensors_close(
                remote_chunk.cpu(), cpu_chunk, rtol=1e-8, atol=1e-8
            )

    @pytest.mark.fast
    def test_mixed_dimension_stacking(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Test stacking tensors with different dimensions
        cpu_tensors = [torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)]
        remote_tensors = [t.to(machine.device(device_type)) for t in cpu_tensors]

        # Test different stack dimensions
        for dim in range(3):  # 0, 1, 2
            cpu_result = torch.stack(cpu_tensors, dim=dim)
            remote_result = torch.stack(remote_tensors, dim=dim)

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )

    @pytest.mark.fast
    def test_dtype_preservation(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Test with different dtypes
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            if dtype in [torch.int32, torch.int64]:
                cpu_tensor = torch.randint(0, 10, (4, 4), dtype=dtype)
            else:
                cpu_tensor = torch.randn(4, 4, dtype=dtype)

            remote_tensor = cpu_tensor.to(machine.device(device_type))

            cpu_result = cpu_tensor.repeat(2, 1)
            remote_result = remote_tensor.repeat(2, 1)

            assert cpu_result.dtype == remote_result.cpu().dtype
            if dtype in [torch.int32, torch.int64]:
                assert torch.equal(remote_result.cpu(), cpu_result)
            else:
                NumericalTestUtils.assert_tensors_close(
                    remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
                )
