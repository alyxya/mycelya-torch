# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch
from test_utilities import NumericalTestUtils


def generate_valid_shape_dim_combinations(
    shapes, include_none=True, include_negative=True
):
    """Generate only valid (shape, dim) combinations to eliminate test skips."""
    combinations = []

    if include_none:
        for shape in shapes:
            combinations.append((shape, None))

    for shape in shapes:
        ndim = len(shape)
        # Generate valid positive dimensions: 0 to ndim-1
        for dim in range(ndim):
            combinations.append((shape, dim))

        # Generate valid negative dimensions: -ndim to -1
        if include_negative:
            for dim in range(-ndim, 0):
                combinations.append((shape, dim))

    return combinations


class TestBasicReductions:
    """Test sum and mean operations comprehensively."""

    @pytest.mark.parametrize(
        "shape,dim",
        generate_valid_shape_dim_combinations([(10,), (5, 5), (3, 4, 5), (2, 3, 4, 5)]),
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.fast
    def test_sum_variations(self, shared_machines, provider, shape, dim, keepdim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.sum(cpu_tensor, dim=dim, keepdim=keepdim)
        remote_result = torch.sum(remote_tensor, dim=dim, keepdim=keepdim)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize(
        "shape,dim", generate_valid_shape_dim_combinations([(10,), (4, 4), (2, 3, 4)])
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.fast
    def test_mean_variations(self, shared_machines, provider, shape, dim, keepdim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.mean(cpu_tensor, dim=dim, keepdim=keepdim)
        remote_result = torch.mean(remote_tensor, dim=dim, keepdim=keepdim)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )


class TestStatisticalReductions:
    """Test standard deviation, variance, and related operations."""

    @pytest.mark.parametrize(
        "shape,dim", generate_valid_shape_dim_combinations([(10,), (5, 5), (3, 4, 5)])
    )
    @pytest.mark.parametrize("unbiased", [True, False])
    @pytest.mark.fast
    def test_std_operations(self, shared_machines, provider, shape, dim, unbiased):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.std(cpu_tensor, dim=dim, unbiased=unbiased)
        remote_result = torch.std(remote_tensor, dim=dim, unbiased=unbiased)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-4, atol=1e-5
        )

    @pytest.mark.parametrize(
        "shape,dim", generate_valid_shape_dim_combinations([(10,), (4, 4), (2, 3, 4)])
    )
    @pytest.mark.parametrize("unbiased", [True, False])
    @pytest.mark.fast
    def test_var_operations(self, shared_machines, provider, shape, dim, unbiased):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.var(cpu_tensor, dim=dim, unbiased=unbiased)
        remote_result = torch.var(remote_tensor, dim=dim, unbiased=unbiased)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-4, atol=1e-5
        )

    @pytest.mark.parametrize(
        "shape,dim", generate_valid_shape_dim_combinations([(8,), (4, 4)])
    )
    @pytest.mark.fast
    def test_std_mean_combined(self, shared_machines, provider, shape, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_std, cpu_mean = torch.std_mean(cpu_tensor, dim=dim)
        remote_std, remote_mean = torch.std_mean(remote_tensor, dim=dim)

        NumericalTestUtils.assert_tensors_close(
            remote_std.cpu(), cpu_std, rtol=1e-4, atol=1e-5
        )
        NumericalTestUtils.assert_tensors_close(
            remote_mean.cpu(), cpu_mean, rtol=1e-5, atol=1e-6
        )


class TestMinMaxOperations:
    """Test min, max, argmin, argmax operations."""

    @pytest.mark.parametrize(
        "shape,dim", generate_valid_shape_dim_combinations([(10,), (5, 5), (3, 4, 5)])
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.fast
    def test_min_operations(self, shared_machines, provider, shape, dim, keepdim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        if dim is None:
            cpu_result = torch.min(cpu_tensor)
            remote_result = torch.min(remote_tensor)

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )
        else:
            cpu_values, cpu_indices = torch.min(cpu_tensor, dim=dim, keepdim=keepdim)
            remote_values, remote_indices = torch.min(
                remote_tensor, dim=dim, keepdim=keepdim
            )

            NumericalTestUtils.assert_tensors_close(
                remote_values.cpu(), cpu_values, rtol=1e-8, atol=1e-8
            )
            assert torch.equal(remote_indices.cpu(), cpu_indices)

    @pytest.mark.parametrize(
        "shape,dim", generate_valid_shape_dim_combinations([(10,), (4, 4), (2, 3, 4)])
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.fast
    def test_max_operations(self, shared_machines, provider, shape, dim, keepdim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        if dim is None:
            cpu_result = torch.max(cpu_tensor)
            remote_result = torch.max(remote_tensor)

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )
        else:
            cpu_values, cpu_indices = torch.max(cpu_tensor, dim=dim, keepdim=keepdim)
            remote_values, remote_indices = torch.max(
                remote_tensor, dim=dim, keepdim=keepdim
            )

            NumericalTestUtils.assert_tensors_close(
                remote_values.cpu(), cpu_values, rtol=1e-8, atol=1e-8
            )
            assert torch.equal(remote_indices.cpu(), cpu_indices)

    @pytest.mark.parametrize(
        "shape,dim", generate_valid_shape_dim_combinations([(10,), (5, 5)])
    )
    @pytest.mark.fast
    def test_argmin_argmax(self, shared_machines, provider, shape, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test argmin
        cpu_argmin = torch.argmin(cpu_tensor, dim=dim)
        remote_argmin = torch.argmin(remote_tensor, dim=dim)
        assert torch.equal(remote_argmin.cpu(), cpu_argmin)

        # Test argmax
        cpu_argmax = torch.argmax(cpu_tensor, dim=dim)
        remote_argmax = torch.argmax(remote_tensor, dim=dim)
        assert torch.equal(remote_argmax.cpu(), cpu_argmax)

    @pytest.mark.fast
    def test_aminmax(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(5, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_min, cpu_max = torch.aminmax(cpu_tensor)
        remote_min, remote_max = torch.aminmax(remote_tensor)

        NumericalTestUtils.assert_tensors_close(
            remote_min.cpu(), cpu_min, rtol=1e-8, atol=1e-8
        )
        NumericalTestUtils.assert_tensors_close(
            remote_max.cpu(), cpu_max, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.fast
    def test_aminmax_with_dim(self, shared_machines, provider, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_min, cpu_max = torch.aminmax(cpu_tensor, dim=dim)
        remote_min, remote_max = torch.aminmax(remote_tensor, dim=dim)

        NumericalTestUtils.assert_tensors_close(
            remote_min.cpu(), cpu_min, rtol=1e-8, atol=1e-8
        )
        NumericalTestUtils.assert_tensors_close(
            remote_max.cpu(), cpu_max, rtol=1e-8, atol=1e-8
        )


class TestProductReductions:
    """Test product reduction operations."""

    @pytest.mark.parametrize(
        "shape,dim",
        generate_valid_shape_dim_combinations(
            [(5,), (3, 4), (2, 3, 4)], include_none=False
        ),
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.fast
    def test_prod_operations(self, shared_machines, provider, shape, dim, keepdim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Use smaller values to avoid overflow
        cpu_tensor = torch.rand(*shape) + 0.1  # Range [0.1, 1.1]
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.prod(cpu_tensor, dim=dim, keepdim=keepdim)
        remote_result = torch.prod(remote_tensor, dim=dim, keepdim=keepdim)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-4, atol=1e-5
        )


class TestCumulativeOperations:
    """Test cumulative sum and product operations."""

    @pytest.mark.parametrize(
        "shape,dim",
        generate_valid_shape_dim_combinations(
            [(10,), (5, 5), (3, 4)], include_none=False
        ),
    )
    @pytest.mark.fast
    def test_cumsum_operations(self, shared_machines, provider, shape, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.cumsum(cpu_tensor, dim=dim)
        remote_result = torch.cumsum(remote_tensor, dim=dim)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize(
        "shape,dim",
        generate_valid_shape_dim_combinations(
            [(8,), (4, 4), (2, 3)], include_none=False
        ),
    )
    @pytest.mark.fast
    def test_cumprod_operations(self, shared_machines, provider, shape, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Use smaller values to avoid overflow
        cpu_tensor = torch.rand(*shape) * 0.5 + 0.5  # Range [0.5, 1.0]
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.cumprod(cpu_tensor, dim=dim)
        remote_result = torch.cumprod(remote_tensor, dim=dim)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-4, atol=1e-5
        )


class TestSpecializedReductions:
    """Test specialized reduction operations."""

    @pytest.mark.parametrize(
        "shape,dim", generate_valid_shape_dim_combinations([(10,), (5, 5), (3, 4)])
    )
    @pytest.mark.fast
    def test_norm_operations(self, shared_machines, provider, shape, dim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test L2 norm (default)
        cpu_result = torch.norm(cpu_tensor, dim=dim)
        remote_result = torch.norm(remote_tensor, dim=dim)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize("p", [1, 2, float("inf")])
    @pytest.mark.fast
    def test_norm_with_p_values(self, shared_machines, provider, p):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.norm(cpu_tensor, p=p)
        remote_result = torch.norm(remote_tensor, p=p)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )

    @pytest.mark.fast
    def test_logsumexp(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(5, 5)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test reduction over all dimensions
        cpu_result = torch.logsumexp(cpu_tensor, dim=[0, 1])
        remote_result = torch.logsumexp(remote_tensor, dim=[0, 1])

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.fast
    def test_logsumexp_with_dim(self, shared_machines, provider, dim, keepdim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.logsumexp(cpu_tensor, dim=dim, keepdim=keepdim)
        remote_result = torch.logsumexp(remote_tensor, dim=dim, keepdim=keepdim)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )


class TestReductionsWithGradients:
    """Test reduction operations maintain gradients properly."""

    @pytest.mark.parametrize("operation", ["sum", "mean", "std", "var"])
    @pytest.mark.fast
    def test_reductions_with_gradients(self, shared_machines, provider, operation):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create leaf tensors to maintain gradient capabilities
        data = torch.randn(4, 4)
        cpu_tensor = data.clone().requires_grad_(True)
        remote_tensor = (
            data.clone().to(machine.device(device_type)).requires_grad_(True)
        )

        # Forward pass
        if operation in ["std", "var"]:
            cpu_result = getattr(torch, operation)(cpu_tensor, unbiased=False)
            remote_result = getattr(torch, operation)(remote_tensor, unbiased=False)
        else:
            cpu_result = getattr(torch, operation)(cpu_tensor)
            remote_result = getattr(torch, operation)(remote_tensor)

        # Backward pass
        cpu_result.backward()
        remote_result.backward()

        # Check gradients
        NumericalTestUtils.assert_tensors_close(
            remote_tensor.grad.cpu(), cpu_tensor.grad, rtol=1e-4, atol=1e-5
        )

    @pytest.mark.fast
    def test_complex_reduction_expression_with_gradients(
        self, shared_machines, provider
    ):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create leaf tensors to maintain gradient capabilities
        data = torch.randn(5, 5)
        cpu_x = data.clone().requires_grad_(True)
        remote_x = data.clone().to(machine.device(device_type)).requires_grad_(True)

        # Complex expression combining multiple reductions
        cpu_result = torch.sum(cpu_x, dim=0) + torch.mean(cpu_x, dim=1)
        remote_result = torch.sum(remote_x, dim=0) + torch.mean(remote_x, dim=1)

        cpu_loss = cpu_result.sum()
        remote_loss = remote_result.sum()

        cpu_loss.backward()
        remote_loss.backward()

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )
        NumericalTestUtils.assert_tensors_close(
            remote_x.grad.cpu(), cpu_x.grad, rtol=1e-4, atol=1e-5
        )


class TestReductionEdgeCases:
    """Test edge cases for reduction operations."""

    @pytest.mark.fast
    def test_reductions_with_single_element(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.tensor([[2.5]])
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        for operation in ["sum", "mean", "min", "max"]:
            cpu_result = getattr(torch, operation)(cpu_tensor)
            remote_result = getattr(torch, operation)(remote_tensor)

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )

        # Test std/var with unbiased=False to avoid invalid statistical operations
        for operation in ["std", "var"]:
            cpu_result = getattr(torch, operation)(cpu_tensor, unbiased=False)
            remote_result = getattr(torch, operation)(remote_tensor, unbiased=False)

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )

    @pytest.mark.fast
    def test_reductions_with_zeros(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_zeros = torch.zeros(3, 3)
        remote_zeros = cpu_zeros.to(machine.device(device_type))

        # Test operations that should work with zeros
        for operation in ["sum", "mean", "min", "max"]:
            cpu_result = getattr(torch, operation)(cpu_zeros)
            remote_result = getattr(torch, operation)(remote_zeros)

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )

    @pytest.mark.fast
    def test_reductions_preserve_dtype(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Test with integer tensor
        cpu_int_tensor = torch.randint(0, 10, (3, 3), dtype=torch.int32)
        remote_int_tensor = cpu_int_tensor.to(machine.device(device_type))

        cpu_sum = torch.sum(cpu_int_tensor)
        remote_sum = torch.sum(remote_int_tensor)

        assert cpu_sum.dtype == remote_sum.cpu().dtype
        assert torch.equal(remote_sum.cpu(), cpu_sum)
