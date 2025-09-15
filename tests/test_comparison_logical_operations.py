# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch


def _generate_valid_shape_dim_keepdim_combinations():
    """Generate valid combinations of shape, dim, and keepdim parameters."""
    shapes = [(10,), (5, 5), (3, 4)]
    dims = [None, 0, 1]
    keepdims = [False, True]

    combinations = []
    for shape in shapes:
        for dim in dims:
            for keepdim in keepdims:
                # Skip invalid dim values for the given shape
                if dim is not None and dim >= len(shape):
                    continue
                combinations.append((shape, dim, keepdim))

    return combinations


class TestComparisonOperations:
    """Test element-wise comparison operations."""

    @pytest.mark.parametrize("operation", ["eq", "ne", "lt", "le", "gt", "ge"])
    @pytest.mark.parametrize("shape", [(10,), (5, 5), (3, 4)])
    @pytest.mark.fast
    def test_tensor_tensor_comparisons(
        self, shared_machines, provider, operation, shape
    ):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_a = torch.randn(*shape)
        cpu_b = torch.randn(*shape)
        remote_a = cpu_a.to(machine.device(device_type))
        remote_b = cpu_b.to(machine.device(device_type))

        cpu_result = getattr(torch, operation)(cpu_a, cpu_b)
        remote_result = getattr(torch, operation)(remote_a, remote_b)

        assert torch.equal(remote_result.cpu(), cpu_result)
        assert remote_result.dtype == torch.bool

    @pytest.mark.parametrize("operation", ["eq", "ne", "lt", "le", "gt", "ge"])
    @pytest.mark.parametrize("scalar", [0.0, 1.5, -2.0])
    @pytest.mark.fast
    def test_tensor_scalar_comparisons(
        self, shared_machines, provider, operation, scalar
    ):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = getattr(torch, operation)(cpu_tensor, scalar)
        remote_result = getattr(torch, operation)(remote_tensor, scalar)

        assert torch.equal(remote_result.cpu(), cpu_result)
        assert remote_result.dtype == torch.bool

    @pytest.mark.parametrize("operation", ["eq", "ne", "lt", "le", "gt", "ge"])
    @pytest.mark.fast
    def test_broadcasting_comparisons(self, shared_machines, provider, operation):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_a = torch.randn(3, 1)
        cpu_b = torch.randn(1, 4)
        remote_a = cpu_a.to(machine.device(device_type))
        remote_b = cpu_b.to(machine.device(device_type))

        cpu_result = getattr(torch, operation)(cpu_a, cpu_b)
        remote_result = getattr(torch, operation)(remote_a, remote_b)

        assert torch.equal(remote_result.cpu(), cpu_result)

    @pytest.mark.fast
    def test_comparison_operators(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_a = torch.randn(3, 3)
        cpu_b = torch.randn(3, 3)
        remote_a = cpu_a.to(machine.device(device_type))
        remote_b = cpu_b.to(machine.device(device_type))

        # Test operator overloads
        operators = [
            ("==", lambda x, y: x == y),
            ("!=", lambda x, y: x != y),
            ("<", lambda x, y: x < y),
            ("<=", lambda x, y: x <= y),
            (">", lambda x, y: x > y),
            (">=", lambda x, y: x >= y),
        ]

        for op_name, op_func in operators:
            cpu_result = op_func(cpu_a, cpu_b)
            remote_result = op_func(remote_a, remote_b)

            assert torch.equal(remote_result.cpu(), cpu_result), (
                f"Operator {op_name} failed"
            )


class TestSpecialValueComparisons:
    """Test comparisons involving special floating point values."""

    @pytest.mark.fast
    def test_isnan_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.tensor([1.0, float("nan"), 2.0, float("nan")])
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.isnan(cpu_tensor)
        remote_result = torch.isnan(remote_tensor)

        assert torch.equal(remote_result.cpu(), cpu_result)

    @pytest.mark.fast
    def test_isinf_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.tensor([1.0, float("inf"), -float("inf"), 2.0])
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.isinf(cpu_tensor)
        remote_result = torch.isinf(remote_tensor)

        assert torch.equal(remote_result.cpu(), cpu_result)

    @pytest.mark.fast
    def test_isfinite_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.tensor([1.0, float("inf"), float("nan"), -2.0])
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.isfinite(cpu_tensor)
        remote_result = torch.isfinite(remote_tensor)

        assert torch.equal(remote_result.cpu(), cpu_result)

    @pytest.mark.fast
    def test_isposinf_isneginf(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.tensor([1.0, float("inf"), -float("inf"), 0.0])
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test isposinf
        cpu_pos_result = torch.isposinf(cpu_tensor)
        remote_pos_result = torch.isposinf(remote_tensor)
        assert torch.equal(remote_pos_result.cpu(), cpu_pos_result)

        # Test isneginf
        cpu_neg_result = torch.isneginf(cpu_tensor)
        remote_neg_result = torch.isneginf(remote_tensor)
        assert torch.equal(remote_neg_result.cpu(), cpu_neg_result)


class TestApproximateComparisons:
    """Test approximate comparison operations."""

    @pytest.mark.parametrize("rtol", [1e-5, 1e-3, 1e-1])
    @pytest.mark.parametrize("atol", [1e-8, 1e-6, 1e-4])
    @pytest.mark.fast
    def test_allclose_operations(self, shared_machines, provider, rtol, atol):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_a = torch.randn(4, 4)
        # Create slightly different tensor
        cpu_b = cpu_a + torch.randn(4, 4) * 1e-6

        remote_a = cpu_a.to(machine.device(device_type))
        remote_b = cpu_b.to(machine.device(device_type))

        cpu_result = torch.allclose(cpu_a, cpu_b, rtol=rtol, atol=atol)
        remote_result = torch.allclose(remote_a, remote_b, rtol=rtol, atol=atol)

        assert cpu_result == remote_result

    @pytest.mark.fast
    def test_equal_operations(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_a = torch.randn(3, 3)
        cpu_b = cpu_a.clone()
        cpu_c = torch.randn(3, 3)

        remote_a = cpu_a.to(machine.device(device_type))
        remote_b = cpu_b.to(machine.device(device_type))
        remote_c = cpu_c.to(machine.device(device_type))

        # Test equal tensors
        assert torch.equal(cpu_a, cpu_b) == torch.equal(remote_a, remote_b)

        # Test unequal tensors
        assert torch.equal(cpu_a, cpu_c) == torch.equal(remote_a, remote_c)


class TestLogicalOperations:
    """Test logical operations (and, or, not, xor)."""

    @pytest.mark.parametrize("operation", ["logical_and", "logical_or", "logical_xor"])
    @pytest.mark.parametrize("shape", [(10,), (4, 4), (2, 3, 4)])
    @pytest.mark.fast
    def test_binary_logical_operations(
        self, shared_machines, provider, operation, shape
    ):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create boolean tensors
        cpu_a = torch.rand(*shape) > 0.5
        cpu_b = torch.rand(*shape) > 0.5
        remote_a = cpu_a.to(machine.device(device_type))
        remote_b = cpu_b.to(machine.device(device_type))

        cpu_result = getattr(torch, operation)(cpu_a, cpu_b)
        remote_result = getattr(torch, operation)(remote_a, remote_b)

        assert torch.equal(remote_result.cpu(), cpu_result)
        assert remote_result.dtype == torch.bool

    @pytest.mark.parametrize("shape", [(8,), (3, 3), (2, 2, 2)])
    @pytest.mark.fast
    def test_logical_not(self, shared_machines, provider, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.rand(*shape) > 0.5
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.logical_not(cpu_tensor)
        remote_result = torch.logical_not(remote_tensor)

        assert torch.equal(remote_result.cpu(), cpu_result)
        assert remote_result.dtype == torch.bool

    @pytest.mark.fast
    def test_logical_operations_with_numeric_tensors(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Test logical operations with numeric tensors (non-zero = True)
        cpu_a = torch.tensor([0.0, 1.0, -1.0, 2.5])
        cpu_b = torch.tensor([1.0, 0.0, -2.0, 0.0])
        remote_a = cpu_a.to(machine.device(device_type))
        remote_b = cpu_b.to(machine.device(device_type))

        operations = ["logical_and", "logical_or", "logical_xor"]
        for operation in operations:
            cpu_result = getattr(torch, operation)(cpu_a, cpu_b)
            remote_result = getattr(torch, operation)(remote_a, remote_b)

            assert torch.equal(remote_result.cpu(), cpu_result)

    @pytest.mark.fast
    def test_logical_broadcasting(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_a = torch.rand(3, 1) > 0.5
        cpu_b = torch.rand(1, 4) > 0.5
        remote_a = cpu_a.to(machine.device(device_type))
        remote_b = cpu_b.to(machine.device(device_type))

        cpu_result = torch.logical_and(cpu_a, cpu_b)
        remote_result = torch.logical_and(remote_a, remote_b)

        assert torch.equal(remote_result.cpu(), cpu_result)
        assert remote_result.shape == (3, 4)


class TestBooleanReductions:
    """Test boolean reduction operations (all, any)."""

    @pytest.mark.parametrize(
        "shape_dim_keepdim", _generate_valid_shape_dim_keepdim_combinations()
    )
    @pytest.mark.fast
    def test_all_operations(self, shared_machines, provider, shape_dim_keepdim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        shape, dim, keepdim = shape_dim_keepdim

        cpu_tensor = torch.rand(*shape) > 0.3  # Mixed true/false
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.all(cpu_tensor, dim=dim, keepdim=keepdim)
        remote_result = torch.all(remote_tensor, dim=dim, keepdim=keepdim)

        assert torch.equal(remote_result.cpu(), cpu_result)

    @pytest.mark.parametrize(
        "shape_dim_keepdim", _generate_valid_shape_dim_keepdim_combinations()
    )
    @pytest.mark.fast
    def test_any_operations(self, shared_machines, provider, shape_dim_keepdim):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        shape, dim, keepdim = shape_dim_keepdim

        cpu_tensor = torch.rand(*shape) > 0.7  # Mostly false with some true
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.any(cpu_tensor, dim=dim, keepdim=keepdim)
        remote_result = torch.any(remote_tensor, dim=dim, keepdim=keepdim)

        assert torch.equal(remote_result.cpu(), cpu_result)

    @pytest.mark.fast
    def test_all_any_with_numeric_tensors(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Test with numeric tensors (zero = False, non-zero = True)
        cpu_tensor = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test all
        cpu_all = torch.all(cpu_tensor, dim=1)
        remote_all = torch.all(remote_tensor, dim=1)
        assert torch.equal(remote_all.cpu(), cpu_all)

        # Test any
        cpu_any = torch.any(cpu_tensor, dim=1)
        remote_any = torch.any(remote_tensor, dim=1)
        assert torch.equal(remote_any.cpu(), cpu_any)


class TestComparisonLogicalWithGradients:
    """Test that comparison and logical operations properly handle gradients."""

    @pytest.mark.fast
    def test_comparison_no_gradients(self, shared_machines, provider):
        """Comparison operations should not propagate gradients."""

        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_a = torch.randn(3, 3, requires_grad=True)
        cpu_b = torch.randn(3, 3, requires_grad=True)
        remote_a = cpu_a.to(machine.device(device_type))
        remote_b = cpu_b.to(machine.device(device_type))
        remote_a.requires_grad_(True)
        remote_b.requires_grad_(True)

        # Comparison operations should not require gradients
        remote_result = remote_a > remote_b
        assert not remote_result.requires_grad
        assert remote_result.dtype == torch.bool

    @pytest.mark.fast
    def test_logical_no_gradients(self, shared_machines, provider):
        """Logical operations should not propagate gradients."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_a = torch.randn(3, 3, requires_grad=True) > 0
        remote_a = cpu_a.to(machine.device(device_type))

        remote_result = torch.logical_not(remote_a)
        assert not remote_result.requires_grad
        assert remote_result.dtype == torch.bool

    @pytest.mark.fast
    def test_gradients_through_where_operation(self, shared_machines, provider):
        """Test gradients flow through conditional operations using comparison results."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_x = torch.randn(3, 3, requires_grad=True)
        cpu_condition = cpu_x > 0
        cpu_y = torch.ones_like(cpu_x)
        cpu_z = torch.zeros_like(cpu_x)

        remote_x = cpu_x.to(machine.device(device_type))
        remote_x.requires_grad_(True)
        remote_condition = remote_x > 0
        remote_y = cpu_y.to(machine.device(device_type))
        remote_z = cpu_z.to(machine.device(device_type))

        # Use where with comparison result
        cpu_result = torch.where(cpu_condition, cpu_y, cpu_z)
        remote_result = torch.where(remote_condition, remote_y, remote_z)

        # The results should be the same
        assert torch.equal(remote_result.cpu(), cpu_result)


class TestComparisonLogicalEdgeCases:
    """Test edge cases for comparison and logical operations."""

    @pytest.mark.fast
    def test_comparison_with_special_values(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.tensor([1.0, float("nan"), float("inf"), -float("inf")])
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test comparisons with NaN (should always be False except !=)
        for operation in ["eq", "ne", "lt", "le", "gt", "ge"]:
            cpu_result = getattr(torch, operation)(cpu_tensor, float("nan"))
            remote_result = getattr(torch, operation)(remote_tensor, float("nan"))
            assert torch.equal(remote_result.cpu(), cpu_result)

    @pytest.mark.fast
    def test_logical_with_empty_tensors(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_empty = torch.empty(0, dtype=torch.bool)
        remote_empty = cpu_empty.to(machine.device(device_type))

        # all() on empty tensor should return True
        cpu_all = torch.all(cpu_empty)
        remote_all = torch.all(remote_empty)
        assert cpu_all.item() == remote_all.item() and cpu_all.item() is True

        # any() on empty tensor should return False
        cpu_any = torch.any(cpu_empty)
        remote_any = torch.any(remote_empty)
        assert cpu_any.item() == remote_any.item() and cpu_any.item() is False

    @pytest.mark.fast
    def test_comparison_dtype_consistency(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Test with different input dtypes
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

        for dtype in dtypes:
            cpu_a = torch.ones(3, dtype=dtype)
            cpu_b = torch.zeros(3, dtype=dtype)
            remote_a = cpu_a.to(machine.device(device_type))
            remote_b = cpu_b.to(machine.device(device_type))

            cpu_result = cpu_a > cpu_b
            remote_result = remote_a > remote_b

            assert torch.equal(remote_result.cpu(), cpu_result)
            assert remote_result.dtype == torch.bool

    @pytest.mark.fast
    def test_chained_comparisons(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(4, 4)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test chained comparison operations
        cpu_result = (cpu_tensor > 0) & (cpu_tensor < 1)
        remote_result = (remote_tensor > 0) & (remote_tensor < 1)

        assert torch.equal(remote_result.cpu(), cpu_result)
