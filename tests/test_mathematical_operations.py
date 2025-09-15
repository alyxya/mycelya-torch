# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch
from test_utilities import NumericalTestUtils


class TestTrigonometricOperations:
    """Test trigonometric functions."""

    @pytest.mark.parametrize("operation", ["sin", "cos", "tan"])
    @pytest.mark.parametrize("shape", [(10,), (5, 5), (2, 3, 4)])
    @pytest.mark.fast
    def test_basic_trigonometric(self, shared_machines, provider, operation, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create test tensor with values in reasonable range for trig functions
        cpu_tensor = torch.randn(*shape) * 2  # Range roughly [-6, 6]
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test operation
        cpu_result = getattr(torch, operation)(cpu_tensor)
        remote_result = getattr(torch, operation)(remote_tensor)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize("operation", ["asin", "acos", "atan"])
    @pytest.mark.parametrize("shape", [(10,), (3, 3)])
    @pytest.mark.fast
    def test_inverse_trigonometric(self, shared_machines, provider, operation, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create test tensor with values in valid domain
        if operation in ["asin", "acos"]:
            cpu_tensor = torch.rand(*shape) * 2 - 1  # Range [-1, 1]
        else:  # atan
            cpu_tensor = torch.randn(*shape) * 5  # Any real values

        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test operation
        cpu_result = getattr(torch, operation)(cpu_tensor)
        remote_result = getattr(torch, operation)(remote_tensor)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize("operation", ["sinh", "cosh", "tanh"])
    @pytest.mark.parametrize("shape", [(8,), (4, 4)])
    @pytest.mark.fast
    def test_hyperbolic_functions(self, shared_machines, provider, operation, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create test tensor with moderate values to avoid overflow
        cpu_tensor = torch.randn(*shape) * 2
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test operation
        cpu_result = getattr(torch, operation)(cpu_tensor)
        remote_result = getattr(torch, operation)(remote_tensor)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )

    @pytest.mark.fast
    def test_atan2(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        y_cpu = torch.randn(5, 5)
        x_cpu = torch.randn(5, 5)
        y_remote = y_cpu.to(machine.device(device_type))
        x_remote = x_cpu.to(machine.device(device_type))

        cpu_result = torch.atan2(y_cpu, x_cpu)
        remote_result = torch.atan2(y_remote, x_remote)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )


class TestExponentialLogarithmicOperations:
    """Test exponential and logarithmic functions."""

    @pytest.mark.parametrize("operation", ["exp", "exp2", "expm1"])
    @pytest.mark.parametrize("shape", [(10,), (3, 4), (2, 2, 2)])
    @pytest.mark.fast
    def test_exponential_functions(self, shared_machines, provider, operation, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Use smaller values to avoid overflow
        cpu_tensor = torch.randn(*shape) * 2
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test operation
        cpu_result = getattr(torch, operation)(cpu_tensor)
        remote_result = getattr(torch, operation)(remote_tensor)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize("operation", ["log", "log2", "log10", "log1p"])
    @pytest.mark.parametrize("shape", [(10,), (3, 4)])
    @pytest.mark.fast
    def test_logarithmic_functions(self, shared_machines, provider, operation, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Use positive values for log functions
        if operation == "log1p":
            cpu_tensor = torch.rand(*shape) * 5 + 0.1  # Range [0.1, 5.1]
        else:
            cpu_tensor = torch.rand(*shape) * 10 + 0.01  # Range [0.01, 10.01]

        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test operation
        cpu_result = getattr(torch, operation)(cpu_tensor)
        remote_result = getattr(torch, operation)(remote_tensor)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize("shape", [(8,), (4, 4)])
    @pytest.mark.fast
    def test_power_operations(self, shared_machines, provider, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        base_cpu = torch.abs(torch.randn(*shape)) + 0.1  # Positive base
        exp_cpu = torch.randn(*shape) * 2  # Moderate exponents

        base_remote = base_cpu.to(machine.device(device_type))
        exp_remote = exp_cpu.to(machine.device(device_type))

        # Test tensor ** tensor
        cpu_result = torch.pow(base_cpu, exp_cpu)
        remote_result = torch.pow(base_remote, exp_remote)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-4, atol=1e-5
        )

        # Test tensor ** scalar
        scalar_exp = 2.5
        cpu_result_scalar = torch.pow(base_cpu, scalar_exp)
        remote_result_scalar = torch.pow(base_remote, scalar_exp)

        NumericalTestUtils.assert_tensors_close(
            remote_result_scalar.cpu(), cpu_result_scalar, rtol=1e-5, atol=1e-6
        )


class TestRootOperations:
    """Test square root and related operations."""

    @pytest.mark.parametrize("operation", ["sqrt", "rsqrt"])
    @pytest.mark.parametrize("shape", [(10,), (5, 5), (2, 3, 4)])
    @pytest.mark.fast
    def test_root_operations(self, shared_machines, provider, operation, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Use positive values for sqrt operations
        cpu_tensor = torch.rand(*shape) * 10 + 0.01  # Range [0.01, 10.01]
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test operation
        cpu_result = getattr(torch, operation)(cpu_tensor)
        remote_result = getattr(torch, operation)(remote_tensor)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-5, atol=1e-6
        )


class TestRoundingOperations:
    """Test rounding and ceiling/floor operations."""

    @pytest.mark.parametrize("operation", ["round", "floor", "ceil", "trunc"])
    @pytest.mark.parametrize("shape", [(10,), (3, 4), (2, 2, 3)])
    @pytest.mark.fast
    def test_rounding_operations(self, shared_machines, provider, operation, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create tensor with fractional values
        cpu_tensor = torch.randn(*shape) * 10  # Range roughly [-30, 30]
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test operation
        cpu_result = getattr(torch, operation)(cpu_tensor)
        remote_result = getattr(torch, operation)(remote_tensor)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.fast
    def test_frac(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(5, 5) * 10
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.frac(cpu_tensor)
        remote_result = torch.frac(remote_tensor)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )


class TestMiscellaneousMathOperations:
    """Test various other mathematical operations."""

    @pytest.mark.parametrize("shape", [(10,), (4, 4)])
    @pytest.mark.fast
    def test_abs(self, shared_machines, provider, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape) * 10
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.abs(cpu_tensor)
        remote_result = torch.abs(remote_tensor)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("shape", [(8,), (3, 3)])
    @pytest.mark.fast
    def test_clamp(self, shared_machines, provider, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape) * 10
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        min_val, max_val = -5.0, 5.0

        cpu_result = torch.clamp(cpu_tensor, min_val, max_val)
        remote_result = torch.clamp(remote_tensor, min_val, max_val)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("shape", [(10,), (5, 5)])
    @pytest.mark.fast
    def test_sign(self, shared_machines, provider, shape):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(*shape) * 10
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        cpu_result = torch.sign(cpu_tensor)
        remote_result = torch.sign(remote_tensor)

        NumericalTestUtils.assert_tensors_close(
            remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
        )


class TestMathOperationsWithGradients:
    """Test mathematical operations maintain gradients properly."""

    @pytest.mark.parametrize("operation", ["sin", "cos", "exp", "log", "sqrt"])
    @pytest.mark.fast
    def test_math_operations_with_gradients(self, shared_machines, provider, operation):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Create tensors requiring gradients
        if operation == "log":
            cpu_tensor = torch.rand(3, 3) + 0.1
        elif operation == "sqrt":
            cpu_tensor = torch.rand(3, 3) + 0.01
        else:
            cpu_tensor = torch.randn(3, 3)

        cpu_tensor = cpu_tensor.detach().requires_grad_()
        remote_tensor = (
            cpu_tensor.to(machine.device(device_type)).detach().requires_grad_()
        )

        # Forward pass
        cpu_result = getattr(torch, operation)(cpu_tensor)
        remote_result = getattr(torch, operation)(remote_tensor)

        # Backward pass
        cpu_loss = cpu_result.sum()
        remote_loss = remote_result.sum()

        cpu_loss.backward()
        remote_loss.backward()

        # Check gradients
        NumericalTestUtils.assert_tensors_close(
            remote_tensor.grad.cpu(), cpu_tensor.grad, rtol=1e-4, atol=1e-5
        )

    @pytest.mark.fast
    def test_complex_math_expression_with_gradients(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_x = torch.randn(4, 4, requires_grad=True)
        remote_x = cpu_x.to(machine.device(device_type)).detach().requires_grad_()

        # Complex expression: sin(x^2) + exp(-abs(x))
        cpu_result = torch.sin(cpu_x.pow(2)) + torch.exp(-torch.abs(cpu_x))
        remote_result = torch.sin(remote_x.pow(2)) + torch.exp(-torch.abs(remote_x))

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


class TestMathOperationsEdgeCases:
    """Test edge cases for mathematical operations."""

    @pytest.mark.fast
    def test_operations_with_special_values(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Test with values that might cause numerical issues
        cpu_tensor = torch.tensor([0.0, 1.0, -1.0, float("inf"), -float("inf")])
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        # Test operations that handle special values
        for operation in ["abs", "sign"]:
            cpu_result = getattr(torch, operation)(cpu_tensor)
            remote_result = getattr(torch, operation)(remote_tensor)

            # Compare finite values
            finite_mask = torch.isfinite(cpu_result)
            if finite_mask.any():
                NumericalTestUtils.assert_tensors_close(
                    remote_result.cpu()[finite_mask],
                    cpu_result[finite_mask],
                    rtol=1e-8,
                    atol=1e-8,
                )

    @pytest.mark.fast
    def test_operations_with_zero_tensors(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_zeros = torch.zeros(3, 3)
        remote_zeros = cpu_zeros.to(machine.device(device_type))

        # Test operations that should work with zeros
        for operation in ["abs", "sign", "sin", "cos", "tanh"]:
            cpu_result = getattr(torch, operation)(cpu_zeros)
            remote_result = getattr(torch, operation)(remote_zeros)

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-8, atol=1e-8
            )

    @pytest.mark.fast
    def test_operations_with_single_element_tensors(self, shared_machines, provider):
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Test various operations
        for operation in ["sin", "exp", "log", "sqrt", "abs"]:
            if operation in ["log", "sqrt"]:
                test_val = torch.tensor([2.5])  # Positive value
            else:
                test_val = torch.tensor([1.5])

            cpu_test = test_val
            remote_test = test_val.to(machine.device(device_type))

            cpu_result = getattr(torch, operation)(cpu_test)
            remote_result = getattr(torch, operation)(remote_test)

            NumericalTestUtils.assert_tensors_close(
                remote_result.cpu(), cpu_result, rtol=1e-6, atol=1e-7
            )
