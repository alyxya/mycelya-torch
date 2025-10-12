# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test suite for @remote decorator functionality.

This module tests the remote function execution system including:
- Basic remote function execution
- Automatic machine inference
- Multiple return values
- Parameterized functions
- Error handling
"""

import pytest
import torch

import mycelya_torch


@pytest.mark.fast
class TestRemoteDecoratorBasic:
    """Basic tests for @remote decorator functionality."""

    def test_simple_addition(self, shared_machines, provider):
        """Test simple addition with @remote decorator."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote
        def simple_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        a = torch.randn(3, 3, device=device)
        b = torch.randn(3, 3, device=device)

        result = simple_add(a, b)

        assert result.device.type == "mycelya"
        assert result.shape == (3, 3)

        # Verify correctness
        result_cpu = result.cpu()
        expected = (a.cpu() + b.cpu())
        torch.testing.assert_close(result_cpu, expected, rtol=1e-4, atol=1e-6)

    def test_matrix_multiplication(self, shared_machines, provider):
        """Test matrix multiplication with @remote decorator."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote()
        def matrix_multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.matmul(x, y)

        a = torch.randn(4, 4, device=device)
        b = torch.randn(4, 4, device=device)

        result = matrix_multiply(a, b)

        assert result.device.type == "mycelya"
        assert result.shape == (4, 4)

    def test_scalar_multiplication(self, shared_machines, provider):
        """Test operations with scalar parameters."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote
        def scale_tensor(x: torch.Tensor, scale: float) -> torch.Tensor:
            return x * scale

        x = torch.randn(3, 3, device=device)

        result = scale_tensor(x, 2.5)

        assert result.device.type == "mycelya"
        assert result.shape == (3, 3)

        # Verify correctness
        result_cpu = result.cpu()
        expected = x.cpu() * 2.5
        torch.testing.assert_close(result_cpu, expected, rtol=1e-4, atol=1e-6)

    def test_multiple_operations(self, shared_machines, provider):
        """Test function with multiple tensor operations."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote()
        def complex_computation(x: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
            y = x * scale
            z = torch.relu(y)
            w = torch.softmax(z, dim=-1)
            return w.sum(dim=0)

        x = torch.randn(8, 5, device=device)

        result = complex_computation(x, scale=3.0)

        assert result.device.type == "mycelya"
        assert result.shape == (5,)


@pytest.mark.fast
class TestRemoteDecoratorMultipleReturns:
    """Tests for @remote decorator with multiple return values."""

    def test_tuple_return(self, shared_machines, provider):
        """Test function returning tuple of tensors."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote
        def split_and_process(
            tensor: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            mid = tensor.shape[0] // 2
            first_half = tensor[:mid]
            second_half = tensor[mid:]

            processed_first = torch.relu(first_half)
            processed_second = torch.sigmoid(second_half)

            return processed_first, processed_second

        x = torch.randn(8, 5, device=device)

        first, second = split_and_process(x)

        assert first.device.type == "mycelya"
        assert second.device.type == "mycelya"
        assert first.shape == (4, 5)
        assert second.shape == (4, 5)

    def test_multiple_tensor_operations(self, shared_machines, provider):
        """Test function returning multiple processed tensors."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote()
        def compute_stats(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mean = x.mean(dim=0)
            std = x.std(dim=0)
            max_vals = x.max(dim=0)[0]
            return mean, std, max_vals

        x = torch.randn(10, 5, device=device)

        mean, std, max_vals = compute_stats(x)

        assert mean.device.type == "mycelya"
        assert std.device.type == "mycelya"
        assert max_vals.device.type == "mycelya"
        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert max_vals.shape == (5,)


@pytest.mark.fast
class TestRemoteDecoratorParameters:
    """Tests for @remote decorator with various parameter types."""

    def test_keyword_arguments(self, shared_machines, provider):
        """Test function with keyword arguments."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote
        def scaled_operation(
            tensor: torch.Tensor, scale: float, operation: str = "multiply"
        ) -> torch.Tensor:
            if operation == "multiply":
                result = tensor * scale
            elif operation == "add":
                result = tensor + scale
            else:
                result = tensor

            return torch.tanh(result)

        x = torch.randn(5, 5, device=device)

        result1 = scaled_operation(x, scale=2.5, operation="multiply")
        result2 = scaled_operation(x, 1.0, operation="add")

        assert result1.device.type == "mycelya"
        assert result2.device.type == "mycelya"
        assert result1.shape == (5, 5)
        assert result2.shape == (5, 5)

    def test_default_arguments(self, shared_machines, provider):
        """Test function with default arguments."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote()
        def process_with_defaults(
            x: torch.Tensor, alpha: float = 1.0, beta: float = 0.0
        ) -> torch.Tensor:
            return x * alpha + beta

        x = torch.randn(3, 3, device=device)

        result1 = process_with_defaults(x)
        result2 = process_with_defaults(x, alpha=2.0, beta=1.0)

        assert result1.device.type == "mycelya"
        assert result2.device.type == "mycelya"


class TestRemoteDecoratorAdvanced:
    """Advanced tests for @remote decorator."""

    def test_torch_nn_functional(self, shared_machines, provider):
        """Test function using torch.nn.functional."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote
        def advanced_torch_ops(tensor: torch.Tensor) -> torch.Tensor:
            normalized = torch.nn.functional.normalize(tensor, p=2, dim=-1)
            softmaxed = torch.softmax(normalized, dim=-1)
            return softmaxed.sum(dim=0)

        x = torch.randn(8, 5, device=device)

        result = advanced_torch_ops(x)

        assert result.device.type == "mycelya"
        assert result.shape == (5,)

    def test_nested_operations(self, shared_machines, provider):
        """Test function with nested tensor operations."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote()
        def nested_computation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # Nested operations
            x = torch.matmul(a, b.t())
            y = torch.relu(x)
            z = torch.sigmoid(y)
            return z.mean()

        a = torch.randn(4, 3, device=device)
        b = torch.randn(4, 3, device=device)

        result = nested_computation(a, b)

        assert result.device.type == "mycelya"
        assert result.dim() == 0  # Scalar result

    def test_tensor_creation_in_remote(self, shared_machines, provider):
        """Test creating new tensors inside remote function."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote
        def create_and_compute(x: torch.Tensor) -> torch.Tensor:
            # Create new tensors inside the remote function
            ones = torch.ones_like(x)
            zeros = torch.zeros_like(x)
            result = (x + ones) * 2 + zeros
            return result

        x = torch.randn(3, 3, device=device)

        result = create_and_compute(x)

        assert result.device.type == "mycelya"
        assert result.shape == (3, 3)

        # Verify correctness
        result_cpu = result.cpu()
        expected = (x.cpu() + 1.0) * 2
        torch.testing.assert_close(result_cpu, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.fast
class TestRemoteDecoratorAsync:
    """Tests for @remote decorator with run_async=True."""

    def test_async_basic(self, shared_machines, provider):
        """Test basic async execution with run_async=True."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote(run_async=True)
        def async_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        a = torch.randn(3, 3, device=device)
        b = torch.randn(3, 3, device=device)

        # Should return a Future immediately
        future = async_add(a, b)

        # Check that it's a Future object
        from concurrent.futures import Future
        assert isinstance(future, Future)

        # Get the result by blocking
        result = future.result()

        assert result.device.type == "mycelya"
        assert result.shape == (3, 3)

        # Verify correctness
        result_cpu = result.cpu()
        expected = (a.cpu() + b.cpu())
        torch.testing.assert_close(result_cpu, expected, rtol=1e-4, atol=1e-6)

    def test_async_multiple_operations(self, shared_machines, provider):
        """Test async execution with multiple concurrent operations."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote(run_async=True)
        def async_operation(x: torch.Tensor, scale: float) -> torch.Tensor:
            return torch.relu(x * scale)

        x = torch.randn(5, 5, device=device)

        # Launch multiple async operations
        future1 = async_operation(x, 2.0)
        future2 = async_operation(x, 3.0)
        future3 = async_operation(x, 4.0)

        # Get results
        result1 = future1.result()
        result2 = future2.result()
        result3 = future3.result()

        # All should be valid mycelya tensors
        assert result1.device.type == "mycelya"
        assert result2.device.type == "mycelya"
        assert result3.device.type == "mycelya"

    def test_async_vs_sync_comparison(self, shared_machines, provider):
        """Test that async and sync modes produce same results."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote
        def sync_func(x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)

        @mycelya_torch.remote(run_async=True)
        def async_func(x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)

        x = torch.randn(4, 4, device=device)

        # Execute both
        sync_result = sync_func(x)
        async_future = async_func(x)
        async_result = async_future.result()

        # Compare results
        sync_cpu = sync_result.cpu()
        async_cpu = async_result.cpu()
        torch.testing.assert_close(sync_cpu, async_cpu, rtol=1e-4, atol=1e-6)

    def test_async_multiple_returns(self, shared_machines, provider):
        """Test async execution with multiple return values."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"
        device = machine.device(device_type)

        @mycelya_torch.remote(run_async=True)
        def async_split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x * 2, x * 3

        x = torch.randn(3, 3, device=device)

        future = async_split(x)
        result1, result2 = future.result()

        assert result1.device.type == "mycelya"
        assert result2.device.type == "mycelya"
        assert result1.shape == (3, 3)
        assert result2.shape == (3, 3)


class TestRemoteDecoratorErrorHandling:
    """Error handling tests for @remote decorator."""

    def test_mixed_machines_error(self, provider):
        """Test that mixing tensors from different machines raises error."""
        if provider == "mock":
            # Create two separate machines
            machine1 = mycelya_torch.RemoteMachine("mock")
            machine2 = mycelya_torch.RemoteMachine("mock")

            device1 = machine1.device("cpu")
            device2 = machine2.device("cpu")

            @mycelya_torch.remote
            def mixed_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            a = torch.randn(2, 2, device=device1)
            b = torch.randn(2, 2, device=device2)

            # Should raise error about mixed machines
            with pytest.raises(RuntimeError, match="different machines"):
                mixed_add(a, b)

    def test_no_tensor_arguments_error(self, shared_machines, provider):
        """Test that functions without tensor arguments fail appropriately."""
        machine = shared_machines["T4"]

        @mycelya_torch.remote
        def no_tensors(x: int, y: int) -> int:
            return x + y

        # Should raise error about no tensor arguments
        with pytest.raises((RuntimeError, ValueError)):
            no_tensors(1, 2)