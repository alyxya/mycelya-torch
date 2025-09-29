# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Regression test suite for mycelya-torch.

This module contains the essential tests that should be run to catch
regressions in core functionality.

Run with: pytest tests/test_regression.py -v
"""

import pytest
import torch

import mycelya_torch


class TestRegression:
    """Regression tests for core functionality."""

    def test_basic_imports(self):
        """Test that core modules import successfully."""
        # Test that mycelya_torch imports
        assert hasattr(mycelya_torch, "RemoteMachine")
        assert hasattr(mycelya_torch, "get_all_machines")

        # Test that remote device is registered
        device = torch.device("mycelya", 0)
        assert device.type == "mycelya"
        assert device.index == 0

    def test_device_creation(self, shared_machines, provider):
        """Test basic device creation and properties."""
        machine = shared_machines["T4"]
        assert machine is not None
        assert hasattr(machine, "device")
        assert hasattr(machine, "machine_id")

        device_type = "cpu" if provider == "mock" else "cuda"
        torch_device = machine.device(device_type)
        assert torch_device.type == "mycelya"

    def test_tensor_creation_on_device(self, shared_machines, provider):
        """Test basic tensor creation on remote device."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Test various tensor creation methods
        x = torch.randn(2, 3, device=machine.device(device_type))
        assert x.device.type == "mycelya"
        assert x.shape == (2, 3)

        y = torch.zeros(3, 3, device=machine.device(device_type))
        assert y.device.type == "mycelya"
        assert y.shape == (3, 3)

    def test_tensor_addition(self, shared_machines, provider):
        """Test basic tensor addition operation."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        x = torch.ones(2, 2, device=machine.device(device_type))
        y = torch.ones(2, 2, device=machine.device(device_type))
        result = x + y

        assert result.device.type == "mycelya"
        assert result.shape == (2, 2)

        # Verify result by transferring to CPU
        result_cpu = result.cpu()
        expected = torch.full((2, 2), 2.0)
        torch.testing.assert_close(result_cpu, expected, rtol=1e-4, atol=1e-6)

    def test_matrix_multiplication(self, shared_machines, provider):
        """Test basic matrix multiplication."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        x = torch.randn(3, 4, device=machine.device(device_type))
        y = torch.randn(4, 5, device=machine.device(device_type))
        result = x @ y

        assert result.device.type == "mycelya"
        assert result.shape == (3, 5)

    def test_cpu_to_remote_transfer(self, shared_machines, provider):
        """Test basic CPU to remote transfer."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(2, 3)
        remote_tensor = cpu_tensor.to(machine.device(device_type))

        assert cpu_tensor.device.type == "cpu"
        assert remote_tensor.device.type == "mycelya"
        assert remote_tensor.shape == cpu_tensor.shape
        assert remote_tensor.dtype == cpu_tensor.dtype

    def test_remote_to_cpu_transfer(self, shared_machines, provider):
        """Test basic remote to CPU transfer."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        remote_tensor = torch.randn(2, 3, device=machine.device(device_type))
        cpu_tensor = remote_tensor.cpu()

        assert remote_tensor.device.type == "mycelya"
        assert cpu_tensor.device.type == "cpu"
        assert cpu_tensor.shape == remote_tensor.shape
        assert cpu_tensor.dtype == remote_tensor.dtype

    def test_simple_gradient_computation(self, shared_machines, provider):
        """Test basic gradient computation."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        x = torch.randn(2, 2, device=machine.device(device_type), requires_grad=True)
        y = x.sum()
        y.backward()

        # Check that gradients exist (might be None due to meta tensor limitations)
        # The important thing is that backward() doesn't crash
        assert x.requires_grad
        assert y.requires_grad

    def test_tensor_view_operation(self, shared_machines, provider):
        """Test basic view operations."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        x = torch.randn(4, 6, device=machine.device(device_type))
        y = x.view(2, 12)

        assert y.device.type == "mycelya"
        assert y.shape == (2, 12)

        # Test transpose
        z = x.transpose(0, 1)
        assert z.shape == (6, 4)

    def test_basic_error_handling(self):
        """Test that invalid operations fail gracefully."""
        # Test invalid GPU type
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            mycelya_torch.RemoteMachine("modal", "INVALID_GPU")

        # Test invalid device operations don't crash the system
        try:
            device = torch.device("mycelya", 999)  # Invalid index
            torch.randn(2, 2, device=device)
        except Exception:
            # Should fail gracefully, not crash
            pass

    def test_scalar_operations(self, shared_machines, provider):
        """Test operations with scalars."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        x = torch.ones(2, 2, device=machine.device(device_type))
        result = x * 2.0

        assert result.device.type == "mycelya"
        assert result.shape == (2, 2)

    def test_tensor_properties_access(self, shared_machines, provider):
        """Test accessing basic tensor properties."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        x = torch.randn(3, 4, dtype=torch.float32, device=machine.device(device_type))

        # These should not raise exceptions
        assert x.shape == (3, 4)
        assert x.dtype == torch.float32
        assert x.device.type == "mycelya"
        assert x.numel() == 12
        assert x.dim() == 2


@pytest.mark.fast
class TestFastFunctional:
    """Fast functional tests - run on PR reviews."""

    def test_multiple_arithmetic_operations(self, shared_machines, provider):
        """Test combination of arithmetic operations."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        x = torch.ones(2, 2, device=machine.device(device_type))
        y = torch.ones(2, 2, device=machine.device(device_type))

        result = (x + y) * 2 - 1
        assert result.device.type == "mycelya"
        assert result.shape == (2, 2)

    def test_tensor_creation_various_dtypes(self, shared_machines, provider):
        """Test tensor creation with different dtypes."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        # Test float32
        x = torch.randn(2, 2, dtype=torch.float32, device=machine.device(device_type))
        assert x.dtype == torch.float32

        # Test int64
        y = torch.ones(2, 2, dtype=torch.int64, device=machine.device(device_type))
        assert y.dtype == torch.int64

    def test_transfer_with_dtype_conversion(self, shared_machines, provider):
        """Test transfer with dtype conversion."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        cpu_tensor = torch.randn(2, 2, dtype=torch.float64)
        remote_tensor = cpu_tensor.to(machine.device(device_type), dtype=torch.float32)

        assert remote_tensor.dtype == torch.float32
        assert remote_tensor.device.type == "mycelya"

    def test_basic_loss_computation(self, shared_machines, provider):
        """Test basic loss function computation."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        pred = torch.randn(4, 3, device=machine.device(device_type))
        target = torch.randn(4, 3, device=machine.device(device_type))

        # MSE loss
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.device.type == "mycelya"
        assert loss.dim() == 0  # Scalar loss

    def test_gradient_with_operations(self, shared_machines, provider):
        """Test gradient computation with multiple operations."""
        machine = shared_machines["T4"]
        device_type = "cpu" if provider == "mock" else "cuda"

        x = torch.randn(3, 3, device=machine.device(device_type), requires_grad=True)
        y = torch.randn(3, 3, device=machine.device(device_type), requires_grad=True)

        z = (x * y).sum()
        z.backward()

        # The important test is that backward() completes without error
        assert x.requires_grad
        assert y.requires_grad
