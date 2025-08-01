# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for tensor transfer operations in mycelya-torch.

This module tests CPU<->remote transfers, cross-device restrictions,
device conversions, and transfer error handling.
"""

import pytest
import torch
from test_utilities import (
    DeviceTestUtils,
    ErrorTestUtils,
    NumericalTestUtils,
    TestConstants,
)


class TestCPUToRemoteTransfers:
    """Tests for transferring tensors from CPU to remote devices."""

    def test_basic_cpu_to_remote_transfer(self, shared_devices):
        """Test basic CPU to remote device transfer."""
        cpu_tensor = torch.randn(2, 2)
        remote_tensor = cpu_tensor.to(shared_devices["t4"].device())

        assert remote_tensor is not None
        assert remote_tensor.shape == cpu_tensor.shape
        assert remote_tensor.dtype == cpu_tensor.dtype
        DeviceTestUtils.verify_device_properties(remote_tensor, shared_devices["t4"])

    def test_cpu_to_remote_various_shapes(self, shared_devices):
        """Test CPU to remote transfer with various tensor shapes."""
        test_shapes = (
            TestConstants.SMALL_SHAPES
            + TestConstants.TENSOR_3D_SHAPES
            + TestConstants.TENSOR_4D_SHAPES
        )

        for shape in test_shapes:
            cpu_tensor = torch.randn(shape)
            remote_tensor = cpu_tensor.to(shared_devices["t4"].device())

            assert remote_tensor.shape == shape
            DeviceTestUtils.verify_device_properties(
                remote_tensor, shared_devices["t4"]
            )

    def test_cpu_to_remote_with_dtype_conversion(self, shared_devices):
        """Test CPU to remote transfer with dtype conversion."""
        cpu_tensor = torch.randn(3, 3, dtype=torch.float32)

        # Transfer with dtype conversion
        remote_tensor = cpu_tensor.to(
            shared_devices["t4"].device(), dtype=torch.float64
        )

        assert remote_tensor.dtype == torch.float64
        assert remote_tensor.shape == cpu_tensor.shape
        DeviceTestUtils.verify_device_properties(remote_tensor, shared_devices["t4"])

    def test_cpu_to_remote_with_requires_grad(self, shared_devices):
        """Test CPU to remote transfer preserving requires_grad."""
        cpu_tensor = torch.randn(2, 2, requires_grad=True)
        remote_tensor = cpu_tensor.to(shared_devices["t4"].device())

        assert remote_tensor.requires_grad == cpu_tensor.requires_grad
        DeviceTestUtils.verify_device_properties(remote_tensor, shared_devices["t4"])

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
    )
    def test_cpu_to_remote_various_dtypes(self, shared_devices, dtype):
        """Test CPU to remote transfer with various data types."""
        try:
            cpu_tensor = torch.randn(2, 2).to(dtype)
            remote_tensor = cpu_tensor.to(shared_devices["t4"].device())

            assert remote_tensor.dtype == dtype
            DeviceTestUtils.verify_device_properties(
                remote_tensor, shared_devices["t4"]
            )
        except (RuntimeError, NotImplementedError):
            pytest.skip(f"dtype {dtype} not supported on remote device")


class TestRemoteToCPUTransfers:
    """Tests for transferring tensors from remote devices to CPU."""

    def test_basic_remote_to_cpu_transfer(self, shared_devices):
        """Test basic remote to CPU transfer."""
        original_cpu = torch.randn(2, 2)
        remote_tensor = original_cpu.to(shared_devices["t4"].device())
        back_to_cpu = remote_tensor.cpu()

        assert back_to_cpu.device.type == "cpu"
        assert back_to_cpu.shape == original_cpu.shape
        assert back_to_cpu.dtype == original_cpu.dtype

        # Verify numerical consistency
        NumericalTestUtils.assert_tensors_close(back_to_cpu, original_cpu)

    def test_remote_to_cpu_various_shapes(self, shared_devices):
        """Test remote to CPU transfer with various shapes."""
        test_shapes = TestConstants.SMALL_SHAPES + TestConstants.TENSOR_3D_SHAPES

        for shape in test_shapes:
            original_cpu = torch.randn(shape)
            remote_tensor = original_cpu.to(shared_devices["t4"].device())
            back_to_cpu = remote_tensor.cpu()

            assert back_to_cpu.device.type == "cpu"
            assert back_to_cpu.shape == shape
            NumericalTestUtils.assert_tensors_close(back_to_cpu, original_cpu)

    def test_remote_to_cpu_preserves_gradients(self, shared_devices):
        """Test that remote to CPU transfer preserves gradient information."""
        original_cpu = torch.randn(2, 2, requires_grad=True)
        remote_tensor = original_cpu.to(shared_devices["t4"].device())

        # Retain gradients for non-leaf tensor
        remote_tensor.retain_grad()

        # Perform operation and backward pass
        loss = remote_tensor.sum()
        loss.backward()

        # Transfer back to CPU
        back_to_cpu = remote_tensor.cpu()
        grad_back_to_cpu = remote_tensor.grad.cpu()

        assert back_to_cpu.device.type == "cpu"
        assert grad_back_to_cpu.device.type == "cpu"
        assert back_to_cpu.requires_grad == original_cpu.requires_grad

    def test_remote_to_cpu_after_operations(self, shared_devices):
        """Test remote to CPU transfer after performing operations."""
        x_cpu = torch.randn(2, 2)
        y_cpu = torch.randn(2, 2)

        x_remote = x_cpu.to(shared_devices["t4"].device())
        y_remote = y_cpu.to(shared_devices["t4"].device())

        # Perform operations on remote
        result_remote = x_remote + y_remote
        result_cpu_reference = x_cpu + y_cpu

        # Transfer result back to CPU
        result_back_to_cpu = result_remote.cpu()

        assert result_back_to_cpu.device.type == "cpu"
        NumericalTestUtils.assert_tensors_close(
            result_back_to_cpu, result_cpu_reference
        )


class TestCrossDeviceTransferRestrictions:
    """Tests for cross-device transfer restrictions and error handling."""

    def test_cross_device_operation_restriction(self, shared_devices):
        """Test that operations between different remote devices are restricted."""
        # Skip if we don't have multiple devices
        available_devices = [
            k for k in TestConstants.DEVICE_KEYS if k in shared_devices
        ]
        if len(available_devices) < 2:
            pytest.skip("Need at least 2 devices for cross-device testing")

        device1_key, device2_key = available_devices[0], available_devices[1]

        tensor1 = DeviceTestUtils.create_remote_tensor(
            (2, 2), shared_devices, device1_key
        )
        tensor2 = DeviceTestUtils.create_remote_tensor(
            (2, 2), shared_devices, device2_key
        )

        # Test various operations that should fail
        ErrorTestUtils.assert_cross_device_fails(tensor1, tensor2, lambda x, y: x + y)
        ErrorTestUtils.assert_cross_device_fails(tensor1, tensor2, lambda x, y: x.mm(y))

    def test_cross_device_transfer_direct(self, shared_devices):
        """Test direct transfer between different remote devices."""
        available_devices = [
            k for k in TestConstants.DEVICE_KEYS if k in shared_devices
        ]
        if len(available_devices) < 2:
            pytest.skip("Need at least 2 devices for cross-device testing")

        device1_key, device2_key = available_devices[0], available_devices[1]

        tensor_device1 = DeviceTestUtils.create_remote_tensor(
            (2, 2), shared_devices, device1_key
        )

        # Try to transfer directly to another remote device (should fail)
        with pytest.raises(
            RuntimeError,
            match="Cross-device remote transfers are not supported",
        ):
            tensor_device1.to(shared_devices[device2_key].device())

    def test_cross_device_via_cpu_transfer(self, shared_devices):
        """Test transfer between remote devices via CPU."""
        available_devices = [
            k for k in TestConstants.DEVICE_KEYS if k in shared_devices
        ]
        if len(available_devices) < 2:
            pytest.skip("Need at least 2 devices for cross-device testing")

        device1_key, device2_key = available_devices[0], available_devices[1]

        original_data = torch.randn(2, 2)
        tensor_device1 = original_data.to(shared_devices[device1_key].device())

        # Transfer via CPU
        cpu_intermediate = tensor_device1.cpu()
        tensor_device2 = cpu_intermediate.to(shared_devices[device2_key].device())

        # Verify successful transfer
        DeviceTestUtils.verify_device_properties(
            tensor_device2, shared_devices[device2_key]
        )
        NumericalTestUtils.assert_tensors_close(tensor_device2.cpu(), original_data)


class TestTransferWithConversions:
    """Tests for transfers with various conversions."""

    def test_transfer_with_copy_parameter(self, shared_devices):
        """Test transfer behavior with copy parameter."""
        cpu_tensor = torch.randn(2, 2)

        # Transfer with copy=True
        remote_tensor_copy = cpu_tensor.to(shared_devices["t4"].device(), copy=True)
        assert remote_tensor_copy is not None
        DeviceTestUtils.verify_device_properties(
            remote_tensor_copy, shared_devices["t4"]
        )

        # Transfer with copy=False
        remote_tensor_no_copy = cpu_tensor.to(shared_devices["t4"].device(), copy=False)
        assert remote_tensor_no_copy is not None
        DeviceTestUtils.verify_device_properties(
            remote_tensor_no_copy, shared_devices["t4"]
        )

    def test_transfer_with_non_blocking(self, shared_devices):
        """Test transfer behavior with non_blocking parameter."""
        cpu_tensor = torch.randn(2, 2)

        try:
            # Transfer with non_blocking=True
            remote_tensor = cpu_tensor.to(
                shared_devices["t4"].device(), non_blocking=True
            )
            assert remote_tensor is not None
            DeviceTestUtils.verify_device_properties(
                remote_tensor, shared_devices["t4"]
            )
        except (RuntimeError, NotImplementedError):
            pytest.skip("non_blocking parameter not supported")

    def test_transfer_dtype_and_device_combined(self, shared_devices):
        """Test transfer with both device and dtype conversion."""
        cpu_tensor = torch.randn(2, 2, dtype=torch.float32)

        remote_tensor = cpu_tensor.to(
            device=shared_devices["t4"].device(), dtype=torch.float64
        )

        assert remote_tensor.dtype == torch.float64
        DeviceTestUtils.verify_device_properties(remote_tensor, shared_devices["t4"])
        assert remote_tensor.shape == cpu_tensor.shape


class TestTransferErrorHandling:
    """Tests for error handling in transfer operations."""

    def test_invalid_device_transfer(self, shared_devices):
        """Test transfer to invalid device handles errors gracefully."""
        cpu_tensor = torch.randn(2, 2)

        # Try to transfer to non-existent device
        with pytest.raises((RuntimeError, ValueError)):
            cpu_tensor.to("nonexistent_device")

    def test_transfer_large_tensors(self, shared_devices):
        """Test transfer of large tensors."""
        try:
            # Create a moderately large tensor
            large_tensor = torch.randn(100, 100, 10)
            remote_large = large_tensor.to(shared_devices["t4"].device())

            assert remote_large.shape == large_tensor.shape
            DeviceTestUtils.verify_device_properties(remote_large, shared_devices["t4"])

            # Transfer back and verify
            back_to_cpu = remote_large.cpu()
            NumericalTestUtils.assert_tensors_close(back_to_cpu, large_tensor)
        except (RuntimeError, MemoryError):
            pytest.skip("Large tensor transfer not supported or insufficient memory")

    def test_transfer_empty_tensors(self, shared_devices):
        """Test transfer of empty tensors."""
        empty_tensor = torch.empty(0, 2)
        remote_empty = empty_tensor.to(shared_devices["t4"].device())

        assert remote_empty.shape == (0, 2)
        DeviceTestUtils.verify_device_properties(remote_empty, shared_devices["t4"])

        # Transfer back
        back_to_cpu = remote_empty.cpu()
        assert back_to_cpu.shape == (0, 2)
        assert back_to_cpu.device.type == "cpu"


class TestTransferMemoryEfficiency:
    """Tests for memory efficiency of transfer operations."""

    def test_repeated_transfers_memory_cleanup(self, shared_devices):
        """Test that repeated transfers don't leak memory."""
        base_tensor = torch.randn(10, 10)

        # Perform multiple transfer cycles
        for _i in range(10):
            remote_tensor = base_tensor.to(shared_devices["t4"].device())
            back_to_cpu = remote_tensor.cpu()

            # Verify consistency
            NumericalTestUtils.assert_tensors_close(back_to_cpu, base_tensor)

            # Clear references
            del remote_tensor, back_to_cpu

    def test_transfer_with_gradient_memory(self, shared_devices):
        """Test memory behavior of transfers with gradients."""
        base_tensor = torch.randn(5, 5, requires_grad=True)

        # Transfer and perform operations
        remote_tensor = base_tensor.to(shared_devices["t4"].device())

        # Retain gradients for non-leaf tensor
        remote_tensor.retain_grad()

        result = remote_tensor.sum()
        result.backward()

        # Transfer gradients back
        grad_cpu = remote_tensor.grad.cpu()
        tensor_cpu = remote_tensor.cpu()

        assert grad_cpu.device.type == "cpu"
        assert tensor_cpu.device.type == "cpu"
        assert grad_cpu.shape == base_tensor.shape

        # Clean up
        del remote_tensor, result, grad_cpu, tensor_cpu


@pytest.mark.parametrize(
    "transfer_chain",
    [
        ["cpu", "t4", "cpu"],
        ["cpu", "t4", "cpu", "t4", "cpu"],
    ],
)
def test_parametrized_transfer_chains(shared_devices, transfer_chain):
    """Test various transfer chains between devices."""
    original_tensor = torch.randn(3, 3)
    current_tensor = original_tensor.clone()

    for target_device in transfer_chain[1:]:  # Skip first 'cpu' entry
        if target_device == "cpu":
            current_tensor = current_tensor.cpu()
            assert current_tensor.device.type == "cpu"
        else:
            if target_device in shared_devices:
                current_tensor = current_tensor.to(
                    shared_devices[target_device].device()
                )
                DeviceTestUtils.verify_device_properties(
                    current_tensor, shared_devices[target_device]
                )
            else:
                pytest.skip(f"Device {target_device} not available")

    # Final tensor should match original numerically
    final_cpu = (
        current_tensor.cpu() if current_tensor.device.type != "cpu" else current_tensor
    )
    NumericalTestUtils.assert_tensors_close(final_cpu, original_tensor)


@pytest.mark.parametrize("device_key", TestConstants.DEVICE_KEYS)
def test_parametrized_device_transfers(shared_devices, device_key):
    """Test transfers to different device types."""
    if device_key not in shared_devices:
        pytest.skip(f"Device {device_key} not available in test environment")

    cpu_tensor = torch.randn(2, 2)
    remote_tensor = cpu_tensor.to(shared_devices[device_key].device())

    DeviceTestUtils.verify_device_properties(remote_tensor, shared_devices[device_key])

    # Transfer back and verify
    back_to_cpu = remote_tensor.cpu()
    NumericalTestUtils.assert_tensors_close(back_to_cpu, cpu_tensor)
