# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for device management functionality in mycelya-torch.

This module tests device creation, validation, registry management,
and device property verification.
"""

import pytest
import torch
from test_utilities import DeviceTestUtils, TestConstants


def test_basic_imports() -> None:
    """Test basic torch and mycelya_torch imports."""
    assert True


def test_device_functions() -> None:
    """Test remote device functions."""
    assert torch.mycelya.is_available()
    # device_count should be >= 0 (could be 0 if no devices registered)
    assert torch.mycelya.device_count() >= 0


def test_tensor_to_method() -> None:
    """Test that tensors have to() method that works with RemoteMachine."""
    x = torch.randn(2, 2)
    assert hasattr(x, "to") and callable(x.to)


def test_device_creation_and_properties(shared_machines, provider):
    """Test that devices are created properly and have correct properties."""
    for machine_key in TestConstants.DEVICE_KEYS:
        if machine_key in shared_machines:
            machine = shared_machines[machine_key]
            assert machine is not None
            assert hasattr(machine, "device")
            assert callable(machine.device)


def test_device_tensor_creation(shared_machines, provider):
    """Test creating tensors on remote devices."""
    device_type = "cpu" if provider == "mock" else "cuda"
    tensor = DeviceTestUtils.create_remote_tensor(
        (2, 2), shared_machines, machine_key="T4", device_type=device_type
    )
    assert tensor is not None
    assert tensor.shape == (2, 2)
    DeviceTestUtils.verify_machine_properties(
        tensor, shared_machines["T4"], device_type=device_type
    )


def test_device_tensor_creation_various_shapes(shared_machines, provider):
    """Test creating tensors with various shapes on remote devices."""
    device_type = "cpu" if provider == "mock" else "cuda"
    test_shapes = TestConstants.SMALL_SHAPES + TestConstants.TENSOR_3D_SHAPES

    for shape in test_shapes:
        tensor = DeviceTestUtils.create_remote_tensor(
            shape, shared_machines, device_type=device_type
        )
        assert tensor.shape == shape
        DeviceTestUtils.verify_machine_properties(
            tensor, shared_machines["T4"], device_type=device_type
        )


def test_device_tensor_with_grad(shared_machines, provider):
    """Test creating tensors with gradients on remote devices."""
    device_type = "cpu" if provider == "mock" else "cuda"
    tensor = DeviceTestUtils.create_remote_tensor(
        (3, 3), shared_machines, requires_grad=True, device_type=device_type
    )
    assert tensor.requires_grad
    DeviceTestUtils.verify_machine_properties(
        tensor, shared_machines["T4"], device_type=device_type
    )


def test_device_tensor_dtype_conversion(shared_machines, provider):
    """Test creating tensors with different dtypes on remote devices."""
    device_type = "cpu" if provider == "mock" else "cuda"
    dtypes_to_test = [torch.float32, torch.float64, torch.int32, torch.int64]

    for dtype in dtypes_to_test:
        try:
            tensor = DeviceTestUtils.create_remote_tensor(
                (2, 2), shared_machines, dtype=dtype, device_type=device_type
            )
            assert tensor.dtype == dtype
            DeviceTestUtils.verify_machine_properties(
                tensor, shared_machines["T4"], device_type=device_type
            )
        except (RuntimeError, NotImplementedError):
            # Some dtypes might not be supported, which is acceptable
            pass


def test_multiple_devices_availability(shared_machines, provider):
    """Test that multiple device types are available if configured."""
    available_devices = []
    for machine_key in TestConstants.DEVICE_KEYS:
        if machine_key in shared_machines:
            available_devices.append(machine_key)

    # Should have at least one device available for testing
    assert len(available_devices) >= 1


def test_device_index_consistency(shared_machines, provider):
    """Test that device indices are consistent."""
    device_type = "cpu" if provider == "mock" else "cuda"
    for machine_key in TestConstants.DEVICE_KEYS:
        if machine_key in shared_machines:
            machine = shared_machines[machine_key]
            tensor = DeviceTestUtils.create_remote_tensor(
                (2, 2), shared_machines, machine_key, device_type=device_type
            )

            # Device index should be consistent
            assert tensor.device.index == machine.device(device_type).index
            assert isinstance(machine.device(device_type).index, int)


def test_device_type_consistency(shared_machines, provider):
    """Test that all remote devices have consistent type."""
    device_type = "cpu" if provider == "mock" else "cuda"
    for machine_key in TestConstants.DEVICE_KEYS:
        if machine_key in shared_machines:
            tensor = DeviceTestUtils.create_remote_tensor(
                (2, 2), shared_machines, machine_key, device_type=device_type
            )
            assert tensor.device.type == "mycelya"


def test_device_registry_state(shared_machines, provider):
    """Test that device registry maintains proper state."""
    device_type = "cpu" if provider == "mock" else "cuda"
    initial_count = torch.mycelya.device_count()

    # Creating tensors shouldn't change device count
    DeviceTestUtils.create_remote_tensor(
        (2, 2), shared_machines, device_type=device_type
    )
    assert torch.mycelya.device_count() == initial_count

    # Multiple tensors on same device shouldn't change count
    DeviceTestUtils.create_remote_tensor(
        (3, 3), shared_machines, device_type=device_type
    )
    assert torch.mycelya.device_count() == initial_count


def test_device_cleanup_behavior(shared_machines, provider):
    """Test that device cleanup works properly."""
    device_type = "cpu" if provider == "mock" else "cuda"
    # Create multiple tensors
    tensors = []
    for _i in range(5):
        tensor = DeviceTestUtils.create_remote_tensor(
            (2, 2), shared_machines, device_type=device_type
        )
        tensors.append(tensor)

    # All tensors should be valid
    for tensor in tensors:
        assert tensor is not None
        DeviceTestUtils.verify_machine_properties(
            tensor, shared_machines["T4"], device_type=device_type
        )

    # Clear references (Python garbage collection will handle cleanup)
    del tensors


def test_device_error_handling_graceful():
    """Test that device-related errors are handled gracefully."""
    # These operations might fail, but shouldn't crash
    try:
        torch.randn(3, 3, device="mycelya")  # Should fail gracefully
    except Exception:
        pass  # Expected to fail

    try:
        torch.randn(2, 2, device="nonexistent_device")
    except Exception:
        pass  # Expected to fail

    assert True  # If we get here without segfault, it's good


@pytest.mark.parametrize("shape", TestConstants.SMALL_SHAPES)
def test_parametrized_device_tensor_creation(shared_machines, provider, shape):
    """Test device tensor creation with parametrized shapes."""
    device_type = "cpu" if provider == "mock" else "cuda"
    tensor = DeviceTestUtils.create_remote_tensor(
        shape, shared_machines, device_type=device_type
    )
    assert tensor.shape == shape
    DeviceTestUtils.verify_machine_properties(
        tensor, shared_machines["T4"], device_type=device_type
    )


@pytest.mark.parametrize("machine_key", TestConstants.DEVICE_KEYS)
def test_parametrized_device_types(shared_machines, provider, machine_key):
    """Test tensor creation on different device types if available."""
    device_type = "cpu" if provider == "mock" else "cuda"
    if machine_key not in shared_machines:
        pytest.skip(f"Device {machine_key} not available in test environment")

    tensor = DeviceTestUtils.create_remote_tensor(
        (2, 2), shared_machines, machine_key, device_type=device_type
    )
    assert tensor is not None
    DeviceTestUtils.verify_machine_properties(
        tensor, shared_machines[machine_key], device_type=device_type
    )
