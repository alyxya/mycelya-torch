# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for torch-remote tests.

This module provides shared device fixtures to avoid recreating devices
for each test, improving test efficiency and reducing GPU resource usage.
"""

import logging

import pytest
import torch

import torch_remote

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def shared_devices():
    """
    Session-scoped fixture providing a dictionary of shared devices.

    Returns a dict with common device configurations that tests can use.
    Add more device types here as needed.
    """
    devices = {
        "t4": torch_remote.create_modal_machine("T4"),
        "l4": torch_remote.create_modal_machine("L4"),
        "a100": torch_remote.create_modal_machine("A100"),
    }
    yield devices
    # Cleanup if needed


@pytest.fixture
def clean_registry():
    """
    Function-scoped fixture that provides a clean registry for isolated testing.

    Preserves any session-scoped devices that may have been created by other fixtures
    to avoid breaking tests that depend on shared_devices or modal_t4_device.

    Use this for tests that need a clean registry state without creating real devices.
    """
    registry = torch_remote.get_device_registry()

    # Save existing devices from session-scoped fixtures before clearing
    existing_devices = list(registry._devices.values())

    # Clear for clean state
    registry.clear()

    # Re-register any session devices that were previously registered
    # This preserves session fixture devices while giving a clean slate for the test
    for device in existing_devices:
        # Only re-register if this looks like a session device (has a running client)
        try:
            if (
                hasattr(device, "get_client")
                and device.get_client()
                and device.get_client().is_running()
            ):
                registry.register_device(device)
        except Exception:
            # If there's any issue checking device state, skip re-registration
            pass

    yield registry

    # Don't clear after test to preserve session devices for other tests
    # The pytest session cleanup will handle session device cleanup


@pytest.fixture
def sample_tensors():
    """
    Function-scoped fixture providing common test tensors.

    Returns a dict of various tensor shapes and types commonly used in tests.
    """
    return {
        "small_2d": torch.randn(2, 2),
        "medium_2d": torch.randn(3, 3),
        "large_2d": torch.randn(500, 500),
        "small_3d": torch.randn(2, 2, 2),
        "vector": torch.randn(100),
        "scalar": torch.tensor(5.0),
        "int_tensor": torch.ones(2, 2, dtype=torch.int32),
        "float64_tensor": torch.randn(2, 2, dtype=torch.float64),
    }


@pytest.fixture
def device_tensors(shared_devices, sample_tensors):
    """
    Function-scoped fixture providing tensors already on the remote device.

    This provides pre-configured remote tensors for test convenience.
    Note: Function-scoped, so tensors are recreated for each test function.
    """
    remote_tensors = {}
    t4_device = shared_devices["t4"]
    for name, cpu_tensor in sample_tensors.items():
        try:
            remote_tensors[name] = cpu_tensor.to(t4_device.device())
        except Exception as e:
            # Some tensor types might not be supported, skip them
            log.warning(f"Could not create remote tensor for {name}: {e}")

    return remote_tensors


# Test configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU resources")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on patterns."""
    for item in items:
        # Mark tests with large tensors as slow
        if "large" in item.name.lower() or "memory" in item.name.lower():
            item.add_marker(pytest.mark.slow)

        # Mark all torch_remote tests as requiring GPU
        if "torch_remote" in str(item.fspath) or "backend" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
