#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for torch-remote tests.

This module provides shared device fixtures to avoid recreating devices
for each test, improving test efficiency and reducing GPU resource usage.
"""

import pytest
import torch
import torch_remote


@pytest.fixture(scope="session")
def modal_t4_device():
    """
    Session-scoped fixture providing a shared Modal T4 device.
    
    The device is created once per test session and reused across all tests.
    This significantly reduces GPU resource usage and test execution time.
    """
    device = torch_remote.create_modal_device("T4")
    yield device
    # Cleanup if needed (device cleanup is typically handled by torch_remote)


@pytest.fixture(scope="session") 
def shared_devices():
    """
    Session-scoped fixture providing a dictionary of shared devices.
    
    Returns a dict with common device configurations that tests can use.
    Add more device types here as needed.
    """
    devices = {
        "t4": torch_remote.create_modal_device("T4"),
        # Add more device types as needed:
        # "a100": torch_remote.create_modal_device("A100"),
        # "cpu": torch.device("cpu"),  # if CPU device testing is needed
    }
    yield devices
    # Cleanup if needed


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
def device_tensors(modal_t4_device, sample_tensors):
    """
    Function-scoped fixture providing tensors already on the remote device.
    
    This creates remote versions of sample tensors to avoid repeated
    CPU-to-device transfers in tests.
    """
    remote_tensors = {}
    for name, cpu_tensor in sample_tensors.items():
        try:
            remote_tensors[name] = cpu_tensor.to(modal_t4_device.device())
        except Exception as e:
            # Some tensor types might not be supported, skip them
            print(f"Warning: Could not create remote tensor for {name}: {e}")
    
    return remote_tensors


# Test configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU resources"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on patterns."""
    for item in items:
        # Mark tests with large tensors as slow
        if "large" in item.name.lower() or "memory" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark all torch_remote tests as requiring GPU
        if "torch_remote" in str(item.fspath) or "backend" in item.name.lower():
            item.add_marker(pytest.mark.gpu)