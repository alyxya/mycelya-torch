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
