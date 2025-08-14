# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test configuration and fixtures for mycelya-torch tests.
"""

import pytest

import mycelya_torch


@pytest.fixture(scope="session")
def shared_devices():
    """Create shared devices for testing - using mock provider for reliability."""
    devices = {}

    # Create mock devices that work reliably in test environment
    devices["t4"] = mycelya_torch.RemoteMachine("mock", "T4")
    devices["l4"] = mycelya_torch.RemoteMachine("mock", "L4")

    # Start the devices
    for device in devices.values():
        device.start()

    yield devices

    # Clean up devices
    for device in devices.values():
        try:
            device.stop()
        except Exception:
            pass  # Ignore cleanup errors
