# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test configuration and fixtures for mycelya-torch tests.

Provider Selection via Parameterization:
- Run with modal (default): pytest tests/
- Run with mock: pytest tests/ -k "mock"
- Run with both: pytest tests/ (tests are parameterized to run with both providers)
- Run only modal: pytest tests/ -k "modal"
"""

import logging

import pytest

import mycelya_torch

log = logging.getLogger(__name__)


@pytest.fixture(scope="session", params=["modal", "mock"])
def provider(request):
    """
    Parameterized fixture that runs tests with both modal and mock providers.

    Tests will automatically run twice - once with each provider.
    Use -k "mock" or -k "modal" to run with specific provider only.
    """
    provider_name = request.param
    log.info(f"Running tests with provider: {provider_name}")
    return provider_name


@pytest.fixture(scope="session")
def shared_devices(provider):
    """
    Session-scoped fixture providing a dictionary of shared devices.

    Automatically parameterized to test with both modal and mock providers.
    Returns a dict with common device configurations that tests can use.
    """
    machines = {}

    # Create machines with the parameterized provider
    if provider == "mock":
        machines["t4"] = mycelya_torch.RemoteMachine(provider)
        machines["l4"] = mycelya_torch.RemoteMachine(provider)
    else:
        machines["t4"] = mycelya_torch.RemoteMachine(provider, "T4")
        machines["l4"] = mycelya_torch.RemoteMachine(provider, "L4")

    # Start the machines
    for machine in machines.values():
        try:
            machine.start()
            log.info(f"Started machine: {machine.machine_id} (provider: {provider})")
        except Exception as e:
            log.warning(f"Failed to start machine {machine.machine_id}: {e}")

    yield machines

    # Clean up machines
    for machine in machines.values():
        try:
            machine.stop()
            log.info(f"Stopped machine: {machine.machine_id} (provider: {provider})")
        except Exception as e:
            log.warning(f"Failed to stop machine {machine.machine_id}: {e}")


# Test configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "critical: marks tests as critical for regression testing"
    )
    config.addinivalue_line("markers", "fast: marks tests as fast for PR reviews")
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

        # Mark all mycelya_torch tests as requiring GPU
        if "mycelya_torch" in str(item.fspath) or "backend" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
