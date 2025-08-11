# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Device registry management for mycelya_torch.

This module provides the DeviceRegistry for managing RemoteMachine instances
and their device indices for PyTorch integration.
"""

from typing import Dict, Optional

from ._machine import RemoteMachine

# Re-export machine types for backward compatibility
from ._machine import CloudProvider, GPUType, create_mock_machine, create_modal_machine


class DeviceRegistry:
    """
    Registry to manage active RemoteMachine instances.

    Maps device indices directly to RemoteMachine instances for simple lookups.
    """

    def __init__(self) -> None:
        self._devices: Dict[int, RemoteMachine] = {}  # index -> RemoteMachine
        self._next_index = 0

    def register_device(self, machine: RemoteMachine) -> int:
        """
        Register a device and return its index.

        Args:
            machine: The RemoteMachine to register

        Returns:
            The assigned device index
        """
        # Check if device is already registered
        for index, existing_device in self._devices.items():
            if existing_device is machine:
                return index

        # Assign new index
        index = self._next_index
        self._next_index += 1

        # Store direct mapping
        self._devices[index] = machine

        return index

    def get_device_by_index(self, index: int) -> Optional[RemoteMachine]:
        """Get device by its index."""
        return self._devices.get(index)

    def get_device_index(self, machine: RemoteMachine) -> Optional[int]:
        """Get the index of a machine."""
        for index, existing_machine in self._devices.items():
            if existing_machine is machine:
                return index
        return None

    def get_all_machines(self) -> list[RemoteMachine]:
        """Get a list of all registered machines."""
        return list(self._devices.values())


# Global device registry
_device_registry = DeviceRegistry()

# Device cleanup is handled via atexit registration in RemoteMachine.__init__
# Modal handles its own async context cleanup, but we still register explicit cleanup
# for proper resource management in standalone usage scenarios.


def get_device_registry() -> DeviceRegistry:
    """Get the global device registry."""
    return _device_registry


def get_all_machines() -> list[RemoteMachine]:
    """
    Get a list of all created machines.

    Returns:
        List of all RemoteMachine instances that have been created

    Example:
        >>> machine1 = create_modal_machine("T4")
        >>> machine2 = create_modal_machine("A100-40GB")
        >>> machines = get_all_machines()
        >>> print(f"Created {len(machines)} machines")
        Created 2 machines
    """
    return _device_registry.get_all_machines()