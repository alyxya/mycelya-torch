# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Device manager for mycelya_torch.

This module provides the DeviceManager for managing remote device information
and their device indices for PyTorch integration.
"""

from typing import Dict, Tuple

import torch


class DeviceManager:
    """
    Manager for remote device information.

    Maps device indices to tuples of (machine_id, device, index) with reverse mapping.
    """

    def __init__(self) -> None:
        self._devices: Dict[
            int, Tuple[str, str, int]
        ] = {}  # index -> (machine_id, device, index)
        self._reverse_devices: Dict[
            Tuple[str, str, int], int
        ] = {}  # (machine_id, device, index) -> index
        self._next_index = 0

    def register_device(
        self, machine_id: str, device: str = "cuda", index: int = 0
    ) -> int:
        """
        Register a device and return its index.

        Args:
            machine_id: The unique machine identifier
            device: The remote machine's device type (default: "cuda")
            index: The remote machine's device index (default: 0)

        Returns:
            The assigned device index
        """
        device_tuple = (machine_id, device, index)

        # Check if device is already registered
        if device_tuple in self._reverse_devices:
            return self._reverse_devices[device_tuple]

        # Assign new index
        device_index = self._next_index
        self._next_index += 1

        # Store bidirectional mapping
        self._devices[device_index] = device_tuple
        self._reverse_devices[device_tuple] = device_index

        return device_index

    def get_device(
        self, machine_id: str, device: str = "cuda", index: int = 0
    ) -> torch.device:
        """
        Get a torch.device object for the given machine configuration.

        Args:
            machine_id: The unique machine identifier
            device: The remote machine's device type (default: "cuda")
            index: The remote machine's device index (default: 0)

        Returns:
            torch.device object with type "mycelya" and the mapped index
        """
        device_tuple = (machine_id, device, index)
        device_index = self._reverse_devices.get(device_tuple)
        if device_index is None:
            raise RuntimeError(f"Device not registered: {device_tuple}")
        return torch.device("mycelya", device_index)


# Global device manager
_device_manager = DeviceManager()


def get_device_manager() -> DeviceManager:
    """Get the global device manager."""
    return _device_manager
