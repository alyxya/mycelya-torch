# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Device manager for mycelya_torch.

This module provides the DeviceManager for managing remote device information
and their device indices for PyTorch integration.
"""

# No typing imports needed

import torch


class DeviceManager:
    """
    Manager for remote device information.

    Maps local device indices to remote device info with bidirectional lookup.
    """

    def __init__(self) -> None:
        self._local_to_remote_device: dict[
            int, tuple[str, str, int]
        ] = {}  # local_index -> (machine_id, remote_type, remote_index)
        self._remote_to_local_device: dict[
            tuple[str, str, int], int
        ] = {}  # (machine_id, remote_type, remote_index) -> local_index
        self._next_index = 0

    def get_mycelya_device(
        self, machine_id: str, type: str, index: int
    ) -> torch.device:
        """
        Get a torch.device object for the given machine configuration.

        Creates the mapping if it doesn't exist, otherwise returns the existing one.

        Args:
            machine_id: The unique machine identifier
            type: The remote machine's device type (e.g., "cuda")
            index: The remote machine's device index

        Returns:
            torch.device object with type "mycelya" and the mapped index
        """
        device_tuple = (machine_id, type, index)

        # Check if device mapping already exists
        local_index = self._remote_to_local_device.get(device_tuple)
        if local_index is not None:
            return torch.device("mycelya", local_index)

        # Create new mapping
        local_index = self._next_index
        self._next_index += 1

        # Store bidirectional mapping
        self._local_to_remote_device[local_index] = device_tuple
        self._remote_to_local_device[device_tuple] = local_index

        return torch.device("mycelya", local_index)

    def get_remote_device_info(self, device_index: int) -> tuple[str, str, int]:
        """Get remote device info for a given mycelya device index.

        Args:
            device_index: Local mycelya device index

        Returns:
            Tuple of (machine_id, remote_type, remote_index)
        """
        return self._local_to_remote_device[device_index]


# Global device manager
device_manager = DeviceManager()
