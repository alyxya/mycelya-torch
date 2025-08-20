# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Device manager for mycelya_torch.

This module provides the DeviceManager for managing remote device information
and their device indices for PyTorch integration.
"""

from typing import Dict, Optional, Tuple

import torch


class DeviceManager:
    """
    Manager for remote device information.

    Maps local device indices to remote device info with bidirectional lookup.
    """

    def __init__(self) -> None:
        self._local_to_remote_device: Dict[
            int, Tuple[str, str, int]
        ] = {}  # local_index -> (machine_id, remote_type, remote_index)
        self._remote_to_local_device: Dict[
            Tuple[str, str, int], int
        ] = {}  # (machine_id, remote_type, remote_index) -> local_index
        self._next_index = 0

    def get_device(self, machine_id: str, type: str, index: int) -> torch.device:
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

    def get_machine_id_for_device_index(self, device_index: int) -> Optional[str]:
        """Get machine_id for a given device index."""
        device_info = self._local_to_remote_device.get(device_index)
        if device_info is None:
            return None
        return device_info[0]  # machine_id is the first element


# Global device manager
_device_manager = DeviceManager()


def get_device_manager() -> DeviceManager:
    """Get the global device manager."""
    return _device_manager
