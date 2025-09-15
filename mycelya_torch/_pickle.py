# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Custom pickle system for mycelya tensors and remote execution.

This module provides custom pickler/unpickler classes that handle mycelya tensors and devices
properly during serialization for remote execution. It includes:

- Pickler: Converts mycelya tensors to tensor IDs and devices to remote info
- Unpickler: Reconstructs remote execution results back to local mycelya tensors
"""

import io
import pickle
from typing import Any, Optional, Tuple

import cloudpickle
import torch

from ._device import device_manager
from ._logging import get_logger
from ._utils import (
    create_mycelya_tensor_from_metadata,
    get_storage_id,
    get_tensor_id,
)

log = get_logger(__name__)


class Pickler(cloudpickle.Pickler):
    """
    Custom Pickler that handles mycelya tensors and devices for remote execution.

    This pickler converts:
    - Mycelya tensors -> tensor IDs for remote lookup
    - Mycelya devices -> device info tuples for remote mapping

    It maintains internal state about the remote machine being serialized for,
    and validates that all tensors/devices belong to the same machine.
    Uses cloudpickle.Pickler for proper function serialization.
    """

    def __init__(
        self, file: io.BytesIO, storage_manager, protocol: int = None, buffer_callback: Any = None
    ):
        super().__init__(file, protocol=protocol, buffer_callback=buffer_callback)
        self.storage_manager = storage_manager
        self.machine_id: Optional[str] = None
        # Collect tensors that need _maybe_create_tensor called
        self.tensors = []

    def persistent_id(self, obj: Any) -> Optional[Tuple[str, Any]]:
        """
        Handle mycelya tensors and devices during pickling.

        Args:
            obj: Object being pickled

        Returns:
            Tuple of (type_tag, data) for mycelya objects, None for regular objects

        Raises:
            RuntimeError: If tensors/devices from different machines are mixed
        """
        # Handle mycelya tensors
        if isinstance(obj, torch.Tensor) and obj.device.type == "mycelya":
            # Get tensor's machine information
            storage_id = get_storage_id(obj)
            machine_id, remote_type, remote_index = (
                self.storage_manager.get_remote_device_info(storage_id)
            )

            # Validate machine consistency
            if self.machine_id is None:
                self.machine_id = machine_id
            elif self.machine_id != machine_id:
                raise RuntimeError(
                    f"Cannot serialize tensors from different machines: "
                    f"current machine {self.machine_id}, tensor machine {machine_id}"
                )

            # Collect tensor for orchestrator to call _maybe_create_tensor on
            self.tensors.append(obj)
            tensor_id = get_tensor_id(obj)  # Use metadata hash as tensor ID
            return ("mycelya_tensor", tensor_id)

        # Handle mycelya devices
        elif isinstance(obj, torch.device) and obj.type == "mycelya":
            if obj.index is None:
                raise ValueError("Mycelya device must have an index")

            # Get device's machine information
            machine_id, remote_type, remote_index = (
                device_manager.get_remote_device_info(obj.index)
            )

            # Validate machine consistency
            if self.machine_id is None:
                self.machine_id = machine_id
            elif self.machine_id != machine_id:
                raise RuntimeError(
                    f"Cannot serialize devices from different machines: "
                    f"current machine {self.machine_id}, device machine {machine_id}"
                )

            # Return device info for remote mapping
            return ("mycelya_device", (remote_type, remote_index))

        # Not a mycelya object - use normal pickling
        return None


class Unpickler(pickle.Unpickler):
    """
    Unpickler for remote function execution results.

    This unpickler handles the results returned from remote function execution,
    converting remote_tensor metadata back into local mycelya tensors and
    remote_device info back into local mycelya devices.
    """

    def __init__(self, file: io.BytesIO, machine_id: str):
        super().__init__(file)
        self.machine_id = machine_id
        # Collect tensor linking info for orchestrator to handle
        self.tensors_to_link = []  # List of (tensor, temp_key) tuples

    def persistent_load(self, pid: Tuple[str, Any]) -> Any:
        """
        Handle reconstruction of remote execution results.

        Args:
            pid: Persistent ID tuple from remote pickler

        Returns:
            Reconstructed mycelya tensor or device
        """
        type_tag, data = pid

        if type_tag == "remote_tensor":
            metadata = data

            # Get device using device_manager
            device = device_manager.get_mycelya_device(
                self.machine_id, metadata["device_type"], metadata["device_index"]
            )

            # Create mycelya tensor from metadata
            tensor = create_mycelya_tensor_from_metadata(metadata, device)

            # Collect tensor linking info for orchestrator to handle
            temp_key = metadata["temp_key"]
            self.tensors_to_link.append((tensor, temp_key))

            return tensor

        elif type_tag == "remote_device":
            device_type, device_index = data

            # Get device using device_manager
            return device_manager.get_mycelya_device(
                self.machine_id, device_type, device_index
            )

        else:
            raise pickle.PicklingError(f"Unknown persistent ID type: {type_tag}")
