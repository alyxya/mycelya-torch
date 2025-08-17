# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Storage management for remote tensors.

This module manages storage IDs and their lifecycle:
- Storage ID generation and tracking
- Storage-to-machine mappings and resolution
- Cross-device operation validation
- Remote storage creation and cleanup
- Storage statistics and information
"""

import threading
from typing import Dict, List, Optional

from ._device import get_device_registry
from ._logging import get_logger
from ._machine import RemoteMachine

log = get_logger(__name__)


class StorageRegistry:
    """
    Simplified registry to track remote storage IDs only.

    Key concepts:
    - storage_id: Identifies remote memory allocation on clients
    - Uses incremental storage IDs starting from 1 (1, 2, 3, ...)
    - No local device simulation or memory allocation
    - Remote operations: Receive storage_id + tensor metadata (shape, stride, offset, storage_id)
    - Storage cleanup: Handles remote storage cleanup when tensors are freed
    """

    def __init__(self) -> None:
        # Storage ID tracking - maps storage to device
        self.storage_id_to_device: Dict[int, int] = {}  # storage_id -> device_index

        # Storage ID to nbytes mapping - tracks original nbytes from allocator
        self.storage_id_to_nbytes: Dict[int, int] = {}  # storage_id -> nbytes

        # Tensor ID tracking - maps tensor ID to device
        self.tensor_id_to_device: Dict[int, int] = {}  # tensor_id -> device_index

        # Thread-safe counter for generating incremental storage IDs
        self._storage_id_counter = 1
        self._storage_id_lock = threading.Lock()

        log.info("🚀 Storage registry initialized")

    def create_storage(self, nbytes: int, device_index: int) -> int:
        """
        Create remote storage with an incremental unique ID.

        Args:
            nbytes: Number of bytes to allocate
            device_index: Device index to create storage on

        Returns:
            int: The generated storage ID on success, or 0 on failure
        """
        # Generate incremental storage ID in thread-safe manner
        with self._storage_id_lock:
            storage_id = self._storage_id_counter
            self._storage_id_counter += 1

        log.info(f"🆔 GENERATED Storage ID: {storage_id}")

        # Always track the storage ID for all tensors
        self.storage_id_to_device[storage_id] = device_index

        # Store the original nbytes from the allocator
        self.storage_id_to_nbytes[storage_id] = nbytes

        # Storage is now managed locally only - remote side uses tensor IDs exclusively
        # No need to register with orchestrator as tensors will be created on-demand
        log.info(
            f"Created local storage ID {storage_id} for device {device_index} ({nbytes} bytes)"
        )

        log.info(f"Registered storage ID {storage_id} on device {device_index}")
        return storage_id

    def register_tensor_id(self, tensor_id: int, device_index: int) -> None:
        """Register a tensor ID with its device."""
        self.tensor_id_to_device[tensor_id] = device_index
        log.debug(f"Registered tensor ID {tensor_id} on device {device_index}")

    def get_tensor_device(self, tensor_id: int) -> Optional[int]:
        """Get device index for a tensor ID"""
        return self.tensor_id_to_device.get(tensor_id)

    def free_storage_with_id(self, storage_id: int) -> bool:
        """Free storage by storage ID and perform remote cleanup"""
        storage_id = int(storage_id)
        if storage_id == 0:  # Empty storage
            return True

        # Get device information before cleanup
        device_idx = self.storage_id_to_device.get(storage_id)

        if storage_id in self.storage_id_to_device:
            # Clean up storage tracking
            self.storage_id_to_device.pop(storage_id, None)
            self.storage_id_to_nbytes.pop(storage_id, None)

            # Remote cleanup if device information is available
            if device_idx is not None:
                log.info(f"Storage {storage_id} freed, initiating remote cleanup")
                self._cleanup_remote_storage(storage_id, device_idx)
            else:
                log.debug(
                    f"No device index found for storage {storage_id}, "
                    "skipping remote cleanup"
                )

            log.info(f"Freed storage ID {storage_id}")
            return True
        else:
            log.warning(f"Attempted to free unknown storage {storage_id}")
            return False

    def _cleanup_remote_storage(self, storage_id: int, device_idx: int) -> None:
        """Clean up storage on remote GPU device"""
        try:
            # Import here to avoid circular imports
            from ._orchestrator import orchestrator

            # Get the device registry to find the machine
            if orchestrator is None:
                log.warning(
                    f"No orchestrator available for storage {storage_id} cleanup"
                )
                return

            device_registry = get_device_registry()
            device = device_registry.get_device_by_index(device_idx)
            if device is None:
                log.warning(
                    f"No device found for index {device_idx} during storage "
                    f"{storage_id} cleanup"
                )
                return

            # Get all tensor IDs associated with this storage using orchestrator
            tensor_ids = orchestrator.get_tensor_ids_for_storage_by_device(
                device_idx, storage_id
            )

            if tensor_ids:
                log.info(
                    f"Cleaning up {len(tensor_ids)} tensor IDs for storage {storage_id}"
                )

                # Remove tensors from remote side using orchestrator
                orchestrator.remove_tensors_by_device(device_idx, list(tensor_ids))

                # Clean up client-side mapping using orchestrator
                for tensor_id in tensor_ids:
                    orchestrator.remove_tensor_from_storage_mapping_by_device(
                        device_idx, storage_id, tensor_id
                    )

                log.info(
                    f"✅ Successfully cleaned up {len(tensor_ids)} tensors for storage {storage_id} "
                    f"on device {device_idx}"
                )
            else:
                # No tensor IDs found for this storage - it may have already been cleaned up
                log.info(
                    f"No tensor IDs found for storage {storage_id} - may already be cleaned up"
                )

        except Exception as e:
            log.error(
                f"Failed remote cleanup for storage {storage_id} on device {device_idx}: {e}"
            )

    def resize_storage_by_id(self, storage_id: int, nbytes: int) -> bool:
        """Resize remote storage by storage ID"""
        try:
            storage_id = int(storage_id)

            # Get device index for this storage
            device_idx = self.storage_id_to_device.get(storage_id)
            if device_idx is None:
                log.warning(f"No device found for storage {storage_id}")
                return False

            # Use orchestrator for centralized client management
            from ._orchestrator import orchestrator

            orchestrator.resize_storage(storage_id, nbytes)
            log.info(
                f"✅ Successfully resized remote storage {storage_id} to {nbytes} bytes via orchestrator"
            )
            return True

        except Exception as e:
            log.warning(f"Failed to resize remote storage {storage_id}: {e}")
            return False


# Global instances
_storage_registry = StorageRegistry()


# Storage registry access functions
def create_storage(nbytes: int, device_index: int) -> int:
    """Create remote storage with a generated unique ID."""
    return _storage_registry.create_storage(nbytes, device_index)


def free_storage_with_id(storage_id: int) -> bool:
    """Free storage by storage ID."""
    return _storage_registry.free_storage_with_id(storage_id)


def register_tensor_id(tensor_id: int, device_index: int) -> None:
    """Register a tensor ID with its device."""
    return _storage_registry.register_tensor_id(tensor_id, device_index)


def get_tensor_device(tensor_id: int) -> Optional[int]:
    """Get device index for a tensor ID"""
    return _storage_registry.get_tensor_device(tensor_id)


def get_storage_nbytes(storage_id: int) -> Optional[int]:
    """Get the original nbytes for a storage ID from the allocator."""
    return _storage_registry.storage_id_to_nbytes.get(storage_id)


def resize_storage_by_id(storage_id: int, nbytes: int) -> bool:
    """Resize remote storage by storage ID."""
    return _storage_registry.resize_storage_by_id(storage_id, nbytes)


def get_machine_for_storage(storage_id: int) -> RemoteMachine:
    """Get the machine that owns a specific storage ID.

    Args:
        storage_id: Storage ID to resolve

    Returns:
        RemoteMachine that owns the storage

    Raises:
        RuntimeError: If no device or machine found for storage
    """
    # Get device index for this storage
    device_idx = _storage_registry.storage_id_to_device.get(storage_id)
    if device_idx is None:
        raise RuntimeError(f"No device found for storage {storage_id}")

    # Get machine for device index
    registry = get_device_registry()
    machine = registry.get_device_by_index(device_idx)
    if machine is None:
        raise RuntimeError(f"No machine found for device index {device_idx}")

    return machine


def get_machine_for_tensor_id(tensor_id: int) -> RemoteMachine:
    """Get the machine that owns a specific tensor ID.

    Args:
        tensor_id: Tensor ID to resolve

    Returns:
        RemoteMachine that owns the tensor

    Raises:
        RuntimeError: If no device or machine found for tensor
    """
    # Get device index for this tensor ID
    device_idx = get_tensor_device(tensor_id)
    if device_idx is None:
        raise RuntimeError(f"No device found for tensor {tensor_id}")

    # Get machine for device index
    registry = get_device_registry()
    machine = registry.get_device_by_index(device_idx)
    if machine is None:
        raise RuntimeError(f"No machine found for device index {device_idx}")

    return machine


def validate_cross_device_operation_tensor_ids(tensor_ids: List[int]) -> None:
    """Validate that all tensor IDs belong to the same device.

    Args:
        tensor_ids: List of tensor IDs to validate

    Raises:
        RuntimeError: If tensors are on different devices
    """
    if not tensor_ids:
        return

    # Get the first machine as reference
    first_machine = get_machine_for_tensor_id(tensor_ids[0])
    first_device_name = f"{first_machine.provider.value}-{first_machine.gpu_type.value}"

    # Validate all other tensors are on the same machine
    for tensor_id in tensor_ids[1:]:
        machine = get_machine_for_tensor_id(tensor_id)
        current_device_name = f"{machine.provider.value}-{machine.gpu_type.value}"

        if machine.machine_id != first_machine.machine_id:
            raise RuntimeError(
                f"Cannot perform operations between tensors on different remote devices. "
                f"Tensors are on different devices: "
                f'"{first_device_name}" and "{current_device_name}". '
                f"Transfer tensors to the same device first: tensor.cpu().to(target_device)"
            )
