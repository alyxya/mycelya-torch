# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Storage management for torch-remote.

This module manages storage IDs and their lifecycle:
- Storage ID generation and tracking
- Storage-to-machine mappings and resolution
- Cross-device operation validation
- Remote storage creation and cleanup
- Storage statistics and information
"""

import random
from contextlib import contextmanager
from typing import Dict, List, Optional, Set

from ._logging import get_logger
from .device import RemoteMachine, get_device_registry

log = get_logger(__name__)

# Constants for storage ID generation
MAX_ID_GENERATION_ATTEMPTS = 100
MIN_STORAGE_ID = 1
MAX_STORAGE_ID = 2**63 - 1  # 64-bit signed integer max


@contextmanager
def lazy_allocation_mode():
    """Context manager to enable lazy allocation for storage creation within the context"""
    global _lazy_allocation_enabled
    old_value = _lazy_allocation_enabled
    _lazy_allocation_enabled = True
    try:
        yield
    finally:
        _lazy_allocation_enabled = old_value


# Global flag for lazy allocation mode
_lazy_allocation_enabled = False


class StorageRegistry:
    """
    Simplified registry to track remote storage IDs only.

    Key concepts:
    - storage_id: Identifies remote memory allocation on clients
    - No local device simulation or memory allocation
    - Remote operations: Receive storage_id + tensor metadata (shape, stride, offset, storage_id)
    - Storage cleanup: Handles remote storage cleanup when tensors are freed
    """

    def __init__(self) -> None:
        # Storage ID tracking - maps storage to device
        self.storage_id_to_device: Dict[int, int] = {}  # storage_id -> device_index

        # Set for tracking all generated storage IDs to avoid duplicates
        self.generated_storage_ids: Set[int] = set()

        log.info("🚀 Storage registry initialized")

    def create_storage(self, nbytes: int, device_index: int, lazy: bool = False) -> int:
        """
        Create remote storage with a generated unique ID.
        
        Args:
            nbytes: Number of bytes to allocate
            device_index: Device index to create storage on
            lazy: Whether to use lazy allocation
            
        Returns:
            int: The generated storage ID on success, or 0 on failure
        """
        global MAX_ID_GENERATION_ATTEMPTS, _lazy_allocation_enabled
        
        if _lazy_allocation_enabled:
            lazy = True

        # Generate a unique storage ID
        storage_id = 0
        for attempt in range(1, MAX_ID_GENERATION_ATTEMPTS + 1):
            # Generate a random 64-bit integer within the valid range
            candidate_id = random.randint(MIN_STORAGE_ID, MAX_STORAGE_ID)

            # Check if this ID is already in use
            if candidate_id not in self.generated_storage_ids:
                storage_id = candidate_id
                self.generated_storage_ids.add(storage_id)
                log.info(f"🆔 GENERATED Storage ID: {storage_id}")
                break
            else:
                log.debug(
                    f"Generated duplicate storage ID {candidate_id}, retrying (attempt {attempt})"
                )

        if storage_id == 0:
            log.error(f"Failed to generate unique storage ID after {MAX_ID_GENERATION_ATTEMPTS} attempts")
            return 0

        # Always track the storage ID for all tensors
        self.storage_id_to_device[storage_id] = device_index

        # Register the storage with the client immediately for all allocations
        try:
            device_registry = get_device_registry()
            device = device_registry.get_device_by_index(device_index)
            if device is not None:
                client = device._client
                if client:
                    try:
                        # Create storage with exact byte size
                        client.create_storage(storage_id, nbytes, lazy)
                        log.info(
                            f"Registered storage {storage_id} with client "
                            f"({nbytes} bytes, lazy={lazy})"
                        )
                    except Exception as e:
                        log.warning(
                            f"Failed to register storage {storage_id} with client: {e}"
                        )
                        # Clean up the failed storage ID
                        self.storage_id_to_device.pop(storage_id, None)
                        self.generated_storage_ids.discard(storage_id)
                        return 0
        except Exception as e:
            log.warning(f"Failed to register storage {storage_id} with client: {e}")
            # Clean up the failed storage ID
            self.storage_id_to_device.pop(storage_id, None)
            self.generated_storage_ids.discard(storage_id)
            return 0

        log.info(f"Registered storage ID {storage_id} on device {device_index}")
        return storage_id

    def get_storage_device(self, storage_id: int) -> Optional[int]:
        """Get device index for a storage ID"""
        storage_id = int(storage_id)
        return self.storage_id_to_device.get(storage_id)

    def free_storage_with_id(self, storage_id: int) -> None:
        """Free storage by storage ID and perform remote cleanup"""
        storage_id = int(storage_id)
        if storage_id == 0:  # Empty storage
            return

        # Get device information before cleanup
        device_idx = self.storage_id_to_device.get(storage_id)

        if storage_id in self.storage_id_to_device:
            # Clean up storage tracking
            self.storage_id_to_device.pop(storage_id, None)
            self.generated_storage_ids.discard(storage_id)

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
        else:
            log.warning(f"Attempted to free unknown storage {storage_id}")

    def _cleanup_remote_storage(self, storage_id: int, device_idx: int) -> None:
        """Clean up storage on remote GPU device"""
        try:
            # Import here to avoid circular imports
            from ._remote_orchestrator import remote_orchestrator

            # Get the device registry to find the machine
            orchestrator = remote_orchestrator
            if orchestrator is None:
                log.warning(
                    f"No remote orchestrator available for storage {storage_id} cleanup"
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

            # Call cleanup on orchestrator
            log.info(f"Calling remove_storage_from_remote for storage {storage_id}")
            success = orchestrator.remove_tensor_from_remote(storage_id, device)

            if success:
                log.info(
                    f"✅ Successfully cleaned up remote storage {storage_id} "
                    f"on device {device_idx}"
                )
            else:
                log.warning(
                    f"❌ Remote cleanup returned false for storage {storage_id} "
                    f"on device {device_idx}"
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
            device_idx = self.get_storage_device(storage_id)
            if device_idx is None:
                log.warning(f"No device found for storage {storage_id}")
                return False

            # Import here to avoid circular imports
            from ._remote_orchestrator import remote_orchestrator

            # Get the device registry to find the machine
            orchestrator = remote_orchestrator
            if orchestrator is None:
                log.warning(
                    f"No remote orchestrator available for storage {storage_id} resize"
                )
                return False

            device_registry = get_device_registry()
            device = device_registry.get_device_by_index(device_idx)
            if device is None:
                log.warning(
                    f"No device found for index {device_idx} during storage {storage_id} resize"
                )
                return False

            # Get client and call resize_storage
            client = device._client
            if client and client.is_running():
                client.resize_storage(storage_id, nbytes)
                log.info(
                    f"✅ Successfully initiated resize of remote storage {storage_id} to {nbytes} bytes"
                )
                return True
            else:
                log.warning(f"Client not available for storage {storage_id} resize")
                return False

        except Exception as e:
            log.warning(f"Failed to resize remote storage {storage_id}: {e}")
            return False


# Global storage registry instance
_storage_registry = StorageRegistry()


# Storage registry access functions
def create_storage(nbytes: int, device_index: int, lazy: bool = False) -> int:
    """Create remote storage with a generated unique ID."""
    return _storage_registry.create_storage(nbytes, device_index, lazy)


def free_storage_with_id(storage_id: int) -> None:
    """Free storage by storage ID."""
    _storage_registry.free_storage_with_id(storage_id)


def get_storage_device(storage_id: int) -> Optional[int]:
    """Get device index for a storage ID."""
    return _storage_registry.get_storage_device(storage_id)




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
    device_idx = get_storage_device(storage_id)
    if device_idx is None:
        raise RuntimeError(f"No device found for storage {storage_id}")

    # Get machine for device index
    registry = get_device_registry()
    machine = registry.get_device_by_index(device_idx)
    if machine is None:
        raise RuntimeError(f"No machine found for device index {device_idx}")

    return machine


def validate_storage_exists(storage_id: int) -> bool:
    """Validate that a storage ID exists and is accessible.

    Args:
        storage_id: Storage ID to validate

    Returns:
        True if storage exists, False otherwise
    """
    try:
        device_idx = get_storage_device(storage_id)
        return device_idx is not None
    except Exception as e:
        log.warning(f"Failed to validate storage {storage_id}: {e}")
        return False


def validate_cross_device_operation(storage_ids: List[int]) -> None:
    """Validate that all storage IDs belong to the same device.

    Args:
        storage_ids: List of storage IDs to validate

    Raises:
        RuntimeError: If storages are on different devices
    """
    if not storage_ids:
        return

    # Get the first machine as reference
    first_machine = get_machine_for_storage(storage_ids[0])
    first_device_name = f"{first_machine.provider.value}-{first_machine.gpu_type.value}"

    # Validate all other storages are on the same machine
    for storage_id in storage_ids[1:]:
        machine = get_machine_for_storage(storage_id)
        current_device_name = f"{machine.provider.value}-{machine.gpu_type.value}"

        if machine.machine_id != first_machine.machine_id:
            raise RuntimeError(
                f"Cannot perform operations between tensors on different remote devices. "
                f"Tensors are on different devices: "
                f'"{first_device_name}" and "{current_device_name}". '
                f"Transfer tensors to the same device first: tensor.cpu().to(target_device)"
            )


def get_machines_for_storages(storage_ids: List[int]) -> List[RemoteMachine]:
    """Get the machines for multiple storage IDs.

    Args:
        storage_ids: List of storage IDs to resolve

    Returns:
        List of RemoteMachine instances (may contain duplicates)
    """
    machines = []
    for storage_id in storage_ids:
        try:
            machine = get_machine_for_storage(storage_id)
            machines.append(machine)
        except Exception as e:
            log.warning(f"Failed to resolve machine for storage {storage_id}: {e}")

    return machines


def get_unique_machines_for_storages(storage_ids: List[int]) -> Set[RemoteMachine]:
    """Get unique machines for multiple storage IDs.

    Args:
        storage_ids: List of storage IDs to resolve

    Returns:
        Set of unique RemoteMachine instances
    """
    machines = get_machines_for_storages(storage_ids)
    return set(machines)


def get_storage_stats() -> dict:
    """Get statistics about tracked storages.

    Returns:
        Dictionary with storage statistics
    """
    try:
        # Get storage information from device registry
        registry = get_device_registry()

        # The registry doesn't expose its internal _devices dict directly
        # so we'll provide basic stats for now
        stats = {
            "storage_manager_active": True,
            "registry_available": registry is not None,
        }

        return stats

    except Exception as e:
        log.warning(f"Failed to get storage stats: {e}")
        return {"error": str(e)}
