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

import random
from typing import Dict, List, Optional, Set

from ._logging import get_logger
from ._tensor_utils import RemoteTensorMetadata
from .device import RemoteMachine, get_device_registry

log = get_logger(__name__)

# Constants for storage ID generation
MAX_ID_GENERATION_ATTEMPTS = 100
MIN_STORAGE_ID = 1
MAX_STORAGE_ID = 2**63 - 1  # 64-bit signed integer max

# Constants for tensor ID generation
MIN_TENSOR_ID = 1
MAX_TENSOR_ID = 2**63 - 1  # 64-bit signed integer max


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

    def create_storage(self, nbytes: int, device_index: int) -> int:
        """
        Create remote storage with a generated unique ID.

        Args:
            nbytes: Number of bytes to allocate
            device_index: Device index to create storage on

        Returns:
            int: The generated storage ID on success, or 0 on failure
        """
        global MAX_ID_GENERATION_ATTEMPTS

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
            log.error(
                f"Failed to generate unique storage ID after {MAX_ID_GENERATION_ATTEMPTS} attempts"
            )
            return 0

        # Always track the storage ID for all tensors
        self.storage_id_to_device[storage_id] = device_index

        # Register the storage with the orchestrator for centralized client management
        try:
            from ._remote_orchestrator import remote_orchestrator

            # Use orchestrator to create storage with device routing
            remote_orchestrator.create_storage(storage_id, nbytes, device_index)
            log.info(
                f"Registered storage {storage_id} via orchestrator ({nbytes} bytes)"
            )
        except Exception as e:
            log.warning(
                f"Failed to register storage {storage_id} via orchestrator: {e}"
            )
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

    def free_storage_with_id(self, storage_id: int) -> bool:
        """Free storage by storage ID and perform remote cleanup"""
        storage_id = int(storage_id)
        if storage_id == 0:  # Empty storage
            return True

        # Get device information before cleanup
        device_idx = self.storage_id_to_device.get(storage_id)

        if storage_id in self.storage_id_to_device:
            # Clean up associated tensor IDs first
            tensor_cleanup_count = _tensor_id_manager.cleanup_storage_tensors(
                storage_id
            )
            if tensor_cleanup_count > 0:
                log.debug(
                    f"Cleaned up {tensor_cleanup_count} tensor IDs for storage {storage_id}"
                )

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
            return True
        else:
            log.warning(f"Attempted to free unknown storage {storage_id}")
            return False

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

            # Use orchestrator for centralized client management
            from ._remote_orchestrator import remote_orchestrator

            remote_orchestrator.resize_storage(storage_id, nbytes)
            log.info(
                f"✅ Successfully resized remote storage {storage_id} to {nbytes} bytes via orchestrator"
            )
            return True

        except Exception as e:
            log.warning(f"Failed to resize remote storage {storage_id}: {e}")
            return False


class TensorIdManager:
    """
    Registry for passively generated tensor IDs.

    Key concepts:
    - tensor_id: Unique identifier for each tensor (including views)
    - Passive generation: Tensor IDs created on-demand during operations
    - Hierarchical relationship: Each tensor_id maps to one storage_id
    - View support: Multiple tensor_ids can share the same storage_id
    """

    def __init__(self) -> None:
        # Core mappings
        self._tensor_to_storage: Dict[int, int] = {}  # tensor_id -> storage_id
        self._storage_to_tensors: Dict[
            int, Set[int]
        ] = {}  # storage_id -> set[tensor_id]
        self._tensor_to_metadata: Dict[
            int, RemoteTensorMetadata
        ] = {}  # tensor_id -> metadata

        # ID generation tracking
        self._generated_tensor_ids: Set[int] = set()

        log.info("🆔 Tensor ID manager initialized")

    def get_or_create_tensor_id(self, metadata: RemoteTensorMetadata) -> int:
        """
        Generate a unique tensor ID for the given metadata.

        This is the main method for passive tensor ID generation. It creates
        a new tensor ID and establishes the mapping to the storage ID.

        Args:
            metadata: RemoteTensorMetadata containing storage_id and tensor info

        Returns:
            int: Generated tensor ID

        Raises:
            RuntimeError: If unable to generate unique tensor ID
        """
        storage_id = metadata.storage_id

        # Generate unique tensor ID
        tensor_id = self._generate_unique_tensor_id()

        # Establish mappings
        self._tensor_to_storage[tensor_id] = storage_id
        self._tensor_to_metadata[tensor_id] = metadata

        # Track storage -> tensors relationship
        if storage_id not in self._storage_to_tensors:
            self._storage_to_tensors[storage_id] = set()
        self._storage_to_tensors[storage_id].add(tensor_id)

        log.debug(f"Generated tensor_id {tensor_id} for storage_id {storage_id}")
        return tensor_id

    def _generate_unique_tensor_id(self) -> int:
        """Generate a unique tensor ID with collision detection."""
        for attempt in range(1, MAX_ID_GENERATION_ATTEMPTS + 1):
            candidate_id = random.randint(MIN_TENSOR_ID, MAX_TENSOR_ID)

            if candidate_id not in self._generated_tensor_ids:
                self._generated_tensor_ids.add(candidate_id)
                return candidate_id
            else:
                log.debug(
                    f"Generated duplicate tensor ID {candidate_id}, retrying (attempt {attempt})"
                )

        raise RuntimeError(
            f"Failed to generate unique tensor ID after {MAX_ID_GENERATION_ATTEMPTS} attempts"
        )

    def get_storage_id(self, tensor_id: int) -> Optional[int]:
        """Get the storage ID for a tensor ID."""
        return self._tensor_to_storage.get(tensor_id)

    def get_metadata(self, tensor_id: int) -> Optional[RemoteTensorMetadata]:
        """Get the metadata for a tensor ID."""
        return self._tensor_to_metadata.get(tensor_id)

    def get_tensor_ids_for_storage(self, storage_id: int) -> Set[int]:
        """Get all tensor IDs that share the given storage ID."""
        return self._storage_to_tensors.get(storage_id, set())

    def cleanup_tensor_id(self, tensor_id: int) -> bool:
        """
        Clean up a tensor ID when the tensor is no longer needed.

        This removes the tensor ID from all mappings and cleans up
        the storage -> tensors relationship if this was the last tensor.

        Args:
            tensor_id: Tensor ID to clean up

        Returns:
            bool: True if cleanup was successful
        """
        if tensor_id not in self._tensor_to_storage:
            log.warning(f"Attempted to cleanup unknown tensor ID {tensor_id}")
            return False

        storage_id = self._tensor_to_storage[tensor_id]

        # Remove from all mappings
        self._tensor_to_storage.pop(tensor_id, None)
        self._tensor_to_metadata.pop(tensor_id, None)
        self._generated_tensor_ids.discard(tensor_id)

        # Update storage -> tensors mapping
        if storage_id in self._storage_to_tensors:
            self._storage_to_tensors[storage_id].discard(tensor_id)
            # Clean up empty storage entries
            if not self._storage_to_tensors[storage_id]:
                self._storage_to_tensors.pop(storage_id, None)

        log.debug(f"Cleaned up tensor_id {tensor_id}")
        return True

    def cleanup_storage_tensors(self, storage_id: int) -> int:
        """
        Clean up all tensor IDs associated with a storage ID.

        This is called when storage is being freed to clean up
        all associated tensor IDs.

        Args:
            storage_id: Storage ID whose tensors should be cleaned up

        Returns:
            int: Number of tensor IDs cleaned up
        """
        tensor_ids = self.get_tensor_ids_for_storage(storage_id).copy()
        cleaned_count = 0

        for tensor_id in tensor_ids:
            if self.cleanup_tensor_id(tensor_id):
                cleaned_count += 1

        log.debug(f"Cleaned up {cleaned_count} tensor IDs for storage {storage_id}")
        return cleaned_count

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about tensor ID usage."""
        return {
            "active_tensor_ids": len(self._tensor_to_storage),
            "active_storage_mappings": len(self._storage_to_tensors),
            "generated_tensor_ids": len(self._generated_tensor_ids),
        }


# Global instances
_storage_registry = StorageRegistry()
_tensor_id_manager = TensorIdManager()


# Storage registry access functions
def create_storage(nbytes: int, device_index: int) -> int:
    """Create remote storage with a generated unique ID."""
    return _storage_registry.create_storage(nbytes, device_index)


def free_storage_with_id(storage_id: int) -> bool:
    """Free storage by storage ID."""
    return _storage_registry.free_storage_with_id(storage_id)


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


# Tensor ID manager access functions
def get_or_create_tensor_id(metadata: RemoteTensorMetadata) -> int:
    """Generate a unique tensor ID for the given metadata."""
    return _tensor_id_manager.get_or_create_tensor_id(metadata)


def get_storage_id_for_tensor(tensor_id: int) -> Optional[int]:
    """Get the storage ID for a tensor ID."""
    return _tensor_id_manager.get_storage_id(tensor_id)


def get_metadata_for_tensor(tensor_id: int) -> Optional[RemoteTensorMetadata]:
    """Get the metadata for a tensor ID."""
    return _tensor_id_manager.get_metadata(tensor_id)


def get_tensor_ids_for_storage(storage_id: int) -> Set[int]:
    """Get all tensor IDs that share the given storage ID."""
    return _tensor_id_manager.get_tensor_ids_for_storage(storage_id)


def cleanup_tensor_id(tensor_id: int) -> bool:
    """Clean up a tensor ID when the tensor is no longer needed."""
    return _tensor_id_manager.cleanup_tensor_id(tensor_id)


def get_tensor_stats() -> Dict[str, int]:
    """Get statistics about tensor ID usage."""
    return _tensor_id_manager.get_stats()
