# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Storage management for tracking storage-to-machine mappings.

This module manages the mapping between storage IDs and remote machines:
- Storage ID to machine resolution
- Cross-device operation validation
- Storage statistics and information

Handles the local mapping of storage IDs to devices and provides validation
for cross-device operations to ensure tensors are on compatible machines.
Storage lifecycle (creation/destruction) is handled by the C++ allocator.
"""

from typing import List, Set

from ._logging import get_logger
from .device import RemoteMachine, get_device_registry

log = get_logger(__name__)


def get_machine_for_storage(storage_id: int) -> RemoteMachine:
    """Get the machine that owns a specific storage ID.

    Args:
        storage_id: Storage ID to resolve

    Returns:
        RemoteMachine that owns the storage

    Raises:
        RuntimeError: If no device or machine found for storage
    """
    # Import here to avoid circular imports
    from . import driver

    # Get device index for this storage
    device_idx = driver.exec("get_storage_device", storage_id)
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
        from . import driver

        device_idx = driver.exec("get_storage_device", storage_id)
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
