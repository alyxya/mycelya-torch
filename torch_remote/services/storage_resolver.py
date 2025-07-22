# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Storage machine resolver for tracking storage-to-machine mappings.

This service manages the mapping between storage IDs and remote machines:
- Storage ID to machine resolution
- Cross-device operation validation
- Storage registration and tracking
- Storage lifecycle management

Extracted from RemoteOrchestrator to provide better separation of concerns
and centralized storage tracking.
"""

import logging
from typing import List, Set

from ..device import RemoteMachine, get_device_registry

log = logging.getLogger(__name__)


class StorageMachineResolver:
    """Resolves storage IDs to their owning remote machines.
    
    This class provides centralized tracking of which machine owns
    which storage IDs, enabling cross-device validation and storage
    lifecycle management.
    """

    def __init__(self):
        # We'll rely on the device registry and driver for storage-device mapping
        # rather than maintaining our own mapping table
        pass

    def get_machine_for_storage(self, storage_id: int) -> RemoteMachine:
        """Get the machine that owns a specific storage ID.
        
        Args:
            storage_id: Storage ID to resolve
            
        Returns:
            RemoteMachine that owns the storage
            
        Raises:
            RuntimeError: If no device or machine found for storage
        """
        # Import here to avoid circular imports
        from .. import driver

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

    def validate_storage_exists(self, storage_id: int) -> bool:
        """Validate that a storage ID exists and is accessible.
        
        Args:
            storage_id: Storage ID to validate
            
        Returns:
            True if storage exists, False otherwise
        """
        try:
            from .. import driver
            device_idx = driver.exec("get_storage_device", storage_id)
            return device_idx is not None
        except Exception as e:
            log.warning(f"Failed to validate storage {storage_id}: {e}")
            return False

    def validate_cross_device_operation(self, storage_ids: List[int]) -> None:
        """Validate that all storage IDs belong to the same device.
        
        Args:
            storage_ids: List of storage IDs to validate
            
        Raises:
            RuntimeError: If storages are on different devices
        """
        if not storage_ids:
            return

        # Get the first machine as reference
        first_machine = self.get_machine_for_storage(storage_ids[0])
        first_device_name = f"{first_machine.provider.value}-{first_machine.gpu_type.value}"

        # Validate all other storages are on the same machine
        for storage_id in storage_ids[1:]:
            machine = self.get_machine_for_storage(storage_id)
            current_device_name = f"{machine.provider.value}-{machine.gpu_type.value}"

            if machine.machine_id != first_machine.machine_id:
                raise RuntimeError(
                    f"Cannot perform operations between tensors on different remote devices. "
                    f"Tensors are on different devices: "
                    f'"{first_device_name}" and "{current_device_name}". '
                    f"Transfer tensors to the same device first: tensor.cpu().to(target_device)"
                )

    def get_machines_for_storages(self, storage_ids: List[int]) -> List[RemoteMachine]:
        """Get the machines for multiple storage IDs.
        
        Args:
            storage_ids: List of storage IDs to resolve
            
        Returns:
            List of RemoteMachine instances (may contain duplicates)
        """
        machines = []
        for storage_id in storage_ids:
            try:
                machine = self.get_machine_for_storage(storage_id)
                machines.append(machine)
            except Exception as e:
                log.warning(f"Failed to resolve machine for storage {storage_id}: {e}")

        return machines

    def get_unique_machines_for_storages(self, storage_ids: List[int]) -> Set[RemoteMachine]:
        """Get unique machines for multiple storage IDs.
        
        Args:
            storage_ids: List of storage IDs to resolve
            
        Returns:
            Set of unique RemoteMachine instances
        """
        machines = self.get_machines_for_storages(storage_ids)
        return set(machines)

    def register_storage(self, storage_id: int, machine: RemoteMachine) -> None:
        """Register a storage ID with its owning machine.
        
        Note: In the current architecture, storage registration is handled
        by the device daemon when storage is created. This method is provided
        for future extensibility.
        
        Args:
            storage_id: Storage ID to register
            machine: Machine that owns the storage
        """
        # Current implementation relies on the device daemon for registration
        # This method is a placeholder for future enhancements
        log.debug(f"Storage {storage_id} registered with machine {machine.machine_id}")

    def unregister_storage(self, storage_id: int) -> None:
        """Unregister a storage ID.
        
        Note: In the current architecture, storage cleanup is handled
        by the device daemon. This method is provided for future extensibility.
        
        Args:
            storage_id: Storage ID to unregister
        """
        # Current implementation relies on the device daemon for cleanup
        # This method is a placeholder for future enhancements
        log.debug(f"Storage {storage_id} unregistered")

    def get_storage_stats(self) -> dict:
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
                "storage_resolver_active": True,
                "registry_available": registry is not None
            }

            return stats

        except Exception as e:
            log.warning(f"Failed to get storage stats: {e}")
            return {"error": str(e)}
