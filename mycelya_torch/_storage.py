# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Storage management for remote tensors.

This module provides the StorageManager class for managing storage IDs and their lifecycle:
- Storage ID generation and tracking
- Storage-to-machine mappings and resolution
- Storage statistics and information

StorageManager is designed to be used as a property of the Orchestrator class,
not as a global instance. It does not handle remote cleanup or orchestrator interactions.
"""

from typing import Dict, Optional, Tuple

from ._logging import get_logger

log = get_logger(__name__)


class StorageManager:
    """
    Manager for remote storage IDs and their machine mappings.

    Key concepts:
    - storage_id: Identifies remote memory allocation on clients
    - Uses incremental storage IDs starting from 1 (1, 2, 3, ...)
    - Maps storage_id to (machine_id, remote_type, remote_index)
    - No tensor_id tracking (handled by orchestrator)
    - No remote cleanup (handled by orchestrator)
    - Thread-safe storage ID generation
    """

    def __init__(self) -> None:
        # Storage ID tracking - maps storage to machine info
        self.storage_id_to_machine_info: Dict[
            int, Tuple[str, str, int]
        ] = {}  # storage_id -> (machine_id, remote_type, remote_index)

        # Simple counter for generating incremental storage IDs (GIL-protected)
        self._storage_id_counter = 1

        log.info("🚀 Storage manager initialized")

    def create_storage(
        self, machine_id: str, remote_type: str, remote_index: int
    ) -> int:
        """
        Create remote storage with an incremental unique ID.

        Args:
            machine_id: Machine identifier for the storage
            remote_type: Remote device type (e.g., "cuda")
            remote_index: Remote device index

        Returns:
            int: The generated storage ID on success, or 0 on failure
        """
        # Generate incremental storage ID (GIL-protected)
        storage_id = self._storage_id_counter
        self._storage_id_counter += 1

        log.info(f"🆔 GENERATED Storage ID: {storage_id}")

        # Track the storage ID with machine info
        machine_info = (machine_id, remote_type, remote_index)
        self.storage_id_to_machine_info[storage_id] = machine_info

        log.info(f"Created storage ID {storage_id} for machine {machine_id}")
        return storage_id

    def get_machine_info(self, storage_id: int) -> Tuple[str, str, int]:
        """Get machine info for a storage ID.

        Returns:
            Tuple of (machine_id, remote_type, remote_index)

        Raises:
            KeyError: If storage_id not found
        """
        return self.storage_id_to_machine_info[storage_id]

    def free_storage_with_id(self, storage_id: int) -> bool:
        """Free storage by storage ID (local tracking only).

        Note: Remote cleanup is handled by the orchestrator.
        """
        storage_id = int(storage_id)
        if storage_id == 0:  # Empty storage
            return True

        if storage_id in self.storage_id_to_machine_info:
            # Clean up local tracking
            self.storage_id_to_machine_info.pop(storage_id, None)

            log.info(f"Freed storage ID {storage_id} from local tracking")
            return True
        else:
            log.warning(f"Attempted to free unknown storage {storage_id}")
            return False
