# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Storage management for remote tensors.

This module provides the StorageManager class for managing storage IDs and their lifecycle:
- Storage ID generation and tracking
- Storage-to-machine mappings and resolution
- Storage-to-tensor mappings for remote cleanup
- Storage statistics and information

StorageManager is designed to be used as a property of the Orchestrator class,
not as a global instance. It does not handle remote cleanup or orchestrator interactions.
"""

from concurrent.futures import Future
from typing import Dict, List, Set, Tuple

import torch

from ._logging import get_logger
from ._utils import get_storage_id, get_tensor_id

log = get_logger(__name__)


class StorageManager:
    """
    Manager for remote storage IDs and their machine mappings.

    Key concepts:
    - storage_id: Identifies remote memory allocation on clients
    - Uses incremental storage IDs starting from 1 (1, 2, 3, ...)
    - Maps storage_id to (machine_id, remote_type, remote_index)
    - Maps storage_id to tensor_ids for remote cleanup
    - Thread-safe storage ID generation
    """

    def __init__(self) -> None:
        # Storage ID tracking - maps storage to remote device info
        self.storage_id_to_remote_device: Dict[
            int, Tuple[str, str, int]
        ] = {}  # storage_id -> (machine_id, remote_type, remote_index)

        # Storage cache (storage_id -> Future[bytes])
        self._storage_cache: Dict[int, Future[bytes]] = {}

        # Storage ID to tensor IDs mapping for remote cleanup
        self._storage_to_tensors_map: Dict[int, Set[int]] = {}

        # Simple counter for generating incremental storage IDs (GIL-protected)
        self._storage_id_counter = 1

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

        # Track the storage ID with remote device info
        remote_device_info = (machine_id, remote_type, remote_index)
        self.storage_id_to_remote_device[storage_id] = remote_device_info

        return storage_id

    def get_remote_device_info(self, storage_id: int) -> Tuple[str, str, int]:
        """Get remote device info for a storage ID.

        Returns:
            Tuple of (machine_id, remote_type, remote_index)

        Raises:
            KeyError: If storage_id not found
        """
        return self.storage_id_to_remote_device[storage_id]

    def free_storage(self, storage_id: int) -> None:
        """Free storage by storage ID (local tracking only).

        Note: Remote cleanup is handled by the orchestrator.
        """
        self.storage_id_to_remote_device.pop(storage_id, None)
        self._storage_cache.pop(storage_id, None)
        self._storage_to_tensors_map.pop(storage_id, None)

    def cache_storage(self, storage_id: int, data_future: Future[bytes]) -> None:
        """Cache storage future by storage ID.

        Args:
            storage_id: The storage ID to cache
            data_future: Future that will resolve to raw bytes
        """
        self._storage_cache[storage_id] = data_future

    def get_cached_storage(self, storage_id: int) -> Future[bytes] | None:
        """Get cached storage future by storage ID.

        Args:
            storage_id: The storage ID to retrieve from cache

        Returns:
            Future[bytes] if cached, None if not in cache
        """
        return self._storage_cache.get(storage_id)

    def invalidate_storage_caches(self, storage_ids: List[int]) -> None:
        """Invalidate cache entries for one or more storage IDs.

        Args:
            storage_ids: List of storage IDs to remove from cache (can be single element)
        """
        for storage_id in storage_ids:
            self._storage_cache.pop(storage_id, None)

    def register_tensor(self, tensor: torch.Tensor) -> None:
        """Register a tensor as using its associated storage.

        Args:
            tensor: The tensor to register (extracts storage_id and tensor_id internally)
        """
        storage_id = get_storage_id(tensor)
        tensor_id = get_tensor_id(tensor)
        self._storage_to_tensors_map.setdefault(storage_id, set()).add(tensor_id)

    def get_tensors_for_storage(self, storage_id: int) -> Set[int]:
        """Get all tensor IDs for a given storage ID.

        Args:
            storage_id: The storage ID to get tensors for

        Returns:
            Set of tensor IDs using the storage (empty set if none)
        """
        return self._storage_to_tensors_map.get(storage_id, set())

    def get_or_create_tensor_set(self, storage_id: int) -> Set[int]:
        """Get or create the tensor set for a storage ID.

        Args:
            storage_id: The storage ID to get/create tensor set for

        Returns:
            Set of tensor IDs using the storage (creates empty set if none exists)
        """
        return self._storage_to_tensors_map.setdefault(storage_id, set())
