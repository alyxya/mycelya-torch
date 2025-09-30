# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Storage management for remote tensors.

This module provides the StorageManager class for managing storage IDs and their lifecycle:
- Storage ID generation and tracking
- Storage-to-machine mappings and resolution
- Storage-to-tensor mappings to track materialized tensors
- Storage statistics and information

StorageManager is designed to be used as a property of the Orchestrator class,
not as a global instance. It does not handle remote cleanup or orchestrator interactions.
"""

import weakref
from concurrent.futures import Future

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
    - Maps storage_id to tensor_ids to track which tensors are materialized remotely
    - Maps tensor_id to tensors using weak references for automatic cleanup
    - Thread-safe storage ID generation
    """

    def __init__(self) -> None:
        # Storage ID tracking - maps storage to remote device info
        self.storage_id_to_remote_device: dict[
            int, tuple[str, str, int]
        ] = {}  # storage_id -> (machine_id, remote_type, remote_index)

        # Storage cache (storage_id -> Future[bytes])
        self._storage_cache: dict[int, Future[bytes]] = {}

        # Storage ID to tensor IDs mapping - tracks which tensors are materialized on remote machines
        self._storage_to_tensors_map: dict[int, set[int]] = {}

        # Tensor ID to untyped storage mapping - weak references automatically remove deleted storages
        self._tensor_id_to_storage: weakref.WeakValueDictionary[
            int, torch.UntypedStorage
        ] = weakref.WeakValueDictionary()

        # Temporary ID to storage mapping - used during tensor creation from metadata
        # Maps temporary IDs (strings) to storages for handling tensor aliases
        self._temp_id_to_storage: dict[str, torch.UntypedStorage] = {}

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

    def get_remote_device_info(
        self, storage_id_or_tensor: int | torch.Tensor
    ) -> tuple[str, str, int]:
        """Get remote device info for a storage ID or tensor.

        Args:
            storage_id_or_tensor: Either a storage ID (int) or tensor (extracts storage_id internally)

        Returns:
            tuple of (machine_id, remote_type, remote_index)

        Raises:
            KeyError: If storage_id not found
        """
        if isinstance(storage_id_or_tensor, torch.Tensor):
            storage_id = get_storage_id(storage_id_or_tensor)
        else:
            storage_id = storage_id_or_tensor

        return self.storage_id_to_remote_device[storage_id]

    def free_storage(self, storage_id: int) -> list[int]:
        """Free storage by storage ID (local tracking only).

        Args:
            storage_id: The storage ID to free

        Returns:
            list of tensor IDs that were using this storage

        Note: Remote cleanup is handled by the orchestrator.
        """
        self.storage_id_to_remote_device.pop(storage_id, None)
        self._storage_cache.pop(storage_id, None)
        tensor_set = self._storage_to_tensors_map.pop(storage_id, set())
        return list(tensor_set)

    def cache_storage(self, tensor: torch.Tensor, data_future: Future[bytes]) -> None:
        """Cache storage future by tensor.

        Args:
            tensor: The tensor to cache storage for (extracts storage_id internally)
            data_future: Future that will resolve to raw bytes
        """
        storage_id = get_storage_id(tensor)
        self._storage_cache[storage_id] = data_future

    def get_cached_storage(self, tensor: torch.Tensor) -> Future[bytes] | None:
        """Get cached storage future by tensor.

        Args:
            tensor: The tensor to get cached storage for (extracts storage_id internally)

        Returns:
            Future[bytes] if cached, None if not in cache
        """
        storage_id = get_storage_id(tensor)
        return self._storage_cache.get(storage_id)

    def invalidate_cache(self, storage_id_or_tensor: int | torch.Tensor) -> None:
        """Invalidate cache entry for a storage ID or tensor.

        Args:
            storage_id_or_tensor: Either a storage ID (int) or tensor (extracts storage_id internally)
        """
        if isinstance(storage_id_or_tensor, torch.Tensor):
            storage_id = get_storage_id(storage_id_or_tensor)
        else:
            storage_id = storage_id_or_tensor

        self._storage_cache.pop(storage_id, None)

    def register_tensor(self, tensor: torch.Tensor) -> None:
        """Register a tensor as using its associated storage.

        Args:
            tensor: The tensor to register (extracts storage_id and tensor_id internally)
        """
        storage_id = get_storage_id(tensor)
        tensor_id = get_tensor_id(tensor)
        self._storage_to_tensors_map.setdefault(storage_id, set()).add(tensor_id)
        self._tensor_id_to_storage[tensor_id] = tensor.untyped_storage()

    def get_tensor_for_storage(self, storage_id: int) -> int | None:
        """Get a tensor ID for a given storage ID.

        Args:
            storage_id: The storage ID to get tensor for

        Returns:
            tensor ID using the storage, or None if no tensors use this storage
        """
        tensor_set = self._storage_to_tensors_map.get(storage_id)
        if tensor_set:
            return next(iter(tensor_set))
        return None

    def get_alias_tensor_id(self, tensor: torch.Tensor) -> int | None:
        """Get alias tensor ID for materialization logic.

        Args:
            tensor: The tensor to get alias for (extracts storage_id and tensor_id internally)

        Returns:
            - None: if tensor's storage isn't in the storage-to-tensors map (new storage case)
            - tensor_id: if the input tensor's ID is already in the map (tensor already exists)
            - first_tensor_id: otherwise, return the first tensor ID for this storage (view case)
        """
        storage_id = get_storage_id(tensor)
        tensor_id = get_tensor_id(tensor)

        # Get tensor set for this storage
        tensor_set = self._storage_to_tensors_map.get(storage_id)

        if tensor_set is None:
            # Storage not in map - new storage case
            return None
        elif tensor_id in tensor_set:
            # Tensor already exists in map
            return tensor_id
        else:
            # Storage exists but tensor doesn't - return first tensor for view creation
            return next(iter(tensor_set))

    def clear_temp_storage_map(self) -> None:
        """Clear the temporary ID to storage mapping.

        This should be called after tensor linking is complete to free up
        the temporary storage references.
        """
        self._temp_id_to_storage.clear()
