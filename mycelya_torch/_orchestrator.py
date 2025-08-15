# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import torch

# Removed batching imports
from ._logging import get_logger
from .backends.client_base import Client

log = get_logger(__name__)


# Exception handling is done through standard RuntimeError
# Custom exceptions removed as they were not used elsewhere in the codebase


class Orchestrator:
    """Orchestrates remote execution of aten operations across remote machines.

    This class coordinates operation execution between local tensors and remote
    machines, handling tensor transfers, device communication, and distributed
    execution flow. Currently supports Modal as the primary provider.

    No background threading - all operations are synchronous.
    """

    def __init__(self):
        # Simple utility-based architecture - no service objects needed

        # Centralized client management by device index
        self._clients: Dict[int, Client] = {}  # device_index -> client

        # Orchestrator-level storage cache (storage_id -> raw_bytes)
        self._storage_cache: Dict[int, bytes] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Tensor ID to Storage ID mapping for cache coordination
        self._tensor_to_storage_map: Dict[int, int] = {}
        self._storage_to_tensors_map: Dict[int, Set[int]] = {}

    # Client management methods
    def register_client(self, device_index: int, client: Client) -> None:
        """Register a client for a specific device index."""
        if device_index in self._clients:
            # Stop existing client if it exists
            existing_client = self._clients[device_index]
            if existing_client.is_running():
                existing_client.stop()

        self._clients[device_index] = client
        log.info(f"✅ ORCHESTRATOR: Registered client for device index {device_index}")

    def unregister_client(self, device_index: int) -> None:
        """Unregister a client for a specific device index (stops and removes from registry).

        Note: This should only be used during final cleanup/destruction.
        For normal operation, use stop_client() to stop without unregistering.
        """
        if device_index in self._clients:
            client = self._clients.pop(device_index)
            if client.is_running():
                client.stop()
            log.info(
                f"✅ ORCHESTRATOR: Unregistered client for device index {device_index}"
            )

    def get_client_by_device_index(self, device_index: int) -> Client:
        """Get client by device index."""
        client = self._clients.get(device_index)
        if client is None:
            raise RuntimeError(f"No client registered for device index {device_index}")
        if not client.is_running():
            raise RuntimeError(f"Client for device index {device_index} is not running")
        return client

    def start_client(self, device_index: int) -> None:
        """Start a client by device index."""
        client = self._clients.get(device_index)
        if client is None:
            raise RuntimeError(f"No client registered for device index {device_index}")
        if not client.is_running():
            client.start()
            log.info(f"✅ ORCHESTRATOR: Started client for device index {device_index}")

    def stop_client(self, device_index: int) -> None:
        """Stop a client by device index (but keep it registered)."""
        client = self._clients.get(device_index)
        if client is not None and client.is_running():
            client.stop()
            log.info(f"✅ ORCHESTRATOR: Stopped client for device index {device_index}")

    def is_client_running(self, device_index: int) -> bool:
        """Check if a client is running for a device index."""
        client = self._clients.get(device_index)
        return client is not None and client.is_running()

    # Removed all background thread management - no more batching

    def _invalidate_cache_for_storage(self, storage_id: int) -> None:
        """Invalidate cache entry for a specific storage ID.

        Args:
            storage_id: Storage ID to remove from cache
        """
        if storage_id in self._storage_cache:
            del self._storage_cache[storage_id]
            log.debug(f"🗑️ Invalidated orchestrator cache for storage {storage_id}")

            # Note: Do NOT remove tensor-storage mappings during cache invalidation
            # The mappings should only be updated when tensors are actually created/destroyed via RPC
            # Cache invalidation is separate from tensor existence tracking

    def _get_cached_storage_data(self, storage_id: int) -> Optional[bytes]:
        """Get cached storage data by storage ID.

        Args:
            storage_id: The storage ID to retrieve from cache

        Returns:
            Raw bytes if cached, None if not in cache
        """
        if storage_id in self._storage_cache:
            self._cache_hits += 1
            log.debug(f"🎯 CACHE HIT for storage {storage_id}")
            return self._storage_cache[storage_id]
        else:
            self._cache_misses += 1
            log.debug(f"❌ CACHE MISS for storage {storage_id}")
            return None

    def _cache_storage_data(self, storage_id: int, data: bytes) -> None:
        """Cache storage data by storage ID.

        Args:
            storage_id: The storage ID to cache
            data: Raw bytes to cache
        """
        self._storage_cache[storage_id] = data
        log.debug(f"💾 CACHED storage {storage_id} ({len(data)} bytes)")

    def _invalidate_multiple_storage_caches(self, storage_ids: List[int]) -> int:
        """Invalidate cache entries for multiple storage IDs and clean up mappings.

        Args:
            storage_ids: List of storage IDs to invalidate

        Returns:
            Number of cache entries that were actually invalidated
        """
        invalidated_count = 0
        for storage_id in storage_ids:
            if storage_id in self._storage_cache:
                del self._storage_cache[storage_id]
                invalidated_count += 1
                log.debug(f"🗑️ INVALIDATED cache for storage {storage_id}")

            # Note: Do NOT clean up tensor→storage mappings during cache invalidation
            # The mappings should only be updated when tensors are actually created/destroyed via RPC
            # Cache invalidation is separate from tensor existence tracking

        return invalidated_count

    def _reconstruct_tensor_from_cached_storage(
        self,
        cached_bytes: bytes,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
    ) -> torch.Tensor:
        """Reconstruct tensor from cached storage bytes with metadata.

        Args:
            cached_bytes: Raw storage bytes from cache
            shape: Tensor shape
            stride: Tensor stride
            storage_offset: Storage offset for tensor view
            dtype: Tensor data type

        Returns:
            Reconstructed CPU tensor
        """
        # Convert dtype string to torch dtype
        torch_dtype = getattr(torch, dtype.replace("torch.", ""))

        # Create untyped storage from cached bytes
        untyped_storage = torch.UntypedStorage.from_buffer(
            cached_bytes, dtype=torch.uint8
        )

        # Create empty tensor and set storage with view parameters
        tensor = torch.empty(0, dtype=torch_dtype, device="cpu")
        tensor.set_(untyped_storage, storage_offset, shape, stride)

        return tensor

    def _register_tensor_storage_mapping(self, tensor_id: int, storage_id: int) -> None:
        """Register a mapping between tensor ID and storage ID for cache coordination.

        Args:
            tensor_id: The tensor ID (metadata hash)
            storage_id: The storage ID (storage pointer)
        """
        # Update tensor -> storage mapping
        self._tensor_to_storage_map[tensor_id] = storage_id

        # Update storage -> tensors mapping
        if storage_id not in self._storage_to_tensors_map:
            self._storage_to_tensors_map[storage_id] = set()
        self._storage_to_tensors_map[storage_id].add(tensor_id)

        log.debug(f"📋 Registered mapping: tensor {tensor_id} -> storage {storage_id}")

    def _get_storage_id_for_tensor(self, tensor_id: int) -> Optional[int]:
        """Get storage ID for a tensor ID if mapping exists.

        Args:
            tensor_id: The tensor ID to look up

        Returns:
            Storage ID if mapping exists, None otherwise
        """
        return self._tensor_to_storage_map.get(tensor_id)

    def _get_tensor_id_for_storage(self, storage_id: int) -> Optional[int]:
        """Get a tensor ID for a storage ID if mapping exists.

        Args:
            storage_id: The storage ID to look up

        Returns:
            Any tensor ID that maps to this storage, None if no mapping exists
        """
        tensor_set = self._storage_to_tensors_map.get(storage_id)
        if tensor_set:
            return next(iter(tensor_set))  # Return any tensor ID from the set
        return None

    def get_tensor_ids_for_storage_by_device(self, device_index: int, storage_id: int) -> List[int]:
        """Get all tensor IDs associated with a storage ID by device index.

        Args:
            device_index: Device index (currently not used, maintained for compatibility)
            storage_id: The storage ID to look up

        Returns:
            List of tensor IDs that map to this storage, empty list if no mapping exists
        """
        tensor_set = self._storage_to_tensors_map.get(storage_id)
        if tensor_set:
            return list(tensor_set)
        return []

    def _get_device_index_for_client(self, client) -> int:
        """Get device index for a client.

        Args:
            client: The client to get device index for

        Returns:
            Device index for the client
        """
        # Look up device index by matching client machine_id
        for device_index, existing_client in self._clients.items():
            if existing_client is client:
                return device_index
        raise RuntimeError(f"Client {client} not found in registered clients")

    def _invalidate_output_tensor_caches(self, output_tensor_ids: List[int]) -> None:
        """Invalidate cache for all output tensors of an operation.

        This is the fundamental approach: treat all output tensors as mutated and evict from cache.
        Much simpler and more robust than trying to detect in-place operations.

        Args:
            output_tensor_ids: List of tensor IDs for output tensors
        """
        storage_ids_to_invalidate = []

        for tensor_id in output_tensor_ids:
            # Get storage ID for this tensor if we have a mapping
            storage_id = self._get_storage_id_for_tensor(tensor_id)
            if storage_id is not None:
                storage_ids_to_invalidate.append(storage_id)

        # Batch invalidation optimization: remove duplicates and process efficiently
        if storage_ids_to_invalidate:
            unique_storage_ids = list(set(storage_ids_to_invalidate))
            invalidated_count = self._invalidate_multiple_storage_caches(
                unique_storage_ids
            )
            log.debug(f"🗑️ Invalidated {invalidated_count} output tensor caches")

    def _get_client_for_storage(self, storage_id: int) -> Client:
        """Get the client for a specific storage ID with validation.

        Args:
            storage_id: Storage ID to resolve to client

        Returns:
            Client: The client managing this storage

        Raises:
            RuntimeError: If storage, machine, or client not found/available
        """
        try:
            from ._storage import _storage_registry

            device_index = _storage_registry.storage_id_to_device.get(storage_id)
            if device_index is None:
                raise RuntimeError(f"No device found for storage {storage_id}")

            return self.get_client_by_device_index(device_index)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve client for storage {storage_id}: {e}"
            ) from e

    def _get_client_for_tensor_id(self, tensor_id: int) -> Client:
        """Get the client for a specific tensor ID with validation.

        Args:
            tensor_id: Tensor ID to resolve to client

        Returns:
            Client: The client managing this tensor

        Raises:
            RuntimeError: If tensor, machine, or client not found/available
        """
        try:
            from ._storage import get_tensor_device

            device_index = get_tensor_device(tensor_id)
            if device_index is None:
                raise RuntimeError(f"No device found for tensor {tensor_id}")

            return self.get_client_by_device_index(device_index)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve client for tensor {tensor_id}: {e}"
            ) from e

    def _get_validated_client_by_device_index(self, device_index: int) -> Client:
        """Get a validated client for a device index, ensuring it's running.

        Args:
            device_index: Device index to get client for

        Returns:
            Client: The validated, running client

        Raises:
            RuntimeError: If client is None or not running
        """
        return self.get_client_by_device_index(device_index)

    # Storage management methods - mirroring Client

    def get_tensor_by_id(
        self,
        tensor_id: int,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
    ) -> "torch.Tensor":
        """Get tensor data by tensor ID with specified view parameters.

        This method retrieves tensor data using tensor IDs with storage-level caching
        via tensor-to-storage mapping when available.

        Args:
            tensor_id: The tensor ID to retrieve
            shape: Tensor shape for view
            stride: Tensor stride for view
            storage_offset: Storage offset for view
            dtype: Tensor data type

        Returns:
            CPU tensor reconstructed from tensor data with specified view

        Raises:
            RuntimeError: If tensor or client not available
        """
        # Try to use storage cache via tensor→storage mapping first
        storage_id = self._get_storage_id_for_tensor(tensor_id)
        if storage_id is not None:
            # We have a mapping, try cache first
            cached_bytes = self._get_cached_storage_data(storage_id)
            if cached_bytes is not None:
                # Cache hit via mapping
                result = self._reconstruct_tensor_from_cached_storage(
                    cached_bytes, shape, stride, storage_offset, dtype
                )
                log.info(
                    f"✅ ORCHESTRATOR: Retrieved tensor {tensor_id} from cache (storage {storage_id})"
                )
                return result

        # Cache miss or no mapping - fall back to client retrieval
        client = self._get_client_for_tensor_id(tensor_id)
        result = client.get_tensor_by_id(
            tensor_id, shape, stride, storage_offset, dtype
        )

        # Try to establish mapping and cache for future requests
        try:
            # Extract storage_id from reconstructed tensor and cache/map it
            actual_storage_id = result._get_storage_id()
            raw_bytes = result.numpy().tobytes()
            self._cache_storage_data(actual_storage_id, raw_bytes)
            self._register_tensor_storage_mapping(tensor_id, actual_storage_id)
            log.info(
                f"✅ ORCHESTRATOR: Retrieved tensor {tensor_id}, cached as storage {actual_storage_id}"
            )
        except Exception as e:
            log.debug(f"Could not establish mapping/cache for tensor {tensor_id}: {e}")
            log.info(f"✅ ORCHESTRATOR: Retrieved tensor {tensor_id} (no caching)")

        return result

    def resize_storage(self, storage_id: int, nbytes: int) -> None:
        """Resize storage to accommodate new byte size.

        Args:
            storage_id: The storage ID to resize
            nbytes: The number of bytes needed for the new storage size

        Raises:
            RuntimeError: If storage or client not available
        """
        client = self._get_client_for_storage(storage_id)
        client.resize_storage(storage_id, nbytes)

        # Invalidate orchestrator cache for the resized storage
        self._invalidate_cache_for_storage(storage_id)
        log.info(f"✅ ORCHESTRATOR: Resized storage {storage_id} to {nbytes} bytes")

    # New tensor-based methods for the refactored architecture
    def update_tensor(
        self,
        tensor_id: int,
        storage_tensor: torch.Tensor,
        source_shape: List[int],
        source_stride: List[int],
        source_storage_offset: int,
        source_dtype: str,
    ) -> None:
        """Update tensor data by tensor ID with raw data and tensor metadata.

        Args:
            tensor_id: Tensor ID to update
            storage_tensor: CPU tensor wrapping the storage data
            source_shape: Shape of the source data
            source_stride: Stride of the source data
            source_storage_offset: Storage offset of the source data
            source_dtype: Data type of the source data
            target_shape: Shape of the target view
            target_stride: Stride of the target view
            target_storage_offset: Storage offset of the target view
            target_dtype: Data type of the target view

        Raises:
            RuntimeError: If tensor or client not available
        """
        client = self._get_client_for_tensor_id(tensor_id)
        # Convert storage tensor to raw bytes for tensor-only interface
        raw_data = storage_tensor.numpy().tobytes()

        client.update_tensor(
            tensor_id,
            raw_data,
            source_shape,
            source_stride,
            source_storage_offset,
            source_dtype,
        )

        # Try to invalidate cache by finding storage_id for this tensor_id
        # Note: This is best-effort since tensor_id to storage_id mapping may not be available
        try:
            from ._storage import get_tensor_device

            device_idx = get_tensor_device(tensor_id)
            if device_idx is not None:
                # We can't easily map tensor_id to storage_id, so we skip cache invalidation
                # for tensor-based updates. Storage-based updates handle cache invalidation properly.
                log.debug(
                    f"Cache invalidation skipped for tensor {tensor_id} (tensor_id to storage_id mapping not available)"
                )
        except Exception as e:
            log.debug(f"Could not invalidate cache for tensor {tensor_id}: {e}")

        log.info(f"✅ ORCHESTRATOR: Updated tensor {tensor_id}")

    def execute_aten_operation(
        self,
        op_name: str,
        input_tensors: List[torch.Tensor],
        output_tensors: List[torch.Tensor],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        return_metadata: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute remote operation with tensor objects passed directly.

        Args:
            op_name: Name of the operation to execute
            input_tensors: List of input mycelya tensors
            output_tensors: List of output mycelya tensors to store results
            args: Processed args with tensor IDs replacing tensors
            kwargs: Processed kwargs with tensor IDs replacing tensors
            tensor_mask: Boolean mask indicating which positions in args/kwargs had tensors
            return_metadata: If True, return output tensor metadata instead of None

        Returns:
            None for normal operations, or List[Dict] of output tensor metadata if return_metadata=True
        """
        log.info(
            f"🎯 ORCHESTRATOR: Executing {op_name} with {len(input_tensors)} input tensors and {len(output_tensors)} output tensors"
        )
        log.debug(f"Input tensor IDs: {[t._get_tensor_id() for t in input_tensors]}")
        log.debug(f"Output tensor IDs: {[t._get_tensor_id() for t in output_tensors]}")

        # Extract tensor IDs for validation and cache management
        output_tensor_ids = [tensor._get_tensor_id() for tensor in output_tensors]

        # Validate that we have input tensors
        if not input_tensors:
            raise RuntimeError(f"No input tensors provided for operation {op_name}")

        # Collect all tensor IDs for cross-device validation
        all_tensor_ids = []
        for tensor in input_tensors:
            tensor_id = tensor._get_tensor_id()
            all_tensor_ids.append(tensor_id)
        for tensor in output_tensors:
            tensor_id = tensor._get_tensor_id()
            all_tensor_ids.append(tensor_id)

        # Validate all tensor IDs are on the same device
        if all_tensor_ids:
            from ._storage import validate_cross_device_operation_tensor_ids

            validate_cross_device_operation_tensor_ids(all_tensor_ids)

        # Note: Do not proactively register tensor mappings here
        # Mappings should only be registered when tensors are actually created via RPC calls
        # to maintain sync between orchestrator mapping and server tensor registry

        # Get the client using the first input tensor's tensor ID
        tensor_id = input_tensors[0]._get_tensor_id()
        client = self._get_client_for_tensor_id(tensor_id)

        # Ensure all input tensors exist on remote before execution
        log.debug(f"Ensuring {len(input_tensors)} input tensors exist on client")
        for tensor in input_tensors:
            tensor_id = tensor._get_tensor_id()
            log.debug(f"Ensuring tensor {tensor_id} exists on client")
            self._ensure_tensor_exists_on_client(client, tensor)

        # Execute with separated input/output interface
        result = client.execute_aten_operation(
            op_name,
            input_tensors,
            output_tensors,
            args,
            kwargs,
            tensor_mask,
            return_metadata,
        )

        # Register tensor-storage mappings for output tensors
        for output_tensor in output_tensors:
            try:
                tensor_id = output_tensor._get_tensor_id()
                storage_id = output_tensor._get_storage_id()
                # Register mapping if not already known
                existing_storage = self._get_storage_id_for_tensor(tensor_id)
                if existing_storage != storage_id:
                    self._register_tensor_storage_mapping(tensor_id, storage_id)
                    log.debug(
                        f"🗺️ Registered output tensor {tensor_id} -> storage {storage_id}"
                    )
            except Exception as e:
                log.debug(f"Could not register output tensor-storage mapping: {e}")

        # Simple and robust cache invalidation: treat all output tensors as mutated
        # This approach is much simpler than trying to detect in-place operations
        self._invalidate_output_tensor_caches(output_tensor_ids)

        if return_metadata:
            log.info(f"✅ ORCHESTRATOR: Completed {op_name} with metadata return")
            return result
        else:
            log.info(f"✅ ORCHESTRATOR: Completed {op_name} with separated interface")
            return None


    # HuggingFace integration methods
    def prepare_huggingface_model_by_device(
        self,
        device_index: int,
        checkpoint: str,
        torch_dtype: str = None,
        trust_remote_code: bool = False,
    ) -> dict:
        """Prepare a HuggingFace model on remote machine by device index."""
        client = self._get_validated_client_by_device_index(device_index)
        return client.prepare_huggingface_model(
            checkpoint=checkpoint,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    def ensure_tensor_exists_by_device(
        self, device_index: int, tensor: "torch.Tensor"
    ) -> None:
        """Ensure tensor exists on remote machine by device index."""
        client = self._get_validated_client_by_device_index(device_index)
        self._ensure_tensor_exists_on_client(client, tensor)

    def _ensure_tensor_exists_on_client(self, client, tensor: "torch.Tensor") -> None:
        """Ensure tensor exists on remote client using storage mapping logic.

        Logic:
        - If storage ID isn't in mapping, call create_empty_tensor
        - If storage ID exists but not tensor ID, call create_tensor_view
        - Otherwise the tensor already exists
        """
        tensor_id = tensor._get_tensor_id()
        storage_id = tensor._get_storage_id()

        log.debug(
            f"Ensuring tensor {tensor_id} with storage {storage_id} exists on client"
        )

        # Both storage_id and tensor_id should always be valid for mycelya tensors
        # The _get_tensor_id() and _get_storage_id() methods will raise errors for non-mycelya tensors

        # Check orchestrator's storage mapping to decide what to create on remote
        if storage_id not in self._storage_to_tensors_map:
            # Storage doesn't exist - create empty tensor on remote
            log.debug(f"Creating empty tensor {tensor_id} for new storage {storage_id}")
            client.create_empty_tensor(
                tensor_id=tensor_id,
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(),
                dtype=str(tensor.dtype).replace("torch.", ""),
            )
            # Register the mapping in orchestrator
            self._register_tensor_storage_mapping(tensor_id, storage_id)
        else:
            # Storage exists - check if this specific tensor ID exists in orchestrator mapping
            if tensor_id not in self._storage_to_tensors_map[storage_id]:
                # Need to create a view of an existing tensor
                log.debug(
                    f"Creating tensor view {tensor_id} from existing storage {storage_id}"
                )
                # Find any existing tensor ID for this storage as the base
                existing_tensor_ids = self._storage_to_tensors_map[storage_id]
                base_tensor_id = next(
                    iter(existing_tensor_ids)
                )  # Get any existing tensor
                client.create_tensor_view(
                    new_tensor_id=tensor_id,
                    base_tensor_id=base_tensor_id,
                    shape=list(tensor.shape),
                    stride=list(tensor.stride()),
                    offset=tensor.storage_offset(),
                )
                # Register the new tensor in orchestrator mapping
                self._register_tensor_storage_mapping(tensor_id, storage_id)
            else:
                # Tensor already exists in orchestrator mapping, assume it exists on server
                log.debug(f"Tensor {tensor_id} already exists in orchestrator mapping")

    def link_model_tensors_by_device(
        self, device_index: int, local_storage_ids: list, parameter_names: list
    ) -> None:
        """Link model tensors on remote machine by device index."""
        client = self._get_validated_client_by_device_index(device_index)
        client.link_model_tensors(local_storage_ids, parameter_names)

    # Storage cleanup methods

    def remove_tensors_by_device(self, device_index: int, tensor_ids: list) -> None:
        """Remove tensors from remote machine by device index."""
        if not self.is_client_running(device_index):
            log.debug(
                f"Client for device index {device_index} not running, skipping tensor removal"
            )
            return

        client = self.get_client_by_device_index(device_index)
        client.remove_tensors(tensor_ids)

    def remove_tensor_from_storage_mapping_by_device(
        self, device_index: int, storage_id: int, tensor_id: int
    ) -> None:
        """Remove tensor from storage mapping by device index."""
        if not self.is_client_running(device_index):
            log.debug(
                f"Client for device index {device_index} not running, skipping storage mapping removal"
            )
            return

        client = self.get_client_by_device_index(device_index)
        client._remove_tensor_from_storage_mapping(storage_id, tensor_id)

    # Removed batch queue checking - no more batching


# Global orchestrator instance (Modal provider implementation)
orchestrator = Orchestrator()
