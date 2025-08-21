# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from ._logging import get_logger
from ._storage import StorageManager
from ._utils import get_storage_id, get_tensor_id
from .backends.base_client import Client

log = get_logger(__name__)


# Exception handling is done through standard RuntimeError


class Orchestrator:
    """Orchestrates remote execution of aten operations across remote machines.

    This class coordinates operation execution between local tensors and remote
    machines, handling tensor transfers, device communication, and distributed
    execution flow. Currently supports Modal as the primary provider.

    Includes background thread for periodic maintenance tasks like resolving futures.
    """

    def __init__(self):
        # Simple utility-based architecture - no service objects needed

        # Storage management
        self.storage = StorageManager()

        # Centralized client management by machine ID
        self._clients: Dict[str, Client] = {}  # machine_id -> client

        # Orchestrator-level storage cache (storage_id -> raw_bytes)
        self._storage_cache: Dict[int, bytes] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Tensor ID tracking - maps tensor ID to machine info
        self._tensor_id_to_machine_info: Dict[
            int, Tuple[str, str, int]
        ] = {}  # tensor_id -> (machine_id, remote_type, remote_index)

        # Tensor ID to Storage ID mapping for cache coordination
        self._tensor_to_storage_map: Dict[int, int] = {}
        self._storage_to_tensors_map: Dict[int, Set[int]] = {}

        # Background thread for periodic maintenance tasks
        self._main_thread_waiting = threading.Event()
        self._background_thread = threading.Thread(
            target=self._background_loop, daemon=True
        )
        self._background_thread.start()

    # Client management methods
    def create_client(
        self,
        machine_id: str,
        provider: str,
        gpu_type: str,
        batching: bool = True,
    ) -> None:
        """Create and register a client for a machine.

        Args:
            machine_id: Unique machine identifier
            provider: Provider type ("modal" or "mock")
            gpu_type: GPU type string
            batching: Whether to enable batching
        """
        if provider == "modal":
            from .backends.modal.client import ModalClient

            client = ModalClient(gpu_type, machine_id, 300, batching)
        elif provider == "mock":
            from .backends.mock.client import MockClient

            client = MockClient(gpu_type, machine_id, 300, batching)
        else:
            raise ValueError(f"Provider {provider} not implemented yet")

        # Store client mapping
        self._clients[machine_id] = client
        log.info(
            f"✅ ORCHESTRATOR: Created and registered client for machine {machine_id}"
        )

    def start_client(self, machine_id: str) -> None:
        """Start a client for the given machine."""
        client = self._clients[machine_id]
        if not client.is_running():
            client.start()
            log.info(f"✅ ORCHESTRATOR: Started client for machine {machine_id}")

    def stop_client(self, machine_id: str) -> None:
        """Stop a client for the given machine."""
        client = self._clients[machine_id]
        if client.is_running():
            client.stop()
            log.info(f"✅ ORCHESTRATOR: Stopped client for machine {machine_id}")

    def get_client(self, machine_id: str) -> Client:
        """Get client for the given machine."""
        client = self._clients[machine_id]
        if not client.is_running():
            raise RuntimeError(f"Client for machine {machine_id} is not running")
        return client

    def is_client_running(self, machine_id: str) -> bool:
        """Check if a client is running for the given machine."""
        client = self._clients[machine_id]
        return client.is_running()

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

    def _invalidate_multiple_storage_caches(self, storage_ids: List[int]) -> None:
        """Invalidate cache entries for multiple storage IDs and clean up mappings.

        Args:
            storage_ids: List of storage IDs to invalidate
        """
        for storage_id in storage_ids:
            if storage_id in self._storage_cache:
                del self._storage_cache[storage_id]
                log.debug(f"🗑️ INVALIDATED cache for storage {storage_id}")

            # Note: Do NOT clean up tensor→storage mappings during cache invalidation
            # The mappings should only be updated when tensors are actually created/destroyed via RPC
            # Cache invalidation is separate from tensor existence tracking

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
        torch_dtype = getattr(torch, dtype)

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

    def get_tensor_ids_for_storage(self, storage_id: int) -> List[int]:
        """Get all tensor IDs associated with a storage ID.

        Args:
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
            self._invalidate_multiple_storage_caches(unique_storage_ids)

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
            machine_info = self.storage.get_machine_info(storage_id)
            if machine_info is None:
                raise RuntimeError(f"No machine info found for storage {storage_id}")

            machine_id, remote_type, remote_index = machine_info
            return self.get_client(machine_id)
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
            machine_info = self.get_machine_info_for_tensor(tensor_id)
            if machine_info is None:
                raise RuntimeError(f"No machine info found for tensor {tensor_id}")

            machine_id, remote_type, remote_index = machine_info
            return self.get_client(machine_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve client for tensor {tensor_id}: {e}"
            ) from e


    # Storage management methods

    def create_storage(self, nbytes: int, device_index: int) -> int:
        """Create storage using device index.

        Args:
            nbytes: Number of bytes to allocate
            device_index: Device index to resolve to machine

        Returns:
            Storage ID on success, 0 on failure
        """
        from ._device import device_manager
        
        # Get machine info from device index
        machine_id = device_manager.get_machine_id_for_device_index(device_index)
        if machine_id is None:
            raise RuntimeError(f"No machine ID found for device index {device_index}")

        # For now, assume cuda:0 - this could be made more sophisticated later
        remote_type = "cuda"
        remote_index = 0

        return self.storage.create_storage(nbytes, machine_id, remote_type, remote_index)

    def free_storage_with_id(self, storage_id: int) -> bool:
        """Free storage by storage ID with remote cleanup.

        Args:
            storage_id: Storage ID to free

        Returns:
            True if freed successfully, False otherwise
        """
        # Get machine info for remote cleanup
        machine_info = self.storage.get_machine_info(storage_id)

        # Free from local tracking first
        success = self.storage.free_storage_with_id(storage_id)

        if success and machine_info:
            machine_id, remote_type, remote_index = machine_info
            try:
                # Perform remote cleanup
                self._cleanup_remote_storage(storage_id, machine_id)
            except Exception as e:
                log.error(f"Failed remote cleanup for storage {storage_id}: {e}")

        return success

    def resize_storage_by_id(self, storage_id: int, nbytes: int) -> bool:
        """Resize storage by storage ID with remote operation.

        Args:
            storage_id: Storage ID to resize
            nbytes: New size in bytes

        Returns:
            True if resized successfully, False otherwise
        """
        machine_info = self.storage.get_machine_info(storage_id)
        if machine_info is None:
            log.warning(f"No machine info found for storage {storage_id}")
            return False

        machine_id, remote_type, remote_index = machine_info

        try:
            # Perform remote resize
            client = self.get_client(machine_id)
            client.resize_storage(storage_id, nbytes)

            # Update local tracking
            success = self.storage.resize_storage_by_id(storage_id, nbytes)

            # Invalidate orchestrator cache for the resized storage
            self._invalidate_cache_for_storage(storage_id)

            if success:
                log.info(
                    f"✅ ORCHESTRATOR: Resized storage {storage_id} to {nbytes} bytes"
                )

            return success
        except Exception as e:
            log.error(f"Failed to resize storage {storage_id}: {e}")
            return False


    def get_machine_info_for_storage(
        self, storage_id: int
    ) -> Optional[Tuple[str, str, int]]:
        """Get machine info for a storage ID.

        Args:
            storage_id: Storage ID to query

        Returns:
            Tuple of (machine_id, remote_type, remote_index) or None if not found
        """
        return self.storage.get_machine_info(storage_id)

    # Tensor management methods

    def register_tensor_id(
        self, tensor_id: int, machine_id: str, remote_type: str, remote_index: int
    ) -> None:
        """Register a tensor ID with its machine info.

        Args:
            tensor_id: Tensor ID to register
            machine_id: Machine identifier
            remote_type: Remote device type (e.g., "cuda")
            remote_index: Remote device index
        """
        machine_info = (machine_id, remote_type, remote_index)
        self._tensor_id_to_machine_info[tensor_id] = machine_info
        log.debug(f"Registered tensor ID {tensor_id} with machine {machine_id}")

    def get_machine_info_for_tensor(
        self, tensor_id: int
    ) -> Optional[Tuple[str, str, int]]:
        """Get machine info for a tensor ID.

        Args:
            tensor_id: Tensor ID to query

        Returns:
            Tuple of (machine_id, remote_type, remote_index) or None if not found
        """
        return self._tensor_id_to_machine_info.get(tensor_id)

    def _cleanup_remote_storage(self, storage_id: int, machine_id: str) -> None:
        """Clean up storage on remote GPU device.

        Args:
            storage_id: Storage ID to clean up
            machine_id: Machine identifier for the storage
        """
        try:
            if not self.is_client_running(machine_id):
                log.warning(
                    f"Client for machine {machine_id} not running, skipping cleanup"
                )
                return

            # Get all tensor IDs associated with this storage using orchestrator
            tensor_ids = self.get_tensor_ids_for_storage(storage_id)

            if tensor_ids:
                log.info(
                    f"Cleaning up {len(tensor_ids)} tensor IDs for storage {storage_id}"
                )

                client = self.get_client(machine_id)
                # Remove tensors from remote side
                client.remove_tensors(list(tensor_ids))

                log.info(
                    f"✅ Successfully cleaned up {len(tensor_ids)} tensors for storage {storage_id}"
                )
            else:
                log.info(
                    f"No tensor IDs found for storage {storage_id} - may already be cleaned up"
                )

        except Exception as e:
            log.error(
                f"Failed remote cleanup for storage {storage_id} on machine {machine_id}: {e}"
            )

    # Legacy storage methods - mirroring Client

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
            actual_storage_id = get_storage_id(result)
            raw_bytes = result.detach().numpy().tobytes()
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
        raw_data = storage_tensor.detach().numpy().tobytes()

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
            machine_info = self.get_machine_info_for_tensor(tensor_id)
            if machine_info is not None:
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
        log.debug(f"Input tensor IDs: {[get_tensor_id(t) for t in input_tensors]}")
        log.debug(f"Output tensor IDs: {[get_tensor_id(t) for t in output_tensors]}")

        # Validate that we have input tensors
        if not input_tensors:
            raise RuntimeError(f"No input tensors provided for operation {op_name}")

        # Validate all tensors are on the same device using their device attributes
        all_tensors = input_tensors + output_tensors
        if len(all_tensors) > 1:
            first_device = all_tensors[0].device
            for tensor in all_tensors[1:]:
                if tensor.device != first_device:
                    raise RuntimeError(
                        f"Cannot perform operations between tensors on different devices. "
                        f"Found tensors on devices: {first_device} and {tensor.device}. "
                        f"Transfer tensors to the same device first: tensor.to(target_device)"
                    )

        # Extract tensor IDs for cache management
        output_tensor_ids = [get_tensor_id(tensor) for tensor in output_tensors]

        # Note: Do not proactively register tensor mappings here
        # Mappings should only be registered when tensors are actually created via RPC calls
        # to maintain sync between orchestrator mapping and server tensor registry

        # Get the client using the first input tensor's tensor ID
        tensor_id = get_tensor_id(input_tensors[0])
        client = self._get_client_for_tensor_id(tensor_id)

        # Ensure all input tensors exist on remote before execution
        log.debug(f"Ensuring {len(input_tensors)} input tensors exist on client")
        for tensor in input_tensors:
            tensor_id = get_tensor_id(tensor)
            log.debug(f"Ensuring tensor {tensor_id} exists on client")
            self._ensure_tensor_exists_on_client(client, tensor)

        # Execute with separated input/output interface
        result_future = client.execute_aten_operation(
            op_name,
            input_tensors,
            output_tensors,
            args,
            kwargs,
            tensor_mask,
            return_metadata,
        )

        # Get result from future if one was returned
        if result_future is not None:
            # Signal background thread that main thread is waiting on a future
            self._main_thread_waiting.set()
            result = result_future.result()
            self._main_thread_waiting.clear()
        else:
            result = None

        # Register tensor-storage mappings for output tensors
        for output_tensor in output_tensors:
            try:
                tensor_id = get_tensor_id(output_tensor)
                storage_id = get_storage_id(output_tensor)
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


    def _ensure_tensor_exists_on_client(self, client, tensor: "torch.Tensor") -> None:
        """Ensure tensor exists on remote client using storage mapping logic.

        Logic:
        - If storage ID isn't in mapping, call create_empty_tensor
        - If storage ID exists but not tensor ID, call create_tensor_view
        - Otherwise the tensor already exists
        """
        tensor_id = get_tensor_id(tensor)
        storage_id = get_storage_id(tensor)

        log.debug(
            f"Ensuring tensor {tensor_id} with storage {storage_id} exists on client"
        )

        # Both storage_id and tensor_id should always be valid for mycelya tensors
        # The _get_tensor_id() and _get_storage_id() methods will raise errors for non-mycelya tensors

        # Check orchestrator's storage mapping to decide what to create on remote
        if storage_id not in self._storage_to_tensors_map:
            # Storage doesn't exist - create empty tensor on remote
            log.debug(f"Creating empty tensor {tensor_id} for new storage {storage_id}")

            # Get nbytes from the tensor's untyped storage
            nbytes = tensor.untyped_storage().nbytes()

            # Import helper function for dtype conversion
            from ._utils import dtype_to_str

            client.create_empty_tensor(
                tensor_id=tensor_id,
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(),
                dtype=dtype_to_str(tensor.dtype),
                nbytes=nbytes,
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



    def _background_loop(self):
        """Background thread for batch execution and future resolution.

        Currently handles:
        - Executing pending batch operations for all clients
        - Resolving pending futures for all clients

        Future tasks may include:
        - Cache cleanup
        - Connection health checks
        - Metrics collection
        """
        while True:
            for client in self._clients.values():
                if client.is_running():
                    try:
                        # Execute any pending batched operations first
                        client.execute_batch()

                        # Then resolve any pending futures
                        client.resolve_futures()
                    except Exception as e:
                        log.error(f"Error in background maintenance for client: {e}")

            # Yield to the main thread before waiting
            time.sleep(0)
            
            # Wait up to 0.1 seconds, but wake up immediately if main thread is waiting
            self._main_thread_waiting.wait(timeout=0.1)


# Global orchestrator instance (Modal provider implementation)
orchestrator = Orchestrator()
