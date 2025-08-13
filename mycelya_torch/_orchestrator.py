# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""

import atexit
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from ._batching import BatchProcessor
from ._logging import get_logger
from ._machine import RemoteMachine
from ._storage import get_machine_for_storage
from .backends.client_interface import ClientInterface

log = get_logger(__name__)


# Exception handling is done through standard RuntimeError
# Custom exceptions removed as they were not used elsewhere in the codebase


class Orchestrator:
    """Orchestrates remote execution of aten operations across remote machines.

    This class coordinates operation execution between local tensors and remote
    machines, handling tensor transfers, device communication, and distributed
    execution flow. Currently supports Modal as the primary provider.

    Also manages background thread for batching RPCs to improve performance.
    """

    def __init__(self):
        # Simple utility-based architecture - no service objects needed

        # RPC Batching System
        self._batch_clients: Set[ClientInterface] = set()
        self._batch_lock = threading.RLock()
        self._batch_thread: Optional[threading.Thread] = None
        self._batch_shutdown = threading.Event()
        self._batch_wakeup = (
            threading.Event()
        )  # Wake up thread immediately for blocking calls
        self._batch_interval = 0.1  # Process batches every 100ms

        # Orchestrator-level storage cache (storage_id -> raw_bytes)
        self._storage_cache: Dict[int, bytes] = {}
        self._cache_lock = threading.RLock()  # Dedicated lock for cache operations
        self._cache_hits = 0
        self._cache_misses = 0

        # Tensor ID to Storage ID mapping for cache coordination
        self._tensor_to_storage_map: Dict[int, int] = {}
        self._storage_to_tensors_map: Dict[int, Set[int]] = {}

        # Start background thread for batch processing
        self._start_batch_thread()

        # Register cleanup on exit
        atexit.register(self._cleanup_batch_thread)

    def _get_device_client(self, machine: "RemoteMachine"):
        """Get the active client for a specific machine."""
        return machine._client

    # Background thread management for RPC batching
    def _start_batch_thread(self) -> None:
        """Start the background thread for processing RPC batches."""
        if self._batch_thread is None or not self._batch_thread.is_alive():
            self._batch_shutdown.clear()
            self._batch_thread = threading.Thread(
                target=self._batch_processing_loop,
                name="RPCBatchProcessor",
                daemon=True,
            )
            self._batch_thread.start()
            log.info("🧵 Started RPC batch processing thread")

    def _cleanup_batch_thread(self) -> None:
        """Clean up the background batch processing thread."""
        if self._batch_thread and self._batch_thread.is_alive():
            log.info("🛑 Shutting down RPC batch processing thread")
            self._batch_shutdown.set()
            self._batch_thread.join(timeout=2.0)
            if self._batch_thread.is_alive():
                log.warning("⚠️ Batch processing thread did not shutdown cleanly")

    def _batch_processing_loop(self) -> None:
        """Main loop for the background batch processing thread."""
        log.info("🚀 RPC batch processing loop started")

        while not self._batch_shutdown.is_set():
            try:
                # Process batches for all registered clients
                with self._batch_lock:
                    clients_to_process = list(self._batch_clients)

                for client in clients_to_process:
                    try:
                        self._process_client_batch(client)
                    except Exception as e:
                        log.error(f"❌ Error processing batch for client {client}: {e}")

                # Wait for next batch interval OR immediate wakeup for blocking calls
                woken_early = self._batch_wakeup.wait(self._batch_interval)
                if woken_early:
                    # Clear the event for next time and process immediately
                    self._batch_wakeup.clear()
                    log.debug("🚀 Background thread woken early for blocking RPC")

            except Exception as e:
                log.error(f"❌ Error in batch processing loop: {e}")

        log.info("🏁 RPC batch processing loop terminated")

    def _process_client_batch(self, client: ClientInterface) -> None:
        """Process a batch of RPCs for a specific client."""
        if not hasattr(client, "_batch_queue"):
            return

        batch = client._batch_queue.get_batch()
        if not batch:
            return

        try:
            # Execute the batch
            result = BatchProcessor.execute_batch(client._server_instance, batch)

            log.debug(
                f"📊 Batch processed for {client}: "
                f"{result.success_count} success, {result.error_count} errors, "
                f"{result.execution_time:.3f}s"
            )

        except Exception as e:
            log.error(f"❌ Batch execution failed for client {client}: {e}")

            # Cancel all futures in the batch
            for call in batch:
                if call.future and not call.future.done():
                    call.future.set_exception(e)

    def register_client_for_batching(self, client: ClientInterface) -> None:
        """Register a client for RPC batching."""
        with self._batch_lock:
            self._batch_clients.add(client)
            log.info(f"📝 Registered client for batching: {client}")

    def unregister_client_for_batching(self, client: ClientInterface) -> None:
        """Unregister a client from RPC batching."""
        with self._batch_lock:
            self._batch_clients.discard(client)
            log.info(f"🗑️ Unregistered client from batching: {client}")

    def wake_batch_thread_for_blocking_rpc(self) -> None:
        """Wake up the background thread immediately for processing blocking RPCs."""
        if self._batch_thread and self._batch_thread.is_alive():
            self._batch_wakeup.set()
            log.debug("💨 Signaled batch thread to wake up for blocking RPC")

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get statistics about RPC batching across all clients."""
        with self._batch_lock:
            stats = {
                "registered_clients": len(self._batch_clients),
                "batch_interval": self._batch_interval,
                "thread_alive": self._batch_thread.is_alive()
                if self._batch_thread
                else False,
                "clients": [],
            }

            for client in self._batch_clients:
                if hasattr(client, "_batch_queue"):
                    client_stats = client._batch_queue.get_stats()
                    stats["clients"].append(client_stats)

            return stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator cache statistics and diagnostics.

        Returns:
            Dictionary containing detailed cache statistics including mappings and efficiency metrics
        """
        with self._cache_lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

            # Calculate total cache memory usage
            total_bytes = sum(len(data) for data in self._storage_cache.values())
            cache_entries = len(self._storage_cache)

            return {
                "cache_entries": cache_entries,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate_percent": round(hit_rate * 100, 2),
                "total_requests": total_requests,
                "total_bytes_cached": total_bytes,
                "average_bytes_per_entry": round(total_bytes / cache_entries, 2)
                if cache_entries > 0
                else 0.0,
                "tensor_storage_mappings": len(self._tensor_to_storage_map),
                "storage_tensor_mappings": len(self._storage_to_tensors_map),
                "mapping_efficiency": round(
                    len(self._tensor_to_storage_map) / cache_entries, 2
                )
                if cache_entries > 0
                else 0.0,
            }

    def clear_cache(self) -> None:
        """Clear orchestrator cache and reset statistics."""
        with self._cache_lock:
            self._storage_cache.clear()
            self._tensor_to_storage_map.clear()
            self._storage_to_tensors_map.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            log.info("🗑️ Cleared orchestrator storage cache and mappings")

    def _invalidate_cache_for_storage(self, storage_id: int) -> None:
        """Invalidate cache entry for a specific storage ID.

        Args:
            storage_id: Storage ID to remove from cache
        """
        with self._cache_lock:
            if storage_id in self._storage_cache:
                del self._storage_cache[storage_id]
                log.debug(f"🗑️ Invalidated orchestrator cache for storage {storage_id}")

                # Also remove tensor-storage mappings for this storage
                if storage_id in self._storage_to_tensors_map:
                    tensor_ids = self._storage_to_tensors_map.pop(storage_id)
                    for tensor_id in tensor_ids:
                        self._tensor_to_storage_map.pop(tensor_id, None)

    def _get_cached_storage_data(self, storage_id: int) -> Optional[bytes]:
        """Get cached storage data by storage ID.

        Args:
            storage_id: The storage ID to retrieve from cache

        Returns:
            Raw bytes if cached, None if not in cache
        """
        with self._cache_lock:
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
        with self._cache_lock:
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
        with self._cache_lock:
            total_mappings_cleaned = 0
            for storage_id in storage_ids:
                if storage_id in self._storage_cache:
                    del self._storage_cache[storage_id]
                    invalidated_count += 1
                    log.debug(f"🗑️ INVALIDATED cache for storage {storage_id}")

                # Clean up tensor→storage mappings for this storage
                if storage_id in self._storage_to_tensors_map:
                    tensor_ids = self._storage_to_tensors_map.pop(storage_id)
                    for tensor_id in tensor_ids:
                        self._tensor_to_storage_map.pop(tensor_id, None)
                    total_mappings_cleaned += len(tensor_ids)

            if total_mappings_cleaned > 0:
                log.debug(
                    f"🧹 Cleaned up {total_mappings_cleaned} tensor→storage mappings for {invalidated_count}/{len(storage_ids)} storages"
                )

        return invalidated_count

    def _get_storage_id_from_tensor(self, tensor: torch.Tensor) -> int:
        """Extract storage ID from a tensor.

        Args:
            tensor: The tensor to extract storage ID from

        Returns:
            The storage ID
        """
        return tensor.untyped_storage().data_ptr()

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
        with self._cache_lock:
            # Update tensor -> storage mapping
            self._tensor_to_storage_map[tensor_id] = storage_id

            # Update storage -> tensors mapping
            if storage_id not in self._storage_to_tensors_map:
                self._storage_to_tensors_map[storage_id] = set()
            self._storage_to_tensors_map[storage_id].add(tensor_id)

            log.debug(
                f"📋 Registered mapping: tensor {tensor_id} -> storage {storage_id}"
            )

    def _get_storage_id_for_tensor(self, tensor_id: int) -> Optional[int]:
        """Get storage ID for a tensor ID if mapping exists.

        Args:
            tensor_id: The tensor ID to look up

        Returns:
            Storage ID if mapping exists, None otherwise
        """
        with self._cache_lock:
            return self._tensor_to_storage_map.get(tensor_id)

    def _unregister_tensor_storage_mapping(self, tensor_id: int) -> None:
        """Remove tensor-storage mapping when tensor is freed.

        Args:
            tensor_id: The tensor ID to unregister
        """
        with self._cache_lock:
            storage_id = self._tensor_to_storage_map.pop(tensor_id, None)
            if storage_id is not None:
                tensor_set = self._storage_to_tensors_map.get(storage_id)
                if tensor_set is not None:
                    tensor_set.discard(tensor_id)
                    # Clean up empty sets
                    if not tensor_set:
                        del self._storage_to_tensors_map[storage_id]
                log.debug(
                    f"📋 Unregistered mapping: tensor {tensor_id} -> storage {storage_id}"
                )

    def _get_tensor_ids_for_storage(self, storage_id: int) -> Set[int]:
        """Get all tensor IDs that map to a specific storage ID.

        Args:
            storage_id: The storage ID to look up

        Returns:
            Set of tensor IDs that map to this storage, empty set if none
        """
        with self._cache_lock:
            return self._storage_to_tensors_map.get(storage_id, set()).copy()

    def _invalidate_output_tensor_caches(
        self, output_tensor_ids: List[Optional[int]]
    ) -> None:
        """Invalidate cache for all output tensors of an operation.

        This is the fundamental approach: treat all output tensors as mutated and evict from cache.
        Much simpler and more robust than trying to detect in-place operations.

        Args:
            output_tensor_ids: List of tensor IDs for output tensors
        """
        storage_ids_to_invalidate = []

        for tensor_id in output_tensor_ids:
            if tensor_id is not None:
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

    def _get_client_for_storage(self, storage_id: int) -> ClientInterface:
        """Get the client for a specific storage ID with validation.

        Args:
            storage_id: Storage ID to resolve to client

        Returns:
            ClientInterface: The client managing this storage

        Raises:
            RuntimeError: If storage, machine, or client not found/available
        """
        try:
            machine = get_machine_for_storage(storage_id)
            return self._get_validated_client(machine)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve client for storage {storage_id}: {e}"
            ) from e

    def _get_client_for_tensor_id(self, tensor_id: int) -> ClientInterface:
        """Get the client for a specific tensor ID with validation.

        Args:
            tensor_id: Tensor ID to resolve to client

        Returns:
            ClientInterface: The client managing this tensor

        Raises:
            RuntimeError: If tensor, machine, or client not found/available
        """
        try:
            from ._storage import get_machine_for_tensor_id

            machine = get_machine_for_tensor_id(tensor_id)
            return self._get_validated_client(machine)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve client for tensor {tensor_id}: {e}"
            ) from e

    def _get_client_for_machine(self, machine: RemoteMachine) -> ClientInterface:
        """Get the client for a specific machine with validation.

        Args:
            machine: RemoteMachine to get client for

        Returns:
            ClientInterface: The validated client for this machine

        Raises:
            RuntimeError: If client not available or not running
        """
        return self._get_validated_client(machine)

    def _get_validated_client(self, machine: RemoteMachine) -> ClientInterface:
        """Get a validated client for a machine, ensuring it's running.

        Args:
            machine: RemoteMachine to get client for

        Returns:
            ClientInterface: The validated, running client

        Raises:
            RuntimeError: If client is None or not running
        """
        client = machine._client
        if client is None:
            raise RuntimeError(f"No client available for machine {machine.machine_id}")

        if not client.is_running():
            raise RuntimeError(
                f"Client for machine {machine.machine_id} is not running"
            )

        return client

    def _ensure_client_running(self, client: ClientInterface) -> None:
        """Ensure a client is running, with basic retry logic.

        Args:
            client: Client to validate

        Raises:
            RuntimeError: If client cannot be started or validated
        """
        if not client.is_running():
            log.warning(f"Client not running, attempting to start: {client}")
            try:
                client.start()
                if not client.is_running():
                    raise RuntimeError(f"Failed to start client: {client}")
                log.info(f"Successfully started client: {client}")
            except Exception as e:
                raise RuntimeError(f"Failed to start client {client}: {e}") from e

    # Storage management methods - mirroring ClientInterface
    def create_storage(self, storage_id: int, nbytes: int, device_index: int) -> None:
        """Create storage on remote machine using device index routing.

        Args:
            storage_id: Specific ID to use for the storage
            nbytes: Number of bytes to allocate
            device_index: Device index to create storage on

        Raises:
            RuntimeError: If device or client not available
        """
        from ._device import get_device_registry

        registry = get_device_registry()
        machine = registry.get_device_by_index(device_index)
        if machine is None:
            raise RuntimeError(f"No machine found for device index {device_index}")

        client = self._get_validated_client(machine)
        client.create_storage(storage_id, nbytes)
        log.info(
            f"✅ ORCHESTRATOR: Created storage {storage_id} on device {device_index}"
        )

        # Note: No cache invalidation needed for create_storage - new storage has no cached data

    def update_storage(
        self,
        storage_id: int,
        storage_tensor: torch.Tensor,
        source_shape: List[int],
        source_stride: List[int],
        source_storage_offset: int,
        source_dtype: str,
        target_shape: List[int],
        target_stride: List[int],
        target_storage_offset: int,
        target_dtype: str,
    ) -> None:
        """Update existing storage with storage tensor data.

        Args:
            storage_id: Storage ID to update
            storage_tensor: CPU tensor wrapping the storage data (tensor metadata is ignored, only untyped_storage matters)
            source_shape: Shape of the source data
            source_stride: Stride of the source data
            source_storage_offset: Storage offset of the source data
            source_dtype: Data type of the source data
            target_shape: Shape of the target view in storage
            target_stride: Stride of the target view in storage
            target_storage_offset: Storage offset of the target view in storage
            target_dtype: Data type of the target view in storage

        Raises:
            RuntimeError: If storage or client not available
        """
        client = self._get_client_for_storage(storage_id)
        client.update_storage(
            storage_id,
            storage_tensor,
            source_shape,
            source_stride,
            source_storage_offset,
            source_dtype,
            target_shape,
            target_stride,
            target_storage_offset,
            target_dtype,
        )

        # Invalidate orchestrator cache for the updated storage
        self._invalidate_cache_for_storage(storage_id)
        log.info(f"✅ ORCHESTRATOR: Updated storage {storage_id}")

    def get_storage_tensor(
        self,
        storage_id: int,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
    ) -> "torch.Tensor":
        """Get storage data as a tensor with specified view parameters.

        This is a convenience method that combines _get_storage_data() with tensor
        reconstruction. Now checks orchestrator cache first.

        Args:
            storage_id: The storage ID to retrieve
            shape: Tensor shape for view
            stride: Tensor stride for view
            storage_offset: Storage offset for view
            dtype: Tensor data type

        Returns:
            CPU tensor reconstructed from storage with specified view

        Raises:
            RuntimeError: If storage or client not available
        """
        # Check orchestrator cache first using helper method
        cached_bytes = self._get_cached_storage_data(storage_id)
        if cached_bytes is not None:
            # Cache hit - reconstruct tensor from cached bytes
            result = self._reconstruct_tensor_from_cached_storage(
                cached_bytes, shape, stride, storage_offset, dtype
            )
            log.debug(
                f"✅ ORCHESTRATOR: Retrieved cached tensor for storage {storage_id}"
            )
            return result

        # Cache miss - retrieve from client
        client = self._get_client_for_storage(storage_id)
        result = client.get_storage_tensor(
            storage_id, shape, stride, storage_offset, dtype
        )

        # Cache the result as raw bytes using helper method
        raw_bytes = result.numpy().tobytes()
        self._cache_storage_data(storage_id, raw_bytes)

        log.info(
            f"✅ ORCHESTRATOR: Retrieved and cached tensor for storage {storage_id}"
        )
        return result

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
            actual_storage_id = result.untyped_storage().data_ptr()
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

    def remove_storage(self, storage_id: int) -> None:
        """Remove storage from remote machine.

        Args:
            storage_id: The storage ID to remove

        Raises:
            RuntimeError: If storage or client not available
        """
        client = self._get_client_for_storage(storage_id)
        client.remove_storage(storage_id)

        # Invalidate orchestrator cache for the removed storage
        self._invalidate_cache_for_storage(storage_id)
        log.info(f"✅ ORCHESTRATOR: Removed storage {storage_id}")

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

    def get_tensor_data(self, tensor_id: int, tensor: torch.Tensor) -> torch.Tensor:
        """Get tensor data from remote using tensor-based approach.

        Args:
            tensor_id: The tensor ID (metadata hash)
            tensor: Reference tensor to get client routing information

        Returns:
            CPU tensor with the retrieved data

        Raises:
            RuntimeError: If tensor or client not available
        """
        # Get client by first finding the storage ID for the tensor
        storage_id = tensor.untyped_storage().data_ptr()
        client = self._get_client_for_storage(storage_id)

        # Get raw bytes from client
        raw_bytes = client.get_tensor_data(tensor_id)

        # Reconstruct CPU tensor from raw bytes
        result = torch.frombuffer(bytearray(raw_bytes), dtype=tensor.dtype).reshape(
            tensor.shape
        )

        log.info(f"✅ ORCHESTRATOR: Retrieved tensor data for tensor {tensor_id}")
        return result

    def execute_aten_operation(
        self,
        op_name: str,
        input_tensors: List[torch.Tensor],
        output_tensor_ids: List[Optional[int]],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        return_metadata: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute remote operation with tensor objects passed directly.

        Args:
            op_name: Name of the operation to execute
            input_tensors: List of input mycelya tensors
            output_tensor_ids: Tensor IDs for all output tensors (both new and reused)
            args: Processed args with tensor IDs replacing tensors
            kwargs: Processed kwargs with tensor IDs replacing tensors
            tensor_mask: Boolean mask indicating which positions in args/kwargs had tensors
            return_metadata: If True, return output tensor metadata instead of None

        Returns:
            None for normal operations, or List[Dict] of output tensor metadata if return_metadata=True
        """
        log.info(f"🎯 ORCHESTRATOR: Executing {op_name} with tensor objects")

        # Convert tensors to metadata dicts for client interface
        input_tensor_metadata_dicts = []
        for tensor in input_tensors:
            metadata_dict = {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "storage_offset": tensor.storage_offset(),
                "dtype": str(tensor.dtype).split(".")[-1],
                "tensor_id": tensor.get_metadata_hash(),
            }
            input_tensor_metadata_dicts.append(metadata_dict)

        # Validate that we have input tensors
        if not input_tensors:
            raise RuntimeError(f"No input tensors provided for operation {op_name}")

        # Collect all tensor IDs for cross-device validation
        all_tensor_ids = []
        for tensor in input_tensors:
            tensor_id = tensor.get_metadata_hash()
            if tensor_id is not None:
                all_tensor_ids.append(tensor_id)
        for tensor_id in output_tensor_ids:
            if tensor_id is not None:
                all_tensor_ids.append(tensor_id)

        # Validate all tensor IDs are on the same device
        if all_tensor_ids:
            from ._storage import validate_cross_device_operation_tensor_ids

            validate_cross_device_operation_tensor_ids(all_tensor_ids)

        # Proactive tensor-storage mapping registration for better cache coordination
        for tensor in input_tensors:
            try:
                tensor_id = tensor.get_metadata_hash()
                storage_id = tensor.untyped_storage().data_ptr()
                if tensor_id is not None and storage_id is not None:
                    # Register mapping if not already known
                    existing_storage = self._get_storage_id_for_tensor(tensor_id)
                    if existing_storage != storage_id:
                        self._register_tensor_storage_mapping(tensor_id, storage_id)
            except Exception as e:
                log.debug(f"Could not register tensor-storage mapping: {e}")

        # Get the client using the first input tensor's tensor ID
        tensor_id = input_tensors[0].get_metadata_hash()
        client = self._get_client_for_tensor_id(tensor_id)

        # Execute with separated input/output interface
        result = client.execute_aten_operation(
            op_name,
            input_tensor_metadata_dicts,
            output_tensor_ids,
            args,
            kwargs,
            tensor_mask,
            return_metadata,
        )

        # Simple and robust cache invalidation: treat all output tensors as mutated
        # This approach is much simpler than trying to detect in-place operations
        self._invalidate_output_tensor_caches(output_tensor_ids)

        if return_metadata:
            log.info(f"✅ ORCHESTRATOR: Completed {op_name} with metadata return")
            return result
        else:
            log.info(f"✅ ORCHESTRATOR: Completed {op_name} with separated interface")
            return None

    def _remote_tensor_to_cpu(self, remote_tensor: torch.Tensor) -> torch.Tensor:
        """Convert remote tensor to CPU tensor by retrieving data from remote GPU."""
        if remote_tensor.device.type != "mycelya":
            raise ValueError(
                f"Expected remote tensor, got device: {remote_tensor.device}"
            )

        # Get device registry to find the machine
        from ._device import get_device_registry

        registry = get_device_registry()
        machine = registry.get_device_by_index(remote_tensor.device.index)

        if machine is None:
            raise RuntimeError(
                f"No RemoteMachine found for remote device index {remote_tensor.device.index}"
            )

        # Get tensor data using storage ID with internal client resolution
        storage_id = remote_tensor.untyped_storage().data_ptr()

        # Get tensor data from remote storage using new interface
        return self.get_storage_tensor(
            storage_id,
            shape=list(remote_tensor.shape),
            stride=list(remote_tensor.stride()),
            storage_offset=remote_tensor.storage_offset(),
            dtype=str(remote_tensor.dtype),
        )

    def remove_tensor_from_remote(
        self, storage_id: int, machine: "RemoteMachine"
    ) -> bool:
        """Remove a tensor from remote storage."""
        try:
            # Use internal client resolution for consistent error handling
            client = self._get_validated_client(machine)
            client.remove_storage(storage_id)
            log.info(f"✅ ORCHESTRATOR: Removed storage {storage_id} from remote")
            return True
        except Exception as e:
            log.warning(f"Failed to remove storage {storage_id}: {e}")
            return False


# Global orchestrator instance (Modal provider implementation)
orchestrator = Orchestrator()
