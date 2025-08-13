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
from .backends.client_interface import Client

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

        # Centralized client management by device index
        self._clients: Dict[int, Client] = {}  # device_index -> client
        self._client_lock = threading.RLock()

        # RPC Batching System
        self._batch_clients: Set[Client] = set()
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


    # Client management methods
    def register_client(self, device_index: int, client: Client) -> None:
        """Register a client for a specific device index."""
        with self._client_lock:
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
        with self._client_lock:
            if device_index in self._clients:
                client = self._clients.pop(device_index)
                if client.is_running():
                    client.stop()
                log.info(f"✅ ORCHESTRATOR: Unregistered client for device index {device_index}")

    def get_client_by_device_index(self, device_index: int) -> Client:
        """Get client by device index."""
        with self._client_lock:
            client = self._clients.get(device_index)
            if client is None:
                raise RuntimeError(f"No client registered for device index {device_index}")
            if not client.is_running():
                raise RuntimeError(f"Client for device index {device_index} is not running")
            return client

    def start_client(self, device_index: int) -> None:
        """Start a client by device index."""
        with self._client_lock:
            client = self._clients.get(device_index)
            if client is None:
                raise RuntimeError(f"No client registered for device index {device_index}")
            if not client.is_running():
                client.start()
                log.info(f"✅ ORCHESTRATOR: Started client for device index {device_index}")

    def stop_client(self, device_index: int) -> None:
        """Stop a client by device index (but keep it registered)."""
        with self._client_lock:
            client = self._clients.get(device_index)
            if client is not None and client.is_running():
                client.stop()
                log.info(f"✅ ORCHESTRATOR: Stopped client for device index {device_index}")

    def is_client_running(self, device_index: int) -> bool:
        """Check if a client is running for a device index."""
        with self._client_lock:
            client = self._clients.get(device_index)
            return client is not None and client.is_running()

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
        """Clean up the background batch processing thread and all clients."""
        # Stop all clients first
        with self._client_lock:
            for device_index, client in list(self._clients.items()):
                if client.is_running():
                    try:
                        client.stop()
                        log.info(f"🛑 Stopped client for device index {device_index}")
                    except Exception as e:
                        log.warning(f"⚠️ Error stopping client for device index {device_index}: {e}")
            self._clients.clear()

        # Then stop the batch processing thread
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

    def _process_client_batch(self, client: Client) -> None:
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

    def register_client_for_batching(self, client: Client) -> None:
        """Register a client for RPC batching."""
        with self._batch_lock:
            self._batch_clients.add(client)
            log.info(f"📝 Registered client for batching: {client}")

    def unregister_client_for_batching(self, client: Client) -> None:
        """Unregister a client from RPC batching."""
        with self._batch_lock:
            self._batch_clients.discard(client)
            log.info(f"🗑️ Unregistered client from batching: {client}")

    def wake_batch_thread_for_blocking_rpc(self) -> None:
        """Wake up the background thread immediately for processing blocking RPCs."""
        if self._batch_thread and self._batch_thread.is_alive():
            self._batch_wakeup.set()
            log.debug("💨 Signaled batch thread to wake up for blocking RPC")




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
            machine = get_machine_for_storage(storage_id)
            return self._get_validated_client(machine)
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
            from ._storage import get_machine_for_tensor_id

            machine = get_machine_for_tensor_id(tensor_id)
            return self._get_validated_client(machine)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve client for tensor {tensor_id}: {e}"
            ) from e


    def _get_validated_client(self, machine: RemoteMachine) -> Client:
        """Get a validated client for a machine, ensuring it's running.

        Args:
            machine: RemoteMachine to get client for

        Returns:
            Client: The validated, running client

        Raises:
            RuntimeError: If client is None or not running
        """
        device_index = machine.remote_index
        if device_index is None:
            raise RuntimeError(f"Machine {machine.machine_id} not registered with device registry")

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
