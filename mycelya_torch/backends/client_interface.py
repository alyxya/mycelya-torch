# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Abstract client interface for mycelya_torch cloud providers.

This module defines the base interface that all cloud provider clients must implement,
ensuring consistent API across different backends (Modal, AWS, GCP, Azure, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch

from .._batching import RPCBatchQueue


class ClientInterface(ABC):
    """
    Abstract base class defining the interface for cloud provider clients.

    All cloud provider clients (ModalClient, AWSClient, etc.) must inherit from this
    class and implement all abstract methods to ensure consistent API across providers.
    """

    def __init__(self, gpu_type: str, machine_id: str):
        """
        Initialize the client with GPU type, machine ID, and configuration.

        Args:
            gpu_type: The GPU type (e.g., "T4", "A100-40GB")
            machine_id: Unique machine identifier
        """
        self.gpu_type = gpu_type
        self.machine_id = machine_id

        # Storage cache: storage_id -> underlying 1D uint8 CPU tensor
        self._storage_cache: Dict[int, torch.Tensor] = {}

        # Track cache statistics for debugging
        self._cache_hits = 0
        self._cache_misses = 0

        # RPC batching queue
        self._batch_queue = RPCBatchQueue(client_id=machine_id)

        # Register with orchestrator for batching (will be done in subclass start())
        self._registered_for_batching = False

    @abstractmethod
    def start(self) -> None:
        """
        Start the cloud provider's compute resources.

        This method should initialize and start the remote client,
        making it ready to accept operations.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the cloud provider's compute resources.

        This method should cleanly shutdown the remote client
        and release any associated resources.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the machine is currently running and ready.

        Returns:
            True if the machine is running and can accept operations, False otherwise
        """
        pass

    def health_check(self) -> bool:
        """
        Perform a health check on the client connection.

        Default implementation delegates to is_running(). Providers can override
        for more sophisticated health checking.

        Returns:
            True if client is healthy, False otherwise
        """
        return self.is_running()

    def reconnect(self) -> bool:
        """
        Attempt to reconnect the client.

        Default implementation stops and starts the client. Providers can override
        for more sophisticated reconnection logic.

        Returns:
            True if reconnection succeeded, False otherwise
        """
        try:
            self.stop()
            self.start()
            return self.is_running()
        except Exception:
            return False

    # Storage management methods
    @abstractmethod
    def create_storage(self, storage_id: int, nbytes: int) -> None:
        """
        Create a storage on the remote machine.

        Args:
            storage_id: Specific ID to use for the storage (required)
            nbytes: Number of bytes to allocate for the storage

        Returns:
            None
        """
        pass

    @abstractmethod
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
        """
        Update an existing storage with raw tensor data.

        Supports both full storage replacement and partial in-place updates using
        dual tensor metadata to specify source data layout and target storage view.

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

        Returns:
            None
        """
        pass

    @abstractmethod
    def _get_storage_data(
        self,
        storage_id: int,
    ) -> bytes:
        """
        Retrieve raw storage data by ID.

        Returns the complete raw untyped storage bytes. The client interface layer
        will handle tensor reconstruction from metadata and these raw bytes.

        This is a private method used internally by get_storage_tensor().

        Args:
            storage_id: The storage ID to retrieve

        Returns:
            Raw untyped storage bytes
        """
        pass

    @abstractmethod
    def _get_storage_tensor_for_cache(
        self,
        storage_id: int,
    ) -> torch.Tensor:
        """
        Retrieve storage data as a 1D uint8 CPU tensor for caching.

        This method should return the underlying storage as a 1D uint8 tensor
        that can be cached and used to create views. Implementations should
        use torch.save/torch.load for serialization.

        Args:
            storage_id: The storage ID to retrieve

        Returns:
            1D uint8 CPU tensor representing the underlying storage
        """
        pass

    def get_storage_tensor(
        self,
        storage_id: int,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
    ) -> torch.Tensor:
        """
        Retrieve storage data as a tensor with specified view parameters.

        This method implements caching by:
        1. Checking if storage_id is in cache
        2. If cached, creating view from cached tensor
        3. If not cached, making RPC and caching result

        Args:
            storage_id: The storage ID to retrieve
            shape: Tensor shape for view
            stride: Tensor stride for view
            storage_offset: Storage offset for view
            dtype: Tensor data type

        Returns:
            CPU tensor reconstructed from storage with specified view
        """
        # Check cache first
        if storage_id in self._storage_cache:
            self._cache_hits += 1

            # Get cached underlying tensor
            cached_tensor = self._storage_cache[storage_id]

            # Create view from cached tensor
            return self._create_view_from_cached_tensor(
                cached_tensor, shape, stride, storage_offset, dtype
            )

        # Cache miss - make RPC
        self._cache_misses += 1

        # Get the actual tensor from the subclass implementation
        underlying_tensor = self._get_storage_tensor_for_cache(storage_id)

        # Cache the underlying tensor
        self._storage_cache[storage_id] = underlying_tensor

        # Create view from cached tensor
        return self._create_view_from_cached_tensor(
            underlying_tensor, shape, stride, storage_offset, dtype
        )

    @abstractmethod
    def resize_storage(self, storage_id: int, nbytes: int) -> None:
        """
        Resize a storage to accommodate new byte size.

        This handles the case where resize_ needs more storage space than currently allocated.
        Only resizes if nbytes > current storage size.

        Args:
            storage_id: The storage ID to resize
            nbytes: The number of bytes needed for the new storage size

        Returns:
            None
        """
        pass

    @abstractmethod
    def remove_storage(self, storage_id: int) -> None:
        """
        Remove a storage from the remote machine.

        Args:
            storage_id: The storage ID to remove

        Returns:
            None
        """
        pass

    # Operation execution methods
    @abstractmethod
    def execute_aten_operation(
        self,
        op_name: str,
        input_tensor_metadata: List[Dict[str, Any]],
        output_storage_ids: List[Union[int, None]],
        args: List[Any],
        kwargs: Dict[str, Any],
        return_metadata: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute an aten operation on the remote machine with separated input/output specification.

        Args:
            op_name: The aten operation name to execute
            input_tensor_metadata: Metadata for reconstructing input tensors only
            output_storage_ids: List of storage IDs to update with results (all output tensors)
            args: Operation arguments (may contain tensor placeholders)
            kwargs: Operation keyword arguments (may contain tensor placeholders)
            return_metadata: If True, return output tensor metadata instead of None

        Returns:
            None for normal operations, or List[Dict] of output tensor metadata if return_metadata=True
        """
        pass

    def _create_view_from_cached_tensor(
        self,
        cached_tensor: torch.Tensor,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
    ) -> torch.Tensor:
        """
        Create a tensor copy from a cached underlying tensor.

        This method creates a copy of the data (not a view) to protect
        the cached tensor from accidental mutations.

        Args:
            cached_tensor: The cached 1D uint8 underlying tensor
            shape: Desired tensor shape
            stride: Desired tensor stride
            storage_offset: Desired storage offset
            dtype: Desired tensor data type

        Returns:
            CPU tensor copy with specified parameters
        """
        # Convert dtype string to torch.dtype
        dtype_name = dtype.replace("torch.", "")
        torch_dtype = getattr(torch, dtype_name)

        # Create temporary view from cached tensor to get the data
        temp_tensor = torch.empty(0, dtype=torch_dtype, device="cpu")
        temp_tensor.set_(cached_tensor.untyped_storage(), storage_offset, shape, stride)

        # Return a copy to protect the cache from mutations
        return temp_tensor.clone()

    # RPC batching helper methods
    def _queue_rpc(
        self,
        method_name: str,
        call_type: str,
        args: tuple,
        kwargs: dict,
        return_future: bool = False,
        invalidate_storage_ids: Optional[List[int]] = None,
    ) -> Optional[Any]:
        """
        Helper method to queue an RPC for batching.

        Args:
            method_name: Name of the RPC method to call
            call_type: "spawn" for fire-and-forget, "remote" for blocking
            args: Arguments for the RPC method
            kwargs: Keyword arguments for the RPC method
            return_future: Whether to return a Future for this call
            invalidate_storage_ids: Storage IDs to invalidate immediately (at queue time)

        Returns:
            Future object if return_future=True or call_type="remote", None otherwise
        """
        # Invalidate cache immediately for storage-modifying operations
        if invalidate_storage_ids:
            for storage_id in invalidate_storage_ids:
                if storage_id in self._storage_cache:
                    del self._storage_cache[storage_id]

        # Queue the RPC for batching
        future = self._batch_queue.enqueue_call(
            call_type=call_type,
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            return_future=return_future,
        )

        # Wake up background thread immediately for blocking calls to reduce latency
        if call_type == "remote" or return_future:
            from .._remote_orchestrator import remote_orchestrator
            remote_orchestrator.wake_batch_thread_for_blocking_rpc()

        return future

    def _register_for_batching(self) -> None:
        """Register this client with the orchestrator for batching."""
        if not self._registered_for_batching:
            from .._remote_orchestrator import remote_orchestrator

            remote_orchestrator.register_client_for_batching(self)
            self._registered_for_batching = True

    def _unregister_for_batching(self) -> None:
        """Unregister this client from the orchestrator for batching."""
        if self._registered_for_batching:
            from .._remote_orchestrator import remote_orchestrator

            remote_orchestrator.unregister_client_for_batching(self)
            self._registered_for_batching = False

    # Cache invalidation methods (updated for batching timing)
    def invalidate_storage_cache(self, storage_id: int) -> None:
        """
        Invalidate cache entry for a specific storage ID.

        This method should be called whenever a storage has been modified
        on the remote side to ensure cache consistency. With batching,
        invalidation happens at queue time to maintain correct semantics.

        Args:
            storage_id: Storage ID to invalidate
        """
        if storage_id in self._storage_cache:
            del self._storage_cache[storage_id]

    def invalidate_multiple_storage_caches(self, storage_ids: List[int]) -> None:
        """
        Invalidate cache entries for multiple storage IDs.

        This method provides efficient batch invalidation for operations
        that modify multiple storages.

        Args:
            storage_ids: List of storage IDs to invalidate
        """
        for storage_id in storage_ids:
            if storage_id in self._storage_cache:
                del self._storage_cache[storage_id]

    def clear_storage_cache(self) -> None:
        """
        Clear all cached storage data.

        This method can be used for cleanup or when the client is stopped.
        """
        self._storage_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics for debugging and monitoring.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self._storage_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    # Context manager methods (optional to override, but provide default behavior)
    def __enter__(self):
        """Context manager entry - starts the machine."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the machine."""
        self.stop()
        self.clear_storage_cache()
        self._unregister_for_batching()

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation of the client.

        Returns:
            Human-readable string describing the client state
        """
        pass
