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
from collections import deque
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from ._device import device_manager
from ._logging import get_logger
from ._storage import StorageManager
from ._utils import dtype_to_str, get_storage_id, get_tensor_id
from .backends.base_client import Client

log = get_logger(__name__)


class Orchestrator:
    """Orchestrates remote execution of aten operations across remote machines.

    This class coordinates operation execution between local tensors and remote
    machines, handling tensor transfers, device communication, and distributed
    execution flow. Currently supports Modal as the primary provider.

    Includes background thread for periodic maintenance tasks like resolving futures.
    """

    def __init__(self):
        # Storage management
        self.storage = StorageManager()

        # Centralized client management by machine ID
        self._clients: Dict[str, Client] = {}  # machine_id -> client

        # Per-client CPU tensor futures deques for async tensor copying
        self._cpu_tensor_futures_deques: Dict[
            str, deque
        ] = {}  # machine_id -> deque of (storage_future, cpu_tensor_future, mycelya_tensor)

        # Storage ID to tensor IDs mapping for remote cleanup
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

        # Initialize CPU tensor futures deque for this client
        self._cpu_tensor_futures_deques[machine_id] = deque()

    def start_client(self, machine_id: str) -> None:
        """Start a client for the given machine."""
        client = self._clients[machine_id]
        if not client.is_running():
            client.start()

    def stop_client(self, machine_id: str) -> None:
        """Stop a client for the given machine."""
        client = self._clients[machine_id]
        if client.is_running():
            client.stop()

    def is_client_running(self, machine_id: str) -> bool:
        """Check if a client is running for the given machine."""
        client = self._clients[machine_id]
        return client.is_running()

    def _register_tensor_storage_mapping(self, tensor_id: int, storage_id: int) -> None:
        """Register a mapping between tensor ID and storage ID for remote cleanup.

        Args:
            tensor_id: The tensor ID (metadata hash)
            storage_id: The storage ID (storage pointer)
        """
        # Update storage -> tensors mapping
        if storage_id not in self._storage_to_tensors_map:
            self._storage_to_tensors_map[storage_id] = set()
        self._storage_to_tensors_map[storage_id].add(tensor_id)

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

    def _invalidate_output_tensor_caches(
        self, output_tensors: List[torch.Tensor]
    ) -> None:
        """Invalidate cache for all output tensors of an operation.

        This is the fundamental approach: treat all output tensors as mutated and evict from cache.
        Much simpler and more robust than trying to detect in-place operations.

        Args:
            output_tensors: List of output tensors
        """
        storage_ids_to_invalidate = []

        for tensor in output_tensors:
            storage_id = get_storage_id(tensor)
            storage_ids_to_invalidate.append(storage_id)

        # Batch invalidation optimization: remove duplicates and process efficiently
        if storage_ids_to_invalidate:
            unique_storage_ids = list(set(storage_ids_to_invalidate))
            self.storage.invalidate_multiple_storage_caches(unique_storage_ids)

    # Storage management methods

    def create_storage(self, nbytes: int, device_index: int) -> int:
        """Create storage using device index.

        Args:
            nbytes: Number of bytes to allocate
            device_index: Device index to resolve to machine

        Returns:
            Storage ID on success, 0 on failure
        """
        # Get machine info from device index
        machine_id, remote_type, remote_index = device_manager.get_remote_device_info(
            device_index
        )

        return self.storage.create_storage(machine_id, remote_type, remote_index)

    def free_storage(self, storage_id: int) -> None:
        """Free storage with remote cleanup.

        Args:
            storage_id: Storage ID to free
        """
        machine_id, _, _ = self.storage.get_remote_device_info(storage_id)
        tensor_set = self._storage_to_tensors_map.get(storage_id)

        if tensor_set:
            self._clients[machine_id].remove_tensors(list(tensor_set))

        self.storage.free_storage(storage_id)

    def resize_storage(self, storage_id: int, nbytes: int) -> None:
        """Resize storage with remote operation.

        Args:
            storage_id: Storage ID to resize
            nbytes: New size in bytes
        """
        machine_id, _, _ = self.storage.get_remote_device_info(storage_id)
        client = self._clients[machine_id]
        client.resize_storage(storage_id, nbytes)

        # Invalidate cache for the resized storage
        self.storage.invalidate_storage_cache(storage_id)

    # Tensor management methods

    def copy_tensor_to_cpu(self, tensor: torch.Tensor) -> Future[torch.Tensor]:
        """Copy a remote tensor to CPU asynchronously.

        This method initiates an asynchronous copy of a remote tensor to CPU. The copy
        is handled by the background thread to avoid blocking the main thread.

        Args:
            tensor: The mycelya tensor to copy to CPU

        Returns:
            Future[torch.Tensor]: Future that will resolve to the CPU tensor

        Raises:
            RuntimeError: If tensor is not a mycelya tensor or client not available
        """
        if tensor.device.type != "mycelya":
            raise RuntimeError(
                f"copy_tensor_to_cpu() can only be called on mycelya tensors, got {tensor.device.type}"
            )

        # Get tensor and storage IDs
        tensor_id = get_tensor_id(tensor)
        storage_id = get_storage_id(tensor)

        # Get machine_id from storage manager
        machine_id, _, _ = self.storage.get_remote_device_info(storage_id)

        # First try to get cached storage future
        storage_future = self.storage.get_cached_storage(storage_id)

        if storage_future is None:
            # Cache miss - get data from client and cache the future
            client = self._clients[machine_id]
            storage_future = client.get_storage_data(tensor_id)
            self.storage.cache_storage(storage_id, storage_future)

        # Create future for CPU tensor result
        cpu_tensor_future = Future()

        # Add to the CPU tensor futures deque for this client
        copy_entry = (storage_future, cpu_tensor_future, tensor)
        self._cpu_tensor_futures_deques[machine_id].append(copy_entry)

        return cpu_tensor_future

    def update_tensor(
        self,
        target_tensor: torch.Tensor,
        storage_tensor: torch.Tensor,
        source_shape: List[int],
        source_stride: List[int],
        source_storage_offset: int,
        source_dtype: str,
    ) -> None:
        """Update tensor data with raw data and tensor metadata.

        Args:
            target_tensor: Target tensor to update
            storage_tensor: CPU tensor wrapping the storage data
            source_shape: Shape of the source data
            source_stride: Stride of the source data
            source_storage_offset: Storage offset of the source data
            source_dtype: Data type of the source data

        Raises:
            RuntimeError: If tensor or client not available
        """
        tensor_id = get_tensor_id(target_tensor)
        storage_id = get_storage_id(target_tensor)
        machine_id, _, _ = self.storage.get_remote_device_info(storage_id)
        client = self._clients[machine_id]
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

        # Invalidate cache for the updated storage
        self.storage.invalidate_storage_cache(storage_id)

    def ensure_tensor_exists_and_update(
        self,
        target_tensor: torch.Tensor,
        source_tensor: torch.Tensor,
    ) -> None:
        """Ensure target tensor exists on remote and update it with source data.

        This is a convenience method that combines tensor existence checking
        with tensor updating in a single public interface.

        Args:
            target_tensor: The mycelya tensor to update (on remote device)
            source_tensor: The CPU tensor containing the data to copy

        Raises:
            RuntimeError: If tensors are not valid or operation fails
        """
        if target_tensor.device.type != "mycelya":
            raise RuntimeError("Target tensor must be a mycelya tensor")
        if source_tensor.device.type != "cpu":
            raise RuntimeError("Source tensor must be a CPU tensor")

        # Get storage info
        storage_id = get_storage_id(target_tensor)

        # Get client
        machine_id, _, _ = self.storage.get_remote_device_info(storage_id)
        client = self._clients[machine_id]

        # Ensure tensor exists on remote
        self._ensure_tensor_exists_on_client(client, target_tensor)

        # Update tensor with source data
        self.update_tensor(
            target_tensor,
            source_tensor,
            source_shape=list(source_tensor.shape),
            source_stride=list(source_tensor.stride()),
            source_storage_offset=source_tensor.storage_offset(),
            source_dtype=dtype_to_str(source_tensor.dtype),
        )

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

        # Get the client using the first input tensor's storage ID
        storage_id = get_storage_id(input_tensors[0])
        machine_id, _, _ = self.storage.get_remote_device_info(storage_id)
        client = self._clients[machine_id]

        # Ensure all input tensors exist on remote before execution
        for tensor in input_tensors:
            tensor_id = get_tensor_id(tensor)
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
                self._register_tensor_storage_mapping(tensor_id, storage_id)
            except Exception as e:
                log.debug(f"Could not register output tensor-storage mapping: {e}")

        # Simple and robust cache invalidation: treat all output tensors as mutated
        # This approach is much simpler than trying to detect in-place operations
        self._invalidate_output_tensor_caches(output_tensors)

        if return_metadata:
            return result
        else:
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

        # Check orchestrator's storage mapping to decide what to create on remote
        if storage_id not in self._storage_to_tensors_map:
            # Storage doesn't exist - create empty tensor on remote
            # Get nbytes from the tensor's untyped storage
            nbytes = tensor.untyped_storage().nbytes()

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
                base_tensor_id = next(iter(existing_tensor_ids))
                client.create_tensor_view(
                    new_tensor_id=tensor_id,
                    base_tensor_id=base_tensor_id,
                    shape=list(tensor.shape),
                    stride=list(tensor.stride()),
                    offset=tensor.storage_offset(),
                )
                # Register the new tensor in orchestrator mapping
                self._register_tensor_storage_mapping(tensor_id, storage_id)

    def _resolve_cpu_tensor_futures(self, machine_id: str) -> None:
        """Resolve pending CPU tensor futures for a client."""
        cpu_futures_deque = self._cpu_tensor_futures_deques.get(machine_id)

        while cpu_futures_deque:
            storage_future, cpu_tensor_future, mycelya_tensor = cpu_futures_deque[0]

            if not storage_future.done():
                break

            cpu_futures_deque.popleft()

            raw_bytes = storage_future.result()

            # Reconstruct CPU tensor from raw bytes
            untyped_storage = torch.UntypedStorage.from_buffer(
                raw_bytes, dtype=torch.uint8
            )
            cpu_tensor = torch.empty(0, dtype=mycelya_tensor.dtype, device="cpu")
            cpu_tensor.set_(
                untyped_storage,
                mycelya_tensor.storage_offset(),
                mycelya_tensor.shape,
                mycelya_tensor.stride(),
            )

            cpu_tensor_future.set_result(cpu_tensor)

    def _background_loop(self):
        """Background thread for batch execution and future resolution.

        Currently handles:
        - Executing pending batch operations for all clients
        - Resolving pending futures for all clients
        - Resolving pending CPU tensor futures for tensor copying

        Future tasks may include:
        - Cache cleanup
        - Connection health checks
        - Metrics collection
        """
        while True:
            for machine_id, client in self._clients.items():
                if client.is_running():
                    try:
                        # Execute any pending batched operations first
                        client.execute_batch()

                        # Then resolve any pending futures
                        client.resolve_futures()

                        # Process CPU tensor futures for this client
                        self._resolve_cpu_tensor_futures(machine_id)
                    except Exception as e:
                        log.error(f"Error in background maintenance for client: {e}")

            # Yield to the main thread before waiting
            time.sleep(0)

            # Wait up to 0.1 seconds, but wake up immediately if main thread is waiting
            self._main_thread_waiting.wait(timeout=0.1)


# Global orchestrator instance (Modal provider implementation)
orchestrator = Orchestrator()
