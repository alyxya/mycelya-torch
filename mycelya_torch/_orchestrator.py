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
from ._utils import dtype_to_str, get_storage_id, get_tensor_id, map_args_kwargs
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

    # Storage management methods

    def _get_tensor_id_for_storage(self, storage_id: int) -> Optional[int]:
        """Get any tensor ID for a storage ID if mapping exists."""
        tensor_set = self._storage_to_tensors_map.get(storage_id)
        return next(iter(tensor_set)) if tensor_set else None

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
        # Get a tensor ID for this storage if mapping exists
        tensor_id = self._get_tensor_id_for_storage(storage_id)
        if tensor_id is not None:
            machine_id, _, _ = self.storage.get_remote_device_info(storage_id)
            client = self._clients[machine_id]
            client.resize_storage(tensor_id, nbytes)

        # Invalidate cache for the resized storage
        self.storage.invalidate_storage_caches([storage_id])

    # Tensor methods

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
        
        # Ensure tensor exists on remote before copying
        self._maybe_create_tensor(tensor)

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
        source_tensor: torch.Tensor,
    ) -> None:
        """Ensure target tensor exists on remote and update it with source data.

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
        self._maybe_create_tensor(target_tensor)

        # Get tensor ID and prepare data for update
        tensor_id = get_tensor_id(target_tensor)
        # Get the full storage bytes, not just the tensor view bytes
        storage = source_tensor.untyped_storage()
        storage_tensor = torch.empty(0, dtype=torch.uint8, device=source_tensor.device)
        storage_tensor.set_(
            storage, storage_offset=0, size=(storage.nbytes(),), stride=(1,)
        )
        raw_data = storage_tensor.detach().numpy().tobytes()

        # Update tensor with source data
        client.update_tensor(
            tensor_id,
            raw_data,
            list(source_tensor.shape),
            list(source_tensor.stride()),
            source_tensor.storage_offset(),
            dtype_to_str(source_tensor.dtype),
        )

        # Invalidate cache for the updated storage
        self.storage.invalidate_storage_caches([storage_id])

    def copy_tensor(
        self,
        source_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> None:
        """Copy tensor data from source to target on the same remote machine.

        Args:
            source_tensor: The mycelya tensor to copy from
            target_tensor: The mycelya tensor to copy to

        Raises:
            RuntimeError: If tensors are not on the same machine or operation fails
        """
        if source_tensor.device.type != "mycelya":
            raise RuntimeError("Source tensor must be a mycelya tensor")
        if target_tensor.device.type != "mycelya":
            raise RuntimeError("Target tensor must be a mycelya tensor")

        # Get storage info for both tensors
        source_storage_id = get_storage_id(source_tensor)
        target_storage_id = get_storage_id(target_tensor)

        # Get machine info for both tensors
        source_machine_id, _, _ = self.storage.get_remote_device_info(source_storage_id)
        target_machine_id, _, _ = self.storage.get_remote_device_info(target_storage_id)

        # Validate they're on the same machine
        if source_machine_id != target_machine_id:
            raise RuntimeError(
                f"Cross-machine remote transfers are not supported. "
                f"Source machine: {source_machine_id}, Target machine: {target_machine_id}. "
                f"Only CPU↔remote and same-machine transfers are allowed. Use CPU as intermediate."
            )

        # Ensure both tensors exist on remote before copying
        self._maybe_create_tensor(source_tensor)
        self._maybe_create_tensor(target_tensor)
        
        # Get client and perform copy
        client = self._clients[source_machine_id]
        source_tensor_id = get_tensor_id(source_tensor)
        target_tensor_id = get_tensor_id(target_tensor)
        client.copy_tensor(source_tensor_id, target_tensor_id)

        # Invalidate cache for the target storage since it was modified
        self.storage.invalidate_storage_caches([target_storage_id])

    def execute_aten_operation(
        self,
        op_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        output_tensors: Optional[List[torch.Tensor]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute remote operation with tensor objects in args/kwargs.

        Args:
            op_name: Name of the operation to execute
            args: Operation args containing original tensors
            kwargs: Operation kwargs containing original tensors
            output_tensors: List of output tensors with proper shapes/dtypes for static operations,
                           or None for dynamic operations

        Returns:
            For dynamic operations (output_tensors=None): List[Dict] metadata with temp_key embedded
            For static operations: None
        """
        # Process args/kwargs: validate, collect tensors, replace with IDs
        input_tensors, tensor_mask = [], []
        remote_device_info = None

        def process_tensor(obj):
            nonlocal remote_device_info
            if isinstance(obj, torch.Tensor):
                input_tensors.append(obj)
                tensor_mask.append(True)
                
                # Validate and get device info through storage
                storage_id = get_storage_id(obj)
                tensor_device_info = self.storage.get_remote_device_info(storage_id)
                
                if remote_device_info is None:
                    remote_device_info = tensor_device_info
                elif remote_device_info != tensor_device_info:
                    raise RuntimeError(
                        f"Cannot perform operation {op_name} between different devices. "
                        f"Expected device {remote_device_info}, got {tensor_device_info}"
                    )
                
                # Ensure tensor exists on remote
                self._maybe_create_tensor(obj)
                return get_tensor_id(obj)
            
            tensor_mask.append(False)
            return obj

        processed_args, processed_kwargs = map_args_kwargs(process_tensor, args, kwargs)
        
        # Validate output tensors separately (they don't need tensor ID processing)
        if output_tensors:
            for output_tensor in output_tensors:
                if isinstance(output_tensor, torch.Tensor):
                    storage_id = get_storage_id(output_tensor)
                    tensor_device_info = self.storage.get_remote_device_info(storage_id)
                    
                    if remote_device_info is None:
                        remote_device_info = tensor_device_info
                    elif remote_device_info != tensor_device_info:
                        raise RuntimeError(
                            f"Cannot perform operation {op_name} between different devices. "
                            f"Expected device {remote_device_info}, got {tensor_device_info}"
                        )
        
        client = self._clients[remote_device_info[0]]  # Extract machine_id from (machine_id, device_type, device_index)

        # Execute with simplified client interface
        output_tensor_ids = [get_tensor_id(t) for t in output_tensors] if output_tensors else None

        result_future = client.execute_aten_operation(
            op_name,
            processed_args,
            processed_kwargs,
            tensor_mask,
            output_tensor_ids,
        )

        # Static operation: register tensor mappings and return None
        if output_tensors is not None:
            for output_tensor in output_tensors:
                tensor_id = get_tensor_id(output_tensor)
                storage_id = get_storage_id(output_tensor)
                self._storage_to_tensors_map.setdefault(storage_id, set()).add(tensor_id)
            
            self.storage.invalidate_storage_caches([get_storage_id(t) for t in output_tensors])
            return None
        
        # Dynamic operation: get result and return metadata for tensor linking
        if result_future is not None:
            self._main_thread_waiting.set()
            result = result_future.result()
            self._main_thread_waiting.clear()
            return result

    def _maybe_create_tensor(self, tensor: torch.Tensor) -> None:
        """Ensure tensor exists on remote client using storage mapping logic.

        Logic:
        - If storage ID isn't in mapping, call create_empty_tensor
        - If storage ID exists but not tensor ID, call create_tensor_view
        - Otherwise the tensor already exists
        """
        tensor_id = get_tensor_id(tensor)
        storage_id = get_storage_id(tensor)


        # Get client from tensor's storage
        machine_id, _, _ = self.storage.get_remote_device_info(storage_id)
        client = self._clients[machine_id]

        # Check orchestrator's storage mapping to decide what to create on remote
        if storage_id not in self._storage_to_tensors_map:
            # Storage doesn't exist - create empty tensor on remote
            # Get nbytes from the tensor's untyped storage
            nbytes = tensor.untyped_storage().nbytes()

            # Get device type and index from storage
            machine_id, device_type, device_index = self.storage.get_remote_device_info(
                storage_id
            )

            client.create_empty_tensor(
                tensor_id=tensor_id,
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(),
                dtype=dtype_to_str(tensor.dtype),
                nbytes=nbytes,
                device_type=device_type,
                device_index=device_index,
            )
            # Register the mapping in orchestrator
            if storage_id not in self._storage_to_tensors_map:
                self._storage_to_tensors_map[storage_id] = set()
            self._storage_to_tensors_map[storage_id].add(tensor_id)
        else:
            # Storage exists - check if this specific tensor ID exists in orchestrator mapping
            if tensor_id not in self._storage_to_tensors_map[storage_id]:
                # Need to create a view of an existing tensor
                # Find any existing tensor ID for this storage as the base
                base_tensor_id = self._get_tensor_id_for_storage(storage_id)
                client.create_tensor_view(
                    new_tensor_id=tensor_id,
                    base_tensor_id=base_tensor_id,
                    shape=list(tensor.shape),
                    stride=list(tensor.stride()),
                    offset=tensor.storage_offset(),
                )
                # Register the new tensor in orchestrator mapping
                if storage_id not in self._storage_to_tensors_map:
                    self._storage_to_tensors_map[storage_id] = set()
                self._storage_to_tensors_map[storage_id].add(tensor_id)

    def link_tensors(self, local_tensors: List[torch.Tensor], temp_keys: List[str]) -> None:
        """Link local tensors to remote tensors from temporary registry.

        Args:
            local_tensors: List of local mycelya tensors to link
            temp_keys: List of temporary keys from remote execution

        Note: All tensors must be on the same device.
        """
        if not local_tensors or not temp_keys:
            return

        if len(local_tensors) != len(temp_keys):
            raise ValueError(
                f"Mismatch between tensors ({len(local_tensors)}) and temp keys ({len(temp_keys)})"
            )

        # Extract tensor IDs from tensors
        local_tensor_ids = [get_tensor_id(tensor) for tensor in local_tensors]

        # Get the machine from the first tensor (all should be on same device)
        first_tensor = local_tensors[0]
        storage_id = get_storage_id(first_tensor)
        machine_id, _, _ = self.storage.get_remote_device_info(storage_id)
        client = self._clients[machine_id]

        # Delegate to client
        client.link_tensors(local_tensor_ids, temp_keys)

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
