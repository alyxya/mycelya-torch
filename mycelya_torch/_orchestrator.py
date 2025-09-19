# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""

import atexit
import io
import threading
import time
from concurrent.futures import Future
from typing import Any, Dict, List, Set, Tuple

import torch

from ._client_manager import ClientManager
from ._device import device_manager
from ._logging import get_logger
from ._package_version import module_name_to_package_name
from ._pickle import Pickler, Unpickler
from ._storage import StorageManager
from ._utils import (
    TensorMetadata,
    dtype_to_str,
    get_storage_id,
    get_tensor_id,
    map_args_kwargs,
)

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

        # Centralized client manager management by machine ID
        self._client_managers: Dict[
            str, ClientManager
        ] = {}  # machine_id -> client manager

        # Storage ID to tensor IDs mapping for remote cleanup
        self._storage_to_tensors_map: Dict[int, Set[int]] = {}

        # Background thread for periodic maintenance tasks
        self._main_thread_waiting = threading.Event()
        self._running_flag = threading.Event()
        self._running_flag.set()  # Start as running
        self._background_thread = threading.Thread(
            target=self._background_loop, daemon=True
        )
        self._background_thread.start()

        # Register shutdown hook
        atexit.register(self._shutdown)

    # Client management methods

    def create_client(
        self,
        machine_id: str,
        provider: str,
        gpu_type: str,
        gpu_count: int,
        batching: bool = True,
        modal_timeout: int | None = None,
    ) -> ClientManager:
        """Create and register a client for a machine.

        Args:
            machine_id: Unique machine identifier
            provider: Provider type ("modal" or "mock")
            gpu_type: GPU type string (required for modal, ignored for mock)
            gpu_count: Number of GPUs (1-8, ignored for mock)
            batching: Whether to enable batching
            modal_timeout: Timeout in seconds (optional for modal, ignored for mock)
        """
        if provider == "modal":
            from .clients.modal.client import ModalClient

            client_impl = ModalClient(machine_id, timeout=modal_timeout)
        elif provider == "mock":
            from .clients.mock.client import MockClient

            client_impl = MockClient(machine_id)
        else:
            raise ValueError(f"Provider {provider} not implemented yet")

        # Create client manager wrapping the client implementation
        client_manager = ClientManager(
            client_impl,
            self._main_thread_waiting,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            batching=batching,
        )

        # Store client manager mapping
        self._client_managers[machine_id] = client_manager

        return client_manager

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
            self._client_managers[machine_id].remove_tensors(list(tensor_set))

        self.storage.free_storage(storage_id)

    def resize_storage(self, storage_id: int, nbytes: int) -> None:
        """Resize storage with remote operation.

        Args:
            storage_id: Storage ID to resize
            nbytes: New size in bytes
        """
        # Get a tensor ID for this storage if mapping exists
        tensor_set = self._storage_to_tensors_map.get(storage_id)
        if tensor_set:
            tensor_id = next(iter(tensor_set))
            machine_id, _, _ = self.storage.get_remote_device_info(storage_id)
            client = self._client_managers[machine_id]
            client.resize_storage(tensor_id, nbytes)

        # Invalidate cache for the resized storage
        self.storage.invalidate_storage_caches([storage_id])

    # Tensor methods

    def copy_tensor_to_cpu_future(self, tensor: torch.Tensor) -> Future[torch.Tensor]:
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
            client = self._client_managers[machine_id]
            storage_future = client.get_storage_data(tensor_id)
            self.storage.cache_storage(storage_id, storage_future)

        # Create future for CPU tensor result
        cpu_tensor_future = Future()

        # Add to the CPU tensor futures deque for this client
        copy_entry = (storage_future, cpu_tensor_future, tensor)
        self._client_managers[machine_id].cpu_tensor_futures_deque.append(copy_entry)

        return cpu_tensor_future

    def copy_tensor_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Copy a remote tensor to CPU synchronously.

        This method waits for the copy operation to complete and returns the CPU tensor directly.

        Args:
            tensor: The mycelya tensor to copy to CPU

        Returns:
            torch.Tensor: The CPU tensor with the copied data

        Raises:
            RuntimeError: If tensor is not a mycelya tensor or client not available
        """
        if tensor.device.type != "mycelya":
            raise RuntimeError(
                f"copy_tensor_to_cpu() can only be called on mycelya tensors, got {tensor.device.type}"
            )

        # Get tensor and storage IDs
        storage_id = get_storage_id(tensor)

        # Fast path: check if storage is already cached and done
        storage_future = self.storage.get_cached_storage(storage_id)
        if storage_future is not None and storage_future.done():
            # Direct reconstruction from cached data
            raw_bytes = storage_future.result()
            untyped_storage = torch.UntypedStorage.from_buffer(
                raw_bytes, dtype=torch.uint8
            )
            cpu_tensor = torch.empty(0, dtype=tensor.dtype, device="cpu")
            cpu_tensor.set_(
                untyped_storage,
                tensor.storage_offset(),
                tensor.shape,
                tensor.stride(),
            )
            return cpu_tensor

        # Slow path: go through async method
        result_future = self.copy_tensor_to_cpu_future(tensor)

        # Wait for result while signaling background thread to continue
        self._main_thread_waiting.set()
        result = result_future.result()
        self._main_thread_waiting.clear()
        return result

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
        client = self._client_managers[machine_id]

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
        client = self._client_managers[source_machine_id]
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
        output_tensors: List[torch.Tensor] | None = None,
    ) -> List[TensorMetadata] | None:
        """Execute remote operation with tensor objects in args/kwargs.

        Args:
            op_name: Name of the operation to execute
            args: Operation args containing original tensors
            kwargs: Operation kwargs containing original tensors
            output_tensors: List of output tensors with proper shapes/dtypes for static operations,
                           or None for dynamic operations

        Returns:
            For dynamic operations (output_tensors=None): List[TensorMetadata] metadata with temp_key embedded
            For static operations: None
        """
        # Process args/kwargs: validate, collect tensors, replace with IDs
        input_tensors, tensor_mask = [], []
        remote_device_info = (
            device_manager.get_remote_device_info(kwargs["device"].index)
            if "device" in kwargs
            else None
        )

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

            # Convert mycelya device arguments to corresponding remote device
            if isinstance(obj, torch.device) and obj.type == "mycelya":
                tensor_mask.append(False)
                device_info = device_manager.get_remote_device_info(obj.index)

                # Update remote_device_info tracking if not already set
                if remote_device_info is None:
                    remote_device_info = device_info
                elif remote_device_info != device_info:
                    raise RuntimeError(
                        f"Cannot perform operation {op_name} with mixed devices. "
                        f"Expected device {remote_device_info}, got device argument for {device_info}"
                    )

                _, remote_type, remote_index = device_info
                return torch.device(remote_type, remote_index)

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

        client = self._client_managers[
            remote_device_info[0]
        ]  # Extract machine_id from (machine_id, device_type, device_index)

        # Execute with simplified client interface
        output_tensor_ids = (
            [get_tensor_id(t) for t in output_tensors] if output_tensors else None
        )

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
                self._storage_to_tensors_map.setdefault(storage_id, set()).add(
                    tensor_id
                )

            self.storage.invalidate_storage_caches(
                [get_storage_id(t) for t in output_tensors]
            )
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

        # Get or create tensor set for this storage
        tensor_set = self._storage_to_tensors_map.setdefault(storage_id, set())

        # Check if tensor already exists
        if tensor_id in tensor_set:
            return

        # Get client and device info from tensor's storage
        machine_id, device_type, device_index = self.storage.get_remote_device_info(
            storage_id
        )
        client = self._client_managers[machine_id]

        if not tensor_set:
            # Storage doesn't exist - create empty tensor on remote
            client.create_empty_tensor(
                tensor_id=tensor_id,
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(),
                dtype=dtype_to_str(tensor.dtype),
                nbytes=tensor.untyped_storage().nbytes(),
                device_type=device_type,
                device_index=device_index,
            )
        else:
            # Storage exists - create view of existing tensor
            base_tensor_id = next(iter(tensor_set))
            client.create_tensor_view(
                new_tensor_id=tensor_id,
                base_tensor_id=base_tensor_id,
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                offset=tensor.storage_offset(),
            )

        # Register the tensor in orchestrator mapping
        tensor_set.add(tensor_id)

    def link_tensors(
        self, local_tensors: List[torch.Tensor], temp_keys: List[str]
    ) -> None:
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
        client = self._client_managers[machine_id]

        # Delegate to client
        client.link_tensors(local_tensor_ids, temp_keys)

        # Update orchestrator's storage mapping to track these linked tensors
        for tensor in local_tensors:
            tensor_id = get_tensor_id(tensor)
            storage_id = get_storage_id(tensor)

            # Register tensor ID in orchestrator mapping
            self._storage_to_tensors_map.setdefault(storage_id, set()).add(tensor_id)

    def execute_function(self, func, args, kwargs) -> Any:
        """
        Execute a pickled function on the remote machine.

        Args:
            func: Function to execute remotely
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            Function result with proper tensor linking
        """
        # Create function bundle and pickle it

        func_bundle = {
            "function": func,
            "args": args,
            "kwargs": kwargs,
        }
        buffer = io.BytesIO()
        pickler = Pickler(buffer, self.storage)
        pickler.dump(func_bundle)
        pickled_func = buffer.getvalue()

        # Handle tensor creation for any tensors collected by pickler
        for tensor in pickler.tensors:
            self._maybe_create_tensor(tensor)

        # Get machine_id from pickler (inferred during pickling)
        machine_id = pickler.machine_id
        if machine_id is None:
            # No mycelya objects found - try to infer from single client
            if len(self._client_managers) == 1:
                machine_id = next(iter(self._client_managers))
            else:
                raise RuntimeError(
                    f"No mycelya tensors or devices found in function arguments. "
                    f"Remote execution requires at least one mycelya object to determine target machine, "
                    f"or exactly one client to exist (found {len(self._client_managers)} clients)."
                )

        # Get client for the target machine
        client = self._client_managers[machine_id]

        # Install module dependencies if any were found
        if pickler.module_dependencies:
            modules_to_install = [
                pkg for mod in pickler.module_dependencies
                if (pkg := module_name_to_package_name(mod))
            ]
            if modules_to_install:
                log.debug(f"Installing module dependencies: {modules_to_install}")
                client.pip_install(modules_to_install)

        # Execute remotely
        result_future = client.execute_function(pickled_func)
        pickled_result = result_future.result()

        # Unpickle result with proper tensor linking

        buffer = io.BytesIO(pickled_result)
        unpickler = Unpickler(buffer, machine_id)
        result = unpickler.load()

        # Handle tensor linking if any tensors were collected
        if unpickler.tensors_to_link:
            tensors, temp_keys = zip(*unpickler.tensors_to_link)
            self.link_tensors(list(tensors), list(temp_keys))

        return result

    def load_huggingface_state_dicts_future(
        self,
        device_index: int,
        checkpoint: str,
        path: str = "",
    ) -> Future[Dict[str, Dict[str, TensorMetadata]]]:
        """Load a HuggingFace state dict on the remote machine associated with device_index.

        Args:
            device_index: Local mycelya device index
            checkpoint: HuggingFace model checkpoint
            path: Path within the repository to load from (default: whole repo)

        Returns:
            Future that resolves to model metadata with temp_keys
        """
        # Get machine info from device index using device manager
        machine_id, remote_device_type, remote_device_index = (
            device_manager.get_remote_device_info(device_index)
        )
        client = self._client_managers[machine_id]

        # Delegate to client's load_huggingface_state_dicts method
        return client.load_huggingface_state_dicts(
            repo=checkpoint,
            path=path,
            device_type=remote_device_type,
            device_index=remote_device_index,
        )

    def load_huggingface_state_dicts(
        self,
        device_index: int,
        checkpoint: str,
        path: str = "",
    ) -> Dict[str, Dict[str, TensorMetadata]]:
        """Load HuggingFace state dicts organized by directory on the remote machine synchronously.

        Args:
            device_index: Local mycelya device index
            checkpoint: HuggingFace model checkpoint
            path: Path within the repository to load from (default: whole repo)

        Returns:
            Dict[str, Dict[str, TensorMetadata]] mapping directory names to state dicts
        """
        # Go through async method
        result_future = self.load_huggingface_state_dicts_future(
            device_index, checkpoint, path
        )

        # Wait for result while signaling background thread to continue
        self._main_thread_waiting.set()
        result = result_future.result()
        self._main_thread_waiting.clear()
        return result

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
        while self._running_flag.is_set():
            for _machine_id, client in self._client_managers.items():
                # Process background tasks for this client
                client.process_background_tasks()

            # Yield to the main thread before waiting
            time.sleep(0)

            # Wait up to 0.1 seconds, but wake up immediately if main thread is waiting or shutdown is requested
            self._main_thread_waiting.wait(timeout=0.1)

        log.info("Background thread shutting down gracefully")
        # Signal that the background loop has finished
        self._running_flag.set()

    def _shutdown(self) -> None:
        """Gracefully shutdown the orchestrator background thread."""
        self._running_flag.clear()
        self._main_thread_waiting.set()  # Wake up the background thread

        # Wait for the background thread to finish (running flag will be set by background loop)
        self._running_flag.wait()
        # Clear the wake-up signal after finishing waiting
        self._main_thread_waiting.clear()


# Global orchestrator instance (Modal provider implementation)
orchestrator = Orchestrator()
