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
from typing import Any

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
        self._client_managers: dict[
            str, ClientManager
        ] = {}  # machine_id -> client manager

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
        idle_timeout: int | None = None,
        modal_timeout: int | None = 3600,
    ) -> ClientManager:
        """Create and register a client for a machine.

        Args:
            machine_id: Unique machine identifier
            provider: Provider type ("modal" or "mock")
            gpu_type: GPU type string (required for modal, ignored for mock)
            gpu_count: Number of GPUs (1-8, ignored for mock)
            batching: Whether to enable batching
            idle_timeout: Number of seconds of inactivity before machine pauses (optional, default None)
            modal_timeout: Timeout in seconds for modal provider (default: 3600, ignored for mock)
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
            self.storage,
            self._main_thread_waiting,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            batching=batching,
            idle_timeout=idle_timeout,
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
        """Free storage with lazy remote cleanup.

        This method marks the storage for removal but does not immediately free it.
        The actual removal (both local and remote) happens when the next client manager
        method is invoked (after _ensure_running()).

        Args:
            storage_id: Storage ID to free
        """
        machine_id, _, _ = self.storage.get_remote_device_info(storage_id)
        self._client_managers[machine_id].mark_storage_for_removal(storage_id)

    def resize_storage(self, storage_id: int, nbytes: int) -> None:
        """Resize storage with remote operation.

        Args:
            storage_id: Storage ID to resize
            nbytes: New size in bytes
        """
        # Get a tensor ID for this storage if mapping exists
        tensor_id = self.storage.get_tensor_for_storage(storage_id)
        if tensor_id is not None:
            machine_id, _, _ = self.storage.get_remote_device_info(storage_id)
            client_manager = self._client_managers[machine_id]
            client_manager.resize_storage(tensor_id, nbytes)

        # Invalidate cache for the resized storage
        self.storage.invalidate_cache(storage_id)

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

        # Get tensor ID
        tensor_id = get_tensor_id(tensor)

        # Get machine_id from storage manager
        machine_id, _, _ = self.storage.get_remote_device_info(tensor)

        # Ensure tensor exists on remote before copying
        self._materialize_tensor(tensor)

        # First try to get cached storage future
        storage_future = self.storage.get_cached_storage(tensor)

        if storage_future is None:
            # Cache miss - get data from client and cache the future
            client_manager = self._client_managers[machine_id]
            storage_future = client_manager.get_storage_data(tensor_id)
            self.storage.cache_storage(tensor, storage_future)

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

        # Fast path: check if storage is already cached and done
        storage_future = self.storage.get_cached_storage(tensor)
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

        # Get client manager
        machine_id, _, _ = self.storage.get_remote_device_info(target_tensor)
        client_manager = self._client_managers[machine_id]

        # Ensure tensor exists on remote
        self._materialize_tensor(target_tensor)

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
        client_manager.update_tensor(
            tensor_id,
            raw_data,
            list(source_tensor.shape),
            list(source_tensor.stride()),
            source_tensor.storage_offset(),
            dtype_to_str(source_tensor.dtype),
        )

        # Invalidate cache for the updated storage
        self.storage.invalidate_cache(target_tensor)

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

        # Get machine info for both tensors
        source_machine_id, _, _ = self.storage.get_remote_device_info(source_tensor)
        target_machine_id, _, _ = self.storage.get_remote_device_info(target_tensor)

        # Validate they're on the same machine
        if source_machine_id != target_machine_id:
            raise RuntimeError(
                f"Cross-machine remote transfers are not supported. "
                f"Source machine: {source_machine_id}, Target machine: {target_machine_id}. "
                f"Only CPU↔remote and same-machine transfers are allowed. Use CPU as intermediate."
            )

        # Ensure both tensors exist on remote before copying
        self._materialize_tensor(source_tensor)
        self._materialize_tensor(target_tensor)

        # Get client manager and perform copy
        client_manager = self._client_managers[source_machine_id]
        source_tensor_id = get_tensor_id(source_tensor)
        target_tensor_id = get_tensor_id(target_tensor)
        client_manager.copy_tensor(source_tensor_id, target_tensor_id)

        # Invalidate cache for the target storage since it was modified
        self.storage.invalidate_cache(target_tensor)

    def execute_aten_operation(
        self,
        op_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output_tensors: list[torch.Tensor | None] | None = None,
    ) -> list[TensorMetadata] | None:
        """Execute remote operation with tensor objects in args/kwargs.

        Args:
            op_name: Name of the operation to execute
            args: Operation args containing original tensors
            kwargs: Operation kwargs containing original tensors
            output_tensors: List of output tensors (or None for unused outputs) with proper shapes/dtypes
                           for static operations, or None for dynamic operations

        Returns:
            For dynamic operations (output_tensors=None): list[TensorMetadata] metadata with temp_id embedded
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
                # Special case: CPU scalar tensors (0-dim) are converted to Python scalars
                # This allows operations like: mycelya_tensor + torch.tensor(2.0)
                if obj.device.type == "cpu" and obj.dim() == 0:
                    tensor_mask.append(False)
                    return obj.item()  # Convert to Python scalar

                input_tensors.append(obj)
                tensor_mask.append(True)

                # Validate and get device info through storage
                tensor_device_info = self.storage.get_remote_device_info(obj)

                if remote_device_info is None:
                    remote_device_info = tensor_device_info
                elif remote_device_info != tensor_device_info:
                    raise RuntimeError(
                        f"Cannot perform operation {op_name} between different devices. "
                        f"Expected device {remote_device_info}, got {tensor_device_info}"
                    )

                # Ensure tensor exists on remote
                self._materialize_tensor(obj)
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
                    tensor_device_info = self.storage.get_remote_device_info(
                        output_tensor
                    )

                    if remote_device_info is None:
                        remote_device_info = tensor_device_info
                    elif remote_device_info != tensor_device_info:
                        raise RuntimeError(
                            f"Cannot perform operation {op_name} between different devices. "
                            f"Expected device {remote_device_info}, got {tensor_device_info}"
                        )

        client_manager = self._client_managers[
            remote_device_info[0]
        ]  # Extract machine_id from (machine_id, device_type, device_index)

        # Execute with simplified client interface
        # Filter out None outputs when creating tensor IDs (None is common in backward operations)
        output_tensor_ids = (
            [get_tensor_id(t) if t is not None else None for t in output_tensors]
            if output_tensors
            else None
        )

        result_future = client_manager.execute_aten_operation(
            op_name,
            processed_args,
            processed_kwargs,
            tensor_mask,
            output_tensor_ids,
        )

        # Static operation: register tensor mappings and return None
        if output_tensors is not None:
            for output_tensor in output_tensors:
                if output_tensor is not None:
                    self.storage.register_tensor(output_tensor)
                    self.storage.invalidate_cache(output_tensor)
            return None

        # Dynamic operation: get result and return metadata for tensor linking
        if result_future is not None:
            self._main_thread_waiting.set()
            result = result_future.result()
            self._main_thread_waiting.clear()
            return result

    def _materialize_tensor(self, tensor: torch.Tensor) -> None:
        """Ensure tensor exists on remote client using storage mapping logic.

        Logic:
        - If storage ID isn't in mapping, create empty tensor (alias_id=None)
        - If storage ID exists but not tensor ID, create tensor view (alias_id=existing tensor ID)
        - Otherwise the tensor already exists
        """
        tensor_id = get_tensor_id(tensor)

        # Get alias tensor ID to determine materialization case
        alias_tensor_id = self.storage.get_alias_tensor_id(tensor)

        if alias_tensor_id == tensor_id:
            # Tensor already exists - nothing to do
            return

        # Get client manager and device info from tensor's storage
        machine_id, device_type, device_index = self.storage.get_remote_device_info(
            tensor
        )
        client_manager = self._client_managers[machine_id]

        # Create metadata for tensor creation
        metadata = TensorMetadata(
            id=tensor_id,
            alias_id=alias_tensor_id,  # None for empty tensor, int for tensor view
            shape=list(tensor.shape),
            stride=list(tensor.stride()),
            dtype=dtype_to_str(tensor.dtype),
            storage_offset=tensor.storage_offset(),
            nbytes=tensor.untyped_storage().nbytes(),
            device_type=device_type,
            device_index=device_index,
        )

        # Use unified create_tensor method
        client_manager.create_tensor(metadata)

        # Register the tensor in storage manager
        self.storage.register_tensor(tensor)

    def link_tensors(
        self, local_tensors: list[torch.Tensor], temp_ids: list[str]
    ) -> None:
        """Link local tensors to remote tensors from temporary registry.

        Args:
            local_tensors: List of local mycelya tensors to link
            temp_ids: list of temporary IDs from remote execution

        Note: All tensors must be on the same device.
        """
        if not local_tensors or not temp_ids:
            return

        if len(local_tensors) != len(temp_ids):
            raise ValueError(
                f"Mismatch between tensors ({len(local_tensors)}) and temp IDs ({len(temp_ids)})"
            )

        # Extract tensor IDs from tensors
        tensor_ids = [get_tensor_id(tensor) for tensor in local_tensors]

        # Get the machine from the first tensor (all should be on same device)
        first_tensor = local_tensors[0]
        machine_id, _, _ = self.storage.get_remote_device_info(first_tensor)
        client_manager = self._client_managers[machine_id]

        # Delegate to client manager
        client_manager.link_tensors(tensor_ids, temp_ids)

        # Update storage manager mapping to track these linked tensors
        for tensor in local_tensors:
            # Register tensor in storage manager
            self.storage.register_tensor(tensor)
            # Invalidate cache since tensor is now linked to fresh remote data
            self.storage.invalidate_cache(tensor)

        # Clear temporary storage mappings after linking is complete
        self.storage.clear_temp_storage_map()

    def execute_function_future(self, func, args, kwargs) -> Future[Any]:
        """
        Execute a pickled function on the remote machine asynchronously.

        This method initiates an asynchronous execution of a function. The unpickling
        and tensor linking are handled by the background thread to avoid blocking the main thread.

        Args:
            func: Function to execute remotely
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            Future[Any]: Future that will resolve to the function result

        Raises:
            RuntimeError: If no machine can be inferred or client not available
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
        for tensor in pickler.tensors.values():
            self._materialize_tensor(tensor)

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

        # Get client manager for the target machine
        client_manager = self._client_managers[machine_id]

        # Install module dependencies if any were found
        if pickler.module_dependencies:
            modules_to_install = [
                pkg
                for mod in pickler.module_dependencies
                if (pkg := module_name_to_package_name(mod))
            ]
            if modules_to_install:
                log.debug(f"Installing module dependencies: {modules_to_install}")
                client_manager.pip_install(modules_to_install)

        # Execute remotely (returns Future[bytes])
        pickled_result_future = client_manager.execute_function(pickled_func)

        # Create future for final result
        final_result_future = Future()

        # Capture pickler tensors for unpickler cache
        pickler_tensors = pickler.tensors

        # Create unpickling callback for background thread
        def unpickle_and_link(pickled_result: bytes) -> Any:
            buffer = io.BytesIO(pickled_result)
            unpickler = Unpickler(buffer, machine_id, self.storage, pickler_tensors)
            result_bundle = unpickler.load()

            # Extract result from bundle (args/kwargs ignored for now)
            result = result_bundle["result"]

            # Handle tensor linking if any tensors were collected
            if unpickler.tensors_to_link:
                tensors, temp_ids = zip(*unpickler.tensors_to_link)
                self.link_tensors(list(tensors), list(temp_ids))

            return result

        # Add to the function result futures deque for this client
        func_entry = (pickled_result_future, final_result_future, unpickle_and_link)
        client_manager.function_result_futures_deque.append(func_entry)

        return final_result_future

    def execute_function(self, func, args, kwargs) -> Any:
        """
        Execute a pickled function on the remote machine synchronously.

        This method waits for the function execution to complete and returns the result directly.

        Args:
            func: Function to execute remotely
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            Function result with proper tensor linking

        Raises:
            RuntimeError: If no machine can be inferred or client not available
        """
        # Use the async version and wait for result
        result_future = self.execute_function_future(func, args, kwargs)

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
        - Resolving pending function result futures for function execution

        Future tasks may include:
        - Cache cleanup
        - Connection health checks
        - Metrics collection
        """
        while self._running_flag.is_set():
            for _machine_id, client_manager in self._client_managers.items():
                # Process background tasks for this client manager
                client_manager.process_background_tasks()

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
