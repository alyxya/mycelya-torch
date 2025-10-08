# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Client manager for mycelya_torch cloud provider clients.

This module provides the ClientManager class that handles the common functionality
for all cloud provider clients, including batching, future management, and
operation coordination.
"""

import re
import threading
from collections import deque
from concurrent.futures import Future
from enum import Enum
from typing import Any

import torch

from ._logging import get_logger
from ._package_version import get_python_version, get_versioned_packages
from ._utils import TensorMetadata
from .clients.base_client import BatchCall, Client

log = get_logger(__name__)


class ClientState(Enum):
    """Enumeration of possible client states."""

    INITIALIZED = "initialized"  # Created but not started
    RUNNING = "running"  # Active and accepting operations
    PAUSED = "paused"  # Temporarily suspended (not implemented yet)
    STOPPED = "stopped"  # Shut down, no longer accepting operations


class ClientManager:
    """
    Manager class that handles common client functionality including batching,
    future management, and operation coordination.

    This class wraps a concrete client implementation and provides the
    batching, validation, and common logic while delegating actual
    implementation to the wrapped client.
    """

    def __init__(
        self,
        client: Client,
        main_thread_waiting: threading.Event,
        gpu_type: str,
        gpu_count: int,
        batching: bool = True,
    ):
        """
        Initialize the client manager with a concrete client implementation.

        Args:
            client: Concrete client implementation (ModalClient, MockClient, etc.)
            main_thread_waiting: Event to signal the background thread for coordination
            gpu_type: GPU type string (required for modal, ignored for mock)
            gpu_count: Number of GPUs (1-8, ignored for mock)
            batching: Whether to enable operation batching (default: True)
        """
        self.client = client
        self.main_thread_waiting = main_thread_waiting
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        # Default packages required for remote execution (versioned)
        default_packages = [
            "numpy",
            "torch",
            "cloudpickle",
        ]
        self.packages = get_versioned_packages(default_packages)
        self.batching = batching
        self.state = ClientState.INITIALIZED

        # Deque for storing pending futures that need to be resolved (returned to caller)
        self._pending_futures = deque()

        # Deque for storing pending results/FunctionCalls from remote operations (1:1 correspondence with futures)
        self._pending_results = deque()

        # Deque for storing pending batch calls when batching is enabled
        self._batch_calls = deque()

        # CPU tensor futures deque for async tensor copying
        self.cpu_tensor_futures_deque = deque()

        # Client stop signal event (initially set - no stop requested)
        self.stop_signal = threading.Event()
        self.stop_signal.set()

        # Stop state for stop operations (None when not stopping)
        self._stop_state = None

    @property
    def machine_id(self) -> str:
        """Get the machine ID from the client."""
        return self.client.machine_id

    def _start(self) -> None:
        """Internal method to start the cloud provider's compute resources."""
        self.client.start(
            gpu_type=self.gpu_type,
            gpu_count=self.gpu_count,
            packages=self.packages,
            python_version=get_python_version(),
        )
        self.state = ClientState.RUNNING

    def _stop(self, target_state: ClientState) -> None:
        """Internal method to signal background thread to stop the cloud provider's compute resources.

        Args:
            target_state: The state to transition to (STOPPED or PAUSED)
        """
        # Set stop state for background thread
        self._stop_state = target_state
        # Clear the event to signal stop request
        self.stop_signal.clear()
        # Wake up background thread
        self.main_thread_waiting.set()
        # Wait for background thread to set it back (confirming stop)
        self.stop_signal.wait()
        # Clear the wake-up signal after finishing waiting
        self.main_thread_waiting.clear()
        # Reset stop state
        self._stop_state = None

    def start(self) -> None:
        """Start the cloud provider's compute resources."""
        if self.state != ClientState.INITIALIZED:
            raise RuntimeError(
                f"Cannot start machine {self.machine_id} - must be in initialized state"
            )

        self._start()

    def stop(self) -> None:
        """Stop the cloud provider's compute resources."""
        if self.state == ClientState.RUNNING:
            self._stop(ClientState.STOPPED)
        elif self.state == ClientState.PAUSED:
            # Already stopped, just transition to STOPPED state
            # TODO: Clean up offload state (remove offloaded files)
            self.state = ClientState.STOPPED
        else:
            raise RuntimeError(
                f"Cannot stop machine {self.machine_id} - not currently running or paused"
            )

    def pause(self) -> None:
        """Pause the client (offload state and stop)."""
        if self.state != ClientState.RUNNING:
            raise RuntimeError(
                f"Cannot pause machine {self.machine_id} - not currently running"
            )

        # Offload tensor state to disk
        self.client.offload()

        # Stop via _stop() method with PAUSED target state
        self._stop(ClientState.PAUSED)

    def resume(self) -> None:
        """Resume the client from paused state (start and reload)."""
        if self.state != ClientState.PAUSED:
            raise RuntimeError(
                f"Cannot resume machine {self.machine_id} - not currently paused"
            )

        # Start via ClientManager's _start() method
        self._start()

        # Reload tensor state from disk
        self.client.reload()

    def is_running(self) -> bool:
        """Check if the machine is currently running and ready."""
        return self.state == ClientState.RUNNING

    def resolve_futures(self) -> None:
        """Resolve any pending futures by fetching results from the queue."""
        if not self.is_running():
            return

        # Poll for completed RPC results and resolve corresponding futures
        while self._pending_results and self._pending_futures:
            rpc_result = self._pending_results[0]  # Peek at first

            # Try to get the result without blocking
            resolved_value = self.client.get_rpc_result(rpc_result, blocking=False)
            if resolved_value is None:
                # Result not ready yet, stop processing
                break

            # Result is ready, remove from pending results
            self._pending_results.popleft()

            # Resolve futures based on batching mode
            if self.batching:
                # Batch result - iterate over list
                for res in resolved_value:
                    self._pending_futures.popleft().set_result(res)
            else:
                # Individual result
                self._pending_futures.popleft().set_result(resolved_value)

        # Also resolve CPU tensor futures
        while self.cpu_tensor_futures_deque:
            storage_future, cpu_tensor_future, mycelya_tensor = (
                self.cpu_tensor_futures_deque[0]
            )
            if not storage_future.done():
                break
            self.cpu_tensor_futures_deque.popleft()
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

    def propagate_exception_to_futures(self, exception: Exception) -> None:
        """Propagate the given exception to all pending futures."""
        # Propagate to all pending futures
        while self._pending_futures:
            future = self._pending_futures.popleft()
            future.set_exception(exception)

        # Clear pending results
        self._pending_results.clear()

        # Also propagate to CPU tensor futures
        while self.cpu_tensor_futures_deque:
            storage_future, cpu_tensor_future, mycelya_tensor = (
                self.cpu_tensor_futures_deque.popleft()
            )
            if not cpu_tensor_future.cancelled():
                cpu_tensor_future.set_exception(exception)

    def _ensure_running(self) -> None:
        """Ensure the machine is running, raise RuntimeError if not."""
        if self.state == ClientState.INITIALIZED:
            raise RuntimeError(
                f"Machine {self.machine_id} is not started. Call start() first."
            )
        elif self.state == ClientState.PAUSED:
            raise RuntimeError(
                f"Machine {self.machine_id} is paused. Call resume() first."
            )
        elif self.state == ClientState.STOPPED:
            raise RuntimeError(
                f"Machine {self.machine_id} is stopped and cannot be restarted."
            )
        # If we get here, state is RUNNING - which is what we want

    def execute_batch(self) -> None:
        """Execute any pending batch calls."""
        if not self._batch_calls:
            return

        # Pop all pending calls from deque to create batch list
        batch_to_execute = []
        while self._batch_calls:
            batch_to_execute.append(self._batch_calls.popleft())

        # Execute the batch via the client and add result to pending results
        result = self.client.execute_batch(batch_to_execute)
        self._pending_results.append(result)

    # Tensor management methods

    def create_tensor(self, metadata: TensorMetadata) -> None:
        """Create a tensor on the remote machine.

        Creates either a new empty tensor or a tensor view based on metadata.alias_id:
        - If alias_id is None: Creates new empty tensor
        - If alias_id is int: Creates tensor view using alias_id as base tensor

        Args:
            metadata: TensorMetadata containing tensor properties and creation info
        """
        self._ensure_running()

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="create_tensor",
                    args=(metadata,),
                    kwargs={},
                )
            )
        else:
            # Direct execution
            self.client.create_tensor(metadata)

    def update_tensor(
        self,
        tensor_id: int,
        raw_data: bytes,
        source_shape: list[int],
        source_stride: list[int],
        source_storage_offset: int,
        source_dtype: str,
    ) -> None:
        """Update an existing tensor with new data and source metadata."""
        self._ensure_running()

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="update_tensor",
                    args=(
                        tensor_id,
                        raw_data,
                        source_shape,
                        source_stride,
                        source_storage_offset,
                        source_dtype,
                    ),
                    kwargs={},
                )
            )
        else:
            # Direct execution
            self.client.update_tensor(
                tensor_id,
                raw_data,
                source_shape,
                source_stride,
                source_storage_offset,
                source_dtype,
            )

    def get_storage_data(self, tensor_id: int) -> Future[bytes]:
        """Get raw storage data by tensor ID."""
        self._ensure_running()

        # Create a Future for the result
        future = Future()
        # Add future to pending futures queue
        self._pending_futures.append(future)

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(method_name="get_storage_data", args=(tensor_id,), kwargs={})
            )
        else:
            # Direct execution and add result to pending results
            result = self.client.get_storage_data(tensor_id)
            self._pending_results.append(result)

        return future

    def remove_tensors(self, tensor_ids: list[int]) -> None:
        """Remove multiple tensors from the remote machine."""
        if not self.is_running():
            # During cleanup/garbage collection, machine may already be shut down
            # This is expected and should not raise an exception
            return

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(method_name="remove_tensors", args=(tensor_ids,), kwargs={})
            )
        else:
            # Direct execution
            self.client.remove_tensors(tensor_ids)

    def resize_storage(self, tensor_id: int, nbytes: int) -> None:
        """Resize the underlying storage for a tensor."""
        self._ensure_running()

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="resize_storage", args=(tensor_id, nbytes), kwargs={}
                )
            )
        else:
            # Direct execution
            self.client.resize_storage(tensor_id, nbytes)

    # Tensor copy methods

    def copy_tensor(
        self,
        source_tensor_id: int,
        target_tensor_id: int,
    ) -> None:
        """Copy tensor data from source to target on the same remote machine."""
        self._ensure_running()

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="copy_tensor",
                    args=(source_tensor_id, target_tensor_id),
                    kwargs={},
                )
            )
        else:
            # Direct execution
            self.client.copy_tensor(source_tensor_id, target_tensor_id)

    # Operation execution methods

    def execute_aten_operation(
        self,
        op_name: str,
        args: list[Any],
        kwargs: dict[str, Any],
        tensor_mask: list[bool],
        output_tensor_ids: list[int] | None = None,
    ) -> Future[list[TensorMetadata]] | None:
        """Execute an aten operation on the remote machine."""
        self._ensure_running()

        # Create future for dynamic operations
        future = None
        if output_tensor_ids is None:
            future = Future()
            self._pending_futures.append(future)

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="execute_aten_operation",
                    args=(
                        op_name,
                        args,
                        kwargs,
                        tensor_mask,
                        output_tensor_ids,
                    ),
                    kwargs={},
                )
            )
        else:
            # Direct execution and add result to pending results only for dynamic operations
            result = self.client.execute_aten_operation(
                op_name,
                args,
                kwargs,
                tensor_mask,
                output_tensor_ids,
            )
            # Only add result to pending results for dynamic operations (when output_tensor_ids is None)
            if output_tensor_ids is None:
                self._pending_results.append(result)

        return future

    def link_tensors(
        self,
        tensor_ids: list[int],
        temp_ids: list[str],
    ) -> None:
        """Link local mycelya tensor IDs to remote tensors from temporary registry."""
        self._ensure_running()

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="link_tensors",
                    args=(tensor_ids, temp_ids),
                    kwargs={},
                )
            )
        else:
            # Direct execution
            self.client.link_tensors(tensor_ids, temp_ids)

    def execute_function(self, pickled_function: bytes) -> Future[bytes]:
        """Execute a pickled function on the remote machine."""
        self._ensure_running()

        # Create a Future for the result
        future = Future()
        # Add future to pending futures queue
        self._pending_futures.append(future)

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="execute_function",
                    args=(pickled_function,),
                    kwargs={},
                )
            )
        else:
            # Direct execution and add result to pending results
            result = self.client.execute_function(pickled_function)
            self._pending_results.append(result)

        return future

    def pip_install(self, packages: list[str]) -> None:
        """Install packages using pip on the remote machine.

        This method intelligently manages the packages list:
        - If client hasn't been started yet, modifies the packages list for initial image creation
        - If client has been started, calls pip install on the running container

        Args:
            packages: List of package names to install (e.g., ["numpy", "scipy"])
        """
        if not packages:
            return

        # Version the input packages and handle overrides
        versioned_packages = get_versioned_packages(packages)

        # Create mapping of existing packages by name
        existing_packages = {}
        for pkg in self.packages:
            pkg_name = re.split(r"[<>=!~]", pkg)[0]
            existing_packages[pkg_name] = pkg

        # Process new packages: add new ones or replace existing with different versions
        packages_to_install = []
        for pkg in versioned_packages:
            pkg_name = re.split(r"[<>=!~]", pkg)[0]
            existing_pkg = existing_packages.get(pkg_name)

            if existing_pkg != pkg:  # New package or different version
                if existing_pkg:
                    # Replace existing package
                    self.packages[self.packages.index(existing_pkg)] = pkg
                else:
                    # Add new package
                    self.packages.append(pkg)
                packages_to_install.append(pkg)

        # If client is already running, install packages at runtime
        if self.state == ClientState.RUNNING and packages_to_install:
            if self.batching:
                # Add to batch
                self._batch_calls.append(
                    BatchCall(
                        method_name="pip_install",
                        args=(packages_to_install,),
                        kwargs={},
                    )
                )
            else:
                # Direct execution
                self.client.pip_install(packages_to_install)
        # If client is not started yet, packages are already added to self.packages
        # and will be included in the initial image when start() is called

    def process_background_tasks(self) -> None:
        """Process background tasks for this client manager."""
        # Check if client should be stopped (stop request signaled by cleared event)
        if not self.stop_signal.is_set():
            if self.is_running():
                # Stop the cloud provider's compute resources
                self.client.stop()
                # Set state based on _stop_state (STOPPED or PAUSED)
                self.state = self._stop_state
            # Set event to signal completion of stop
            self.stop_signal.set()
            return

        # Normal processing for running clients
        if self.is_running():
            try:
                # Execute any pending batched operations first
                self.execute_batch()

                # Then resolve any pending futures (including CPU tensor futures)
                self.resolve_futures()
            except Exception as e:
                log.error(f"Fatal error for client {self.machine_id}: {e}")
                # Propagate the exception to all pending futures for this client (including CPU tensor futures)
                self.propagate_exception_to_futures(e)
