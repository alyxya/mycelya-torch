# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Client manager for mycelya_torch cloud provider clients.

This module provides the ClientManager class that handles the common functionality
for all cloud provider clients, including batching, future management, and
operation coordination.
"""

from collections import deque
from concurrent.futures import Future
from typing import Any, Dict, List

from ._utils import TensorMetadata
from .clients.base_client import BatchCall, Client


class ClientManager:
    """
    Manager class that handles common client functionality including batching,
    future management, and operation coordination.

    This class wraps a concrete client implementation and provides the
    batching, validation, and common logic while delegating actual
    implementation to the wrapped client.
    """

    def __init__(self, client: Client, machine_id: str, batching: bool = True):
        """
        Initialize the client manager with a concrete client implementation.

        Args:
            client: Concrete client implementation (ModalClient, MockClient, etc.)
            machine_id: Unique machine identifier
            batching: Whether to enable operation batching (default: True)
        """
        self.client = client
        self.machine_id = machine_id
        self.batching = batching

        # Deque for storing pending futures that need to be resolved (returned to caller)
        self._pending_futures = deque()

        # Deque for storing pending results/FunctionCalls from remote operations (1:1 correspondence with futures)
        self._pending_results = deque()

        # Deque for storing pending batch calls when batching is enabled
        self._batch_calls = deque()

    def start(self) -> None:
        """Start the cloud provider's compute resources."""
        self.client.start()

    def stop(self) -> None:
        """Stop the cloud provider's compute resources."""
        self.client.stop()

    def is_running(self) -> bool:
        """Check if the machine is currently running and ready."""
        return self.client.is_running()

    def resolve_futures(self) -> None:
        """Resolve any pending futures by fetching results from the queue."""
        # Pass the manager's queues to the client for resolution
        self.client.resolve_futures_with_state(
            self._pending_futures, self._pending_results, self.batching
        )

    def propagate_exception_to_futures(self, exception: Exception) -> None:
        """Propagate the given exception to all pending futures."""
        # Propagate to all pending futures
        while self._pending_futures:
            future = self._pending_futures.popleft()
            future.set_exception(exception)

        # Clear pending results
        self._pending_results.clear()

    def add_pending_result(self, result):
        """Add a pending result to the manager's queue."""
        self._pending_results.append(result)

    def _ensure_running(self) -> None:
        """Ensure the machine is running, raise RuntimeError if not."""
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

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
        if result is not None:
            self._pending_results.append(result)

    # Tensor management methods

    def create_empty_tensor(
        self,
        tensor_id: int,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
        nbytes: int,
        device_type: str,
        device_index: int,
    ) -> None:
        """Create an empty tensor on the remote machine with proper storage layout."""
        self._ensure_running()

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="create_empty_tensor",
                    args=(
                        tensor_id,
                        shape,
                        stride,
                        storage_offset,
                        dtype,
                        nbytes,
                        device_type,
                        device_index,
                    ),
                    kwargs={},
                )
            )
        else:
            # Direct execution
            self.client.create_empty_tensor(
                tensor_id,
                shape,
                stride,
                storage_offset,
                dtype,
                nbytes,
                device_type,
                device_index,
            )

    def create_tensor_view(
        self,
        new_tensor_id: int,
        base_tensor_id: int,
        shape: List[int],
        stride: List[int],
        offset: int,
    ) -> None:
        """Create a tensor view from an existing tensor."""
        self._ensure_running()

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="create_tensor_view",
                    args=(new_tensor_id, base_tensor_id, shape, stride, offset),
                    kwargs={},
                )
            )
        else:
            # Direct execution
            self.client.create_tensor_view(
                new_tensor_id, base_tensor_id, shape, stride, offset
            )

    def update_tensor(
        self,
        tensor_id: int,
        raw_data: bytes,
        source_shape: List[int],
        source_stride: List[int],
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
            if result is not None:
                self._pending_results.append(result)

        return future

    def remove_tensors(self, tensor_ids: List[int]) -> None:
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
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        output_tensor_ids: List[int] | None = None,
    ) -> Future[List[TensorMetadata]] | None:
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

    # HuggingFace model loading methods

    def load_huggingface_state_dicts(
        self,
        repo: str,
        path: str,
        device_type: str,
        device_index: int,
    ) -> Future[Dict[str, Dict[str, TensorMetadata]]]:
        """Load HuggingFace state dicts organized by directory on the remote machine."""
        self._ensure_running()

        # Create a Future for the result
        future = Future()
        # Add future to pending futures queue
        self._pending_futures.append(future)

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="load_huggingface_state_dicts",
                    args=(repo, path, device_type, device_index),
                    kwargs={},
                )
            )
        else:
            # Direct execution and add result to pending results
            result = self.client.load_huggingface_state_dicts(
                repo, path, device_type, device_index
            )
            if result is not None:
                self._pending_results.append(result)

        return future

    def link_tensors(
        self,
        local_tensor_ids: List[int],
        temp_keys: List[str],
    ) -> None:
        """Link local mycelya tensor IDs to remote tensors from temporary registry."""
        self._ensure_running()

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="link_tensors",
                    args=(local_tensor_ids, temp_keys),
                    kwargs={},
                )
            )
        else:
            # Direct execution
            self.client.link_tensors(local_tensor_ids, temp_keys)

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
            if result is not None:
                self._pending_results.append(result)

        return future

    def __repr__(self) -> str:
        """String representation of the client manager."""
        return f"ClientManager(client={self.client}, machine_id={self.machine_id}, batching={self.batching})"
