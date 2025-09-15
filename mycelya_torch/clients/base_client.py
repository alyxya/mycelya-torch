# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Abstract client interface for mycelya_torch cloud providers.

This module defines the base interface that all cloud provider clients must implement,
ensuring consistent API across different backends (Modal, AWS, GCP, Azure, etc.).
"""

from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from .._utils import TensorMetadata


class BatchCall(TypedDict):
    """Structure for a single batched RPC call.

    This TypedDict defines the structure used for batching multiple operations
    into a single RPC call for performance optimization.
    """

    method_name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


class Client(ABC):
    """
    Abstract base class defining the interface for cloud provider clients.

    All cloud provider clients (ModalClient, MockClient, etc.) must inherit from this
    class and implement all abstract methods to ensure consistent API across providers.
    """

    def __init__(self, machine_id: str, batching: bool = True):
        """
        Initialize the client with machine ID and configuration.

        Args:
            machine_id: Unique machine identifier
            batching: Whether to enable operation batching (default: True)
        """
        self.machine_id = machine_id
        self.batching = batching

        # Deque for storing pending futures that need to be resolved (returned to caller)
        self._pending_futures = deque()

        # Deque for storing pending results/FunctionCalls from remote operations (1:1 correspondence with futures)
        self._pending_results = deque()

        # Deque for storing pending batch calls when batching is enabled
        self._batch_calls = deque()

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

    @abstractmethod
    def resolve_futures(self) -> None:
        """
        Resolve any pending futures by fetching results from the queue.

        Called periodically by orchestrator's background thread to process
        futures that were created for asynchronous operations.
        """
        pass

    @abstractmethod
    def propagate_exception_to_futures(self, exception: Exception) -> None:
        """
        Propagate the given exception to all pending futures.

        This method should be called when a fatal error occurs that affects
        all pending operations for this client. All futures in _pending_futures
        will be resolved with the provided exception, and any client-specific
        result queues should also be cleared.

        Args:
            exception: The exception to set on all pending futures

        Returns:
            None
        """
        pass

    @abstractmethod
    def _execute_batch_impl(self, batch_calls: List[BatchCall]) -> None:
        """
        Implementation: Execute a batch of operations.

        Args:
            batch_calls: List of BatchCall objects to execute
        """
        pass

    def execute_batch(self) -> None:
        """
        Execute any pending batch calls.

        Pops all pending calls from the deque to create a batch list,
        then executes all operations in a single batch.
        """
        if not self._batch_calls:
            return

        # Pop all pending calls from deque to create batch list
        batch_to_execute = []
        while self._batch_calls:
            batch_to_execute.append(self._batch_calls.popleft())

        # Execute the batch
        self._execute_batch_impl(batch_to_execute)

    # Tensor management methods
    @abstractmethod
    def _create_empty_tensor_impl(
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
        """Implementation: Create an empty tensor on the remote machine with proper storage layout."""
        pass

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
        """
        Create an empty tensor on the remote machine with proper storage layout.

        Args:
            tensor_id: Unique tensor ID (metadata hash)
            shape: Shape of the tensor
            stride: Stride of the tensor
            storage_offset: Storage offset for the tensor
            dtype: Data type of the tensor (e.g., "float32", "int64")
            nbytes: Number of bytes for the underlying storage (from client allocator)
            device_type: Remote device type (e.g., "cuda", "cpu")
            device_index: Remote device index (e.g., 0, 1)

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

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
            # Direct execution (existing behavior)
            self._create_empty_tensor_impl(
                tensor_id,
                shape,
                stride,
                storage_offset,
                dtype,
                nbytes,
                device_type,
                device_index,
            )

    @abstractmethod
    def _create_tensor_view_impl(
        self,
        new_tensor_id: int,
        base_tensor_id: int,
        shape: List[int],
        stride: List[int],
        offset: int,
    ) -> None:
        """Implementation: Create a tensor view from an existing tensor."""
        pass

    def create_tensor_view(
        self,
        new_tensor_id: int,
        base_tensor_id: int,
        shape: List[int],
        stride: List[int],
        offset: int,
    ) -> None:
        """
        Create a tensor view from an existing tensor.

        Args:
            new_tensor_id: Tensor ID for the new view
            base_tensor_id: Tensor ID of the base tensor
            shape: Shape of the view
            stride: Stride of the view
            offset: Storage offset of the view

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

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
            # Direct execution (existing behavior)
            self._create_tensor_view_impl(
                new_tensor_id, base_tensor_id, shape, stride, offset
            )

    @abstractmethod
    def _update_tensor_impl(
        self,
        tensor_id: int,
        raw_data: bytes,
        source_shape: List[int],
        source_stride: List[int],
        source_storage_offset: int,
        source_dtype: str,
    ) -> None:
        """Implementation: Update an existing tensor with new data and source metadata."""
        pass

    def update_tensor(
        self,
        tensor_id: int,
        raw_data: bytes,
        source_shape: List[int],
        source_stride: List[int],
        source_storage_offset: int,
        source_dtype: str,
    ) -> None:
        """
        Update an existing tensor with new data and source metadata.

        Args:
            tensor_id: Tensor ID to update
            raw_data: Raw bytes of the tensor data
            source_shape: Shape of the source data
            source_stride: Stride of the source data
            source_storage_offset: Storage offset of the source data
            source_dtype: Data type of the source data

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

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
            # Direct execution (existing behavior)
            self._update_tensor_impl(
                tensor_id,
                raw_data,
                source_shape,
                source_stride,
                source_storage_offset,
                source_dtype,
            )

    @abstractmethod
    def _get_storage_data_impl(self, tensor_id: int) -> None:
        """Implementation: Get raw storage data by tensor ID."""
        pass

    def get_storage_data(self, tensor_id: int) -> Future[bytes]:
        """
        Get raw storage data by tensor ID.

        Args:
            tensor_id: The tensor ID to retrieve storage data from

        Returns:
            Future that resolves to raw storage data as bytes
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

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
            # Direct execution (existing behavior)
            self._get_storage_data_impl(tensor_id)

        return future

    @abstractmethod
    def _remove_tensors_impl(self, tensor_ids: List[int]) -> None:
        """Implementation: Remove multiple tensors from the remote machine."""
        pass

    def remove_tensors(self, tensor_ids: List[int]) -> None:
        """
        Remove multiple tensors from the remote machine.

        Args:
            tensor_ids: List of tensor IDs to remove

        Returns:
            None
        """
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
            # Direct execution (existing behavior)
            self._remove_tensors_impl(tensor_ids)

    @abstractmethod
    def _resize_storage_impl(self, tensor_id: int, nbytes: int) -> None:
        """Implementation: Resize the underlying storage for a tensor."""
        pass

    def resize_storage(self, tensor_id: int, nbytes: int) -> None:
        """
        Resize the underlying storage for a tensor.

        Args:
            tensor_id: Tensor ID whose storage to resize
            nbytes: New size in bytes

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        if self.batching:
            # Add to batch
            self._batch_calls.append(
                BatchCall(
                    method_name="resize_storage", args=(tensor_id, nbytes), kwargs={}
                )
            )
        else:
            # Direct execution (existing behavior)
            self._resize_storage_impl(tensor_id, nbytes)

    # Tensor copy methods
    @abstractmethod
    def _copy_tensor_impl(
        self,
        source_tensor_id: int,
        target_tensor_id: int,
    ) -> None:
        """Implementation: Copy tensor data from source to target on the remote machine."""
        pass

    def copy_tensor(
        self,
        source_tensor_id: int,
        target_tensor_id: int,
    ) -> None:
        """
        Copy tensor data from source to target on the same remote machine.

        Both tensors must exist on the same remote machine. This operation
        performs the copy directly on the remote machine without data transfer.

        Args:
            source_tensor_id: ID of the source tensor to copy from
            target_tensor_id: ID of the target tensor to copy to

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

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
            # Direct execution (existing behavior)
            self._copy_tensor_impl(source_tensor_id, target_tensor_id)

    # Operation execution methods
    @abstractmethod
    def _execute_aten_operation_impl(
        self,
        op_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        output_tensor_ids: Optional[List[int]] = None,
    ) -> None:
        """Implementation: Execute an aten operation on the remote machine with tensor IDs."""
        pass

    def execute_aten_operation(
        self,
        op_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        output_tensor_ids: Optional[List[int]] = None,
    ) -> Optional[Future[List[TensorMetadata]]]:
        """
        Execute an aten operation on the remote machine.

        Args:
            op_name: The aten operation name to execute
            args: Operation arguments (with tensor IDs replacing tensors)
            kwargs: Operation keyword arguments (with tensor IDs replacing tensors)
            tensor_mask: Boolean mask indicating which positions in args/kwargs had tensors
            output_tensor_ids: List of output tensor IDs for static operations, or None for dynamic operations

        Returns:
            None for static operations, or Future[List[TensorMetadata]] of output tensor metadata for dynamic operations
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

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
            # Direct execution (existing behavior)
            self._execute_aten_operation_impl(
                op_name,
                args,
                kwargs,
                tensor_mask,
                output_tensor_ids,
            )

        return future

    # HuggingFace model loading methods
    @abstractmethod
    def _load_huggingface_state_dicts_impl(
        self,
        repo: str,
        path: str,
        device_type: str,
        device_index: int,
    ) -> None:
        """Implementation: Load HuggingFace state dicts organized by directory on the remote machine."""
        pass

    def load_huggingface_state_dicts(
        self,
        repo: str,
        path: str,
        device_type: str,
        device_index: int,
    ) -> Future[Dict[str, Dict[str, TensorMetadata]]]:
        """
        Load HuggingFace state dicts organized by directory on the remote machine.

        This method downloads the model weights directly on the remote GPU,
        loads them into GPU memory organized by directory structure, and returns
        metadata needed to create local tensor stubs.

        Args:
            repo: HuggingFace repository ID (e.g., "HuggingFaceTB/SmolLM2-135M-Instruct")
            path: Path within repository to load from (default: whole repo)
            device_type: Device type (e.g., "cuda", "cpu")
            device_index: Device index (e.g., 0 for cuda:0)

        Returns:
            Future that resolves to Dict[str, TensorMetadata] mapping parameter names to tensor metadata
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

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
            # Direct execution (existing behavior)
            self._load_huggingface_state_dicts_impl(
                repo, path, device_type, device_index
            )

        return future

    @abstractmethod
    def _link_tensors_impl(
        self,
        local_tensor_ids: List[int],
        temp_keys: List[str],
    ) -> None:
        """Implementation: Link local mycelya tensor IDs to remote tensors from temporary registry."""
        pass

    def link_tensors(
        self,
        local_tensor_ids: List[int],
        temp_keys: List[str],
    ) -> None:
        """
        Link local mycelya tensor IDs to remote tensors from temporary registry.

        This method is used to connect locally created mycelya tensors to remote tensors
        that were previously stored in the temporary registry by remote operations.

        Args:
            local_tensor_ids: List of local tensor IDs from created mycelya tensors
            temp_keys: List of temporary registry keys corresponding to each tensor ID

        Returns:
            None

        Example:
            # After remote operations that populate temporary registry
            local_tensor_ids = [get_tensor_id(tensor) for tensor in tensors]
            temp_keys = ["temp_key_1", "temp_key_2", ...]
            client.link_tensors(local_tensor_ids, temp_keys)
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

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
            # Direct execution (existing behavior)
            self._link_tensors_impl(local_tensor_ids, temp_keys)

    @abstractmethod
    def _execute_function_impl(self, pickled_function: bytes) -> None:
        """Implementation: Execute a pickled function remotely."""
        pass

    def execute_function(self, pickled_function: bytes) -> Future[bytes]:
        """
        Execute a pickled function on the remote machine.

        This method sends a pickled function (containing code, args, kwargs)
        to the remote machine for execution and returns a Future for the
        pickled result.

        Args:
            pickled_function: Pickled function data containing code and arguments

        Returns:
            Future that resolves to pickled result bytes
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

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
            # Direct execution
            self._execute_function_impl(pickled_function)

        return future

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation of the client.

        Returns:
            Human-readable string describing the client state
        """
        pass
