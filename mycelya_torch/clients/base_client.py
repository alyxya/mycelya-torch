# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Abstract client interface for mycelya_torch cloud providers.

This module defines the base interface that all cloud provider clients must implement,
ensuring consistent API across different backends (Modal, AWS, GCP, Azure, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, TypedDict


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

    This class now contains only abstract method definitions. All concrete functionality
    has been moved to ClientManager in _client_manager.py.
    """

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
    def resolve_futures_with_state(
        self, pending_futures, pending_results, batching: bool
    ) -> None:
        """
        Resolve any pending futures by fetching results from the queue.

        Called periodically by orchestrator's background thread to process
        futures that were created for asynchronous operations.

        Args:
            pending_futures: Deque of pending futures to resolve
            pending_results: Deque of pending results from remote operations
            batching: Whether batching is enabled
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
    def _execute_batch_impl(self, batch_calls: List[BatchCall]) -> Any:
        """
        Implementation: Execute a batch of operations.

        Args:
            batch_calls: List of BatchCall objects to execute

        Returns:
            The result object (e.g., FunctionCall for Modal, direct result for Mock)
        """
        pass

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

    @abstractmethod
    def _get_storage_data_impl(self, tensor_id: int) -> Any:
        """Implementation: Get raw storage data by tensor ID.

        Returns:
            The result object (e.g., FunctionCall for Modal, direct result for Mock)
        """
        pass

    @abstractmethod
    def _remove_tensors_impl(self, tensor_ids: List[int]) -> None:
        """Implementation: Remove multiple tensors from the remote machine."""
        pass

    @abstractmethod
    def _resize_storage_impl(self, tensor_id: int, nbytes: int) -> None:
        """Implementation: Resize the underlying storage for a tensor."""
        pass

    # Tensor copy methods
    @abstractmethod
    def _copy_tensor_impl(
        self,
        source_tensor_id: int,
        target_tensor_id: int,
    ) -> None:
        """Implementation: Copy tensor data from source to target on the remote machine."""
        pass

    # Operation execution methods
    @abstractmethod
    def _execute_aten_operation_impl(
        self,
        op_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        output_tensor_ids: List[int] | None = None,
    ) -> Any | None:
        """Implementation: Execute an aten operation on the remote machine with tensor IDs.

        Returns:
            The result object for dynamic operations (when output_tensor_ids is None),
            or None for static operations
        """
        pass

    # HuggingFace model loading methods
    @abstractmethod
    def _load_huggingface_state_dicts_impl(
        self,
        repo: str,
        path: str,
        device_type: str,
        device_index: int,
    ) -> Any:
        """Implementation: Load HuggingFace state dicts organized by directory on the remote machine.

        Returns:
            The result object (e.g., FunctionCall for Modal, direct result for Mock)
        """
        pass

    @abstractmethod
    def _link_tensors_impl(
        self,
        local_tensor_ids: List[int],
        temp_keys: List[str],
    ) -> None:
        """Implementation: Link local mycelya tensor IDs to remote tensors from temporary registry."""
        pass

    @abstractmethod
    def _execute_function_impl(self, pickled_function: bytes) -> Any:
        """Implementation: Execute a pickled function remotely.

        Returns:
            The result object (e.g., FunctionCall for Modal, direct result for Mock)
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation of the client.

        Returns:
            Human-readable string describing the client state
        """
        pass
