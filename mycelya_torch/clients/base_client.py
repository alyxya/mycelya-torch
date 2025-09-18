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

    def __init__(self, machine_id: str):
        """
        Initialize the client with a machine identifier.

        Args:
            machine_id: Unique identifier for this machine/client instance
        """
        self.machine_id = machine_id

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
    def try_get_rpc_result(self, rpc_result: Any) -> Any | None:
        """
        Non-blocking attempt to get the result from an RPC call.

        This method takes the result object returned by RPC methods (like FunctionCall
        for Modal, direct result for Mock) and returns the resolved value if available,
        or None if the result is not ready yet.

        Args:
            rpc_result: The result object returned by any RPC method

        Returns:
            The resolved actual value if ready, None if not ready yet
        """
        pass

    @abstractmethod
    def execute_batch(self, batch_calls: List[BatchCall]) -> Any:
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
        pass

    @abstractmethod
    def create_tensor_view(
        self,
        new_tensor_id: int,
        base_tensor_id: int,
        shape: List[int],
        stride: List[int],
        offset: int,
    ) -> None:
        """Create a tensor view from an existing tensor."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_storage_data(self, tensor_id: int) -> Any:
        """Get raw storage data by tensor ID.

        Returns:
            The result object (e.g., FunctionCall for Modal, direct result for Mock)
        """
        pass

    @abstractmethod
    def remove_tensors(self, tensor_ids: List[int]) -> None:
        """Remove multiple tensors from the remote machine."""
        pass

    @abstractmethod
    def resize_storage(self, tensor_id: int, nbytes: int) -> None:
        """Resize the underlying storage for a tensor."""
        pass

    # Tensor copy methods
    @abstractmethod
    def copy_tensor(
        self,
        source_tensor_id: int,
        target_tensor_id: int,
    ) -> None:
        """Copy tensor data from source to target on the remote machine."""
        pass

    # Operation execution methods
    @abstractmethod
    def execute_aten_operation(
        self,
        op_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        output_tensor_ids: List[int] | None = None,
    ) -> Any:
        """Execute an aten operation on the remote machine with tensor IDs.

        Returns:
            The result object (e.g., FunctionCall for Modal, direct result for Mock)
        """
        pass

    # HuggingFace model loading methods
    @abstractmethod
    def load_huggingface_state_dicts(
        self,
        repo: str,
        path: str,
        device_type: str,
        device_index: int,
    ) -> Any:
        """Load HuggingFace state dicts organized by directory on the remote machine.

        Returns:
            The result object (e.g., FunctionCall for Modal, direct result for Mock)
        """
        pass

    @abstractmethod
    def link_tensors(
        self,
        local_tensor_ids: List[int],
        temp_keys: List[str],
    ) -> None:
        """Link local mycelya tensor IDs to remote tensors from temporary registry."""
        pass

    @abstractmethod
    def execute_function(self, pickled_function: bytes) -> Any:
        """Execute a pickled function remotely.

        Returns:
            The result object (e.g., FunctionCall for Modal, direct result for Mock)
        """
        pass
