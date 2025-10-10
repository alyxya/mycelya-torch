# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Abstract client interface for mycelya_torch cloud providers.

This module defines the base interface that all cloud provider clients must implement,
ensuring consistent API across different backends (Modal, AWS, GCP, Azure, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from .._utils import TensorMetadata


class BatchCall(TypedDict):
    """Structure for a single batched RPC call.

    This TypedDict defines the structure used for batching multiple operations
    into a single RPC call for performance optimization.
    """

    method_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


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
    def start(
        self,
        gpu_type: str,
        gpu_count: int,
        packages: list[str],
        python_version: str,
    ) -> None:
        """
        Start the cloud provider's compute resources.

        This method should initialize and start the remote client,
        making it ready to accept operations.

        Args:
            gpu_type: GPU type string (required for modal, ignored for mock)
            gpu_count: Number of GPUs (1-8, ignored for mock)
            packages: Versioned package list for modal app (ignored for mock)
            python_version: Python version string (ignored for mock)
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
    def get_rpc_result(self, rpc_result: Any, blocking: bool) -> Any | None:
        """
        Get the result from an RPC call.

        This method takes the result object returned by RPC methods (like FunctionCall
        for Modal, direct result for Mock) and returns the resolved value.

        Args:
            rpc_result: The result object returned by any RPC method
            blocking: If True, wait for result. If False, return None if not ready.

        Returns:
            The resolved actual value, or None if not ready and blocking=False
        """
        pass

    @abstractmethod
    def execute_batch(self, batch_calls: list[BatchCall]) -> Any:
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
    def create_tensor(self, metadata: TensorMetadata) -> None:
        """Create a tensor on the remote machine.

        Creates either a new empty tensor or a tensor view based on metadata.alias_id:
        - If alias_id is None: Creates new empty tensor
        - If alias_id is int: Creates tensor view using alias_id as base tensor

        Args:
            metadata: TensorMetadata containing tensor properties and creation info
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_storage_data(self, tensor_id: int) -> Any:
        """Get raw storage data by tensor ID.

        Returns:
            The result object (e.g., FunctionCall for Modal, direct result for Mock)
        """
        pass

    @abstractmethod
    def remove_tensors(self, tensor_ids: list[int]) -> None:
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
        args: list[Any],
        kwargs: dict[str, Any],
        tensor_mask: list[bool],
        output_tensor_ids: list[int | None] | None = None,
    ) -> Any:
        """Execute an aten operation on the remote machine with tensor IDs.

        Args:
            output_tensor_ids: List of tensor IDs for outputs (None entries indicate unused gradients)

        Returns:
            The result object (e.g., FunctionCall for Modal, direct result for Mock)
        """
        pass

    @abstractmethod
    def link_tensors(
        self,
        tensor_ids: list[int],
        temp_ids: list[str],
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

    @abstractmethod
    def pip_install(self, packages: list[str]) -> None:
        """Install packages using pip on the remote machine.

        Args:
            packages: List of package names to install (e.g., ["numpy", "scipy"])
        """
        pass

    @abstractmethod
    def offload(self) -> None:
        """Offload tensor registry to disk."""
        pass

    @abstractmethod
    def reload(self) -> None:
        """Reload tensor registry from disk."""
        pass
