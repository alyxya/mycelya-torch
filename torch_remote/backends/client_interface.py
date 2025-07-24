# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Abstract client interface for torch_remote cloud providers.

This module defines the base interface that all cloud provider clients must implement,
ensuring consistent API across different backends (Modal, AWS, GCP, Azure, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ClientInterface(ABC):
    """
    Abstract base class defining the interface for cloud provider clients.

    All cloud provider clients (ModalClient, AWSClient, etc.) must inherit from this
    class and implement all abstract methods to ensure consistent API across providers.
    """

    def __init__(self, gpu_type: str, machine_id: str):
        """
        Initialize the client with GPU type, machine ID, and configuration.

        Args:
            gpu_type: The GPU type (e.g., "T4", "A100-40GB")
            machine_id: Unique machine identifier
        """
        self.gpu_type = gpu_type
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
    def is_running(self) -> bool:
        """
        Check if the machine is currently running and ready.

        Returns:
            True if the machine is running and can accept operations, False otherwise
        """
        pass

    def health_check(self) -> bool:
        """
        Perform a health check on the client connection.

        Default implementation delegates to is_running(). Providers can override
        for more sophisticated health checking.

        Returns:
            True if client is healthy, False otherwise
        """
        return self.is_running()

    def reconnect(self) -> bool:
        """
        Attempt to reconnect the client.

        Default implementation stops and starts the client. Providers can override
        for more sophisticated reconnection logic.

        Returns:
            True if reconnection succeeded, False otherwise
        """
        try:
            self.stop()
            self.start()
            return self.is_running()
        except Exception:
            return False

    @abstractmethod
    def create_storage(self, nbytes: int, storage_id: int, lazy: bool = False) -> None:
        """
        Create a storage on the remote machine.

        Args:
            nbytes: Number of bytes to allocate for the storage
            storage_id: Specific ID to use for the storage (required)
            lazy: Whether to defer GPU memory allocation until first use

        Returns:
            None
        """
        pass

    @abstractmethod
    def update_storage(self, tensor_data: bytes, storage_id: int) -> None:
        """
        Update an existing storage with tensor data.

        Args:
            tensor_data: Serialized tensor data to store
            storage_id: Storage ID to update

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_storage_data(
        self,
        storage_id: int,
        shape: List[int] = None,
        stride: List[int] = None,
        storage_offset: int = 0,
        dtype: str = None,
    ) -> bytes:
        """
        Retrieve storage data by ID, optionally as a specific view.

        Args:
            storage_id: The storage ID to retrieve
            shape: Tensor shape for view (if None, returns full storage)
            stride: Tensor stride for view
            storage_offset: Storage offset for view
            dtype: Tensor data type

        Returns:
            Serialized tensor data (contiguous representation of the view)
        """
        pass

    @abstractmethod
    def execute_aten_operation(
        self,
        op_name: str,
        tensor_metadata: List[Dict[str, Any]],
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Execute an aten operation on the remote machine.
        All tensors (input and output) are pre-allocated and passed as arguments.

        Args:
            op_name: The aten operation name to execute
            tensor_metadata: Metadata for reconstructing all tensors (shape, stride, offset, storage_id)
            args: Operation arguments (may contain tensor placeholders)
            kwargs: Operation keyword arguments (may contain tensor placeholders)

        Returns:
            None (operation is executed in-place on pre-allocated tensors)
        """
        pass

    @abstractmethod
    def resize_storage(self, storage_id: int, nbytes: int) -> None:
        """
        Resize a storage to accommodate new byte size.

        This handles the case where resize_ needs more storage space than currently allocated.
        Only resizes if nbytes > current storage size.

        Args:
            storage_id: The storage ID to resize
            nbytes: The number of bytes needed for the new storage size

        Returns:
            None
        """
        pass

    @abstractmethod
    def remove_storage(self, storage_id: int) -> None:
        """
        Remove a storage from the remote machine.

        Args:
            storage_id: The storage ID to remove

        Returns:
            None
        """
        pass

    # Context manager methods (optional to override, but provide default behavior)
    def __enter__(self):
        """Context manager entry - starts the machine."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the machine."""
        self.stop()

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation of the client.

        Returns:
            Human-readable string describing the client state
        """
        pass
