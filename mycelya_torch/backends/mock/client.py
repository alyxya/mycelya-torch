# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Mock client implementation for mycelya_torch.

This module provides the MockClient class that uses Modal's .local() execution
for development and testing without requiring remote cloud resources.
"""

from typing import Any, Dict, List, Optional, Union

import torch

from _mycelya_torch_modal.modal_app import create_modal_app_for_gpu

from ..._logging import get_logger
from ..client_interface import ClientInterface

log = get_logger(__name__)


class MockClient(ClientInterface):
    """
    Client interface for mock execution using Modal's .local() calls.

    This class provides a mock execution environment that reuses the existing Modal app
    but executes all methods locally using .local() instead of .remote() or .spawn().
    This avoids code duplication while providing the same interface.
    """

    def __init__(
        self,
        gpu_type: str,
        machine_id: str,
        timeout: int,
        retries: int,
    ):
        super().__init__(gpu_type, machine_id)
        self._app = None
        self._server_class = None
        self._server_instance = None
        self._is_running = False
        self.timeout = timeout
        self.retries = retries

        # Initialize the Modal app and server for mock execution
        self._initialize()

    def _initialize(self):
        """Initialize the Modal app and server class for mock execution."""
        self._app, self._server_class = create_modal_app_for_gpu(
            self.gpu_type, self.machine_id, self.timeout, self.retries
        )

    def start(self):
        """Start the mock execution environment."""
        if not self._is_running:
            # Create server instance for mock execution
            self._server_instance = self._server_class()
            self._is_running = True

            # Register for RPC batching
            self._register_for_batching()

            log.info(f"Started mock client: {self.machine_id}")

    def stop(self):
        """Stop the mock execution environment."""
        # Unregister from RPC batching first
        self._unregister_for_batching()

        if self._is_running:
            self._server_instance = None
            self._is_running = False
            log.info(f"Stopped mock client: {self.machine_id}")

    def is_running(self) -> bool:
        """Check if the mock client is currently running."""
        return self._is_running

    # Storage management methods
    def create_storage(self, storage_id: int, nbytes: int) -> None:
        """
        Create a storage using mock execution.

        Args:
            storage_id: Specific ID to use for the storage (required)
            nbytes: Number of bytes to allocate for the storage

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        try:
            # Execute using .local() instead of queuing for remote execution
            self._server_instance.create_storage.local(storage_id, nbytes)
        except Exception as e:
            raise RuntimeError(f"Failed to create storage {storage_id}: {e}") from e

    def update_storage(
        self,
        storage_id: int,
        storage_tensor: torch.Tensor,
        source_shape: List[int],
        source_stride: List[int],
        source_storage_offset: int,
        source_dtype: str,
        target_shape: List[int],
        target_stride: List[int],
        target_storage_offset: int,
        target_dtype: str,
    ) -> None:
        """
        Update an existing storage with storage tensor data using mock execution.

        Args:
            storage_id: Storage ID to update
            storage_tensor: CPU tensor wrapping the storage data
            source_shape: Shape of the source data
            source_stride: Stride of the source data
            source_storage_offset: Storage offset of the source data
            source_dtype: Data type of the source data
            target_shape: Shape of the target view in storage
            target_stride: Stride of the target view in storage
            target_storage_offset: Storage offset of the target view in storage
            target_dtype: Data type of the target view in storage

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Serialize storage tensor using numpy approach (same as Modal client)
        from ..._tensor_utils import cpu_tensor_to_numpy_bytes

        numpy_bytes = cpu_tensor_to_numpy_bytes(storage_tensor)

        # Invalidate cache immediately since this modifies storage
        self.invalidate_storage_cache(storage_id)

        # Execute using .local() instead of queuing for remote execution
        self._server_instance.update_storage.local(
            storage_id,
            numpy_bytes,
            source_shape,
            source_stride,
            source_storage_offset,
            source_dtype,
            target_shape,
            target_stride,
            target_storage_offset,
            target_dtype,
        )

    def _get_storage_data(
        self,
        storage_id: int,
    ) -> bytes:
        """
        Get raw storage data by ID using mock execution.

        Args:
            storage_id: The storage ID

        Returns:
            Raw untyped storage bytes
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Execute using .local() instead of remote call
        raw_bytes = self._server_instance.get_storage_data.local(storage_id)

        # Return raw bytes directly - no deserialization needed
        return raw_bytes

    def _get_storage_tensor_for_cache(
        self,
        storage_id: int,
    ) -> torch.Tensor:
        """
        Get storage data as a 1D uint8 CPU tensor for caching.

        Args:
            storage_id: The storage ID

        Returns:
            1D uint8 CPU tensor representing the underlying storage
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Execute using .local() instead of remote call
        raw_bytes = self._server_instance.get_storage_data.local(storage_id)

        if raw_bytes is None:
            raise RuntimeError(
                f"Failed to retrieve storage data for storage {storage_id}"
            )

        # Create 1D uint8 tensor directly from raw bytes for caching
        from ..._tensor_utils import numpy_bytes_to_cpu_tensor

        return numpy_bytes_to_cpu_tensor(raw_bytes, (len(raw_bytes),), torch.uint8)

    def resize_storage(self, storage_id: int, nbytes: int) -> None:
        """
        Resize a storage using mock execution.

        Args:
            storage_id: The storage ID to resize
            nbytes: The number of bytes needed for the new storage size

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Invalidate cache immediately since this modifies storage
        self.invalidate_storage_cache(storage_id)

        # Execute using .local() instead of remote call
        self._server_instance.resize_storage.local(storage_id, nbytes)

    def remove_storage(self, storage_id: int) -> None:
        """
        Remove a storage using mock execution.

        Args:
            storage_id: The storage ID

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Invalidate cache immediately since this removes storage
        self.invalidate_storage_cache(storage_id)

        # Execute using .local() instead of remote call
        self._server_instance.remove_storage.local(storage_id)

    # Operation execution methods
    def execute_aten_operation(
        self,
        op_name: str,
        input_tensor_metadata: List[Dict[str, Any]],
        output_storage_ids: List[Union[int, None]],
        args: List[Any],
        kwargs: Dict[str, Any],
        return_metadata: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute an aten operation using mock execution.

        Args:
            op_name: The aten operation name
            input_tensor_metadata: Metadata for reconstructing input tensors only
            output_storage_ids: List of storage IDs to update with results
            args: Operation arguments
            kwargs: Operation keyword arguments
            return_metadata: If True, return output tensor metadata instead of None

        Returns:
            None for normal operations, or List[Dict] of output tensor metadata if return_metadata=True
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        input_storage_ids = [
            metadata["storage_id"] for metadata in input_tensor_metadata
        ]
        log.info(f"📡 Mock Client sending Input Storage IDs: {input_storage_ids}")
        log.info(f"📡 Mock Client sending Output Storage IDs: {output_storage_ids}")

        # Invalidate cache immediately for storage IDs that will be modified
        modified_storage_ids = [sid for sid in output_storage_ids if sid is not None]
        self.invalidate_multiple_storage_caches(modified_storage_ids)

        # Execute using .local() instead of remote call
        result = self._server_instance.execute_aten_operation.local(
            op_name,
            input_tensor_metadata,
            output_storage_ids,
            args,
            kwargs,
            return_metadata,
        )

        if return_metadata:
            log.info(f"📡 Mock Client received metadata for {op_name}")
            return result
        else:
            return None

    def _queue_rpc(
        self,
        method_name: str,
        call_type: str,
        args: tuple,
        kwargs: dict,
        return_future: bool = False,
        invalidate_storage_ids: Optional[List[int]] = None,
    ) -> Optional[Any]:
        """
        Override the base class method to handle any remaining RPCs.

        This should not be called in the mock client since we execute directly,
        but we provide it for compatibility.
        """
        # Invalidate cache immediately for storage-modifying operations
        if invalidate_storage_ids:
            self.invalidate_multiple_storage_caches(invalidate_storage_ids)

        # For mock client, we should not queue calls, but execute directly
        # This method is mainly for compatibility with the base class
        log.warning(
            f"Mock client received unexpected RPC: {method_name}. Consider using direct method calls."
        )
        return None

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return (
            f'MockClient(gpu_type="{self.gpu_type}", '
            f'machine_id="{self.machine_id}", status="{status}")'
        )
