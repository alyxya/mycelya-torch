# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal client implementation for mycelya_torch.

This module provides the ModalClient class for interfacing with Modal cloud GPUs,
along with related functionality for creating and managing Modal applications.
"""

from typing import Any, Dict, List, Optional, Union

import torch

from _mycelya_torch_modal.modal_app import create_modal_app_for_gpu

from ..._logging import get_logger
from ..client_interface import ClientInterface

log = get_logger(__name__)


class ModalClient(ClientInterface):
    """
    Client interface for Modal cloud GPU execution.

    This class provides a client-side interface to Modal's cloud GPU infrastructure,
    encapsulating Modal app management, server instances, and communication
    protocols while maintaining state and connection management.
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
        self._app_context = None
        self.timeout = timeout
        self.retries = retries

        # Initialize the Modal app and server
        self._initialize()

    def _initialize(self):
        """Initialize the Modal app and server class."""
        self._app, self._server_class = create_modal_app_for_gpu(
            self.gpu_type, self.machine_id, self.timeout, self.retries
        )

    def start(self):
        """Start the Modal app context for this machine."""
        if self._app_context is None:
            self._app_context = self._app.run()
            self._app_context.__enter__()
            # Create server instance when app starts
            self._server_instance = self._server_class()

            # Register for RPC batching
            self._register_for_batching()

    def stop(self):
        """Stop the Modal app context for this machine."""
        # Unregister from RPC batching first
        self._unregister_for_batching()

        if self._app_context is not None:
            try:
                self._app_context.__exit__(None, None, None)
            except Exception:
                # Silently ignore cleanup errors during atexit
                pass
            finally:
                self._app_context = None
                self._server_instance = None

    def is_running(self) -> bool:
        """Check if the machine is currently running."""
        return self._app_context is not None

    # Storage management methods
    def create_storage(self, storage_id: int, nbytes: int) -> None:
        """
        Create a storage on the remote machine.

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
            # Queue the RPC call for batching (fire-and-forget)
            self._queue_rpc_call(
                method_name="create_storage",
                call_type="spawn",
                args=(storage_id, nbytes),
                kwargs={},
            )
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
        Update an existing storage with storage tensor data.

        Args:
            storage_id: Storage ID to update
            storage_tensor: CPU tensor wrapping the storage data (tensor metadata is ignored, only untyped_storage matters)
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

        # Serialize storage tensor using numpy approach
        from ..._tensor_utils import cpu_tensor_to_numpy_bytes

        # Serialize tensor using numpy approach
        numpy_bytes = cpu_tensor_to_numpy_bytes(storage_tensor)

        # Queue the RPC call for batching (fire-and-forget)
        # Invalidate cache immediately since this modifies storage
        self._queue_rpc_call(
            method_name="update_storage",
            call_type="spawn",
            args=(
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
            ),
            kwargs={},
            invalidate_storage_ids=[storage_id],
        )

    def _get_storage_data(
        self,
        storage_id: int,
    ) -> bytes:
        """
        Get raw storage data by ID.

        Args:
            storage_id: The storage ID

        Returns:
            Raw untyped storage bytes
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Queue the RPC call for batching (blocking call that returns raw bytes)
        raw_bytes = self._queue_rpc_call(
            method_name="get_storage_data",
            call_type="remote",
            args=(storage_id,),
            kwargs={},
        )

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

        # Queue the RPC call for batching (blocking call that returns raw bytes)
        future = self._queue_rpc_call(
            method_name="get_storage_data",
            call_type="remote",
            args=(storage_id,),
            kwargs={},
        )

        # Wait for the result from the Future
        raw_bytes = future.result() if future else None
        if raw_bytes is None:
            raise RuntimeError(
                f"Failed to retrieve storage data for storage {storage_id}"
            )

        # Create 1D uint8 tensor directly from raw bytes for caching
        from ..._tensor_utils import numpy_bytes_to_cpu_tensor

        return numpy_bytes_to_cpu_tensor(raw_bytes, (len(raw_bytes),), torch.uint8)

    def resize_storage(self, storage_id: int, nbytes: int) -> None:
        """
        Resize a storage on the remote machine.

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

        # Queue the RPC call for batching (fire-and-forget)
        # Invalidate cache immediately since this modifies storage
        self._queue_rpc_call(
            method_name="resize_storage",
            call_type="spawn",
            args=(storage_id, nbytes),
            kwargs={},
            invalidate_storage_ids=[storage_id],
        )

    def remove_storage(self, storage_id: int) -> None:
        """
        Remove a storage from the remote machine.

        Args:
            storage_id: The storage ID

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Queue the RPC call for batching (fire-and-forget)
        # Invalidate cache immediately since this removes storage
        self._queue_rpc_call(
            method_name="remove_storage",
            call_type="spawn",
            args=(storage_id,),
            kwargs={},
            invalidate_storage_ids=[storage_id],
        )

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
        Execute an aten operation with separated input metadata and output storage IDs.

        Args:
            op_name: The aten operation name
            input_tensor_metadata: Metadata for reconstructing input tensors only
            output_storage_ids: List of storage IDs to update with results (all output tensors)
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
        log.info(f"📡 Modal Client sending Input Storage IDs: {input_storage_ids}")
        log.info(f"📡 Modal Client sending Output Storage IDs: {output_storage_ids}")

        # Determine which storage IDs will be modified by this operation
        # Filter out None values for cache invalidation
        modified_storage_ids = [sid for sid in output_storage_ids if sid is not None]

        if return_metadata:
            # Use remote call type to get return value when metadata is needed
            log.info(f"📡 Modal Client requesting metadata for {op_name}")
            future = self._queue_rpc_call(
                method_name="execute_aten_operation",
                call_type="remote",
                args=(op_name, input_tensor_metadata, output_storage_ids, args, kwargs),
                kwargs={"return_metadata": True},
                invalidate_storage_ids=modified_storage_ids,
            )
            # Wait for the result from the Future
            return future.result() if future else None
        else:
            # Use spawn call type for fire-and-forget execution (original behavior)
            self._queue_rpc_call(
                method_name="execute_aten_operation",
                call_type="spawn",
                args=(op_name, input_tensor_metadata, output_storage_ids, args, kwargs),
                kwargs={},
                invalidate_storage_ids=modified_storage_ids,
            )
            return None

    def __enter__(self):
        """Context manager entry - starts the machine."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the machine."""
        self.stop()

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return (
            f'ModalClient(gpu_type="{self.gpu_type}", '
            f'machine_id="{self.machine_id}", status="{status}")'
        )


# PytorchServer and app creation logic is in _mycelya_torch_modal.modal_app
