# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal client implementation for mycelya_torch.

This module provides the ModalClient class for interfacing with Modal cloud GPUs,
along with related functionality for creating and managing Modal applications.
"""

from typing import Any, Dict, List, Optional, Union

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

    def stop(self):
        """Stop the Modal app context for this machine."""
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
            self._server_instance.create_storage.spawn(storage_id, nbytes)
        except Exception as e:
            raise RuntimeError(f"Failed to create storage {storage_id}: {e}") from e

    def update_storage(
        self,
        storage_id: int,
        raw_data: bytes,
        source_shape: List[int],
        source_stride: List[int],
        source_storage_offset: int,
        source_dtype: str,
        target_shape: List[int],
        target_stride: List[int],
        target_storage_offset: int,
        target_dtype: str
    ) -> None:
        """
        Update an existing storage with raw tensor data.

        Args:
            storage_id: Storage ID to update
            raw_data: Raw untyped storage bytes to store
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

        # Convert raw_data to tensor and serialize with torch.save
        from ..._tensor_utils import (
            cpu_tensor_to_torch_bytes,
            storage_bytes_to_cpu_tensor,
        )

        # Create CPU tensor from raw data using source metadata
        cpu_tensor = storage_bytes_to_cpu_tensor(
            raw_data, source_shape, source_stride, source_storage_offset, source_dtype
        )

        # Serialize tensor using torch.save
        torch_bytes = cpu_tensor_to_torch_bytes(cpu_tensor)

        self._server_instance.update_storage.spawn(
            storage_id, torch_bytes,
            source_shape, source_stride, source_storage_offset, source_dtype,
            target_shape, target_stride, target_storage_offset, target_dtype
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

        # Get torch.save serialized bytes from Modal app
        torch_bytes = self._server_instance.get_storage_data.remote(storage_id)

        # Deserialize tensor and convert back to raw storage bytes
        from ..._tensor_utils import (
            cpu_tensor_to_storage_bytes,
            torch_bytes_to_cpu_tensor,
        )

        cpu_tensor = torch_bytes_to_cpu_tensor(torch_bytes)
        return cpu_tensor_to_storage_bytes(cpu_tensor)

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

        self._server_instance.resize_storage.spawn(storage_id, nbytes)

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

        self._server_instance.remove_storage.spawn(storage_id)

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

        if return_metadata:
            # Use .remote() to get return value when metadata is needed
            log.info(f"📡 Modal Client requesting metadata for {op_name}")
            result = self._server_instance.execute_aten_operation.remote(
                op_name, input_tensor_metadata, output_storage_ids, args, kwargs, return_metadata=True
            )
            return result
        else:
            # Use .spawn() for fire-and-forget execution (original behavior)
            self._server_instance.execute_aten_operation.spawn(
                op_name, input_tensor_metadata, output_storage_ids, args, kwargs
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
