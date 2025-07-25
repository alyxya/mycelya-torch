# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal client implementation for torch_remote.

This module provides the ModalClient class for interfacing with Modal cloud GPUs,
along with related functionality for creating and managing Modal applications.
"""

from typing import Any, Dict, List, Union

from _torch_remote_modal.modal_app import create_modal_app_for_gpu

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
    def create_storage(self, storage_id: int, nbytes: int, lazy: bool = False) -> None:
        """
        Create a storage on the remote machine.

        Args:
            storage_id: Specific ID to use for the storage (required)
            nbytes: Number of bytes to allocate for the storage
            lazy: Whether to defer GPU memory allocation until first use

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        try:
            self._server_instance.create_storage.spawn(storage_id, nbytes, lazy)
        except Exception as e:
            raise RuntimeError(f"Failed to create storage {storage_id}: {e}") from e

    def update_storage(self, storage_id: int, tensor_data: bytes) -> None:
        """
        Update an existing storage with tensor data.

        Args:
            storage_id: Storage ID to update
            tensor_data: Serialized tensor data to store

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        self._server_instance.update_storage.spawn(storage_id, tensor_data)

    def get_storage_data(
        self,
        storage_id: int,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
    ) -> bytes:
        """
        Get storage data by ID for device transfer as a specific view.

        Args:
            storage_id: The storage ID
            shape: Tensor shape for view
            stride: Tensor stride for view
            storage_offset: Storage offset for view
            dtype: Tensor data type

        Returns:
            Serialized tensor data (contiguous representation of the view)
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        return self._server_instance.get_storage_data.remote(
            storage_id, shape, stride, storage_offset, dtype
        )

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
    ) -> None:
        """
        Execute an aten operation with separated input metadata and output storage IDs.

        Args:
            op_name: The aten operation name
            input_tensor_metadata: Metadata for reconstructing input tensors only
            output_storage_ids: List of storage IDs to update with results (None for outputs to ignore)
            args: Operation arguments
            kwargs: Operation keyword arguments

        Returns:
            None (operation results are written to output tensors)
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
        self._server_instance.execute_aten_operation.spawn(
            op_name, input_tensor_metadata, output_storage_ids, args, kwargs
        )

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


# PytorchServer and app creation logic is in _torch_remote_modal.modal_app
