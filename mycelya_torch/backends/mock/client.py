# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Mock client implementation for mycelya_torch.

This module provides the MockClient class that uses Modal's .local() execution
for development and testing without requiring remote cloud resources.
"""

from concurrent.futures import Future
from typing import Any, Dict, List, Union

import torch

from _mycelya_torch_modal.modal_app import create_modal_app_for_gpu

from ..._logging import get_logger
from ..base_client import Client

log = get_logger(__name__)


class MockClient(Client):
    """
    Client interface for mock execution using Modal's .local() calls.

    This class provides a mock execution environment that reuses the existing Modal app
    but executes all methods locally using .local() instead of .remote() or .spawn().
    This mirrors the ModalClient structure exactly for testing consistency.
    """

    def __init__(
        self,
        gpu_type: str,
        machine_id: str,
        timeout: int,
    ):
        super().__init__(gpu_type, machine_id)
        self._server_instance = None
        self._is_running = False
        self.timeout = timeout

        # Initialize the Modal app and server class
        self._app, self._server_class, self._response_queue = create_modal_app_for_gpu(
            self.gpu_type, self.machine_id, self.timeout
        )

    def start(self):
        """Start the mock execution environment."""
        if not self._is_running:
            # Create server instance directly without app context
            self._server_instance = self._server_class()
            self._is_running = True

            log.info(f"Started mock client: {self.machine_id}")

    def stop(self):
        """Stop the mock execution environment."""
        if self._is_running:
            self._server_instance = None
            self._is_running = False
            log.info(f"Stopped mock client: {self.machine_id}")

    def is_running(self) -> bool:
        """Check if the mock client is currently running."""
        return self._is_running

    # Tensor management methods
    def _create_empty_tensor_impl(
        self,
        tensor_id: int,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
        nbytes: int,
    ) -> None:
        """Implementation: Create an empty tensor on the remote machine with proper storage layout."""
        try:
            self._server_instance.create_empty_tensor.local(
                tensor_id, shape, stride, storage_offset, dtype, nbytes
            )
            log.info(
                f"Created empty tensor {tensor_id} with shape {shape} and storage {nbytes} bytes (mock)"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create empty tensor {tensor_id}: {e}") from e

    def _create_tensor_view_impl(
        self,
        new_tensor_id: int,
        base_tensor_id: int,
        shape: List[int],
        stride: List[int],
        offset: int,
    ) -> None:
        """Implementation: Create a tensor view from an existing tensor."""
        try:
            self._server_instance.create_tensor_view.local(
                new_tensor_id, base_tensor_id, shape, stride, offset
            )
            log.info(
                f"Created tensor view {new_tensor_id} from tensor {base_tensor_id} (mock)"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create tensor view {new_tensor_id}: {e}"
            ) from e

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
        # Execute using .local() with queue handling to mirror ModalClient exactly (fire-and-forget)
        self._server_instance.update_tensor.local(
            tensor_id,
            raw_data,
            source_shape,
            source_stride,
            source_storage_offset,
            source_dtype,
        )
        log.info(f"Updated tensor {tensor_id} (mock)")

    def get_tensor_by_id(
        self,
        tensor_id: int,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
    ) -> torch.Tensor:
        """
        Get tensor data by tensor ID and reconstruct as tensor with specified view.

        Args:
            tensor_id: The tensor ID (metadata hash)
            shape: Tensor shape for view
            stride: Tensor stride for view
            storage_offset: Storage offset for view
            dtype: Tensor data type

        Returns:
            CPU tensor reconstructed with specified view
        """
        # Get raw data
        raw_bytes_future = self.get_storage_data(tensor_id)
        raw_bytes = raw_bytes_future.result()

        # Convert raw bytes to tensor with specified view
        torch_dtype = getattr(torch, dtype)
        untyped_storage = torch.UntypedStorage.from_buffer(raw_bytes, dtype=torch.uint8)
        tensor = torch.empty(0, dtype=torch_dtype, device="cpu")
        tensor.set_(untyped_storage, storage_offset, shape, stride)
        return tensor

    def _get_storage_data_impl(self, tensor_id: int) -> Future[bytes]:
        """Implementation: Get raw storage data by tensor ID."""
        # Create a future and set its result immediately
        future = Future[bytes]()

        # Execute using .local() with queue handling to mirror ModalClient exactly
        self._server_instance.get_storage_data.local(tensor_id)

        raw_bytes = self._response_queue.get()
        if raw_bytes is None:
            raise RuntimeError(f"Failed to retrieve tensor data for tensor {tensor_id}")

        # Set the result in the future
        future.set_result(raw_bytes)

        return future

    def _remove_tensors_impl(self, tensor_ids: List[int]) -> None:
        """Implementation: Remove multiple tensors from the remote machine."""
        if not tensor_ids:
            return

        self._server_instance.remove_tensors.local(tensor_ids)
        log.info(f"Removed {len(tensor_ids)} tensors (mock)")

    def _resize_storage_impl(self, tensor_id: int, nbytes: int) -> None:
        """Implementation: Resize the underlying storage for a tensor."""
        self._server_instance.resize_storage.local(tensor_id, nbytes)
        log.info(f"Resized storage for tensor {tensor_id} to {nbytes} bytes (mock)")

    # Operation execution methods
    def _execute_aten_operation_impl(
        self,
        op_name: str,
        input_tensor_ids: List[int],
        output_tensor_ids: List[int],
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        return_metadata: bool = False,
    ) -> Union[Future[List[Dict[str, Any]]], None]:
        """Implementation: Execute an aten operation on the remote machine with tensor IDs."""
        log.info(
            f"Mock Client executing {op_name} with inputs: {input_tensor_ids}, outputs: {output_tensor_ids}"
        )

        # Execute using .local()
        self._server_instance.execute_aten_operation.local(
            op_name,
            input_tensor_ids,
            output_tensor_ids,
            args,
            kwargs,
            tensor_mask,
            return_metadata,
        )

        if return_metadata:
            result = self._response_queue.get()
            # Create a future and set its result immediately
            future = Future[List[Dict[str, Any]]]()
            future.set_result(result)
            return future
        else:
            return None

    # HuggingFace model loading methods
    def _prepare_huggingface_model_impl(
        self,
        checkpoint: str,
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> Future[Dict[str, Any]]:
        """Implementation: Download and prepare a HuggingFace model directly on the remote machine."""
        # Create a future and set its result immediately
        future = Future[Dict[str, Any]]()

        # Execute using .local() with queue handling to mirror ModalClient exactly
        self._server_instance.prepare_huggingface_model.local(
            checkpoint, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
        )

        result = self._response_queue.get()

        if result is None:
            raise RuntimeError(f"Failed to prepare model {checkpoint}")

        # Set the result in the future
        future.set_result(result)

        log.info(f"Prepared HuggingFace model {checkpoint} (mock)")
        return future

    def _link_model_tensors_impl(
        self,
        local_tensor_ids: List[int],
        parameter_names: List[str],
    ) -> None:
        """Implementation: Link local mycelya tensor IDs to remote model parameter tensors."""
        self._server_instance.link_model_tensors.local(
            local_tensor_ids, parameter_names
        )
        log.info(f"Linked {len(local_tensor_ids)} model tensors (mock)")

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return (
            f'MockClient(gpu_type="{self.gpu_type}", '
            f'machine_id="{self.machine_id}", status="{status}")'
        )
