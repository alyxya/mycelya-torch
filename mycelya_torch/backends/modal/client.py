# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal client implementation for mycelya_torch.

This module provides the ModalClient class for interfacing with Modal cloud GPUs,
along with related functionality for creating and managing Modal applications.
"""

from collections import deque
from typing import Any, Dict, List

from _mycelya_torch_modal.modal_app import create_modal_app_for_gpu

from ..._logging import get_logger
from ..base_client import BatchCall, Client

log = get_logger(__name__)


class ModalClient(Client):
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
        batching: bool = True,
    ):
        super().__init__(gpu_type, machine_id, batching)
        self._server_instance = None
        self._app_context = None
        self.timeout = timeout

        # Deque for storing FunctionCall objects that may return values
        self._pending_results = deque()

        # Initialize the Modal app and server class
        self._app, self._server_class, self._response_queue = create_modal_app_for_gpu(
            self.gpu_type, self.machine_id, self.timeout
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

    def resolve_futures(self) -> None:
        """Resolve pending futures using FunctionCall polling."""
        if not self.is_running():
            return

        # Poll FunctionCall objects for completed results
        while self._pending_results and self._pending_futures:
            func_call = self._pending_results[0]  # Peek at first
            try:
                # Try to get result with zero timeout (non-blocking)
                result = func_call.get(timeout=0)
                # Result is ready, remove from deque
                self._pending_results.popleft()

                # Resolve futures based on batching mode
                if self.batching:
                    # Batch result - iterate over list
                    for res in result:
                        self._pending_futures.popleft().set_result(res)
                else:
                    # Individual result
                    self._pending_futures.popleft().set_result(result)

            except TimeoutError:
                break

    def _execute_batch_impl(self, batch_calls: List[BatchCall]) -> None:
        """Execute a batch of operations via Modal."""
        if not batch_calls:
            return

        # Use spawn for non-blocking execution and capture FunctionCall for batch results
        func_call = self._server_instance.execute_batch.spawn(batch_calls)
        self._pending_results.append(func_call)

    # Tensor management methods
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
        self._server_instance.create_empty_tensor.spawn(
            tensor_id,
            shape,
            stride,
            storage_offset,
            dtype,
            nbytes,
            device_type,
            device_index,
        )

    def _create_tensor_view_impl(
        self,
        new_tensor_id: int,
        base_tensor_id: int,
        shape: List[int],
        stride: List[int],
        offset: int,
    ) -> None:
        """Implementation: Create a tensor view on the remote machine from an existing tensor using as_strided."""
        self._server_instance.create_tensor_view.spawn(
            new_tensor_id, base_tensor_id, shape, stride, offset
        )

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
        # Call Modal method directly (fire-and-forget)
        self._server_instance.update_tensor.spawn(
            tensor_id,
            raw_data,
            source_shape,
            source_stride,
            source_storage_offset,
            source_dtype,
        )

    def _get_storage_data_impl(self, tensor_id: int) -> None:
        """Implementation: Get raw storage data by tensor ID."""
        # Trigger the remote call and capture FunctionCall - result will be available via resolve_futures
        func_call = self._server_instance.get_storage_data.spawn(tensor_id)
        self._pending_results.append(func_call)

    def _remove_tensors_impl(self, tensor_ids: List[int]) -> None:
        """Implementation: Remove multiple tensors from the remote machine."""
        if not tensor_ids:
            return

        self._server_instance.remove_tensors.spawn(tensor_ids)

    def _resize_storage_impl(self, tensor_id: int, nbytes: int) -> None:
        """Implementation: Resize the underlying storage for a tensor."""
        self._server_instance.resize_storage.spawn(tensor_id, nbytes)

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
    ) -> None:
        """Implementation: Execute an aten operation on the remote machine with tensor IDs."""
        # Call Modal method and capture FunctionCall if returning metadata
        func_call = self._server_instance.execute_aten_operation.spawn(
            op_name,
            input_tensor_ids,
            output_tensor_ids,
            args,
            kwargs,
            tensor_mask,
            return_metadata,
        )
        # Only track FunctionCall if expecting a return value
        if return_metadata:
            self._pending_results.append(func_call)

    # HuggingFace model loading methods
    def _prepare_huggingface_model_impl(
        self,
        checkpoint: str,
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        """Implementation: Download and prepare a HuggingFace model directly on the remote machine."""
        # Trigger the remote call and capture FunctionCall - result will be available via resolve_futures
        func_call = self._server_instance.prepare_huggingface_model.spawn(
            checkpoint, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
        )
        self._pending_results.append(func_call)

    def _link_model_tensors_impl(
        self,
        local_tensor_ids: List[int],
        parameter_names: List[str],
    ) -> None:
        """Implementation: Link local mycelya tensor IDs to remote model parameter tensors."""
        self._server_instance.link_model_tensors.spawn(
            local_tensor_ids, parameter_names
        )

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return (
            f'ModalClient(gpu_type="{self.gpu_type}", '
            f'machine_id="{self.machine_id}", status="{status}")'
        )
