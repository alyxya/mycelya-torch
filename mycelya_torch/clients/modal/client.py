# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal client implementation for mycelya_torch.

This module provides the ModalClient class for interfacing with Modal cloud GPUs,
along with related functionality for creating and managing Modal applications.
"""

from typing import Any, Dict, List

from ..._logging import get_logger
from ..._package_version import get_python_version, get_versioned_packages
from ...servers.modal.server import create_modal_app_for_gpu
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
        machine_id: str,
        gpu_type: str,
        packages: List[str],
        timeout: int | None = None,
    ):
        super().__init__(machine_id)
        self.gpu_type = gpu_type
        self.timeout = timeout
        self._server_instance = None
        self._app_context = None

        # Get versioned packages and Python version from local environment
        versioned_packages = get_versioned_packages(packages)
        python_version = get_python_version()

        # Initialize the Modal app and server class with synchronized versions
        self._app, self._server_class = create_modal_app_for_gpu(
            gpu_type=self.gpu_type,
            packages=versioned_packages,
            python_version=python_version,
            timeout=self.timeout,
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

    def try_get_rpc_result(self, rpc_result: Any) -> Any | None:
        """Non-blocking attempt to get the result from an RPC call."""
        # For Modal, rpc_result is a FunctionCall object - try with zero timeout
        try:
            return rpc_result.get(timeout=0)
        except TimeoutError:
            return None

    def execute_batch(self, batch_calls: List[BatchCall]) -> Any:
        """Execute a batch of operations via Modal."""
        if not batch_calls:
            return None

        # Use spawn for non-blocking execution and return FunctionCall for batch results
        func_call = self._server_instance.execute_batch.spawn(batch_calls)
        return func_call

    # Tensor management methods
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

    def create_tensor_view(
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

    def update_tensor(
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

    def get_storage_data(self, tensor_id: int) -> Any:
        """Implementation: Get raw storage data by tensor ID."""
        # Trigger the remote call and return FunctionCall - result will be available via resolve_futures
        func_call = self._server_instance.get_storage_data.spawn(tensor_id)
        return func_call

    def remove_tensors(self, tensor_ids: List[int]) -> None:
        """Implementation: Remove multiple tensors from the remote machine."""
        if not tensor_ids:
            return

        self._server_instance.remove_tensors.spawn(tensor_ids)

    def resize_storage(self, tensor_id: int, nbytes: int) -> None:
        """Implementation: Resize the underlying storage for a tensor."""
        self._server_instance.resize_storage.spawn(tensor_id, nbytes)

    def copy_tensor(
        self,
        source_tensor_id: int,
        target_tensor_id: int,
    ) -> None:
        """Implementation: Copy tensor data from source to target on the remote machine."""
        self._server_instance.copy_tensor.spawn(source_tensor_id, target_tensor_id)

    # Operation execution methods
    def execute_aten_operation(
        self,
        op_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        output_tensor_ids: List[int] | None = None,
    ) -> Any:
        """Implementation: Execute an aten operation on the remote machine with tensor IDs."""
        # Call Modal method and return FunctionCall
        func_call = self._server_instance.execute_aten_operation.spawn(
            op_name,
            args,
            kwargs,
            tensor_mask,
            output_tensor_ids,
        )
        return func_call

    # HuggingFace model loading methods
    def load_huggingface_state_dicts(
        self,
        repo: str,
        path: str,
        device_type: str,
        device_index: int,
    ) -> Any:
        """Implementation: Load HuggingFace state dicts organized by directory on the remote machine."""
        # Trigger the remote call and return FunctionCall - result will be available via resolve_futures
        func_call = self._server_instance.load_huggingface_state_dicts.spawn(
            repo, path, device_type, device_index
        )
        return func_call

    def link_tensors(
        self,
        local_tensor_ids: List[int],
        temp_keys: List[str],
    ) -> None:
        """Implementation: Link local mycelya tensor IDs to remote tensors from temporary registry."""
        self._server_instance.link_tensors.spawn(local_tensor_ids, temp_keys)

    def execute_function(self, pickled_function: bytes) -> Any:
        """Implementation: Execute a pickled function remotely."""
        func_call = self._server_instance.execute_function.spawn(pickled_function)
        return func_call
