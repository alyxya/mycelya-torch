# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Mock client implementation for mycelya_torch.

This module provides the MockClient class that uses Modal's .local() execution
for development and testing without requiring remote cloud resources.
"""

from typing import Any, Dict, List, Optional

from _mycelya_torch_modal.modal_app import create_modal_app_for_gpu

from ..._logging import get_logger
from ..._package_version import get_python_version, get_versioned_packages
from ..base_client import BatchCall, Client

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
        batching: bool = True,
    ):
        super().__init__(gpu_type, machine_id, batching)
        self._server_instance = None
        self._is_running = False
        self.timeout = timeout

        # Get synchronized package versions and Python version from local environment
        base_packages = ["numpy", "torch", "huggingface_hub", "safetensors"]
        versioned_packages = get_versioned_packages(base_packages)
        python_version = get_python_version()

        # Initialize the Modal app and server class with synchronized versions (local execution)
        self._app, self._server_class = create_modal_app_for_gpu(
            self.gpu_type, self.timeout, versioned_packages, python_version, is_local=True
        )

    def start(self):
        """Start the mock execution environment."""
        if not self._is_running:
            # Create server instance directly without app context
            self._server_instance = self._server_class()
            self._is_running = True

    def stop(self):
        """Stop the mock execution environment."""
        if self._is_running:
            self._server_instance = None
            self._is_running = False

    def is_running(self) -> bool:
        """Check if the mock client is currently running."""
        return self._is_running

    def resolve_futures(self) -> None:
        """Resolve pending futures using result deque."""
        if not self.is_running():
            return

        # Process all available results from the deque
        while self._pending_results and self._pending_futures:
            result = self._pending_results.popleft()

            # Resolve futures based on batching mode
            if self.batching:
                # Batch result - iterate over list
                for res in result:
                    self._pending_futures.popleft().set_result(res)
            else:
                # Individual result
                self._pending_futures.popleft().set_result(result)

    def propagate_exception_to_futures(self, exception: Exception) -> None:
        """Propagate the given exception to all pending futures."""
        # Set exception on all pending futures
        while self._pending_futures:
            future = self._pending_futures.popleft()
            if not future.cancelled():
                future.set_exception(exception)

        # Clear the pending results queue since they're no longer valid
        self._pending_results.clear()

    def _execute_batch_impl(self, batch_calls: List[BatchCall]) -> None:
        """Execute a batch of operations via Mock."""
        if not batch_calls:
            return

        # Use local execution for batch and store result
        result = self._server_instance.execute_batch.local(batch_calls)
        self._pending_results.append(result)

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
        try:
            self._server_instance.create_empty_tensor.local(
                tensor_id,
                shape,
                stride,
                storage_offset,
                dtype,
                nbytes,
                device_type,
                device_index,
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

    def _get_storage_data_impl(self, tensor_id: int) -> None:
        """Implementation: Get raw storage data by tensor ID."""
        # Execute local call and store result for resolve_futures
        result = self._server_instance.get_storage_data.local(tensor_id)
        self._pending_results.append(result)

    def _remove_tensors_impl(self, tensor_ids: List[int]) -> None:
        """Implementation: Remove multiple tensors from the remote machine."""
        if not tensor_ids:
            return

        self._server_instance.remove_tensors.local(tensor_ids)

    def _resize_storage_impl(self, tensor_id: int, nbytes: int) -> None:
        """Implementation: Resize the underlying storage for a tensor."""
        self._server_instance.resize_storage.local(tensor_id, nbytes)

    def _copy_tensor_impl(
        self,
        source_tensor_id: int,
        target_tensor_id: int,
    ) -> None:
        """Implementation: Copy tensor data from source to target on the remote machine."""
        try:
            self._server_instance.copy_tensor.local(source_tensor_id, target_tensor_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to copy tensor from {source_tensor_id} to {target_tensor_id}: {e}"
            ) from e

    # Operation execution methods
    def _execute_aten_operation_impl(
        self,
        op_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        output_tensor_ids: Optional[List[int]] = None,
    ) -> None:
        """Implementation: Execute an aten operation on the remote machine with tensor IDs."""

        # Execute using .local() and store result if returning metadata
        result = self._server_instance.execute_aten_operation.local(
            op_name,
            args,
            kwargs,
            tensor_mask,
            output_tensor_ids,
        )
        # Only store result if expecting a return value for dynamic operations
        if output_tensor_ids is None and result is not None:
            self._pending_results.append(result)

    # HuggingFace model loading methods
    def _load_huggingface_state_dicts_impl(
        self,
        repo: str,
        path: str,
        device_type: str,
        device_index: int,
    ) -> None:
        """Implementation: Load HuggingFace state dicts organized by directory on the remote machine."""
        # Execute local call and store result for resolve_futures
        result = self._server_instance.load_huggingface_state_dicts.local(
            repo, path, device_type, device_index
        )
        self._pending_results.append(result)

    def _link_tensors_impl(
        self,
        local_tensor_ids: List[int],
        temp_keys: List[str],
    ) -> None:
        """Implementation: Link local mycelya tensor IDs to remote tensors from temporary registry."""
        self._server_instance.link_tensors.local(local_tensor_ids, temp_keys)

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return (
            f'MockClient(gpu_type="{self.gpu_type}", '
            f'machine_id="{self.machine_id}", status="{status}")'
        )
