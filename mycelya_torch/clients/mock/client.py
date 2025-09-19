# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Mock client implementation for mycelya_torch.

This module provides the MockClient class that uses Modal's .local() execution
for development and testing without requiring remote cloud resources.
"""

from typing import Any, Dict, List

from ..._logging import get_logger
from ...servers.mock.server import create_mock_modal_app
from ..base_client import BatchCall, Client

log = get_logger(__name__)


class MockClient(Client):
    """
    Client interface for mock execution using Modal's .local() calls.

    This class provides a mock execution environment that reuses the existing Modal app
    but executes all methods locally using .local() instead of .remote() or .spawn().
    This mirrors the ModalClient structure exactly for testing consistency.
    """

    def __init__(self, machine_id: str):
        super().__init__(machine_id)
        self._server_instance = None

    def start(
        self,
        gpu_type: str,
        gpu_count: int,
        packages: List[str],
        python_version: str,
    ):
        """Start the mock execution environment."""
        # Create mock server instance directly without app context
        _, server_class = create_mock_modal_app()
        self._server_instance = server_class()

    def stop(self):
        """Stop the mock execution environment."""
        self._server_instance = None

    def get_rpc_result(self, rpc_result: Any, blocking: bool) -> Any | None:
        """Get the result from an RPC call."""
        # For Mock, rpc_result is already the resolved value - always available
        return rpc_result

    def execute_batch(self, batch_calls: List[BatchCall]) -> Any:
        """Execute a batch of operations via Mock."""
        return self._server_instance.execute_batch.local(batch_calls)

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

    def create_tensor_view(
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
        # Execute using .local() with queue handling to mirror ModalClient exactly (fire-and-forget)
        self._server_instance.update_tensor.local(
            tensor_id,
            raw_data,
            source_shape,
            source_stride,
            source_storage_offset,
            source_dtype,
        )

    def get_storage_data(self, tensor_id: int) -> Any:
        """Implementation: Get raw storage data by tensor ID."""
        return self._server_instance.get_storage_data.local(tensor_id)

    def remove_tensors(self, tensor_ids: List[int]) -> None:
        """Implementation: Remove multiple tensors from the remote machine."""
        self._server_instance.remove_tensors.local(tensor_ids)

    def resize_storage(self, tensor_id: int, nbytes: int) -> None:
        """Implementation: Resize the underlying storage for a tensor."""
        self._server_instance.resize_storage.local(tensor_id, nbytes)

    def copy_tensor(
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
    def execute_aten_operation(
        self,
        op_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        output_tensor_ids: List[int] | None = None,
    ) -> Any:
        """Implementation: Execute an aten operation on the remote machine with tensor IDs."""
        return self._server_instance.execute_aten_operation.local(
            op_name,
            args,
            kwargs,
            tensor_mask,
            output_tensor_ids,
        )

    # HuggingFace model loading methods
    def load_huggingface_state_dicts(
        self,
        repo: str,
        path: str,
        device_type: str,
        device_index: int,
    ) -> Any:
        """Implementation: Load HuggingFace state dicts organized by directory on the remote machine."""
        return self._server_instance.load_huggingface_state_dicts.local(
            repo, path, device_type, device_index
        )

    def link_tensors(
        self,
        local_tensor_ids: List[int],
        temp_keys: List[str],
    ) -> None:
        """Implementation: Link local mycelya tensor IDs to remote tensors from temporary registry."""
        self._server_instance.link_tensors.local(local_tensor_ids, temp_keys)

    def execute_function(self, pickled_function: bytes) -> Any:
        """Implementation: Execute a pickled function remotely."""
        return self._server_instance.execute_function.local(pickled_function)

    def pip_install(self, packages: List[str]) -> None:
        """Implementation: No-op for mock client - packages are already available locally."""
        # Mock client does nothing for pip install since it uses local execution
        pass
