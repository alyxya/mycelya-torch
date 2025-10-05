# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Mock client implementation for mycelya_torch.

This module provides the MockClient class that uses Modal's .local() execution
for development and testing without requiring remote cloud resources.
"""

from typing import Any

from ..._logging import get_logger
from ..._utils import TensorMetadata
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
        packages: list[str],
        python_version: str,
    ):
        """Start the mock execution environment."""
        # Create mock server instance directly without app context
        _, server_class = create_mock_modal_app(machine_id=self.machine_id)
        self._server_instance = server_class()

    def stop(self):
        """Stop the mock execution environment."""
        self._server_instance = None

    def get_rpc_result(self, rpc_result: Any, blocking: bool) -> Any | None:
        """Get the result from an RPC call."""
        # For Mock, rpc_result is already the resolved value - always available
        return rpc_result

    def execute_batch(self, batch_calls: list[BatchCall]) -> Any:
        """Execute a batch of operations via Mock."""
        return self._server_instance.execute_batch.local(batch_calls)

    # Tensor management methods
    def create_tensor(self, metadata: TensorMetadata) -> None:
        """Implementation: Create a tensor on the remote machine.

        Creates either a new empty tensor or a tensor view based on metadata.alias_id:
        - If alias_id is None: Creates new empty tensor
        - If alias_id is int: Creates tensor view using alias_id as base tensor
        """
        self._server_instance.create_tensor.local(metadata)

    def update_tensor(
        self,
        tensor_id: int,
        raw_data: bytes,
        source_shape: list[int],
        source_stride: list[int],
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

    def remove_tensors(self, tensor_ids: list[int]) -> None:
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
        self._server_instance.copy_tensor.local(source_tensor_id, target_tensor_id)

    # Operation execution methods
    def execute_aten_operation(
        self,
        op_name: str,
        args: list[Any],
        kwargs: dict[str, Any],
        tensor_mask: list[bool],
        output_tensor_ids: list[int] | None = None,
    ) -> Any:
        """Implementation: Execute an aten operation on the remote machine with tensor IDs."""
        return self._server_instance.execute_aten_operation.local(
            op_name,
            args,
            kwargs,
            tensor_mask,
            output_tensor_ids,
        )

    def link_tensors(
        self,
        tensor_ids: list[int],
        temp_ids: list[str],
    ) -> None:
        """Implementation: Link local mycelya tensor IDs to remote tensors from temporary registry."""
        self._server_instance.link_tensors.local(tensor_ids, temp_ids)

    def execute_function(self, pickled_function: bytes) -> Any:
        """Implementation: Execute a pickled function remotely."""
        return self._server_instance.execute_function.local(pickled_function)

    def pip_install(self, packages: list[str]) -> None:
        """Implementation: No-op for mock client - packages are already available locally."""
        # Mock client does nothing for pip install since it uses local execution
        pass

    def offload(self) -> None:
        """Implementation: No-op for mock client - no persistence needed for local execution."""
        pass

    def reload(self) -> None:
        """Implementation: No-op for mock client - no persistence needed for local execution."""
        pass
