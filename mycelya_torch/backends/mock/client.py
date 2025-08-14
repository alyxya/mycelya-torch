# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Mock client implementation for mycelya_torch.

This module provides the MockClient class that uses Modal's .local() execution
for development and testing without requiring remote cloud resources.
"""

from typing import Any, Dict, List, Optional, Set

import torch

from _mycelya_torch_modal.modal_app import create_modal_app_for_gpu

from ..._logging import get_logger
from ..client import Client

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
        retries: int,
    ):
        super().__init__(gpu_type, machine_id)
        self._app = None
        self._server_class = None
        self._server_instance = None
        self._is_running = False
        self.timeout = timeout
        self.retries = retries

        # Track which tensor IDs exist on the remote side (mirroring ModalClient)
        self._remote_tensor_ids: Set[int] = set()

        # Initialize the Modal app and server
        self._initialize()

    def _initialize(self):
        """Initialize the Modal app and server class."""
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

    # Tensor management methods
    def create_empty_tensor(
        self,
        tensor_id: int,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
    ) -> None:
        """
        Create an empty tensor on the remote machine with proper storage layout.

        Args:
            tensor_id: Unique tensor ID (metadata hash)
            shape: Shape of the tensor
            stride: Stride of the tensor
            storage_offset: Storage offset for the tensor
            dtype: Data type of the tensor (e.g., "float32", "int64")

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        try:
            # Execute using .local() instead of queuing for batching (mock behavior)
            self._server_instance.create_empty_tensor.local(
                tensor_id, shape, stride, storage_offset, dtype
            )
            self._remote_tensor_ids.add(tensor_id)
            log.info(f"Created empty tensor {tensor_id} with shape {shape} (mock)")
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
        """
        Create a tensor view from an existing tensor.

        Args:
            new_tensor_id: Tensor ID for the new view
            base_tensor_id: Tensor ID of the base tensor
            shape: Shape of the view
            stride: Stride of the view
            offset: Storage offset of the view

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        try:
            # Execute using .local() instead of queuing for batching (mock behavior)
            self._server_instance.create_tensor_view.local(
                new_tensor_id, base_tensor_id, shape, stride, offset
            )
            self._remote_tensor_ids.add(new_tensor_id)
            log.info(
                f"Created tensor view {new_tensor_id} from tensor {base_tensor_id} (mock)"
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
        """
        Update an existing tensor with new data and source metadata.

        Args:
            tensor_id: Tensor ID to update
            raw_data: Raw bytes of the tensor data
            source_shape: Shape of the source data
            source_stride: Stride of the source data
            source_storage_offset: Storage offset of the source data
            source_dtype: Data type of the source data

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Execute using .local() instead of queuing for batching (mock behavior)
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
        raw_bytes = self.get_storage_data(tensor_id)

        # Convert raw bytes to tensor with specified view
        torch_dtype = getattr(torch, dtype.replace("torch.", ""))
        untyped_storage = torch.UntypedStorage.from_buffer(raw_bytes, dtype=torch.uint8)
        tensor = torch.empty(0, dtype=torch_dtype, device="cpu")
        tensor.set_(untyped_storage, storage_offset, shape, stride)
        return tensor

    def get_storage_data(self, tensor_id: int) -> bytes:
        """
        Get raw storage data by tensor ID.

        Args:
            tensor_id: The tensor ID (metadata hash)

        Returns:
            Raw storage data as bytes
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Execute using .local() instead of remote call (mock behavior)
        raw_bytes = self._server_instance.get_storage_data.local(tensor_id)
        if raw_bytes is None:
            raise RuntimeError(f"Failed to retrieve tensor data for tensor {tensor_id}")

        return raw_bytes

    def remove_tensors(self, tensor_ids: List[int]) -> None:
        """
        Remove multiple tensors from the remote machine.

        Args:
            tensor_ids: List of tensor IDs to remove

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        if not tensor_ids:
            return

        # Execute using .local() instead of queuing for batching (mock behavior)
        self._server_instance.remove_tensors.local(tensor_ids)

        # Remove from tracked tensor IDs
        for tid in tensor_ids:
            self._remote_tensor_ids.discard(tid)

        log.info(f"Removed {len(tensor_ids)} tensors (mock)")

    def resize_storage(self, tensor_id: int, nbytes: int) -> None:
        """
        Resize the underlying storage for a tensor.

        Args:
            tensor_id: The tensor ID
            nbytes: The number of bytes needed for the new storage size

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Execute using .local() instead of queuing for batching (mock behavior)
        self._server_instance.resize_storage.local(tensor_id, nbytes)
        log.info(f"Resized storage for tensor {tensor_id} to {nbytes} bytes (mock)")

    # Operation execution methods
    def execute_aten_operation(
        self,
        op_name: str,
        input_tensors: List["torch.Tensor"],
        output_tensors: List["torch.Tensor"],
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        return_metadata: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute an aten operation on the remote machine with input tensor IDs.

        Args:
            op_name: The aten operation name to execute
            input_tensor_ids: List of input tensor IDs
            output_tensor_ids: List of tensor IDs to store results (all output tensors)
            args: Operation arguments (with tensor IDs replacing tensors)
            kwargs: Operation keyword arguments (with tensor IDs replacing tensors)
            tensor_mask: Boolean mask indicating which positions in args/kwargs had tensors
            return_metadata: If True, return output tensor metadata instead of None

        Returns:
            None for normal operations, or List[Dict] of output tensor metadata if return_metadata=True
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Extract tensor IDs from tensors
        input_tensor_ids = [tensor._get_tensor_id() for tensor in input_tensors]
        output_tensor_ids = [tensor._get_tensor_id() for tensor in output_tensors]

        # Ensure input tensors exist on remote (mirroring ModalClient logic)
        for tensor_id in input_tensor_ids:
            if tensor_id not in self._remote_tensor_ids:
                # Input tensor should already exist on remote
                # But for some operations like randn, empty tensors are created and then filled
                # Log a warning but continue - the server will handle missing tensors appropriately
                log.warning(f"Input tensor {tensor_id} not found in client registry. Server will attempt to find it.")
        log.info(f"Mock Client executing {op_name} with inputs: {input_tensor_ids}, outputs: {output_tensor_ids}")

        # Execute using .local() instead of remote call (mock behavior)
        result = self._server_instance.execute_aten_operation.local(
            op_name,
            input_tensor_ids,
            output_tensor_ids,
            args,
            kwargs,
            tensor_mask,
            return_metadata,
        )

        # Track output tensor IDs (mirroring ModalClient logic)
        for tensor_id in output_tensor_ids:
            self._remote_tensor_ids.add(tensor_id)

        # Return result if requested
        if return_metadata:
            return result
        else:
            return None

    # HuggingFace model loading methods
    def prepare_huggingface_model(
        self,
        checkpoint: str,
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> Dict[str, Any]:
        """
        Download and prepare a HuggingFace model directly on the remote machine.

        Args:
            checkpoint: HuggingFace model checkpoint
            torch_dtype: Data type for model weights
            trust_remote_code: Whether to trust remote code

        Returns:
            Model metadata dictionary
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Execute using .local() instead of remote call (mock behavior)
        result = self._server_instance.prepare_huggingface_model.local(
            checkpoint, torch_dtype, trust_remote_code
        )

        if result is None:
            raise RuntimeError(f"Failed to prepare model {checkpoint}")

        log.info(f"Prepared HuggingFace model {checkpoint} (mock)")
        return result

    def link_model_tensors(
        self,
        local_tensor_ids: List[int],
        parameter_names: List[str],
    ) -> None:
        """
        Link local mycelya tensor IDs to remote model parameter tensors.

        Args:
            local_tensor_ids: List of local tensor IDs from created mycelya tensors
            parameter_names: List of parameter names corresponding to each tensor ID

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Execute using .local() instead of remote call (mock behavior)
        self._server_instance.link_model_tensors.local(
            local_tensor_ids, parameter_names
        )

        # Track the linked tensor IDs (mirroring ModalClient logic)
        for tensor_id in local_tensor_ids:
            self._remote_tensor_ids.add(tensor_id)

        log.info(f"Linked {len(local_tensor_ids)} model tensors (mock)")

    # Helper method to ensure tensor exists on remote (mirroring ModalClient)
    def _ensure_tensor_exists(self, tensor: torch.Tensor) -> int:
        """
        Ensure a tensor exists on the remote side, creating it if necessary.

        This method determines whether to create a new empty tensor or a view
        based on whether other tensors share the same storage.

        Args:
            tensor: The mycelya tensor to ensure exists remotely

        Returns:
            The tensor ID (metadata hash)
        """
        # Get tensor ID as int
        tensor_id = tensor._get_tensor_id()

        # Check if tensor already exists
        if tensor_id in self._remote_tensor_ids:
            return tensor_id

        # TODO: In future, implement view detection by tracking storage relationships
        # For now, we'll always create empty tensors

        # Create empty tensor on remote
        self.create_empty_tensor(
            tensor_id=tensor_id,
            shape=list(tensor.shape),
            stride=list(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            dtype=str(tensor.dtype).replace("torch.", ""),
        )

        return tensor_id

    # Batch execution support (mirroring ModalClient)
    def execute_batch(self) -> List[Any]:
        """
        Execute all queued RPCs in a single batch.

        This is called by the orchestrator's batch processing thread.
        For MockClient, we don't actually queue calls, so this is a no-op.

        Returns:
            Empty list since MockClient executes calls immediately
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # MockClient executes calls immediately, so no batched calls to execute
        # But we need to maintain the same interface as ModalClient
        return []

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return (
            f'MockClient(gpu_type="{self.gpu_type}", '
            f'machine_id="{self.machine_id}", status="{status}")'
        )
