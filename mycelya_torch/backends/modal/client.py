# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal client implementation for mycelya_torch.

This module provides the ModalClient class for interfacing with Modal cloud GPUs,
along with related functionality for creating and managing Modal applications.
"""

from typing import Any, Dict, List, Optional

import torch

from _mycelya_torch_modal.modal_app import create_modal_app_for_gpu

from ..._logging import get_logger
from ..client import Client

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
    ):
        super().__init__(gpu_type, machine_id)
        self._app = None
        self._server_class = None
        self._server_instance = None
        self._response_queue = None
        self._app_context = None
        self.timeout = timeout

        # Note: Tensor ID tracking moved to orchestrator

        # Initialize the Modal app and server
        self._initialize()

    def _initialize(self):
        """Initialize the Modal app and server class."""
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

        # Call Modal method which will write result to queue
        self._server_instance.create_empty_tensor.remote(
            tensor_id, shape, stride, storage_offset, dtype
        )

        # Poll queue for result
        self._response_queue.get()

        # Note: Tensor ID tracking moved to orchestrator
        log.info(f"Created empty tensor {tensor_id} with shape {shape}")

    def create_tensor_view(
        self,
        new_tensor_id: int,
        base_tensor_id: int,
        shape: List[int],
        stride: List[int],
        offset: int,
    ) -> None:
        """
        Create a tensor view on the remote machine from an existing tensor using as_strided.

        Args:
            new_tensor_id: Unique tensor ID (metadata hash) for the new view
            base_tensor_id: Tensor ID of the base tensor to create view from
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

        # Call Modal method which will write result to queue
        self._server_instance.create_tensor_view.remote(
            new_tensor_id, base_tensor_id, shape, stride, offset
        )

        # Poll queue for result
        self._response_queue.get()

        # Note: Tensor ID tracking moved to orchestrator
        log.info(f"Created tensor view {new_tensor_id} from tensor {base_tensor_id}")

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

        # Call Modal method directly (fire-and-forget)
        self._server_instance.update_tensor.remote(
            tensor_id,
            raw_data,
            source_shape,
            source_stride,
            source_storage_offset,
            source_dtype,
        )
        log.info(f"Updated tensor {tensor_id}")

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

        # Call Modal method which will write result to queue
        self._server_instance.get_storage_data.remote(tensor_id)

        # Poll queue for result
        raw_bytes = self._response_queue.get()

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

        # Call Modal method directly (fire-and-forget)
        self._server_instance.remove_tensors.remote(tensor_ids)

        # Note: Tensor ID tracking moved to orchestrator

        log.info(f"Removed {len(tensor_ids)} tensors")

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

        # Call Modal method directly (fire-and-forget)
        self._server_instance.resize_storage.remote(tensor_id, nbytes)
        log.info(f"Resized storage for tensor {tensor_id} to {nbytes} bytes")

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
        Execute an aten operation on the remote machine with input and output tensors.

        Args:
            op_name: The aten operation name to execute
            input_tensors: List of input tensors
            output_tensors: List of output tensors to store results
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

        # Note: Input tensor existence checking moved to orchestrator

        # Call Modal method which will write result to queue
        self._server_instance.execute_aten_operation.remote(
            op_name,
            input_tensor_ids,
            output_tensor_ids,
            args,
            kwargs,
            tensor_mask,
            return_metadata,
        )

        # Poll queue for result
        result = self._response_queue.get()

        # Note: Output tensor ID tracking moved to orchestrator

        # Return result if metadata was requested, otherwise return None
        return result if return_metadata else None

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

        # Call Modal method which will write result to queue
        self._server_instance.prepare_huggingface_model.remote(
            checkpoint, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
        )

        # Poll queue for result
        result = self._response_queue.get()

        if result is None:
            raise RuntimeError(f"Failed to prepare model {checkpoint}")

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

        # Call Modal method directly (fire-and-forget)
        self._server_instance.link_model_tensors.remote(
            local_tensor_ids, parameter_names
        )

        # Note: Tensor ID tracking moved to orchestrator

        log.info(f"Linked {len(local_tensor_ids)} model tensors")

    # Note: _ensure_tensor_exists method removed - tensor existence checking moved to orchestrator

    # Removed batch execution support - using direct calls now

    def _get_tensor_ids_for_storage(self, storage_id: int) -> List[int]:
        """Get tensor IDs associated with a storage ID."""
        # For now, return empty list as this is used for cleanup
        # Could be implemented using storage tracking if needed
        return []

    def remove_storage(self, storage_id: int) -> bool:
        """Remove storage by ID."""
        # For now, just return True as this is used for cleanup
        # Could be implemented properly if needed
        return True

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return (
            f'ModalClient(gpu_type="{self.gpu_type}", '
            f'machine_id="{self.machine_id}", status="{status}")'
        )
