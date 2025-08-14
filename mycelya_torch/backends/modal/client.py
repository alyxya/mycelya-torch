# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal client implementation for mycelya_torch.

This module provides the ModalClient class for interfacing with Modal cloud GPUs,
along with related functionality for creating and managing Modal applications.
"""

from typing import Any, Dict, List, Optional, Set, Union

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
        retries: int,
    ):
        super().__init__(gpu_type, machine_id)
        self._app = None
        self._server_class = None
        self._server_instance = None
        self._app_context = None
        self.timeout = timeout
        self.retries = retries

        # Track which tensor IDs exist on the remote side
        self._remote_tensor_ids: Set[int] = set()

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

            # Register for RPC batching
            self._register_for_batching()

    def stop(self):
        """Stop the Modal app context for this machine."""
        # Unregister from RPC batching first
        self._unregister_for_batching()

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

        try:
            # Queue the RPC for batching (fire-and-forget)
            self._queue_rpc(
                method_name="create_empty_tensor",
                call_type="spawn",
                args=(tensor_id, shape, stride, storage_offset, dtype),
                kwargs={},
            )
            self._remote_tensor_ids.add(tensor_id)
            log.info(f"Queued creation of empty tensor {tensor_id} with shape {shape}")
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

        try:
            # Queue the RPC for batching (fire-and-forget)
            self._queue_rpc(
                method_name="create_tensor_view",
                call_type="spawn",
                args=(new_tensor_id, base_tensor_id, shape, stride, offset),
                kwargs={},
            )
            self._remote_tensor_ids.add(new_tensor_id)
            log.info(
                f"Queued creation of tensor view {new_tensor_id} from tensor {base_tensor_id}"
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

        # Queue the RPC for batching (fire-and-forget)
        self._queue_rpc(
            method_name="update_tensor",
            call_type="spawn",
            args=(
                tensor_id,
                raw_data,
                source_shape,
                source_stride,
                source_storage_offset,
                source_dtype,
            ),
            kwargs={},
            invalidate_tensor_ids=[tensor_id],
        )
        log.info(f"Queued update for tensor {tensor_id}")

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

        # Queue the RPC for batching (blocking call that returns raw bytes)
        future = self._queue_rpc(
            method_name="get_storage_data",
            call_type="remote",
            args=(tensor_id,),
            kwargs={},
        )

        # Wait for the result from the Future
        raw_bytes = future.result() if future else None
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

        # Queue the RPC for batching (fire-and-forget)
        self._queue_rpc(
            method_name="remove_tensors",
            call_type="spawn",
            args=(tensor_ids,),
            kwargs={},
        )

        # Remove from tracked tensor IDs
        for tid in tensor_ids:
            self._remote_tensor_ids.discard(tid)

        log.info(f"Queued removal of {len(tensor_ids)} tensors")

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

        # Queue the RPC for batching (fire-and-forget)
        self._queue_rpc(
            method_name="resize_storage",
            call_type="spawn",
            args=(tensor_id, nbytes),
            kwargs={},
        )
        log.info(f"Queued storage resize for tensor {tensor_id} to {nbytes} bytes")

    # Operation execution methods
    def execute_aten_operation(
        self,
        op_name: str,
        input_tensor_metadata: List[Dict[str, Any]],
        output_tensor_ids: List[Union[int, None]],
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        return_metadata: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute an aten operation on the remote machine with separated input/output specification.

        Args:
            op_name: The aten operation name to execute
            input_tensor_metadata: Metadata for reconstructing input tensors (including tensor_id)
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

        # Ensure input tensors exist on remote
        for metadata in input_tensor_metadata:
            tensor_id = metadata["tensor_id"]
            if tensor_id not in self._remote_tensor_ids:
                # Create the tensor on the remote side if it doesn't exist yet
                self.create_empty_tensor(
                    tensor_id=tensor_id,
                    shape=metadata["shape"],
                    stride=metadata["stride"],
                    storage_offset=metadata["storage_offset"],
                    dtype=metadata["dtype"],
                )

        # Queue the RPC for batching
        # Use 'remote' for operations that return metadata, 'spawn' otherwise
        call_type = "remote" if return_metadata else "spawn"

        future = self._queue_rpc(
            method_name="execute_aten_operation",
            call_type=call_type,
            args=(
                op_name,
                input_tensor_metadata,
                output_tensor_ids,
                args,
                kwargs,
                tensor_mask,
                return_metadata,
            ),
            kwargs={},
        )

        # Track output tensor IDs
        for tensor_id in output_tensor_ids:
            if tensor_id is not None:
                self._remote_tensor_ids.add(tensor_id)

        # Return result if requested
        if return_metadata:
            return future.result() if future else None
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

        # Queue the RPC for batching (blocking call)
        future = self._queue_rpc(
            method_name="prepare_huggingface_model",
            call_type="remote",
            args=(checkpoint,),
            kwargs={"torch_dtype": torch_dtype, "trust_remote_code": trust_remote_code},
        )

        # Wait for the result
        result = future.result() if future else None
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

        # Queue the RPC for batching (fire-and-forget)
        self._queue_rpc(
            method_name="link_model_tensors",
            call_type="spawn",
            args=(local_tensor_ids, parameter_names),
            kwargs={},
        )

        # Track the linked tensor IDs
        for tensor_id in local_tensor_ids:
            self._remote_tensor_ids.add(tensor_id)

        log.info(f"Linked {len(local_tensor_ids)} model tensors")

    # Helper method to ensure tensor exists on remote
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

    # Batch execution support
    def execute_batch(self) -> List[Any]:
        """
        Execute all queued RPCs in a single batch.

        This is called by the orchestrator's batch processing thread.

        Returns:
            List of results from the batched RPCs
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Get all pending calls from the batch queue
        batch_calls = self._batch_queue.get_pending_calls()

        if not batch_calls:
            return []

        # Send batch to Modal server
        try:
            results = self._server_instance.execute_batch(batch_calls)
            return results
        except Exception as e:
            log.error(f"Batch execution failed: {e}")
            # Return None for each call to indicate failure
            return [None] * len(batch_calls)

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return (
            f'ModalClient(gpu_type="{self.gpu_type}", '
            f'machine_id="{self.machine_id}", status="{status}")'
        )
