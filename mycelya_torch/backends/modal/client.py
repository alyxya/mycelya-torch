# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal client implementation for mycelya_torch.

This module provides the ModalClient class for interfacing with Modal cloud GPUs,
along with related functionality for creating and managing Modal applications.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Union

import torch

from _mycelya_torch_modal.modal_app import create_modal_app_for_gpu

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

        # Storage to tensor mapping for the new tensor-based architecture
        self._storage_to_tensor_ids: Dict[int, Set[int]] = defaultdict(set)

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

    # Storage management methods
    def create_storage(self, storage_id: int, nbytes: int) -> None:
        """
        Create a storage on the remote machine.

        Args:
            storage_id: Specific ID to use for the storage (required)
            nbytes: Number of bytes to allocate for the storage

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
                method_name="create_storage",
                call_type="spawn",
                args=(storage_id, nbytes),
                kwargs={},
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create storage {storage_id}: {e}") from e

    def update_storage(
        self,
        storage_id: int,
        storage_tensor: torch.Tensor,
        source_shape: List[int],
        source_stride: List[int],
        source_storage_offset: int,
        source_dtype: str,
        target_shape: List[int],
        target_stride: List[int],
        target_storage_offset: int,
        target_dtype: str,
    ) -> None:
        """
        Update an existing storage with storage tensor data.

        Args:
            storage_id: Storage ID to update
            storage_tensor: CPU tensor wrapping the storage data (tensor metadata is ignored, only untyped_storage matters)
            source_shape: Shape of the source data
            source_stride: Stride of the source data
            source_storage_offset: Storage offset of the source data
            source_dtype: Data type of the source data
            target_shape: Shape of the target view in storage
            target_stride: Stride of the target view in storage
            target_storage_offset: Storage offset of the target view in storage
            target_dtype: Data type of the target view in storage

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Serialize storage tensor using numpy approach
        from ..._tensor_utils import cpu_tensor_to_numpy_bytes

        # Serialize tensor using numpy approach
        numpy_bytes = cpu_tensor_to_numpy_bytes(storage_tensor)

        # Queue the RPC for batching (fire-and-forget)
        # Invalidate cache immediately since this modifies storage
        self._queue_rpc(
            method_name="update_storage",
            call_type="spawn",
            args=(
                storage_id,
                numpy_bytes,
                source_shape,
                source_stride,
                source_storage_offset,
                source_dtype,
                target_shape,
                target_stride,
                target_storage_offset,
                target_dtype,
            ),
            kwargs={},
            invalidate_storage_ids=[storage_id],
        )

    def _get_storage_data(
        self,
        storage_id: int,
    ) -> bytes:
        """
        Get raw storage data by ID.

        Args:
            storage_id: The storage ID

        Returns:
            Raw untyped storage bytes
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Queue the RPC for batching (blocking call that returns raw bytes)
        future = self._queue_rpc(
            method_name="get_storage_data",
            call_type="remote",
            args=(storage_id,),
            kwargs={},
        )

        # Wait for the result from the Future
        raw_bytes = future.result() if future else None
        if raw_bytes is None:
            raise RuntimeError(
                f"Failed to retrieve storage data for storage {storage_id}"
            )

        # Return raw bytes directly - no deserialization needed
        return raw_bytes

    def _get_storage_tensor_for_cache(
        self,
        storage_id: int,
    ) -> torch.Tensor:
        """
        Get storage data as a 1D uint8 CPU tensor for caching.

        Args:
            storage_id: The storage ID

        Returns:
            1D uint8 CPU tensor representing the underlying storage
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Queue the RPC for batching (blocking call that returns raw bytes)
        future = self._queue_rpc(
            method_name="get_storage_data",
            call_type="remote",
            args=(storage_id,),
            kwargs={},
        )

        # Wait for the result from the Future
        raw_bytes = future.result() if future else None
        if raw_bytes is None:
            raise RuntimeError(
                f"Failed to retrieve storage data for storage {storage_id}"
            )

        # Create 1D uint8 tensor directly from raw bytes for caching
        from ..._tensor_utils import numpy_bytes_to_cpu_tensor

        return numpy_bytes_to_cpu_tensor(raw_bytes, (len(raw_bytes),), torch.uint8)

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

        # Queue the RPC for batching (fire-and-forget)
        # Invalidate cache immediately since this modifies storage
        self._queue_rpc(
            method_name="resize_storage",
            call_type="spawn",
            args=(storage_id, nbytes),
            kwargs={},
            invalidate_storage_ids=[storage_id],
        )

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

        # Queue the RPC for batching (fire-and-forget)
        # Invalidate cache immediately since this removes storage
        self._queue_rpc(
            method_name="remove_storage",
            call_type="spawn",
            args=(storage_id,),
            kwargs={},
            invalidate_storage_ids=[storage_id],
        )

    # New tensor-based methods for the refactored architecture
    def create_empty_tensor(self, tensor_id: int, shape: List[int], dtype: str) -> None:
        """
        Create an empty tensor on the remote machine with the given tensor_id.

        Args:
            tensor_id: Unique tensor ID (metadata hash)
            shape: Shape of the tensor to create
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
                args=(tensor_id, shape, dtype),
                kwargs={},
            )
            log.info(f"Queued creation of empty tensor {tensor_id} with shape {shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to create empty tensor {tensor_id}: {e}") from e

    def create_tensor_view(
        self,
        tensor_id: int,
        base_storage_id: int,
        shape: List[int],
        stride: List[int],
        offset: int,
        dtype: str,
    ) -> None:
        """
        Create a tensor view on the remote machine from an existing storage.

        Args:
            tensor_id: Unique tensor ID (metadata hash)
            base_storage_id: Storage ID of the base tensor to create view from
            shape: Shape of the view
            stride: Stride of the view
            offset: Storage offset of the view
            dtype: Data type of the view

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
                args=(tensor_id, base_storage_id, shape, stride, offset, dtype),
                kwargs={},
            )
            log.info(f"Queued creation of tensor view {tensor_id} from storage {base_storage_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to create tensor view {tensor_id}: {e}") from e

    def update_tensor(self, tensor_id: int, tensor: torch.Tensor) -> None:
        """
        Update an existing tensor with new data.

        Args:
            tensor_id: Unique tensor ID (metadata hash)
            tensor: CPU tensor containing the data to upload

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Serialize tensor using numpy approach
        from ..._tensor_utils import cpu_tensor_to_numpy_bytes

        numpy_bytes = cpu_tensor_to_numpy_bytes(tensor)

        # Queue the RPC for batching (fire-and-forget)
        self._queue_rpc(
            method_name="update_tensor",
            call_type="spawn",
            args=(tensor_id, numpy_bytes),
            kwargs={},
            # Note: We don't invalidate by storage_id here since we're using tensor_id
        )
        log.info(f"Queued update for tensor {tensor_id}")

    def get_tensor_data(self, tensor_id: int) -> bytes:
        """
        Get raw tensor data by tensor ID.

        Args:
            tensor_id: The tensor ID (metadata hash)

        Returns:
            Raw tensor data as bytes
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Queue the RPC for batching (blocking call that returns raw bytes)
        future = self._queue_rpc(
            method_name="get_tensor_data",
            call_type="remote",
            args=(tensor_id,),
            kwargs={},
        )

        # Wait for the result from the Future
        raw_bytes = future.result() if future else None
        if raw_bytes is None:
            raise RuntimeError(
                f"Failed to retrieve tensor data for tensor {tensor_id}"
            )

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
        log.info(f"Queued removal of {len(tensor_ids)} tensors")

    def resize_tensor_storage(self, tensor_id: int, nbytes: int) -> None:
        """
        Resize the underlying storage for a tensor (be careful about shared storage).

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
            method_name="resize_tensor_storage",
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
        output_storage_ids: List[Union[int, None]],
        args: List[Any],
        kwargs: Dict[str, Any],
        return_metadata: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute an aten operation with separated input metadata and output storage IDs.

        Args:
            op_name: The aten operation name
            input_tensor_metadata: Metadata for reconstructing input tensors only
            output_storage_ids: List of storage IDs to update with results (all output tensors)
            args: Operation arguments
            kwargs: Operation keyword arguments
            return_metadata: If True, return output tensor metadata instead of None

        Returns:
            None for normal operations, or List[Dict] of output tensor metadata if return_metadata=True
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

        # Determine which storage IDs will be modified by this operation
        # Filter out None values for cache invalidation
        modified_storage_ids = [sid for sid in output_storage_ids if sid is not None]

        if return_metadata:
            # Use remote call type to get return value when metadata is needed
            log.info(f"📡 Modal Client requesting metadata for {op_name}")
            future = self._queue_rpc(
                method_name="execute_aten_operation",
                call_type="remote",
                args=(op_name, input_tensor_metadata, output_storage_ids, args, kwargs),
                kwargs={
                    "return_metadata": True,
                },
                invalidate_storage_ids=modified_storage_ids,
            )
            # Wait for the result from the Future
            return future.result() if future else None
        else:
            # Use spawn call type for fire-and-forget execution (original behavior)
            self._queue_rpc(
                method_name="execute_aten_operation",
                call_type="spawn",
                args=(op_name, input_tensor_metadata, output_storage_ids, args, kwargs),
                kwargs={},
                invalidate_storage_ids=modified_storage_ids,
            )
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
            checkpoint: HuggingFace model checkpoint (e.g., "gpt2", "bert-base-uncased")
            torch_dtype: Data type for model weights ("auto", "float32", "float16", etc.)
            trust_remote_code: Whether to trust remote code for custom models

        Returns:
            Dict containing state_dict_metadata, config, and model_type
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        log.info(f"📡 Modal Client preparing HuggingFace model: {checkpoint}")

        # Use remote call to get model metadata
        future = self._queue_rpc(
            method_name="prepare_huggingface_model",
            call_type="remote",
            args=(checkpoint,),
            kwargs={
                "torch_dtype": str(torch_dtype) if torch_dtype is not None else None,
                "trust_remote_code": trust_remote_code,
            },
        )

        result = future.result() if future else None
        log.info(
            f"✅ Modal Client completed HuggingFace model preparation for {checkpoint}"
        )
        return result

    def link_model_tensors(
        self,
        local_storage_ids: List[int],
        parameter_names: List[str],
    ) -> None:
        """
        Link local mycelya tensor storage IDs to remote model parameter tensors.

        Args:
            local_storage_ids: List of local storage IDs from created mycelya tensors
            parameter_names: List of parameter names corresponding to each storage ID
        """
        log.info(
            f"📡 Modal Client linking {len(local_storage_ids)} local tensors to remote model parameters"
        )

        # Queue the linking RPC - this will execute after all create_storage calls in the batch
        self._queue_rpc(
            "link_model_tensors",
            "spawn",  # No return value needed
            (local_storage_ids, parameter_names),
            {},
        )

        # Don't wait for result here - let it execute in batch with proper ordering
        log.info("✅ Modal Client queued model tensor linking")

    # Helper methods for storage-to-tensor mapping management
    def _get_all_tracked_tensor_ids(self) -> Set[int]:
        """Get all tensor IDs currently tracked by this client."""
        all_tensor_ids: Set[int] = set()
        for tensor_ids in self._storage_to_tensor_ids.values():
            all_tensor_ids.update(tensor_ids)
        return all_tensor_ids

    def _add_tensor_to_storage_mapping(self, storage_id: int, tensor_id: int) -> None:
        """Add a tensor ID to the mapping for the given storage ID."""
        self._storage_to_tensor_ids[storage_id].add(tensor_id)

    def _remove_tensor_from_storage_mapping(self, storage_id: int, tensor_id: int) -> None:
        """Remove a tensor ID from the mapping for the given storage ID."""
        if storage_id in self._storage_to_tensor_ids:
            self._storage_to_tensor_ids[storage_id].discard(tensor_id)
            # Clean up empty sets
            if not self._storage_to_tensor_ids[storage_id]:
                del self._storage_to_tensor_ids[storage_id]

    def _get_tensor_ids_for_storage(self, storage_id: int) -> Set[int]:
        """Get all tensor IDs associated with a storage ID."""
        return self._storage_to_tensor_ids.get(storage_id, set()).copy()

    def _is_storage_new(self, storage_id: int) -> bool:
        """Check if this is the first tensor for the given storage ID."""
        return storage_id not in self._storage_to_tensor_ids or len(self._storage_to_tensor_ids[storage_id]) == 0

    def _ensure_tensor_exists(self, tensor: torch.Tensor) -> int:
        """
        Ensure that a tensor exists on the remote side, creating it if necessary.
        
        This method implements the core implicit tensor creation logic:
        - Calculate tensor_id from metadata hash
        - Check if tensor already exists
        - If not, add to storage mapping and create on remote side
        - First tensor for storage -> create empty tensor
        - Additional tensors for storage -> create view
        
        Args:
            tensor: The tensor to ensure exists remotely
            
        Returns:
            tensor_id: The tensor ID (metadata hash)
        """
        # Calculate tensor ID from metadata hash
        tensor_id = tensor.get_metadata_hash()
        storage_id = tensor.untyped_storage().data_ptr()
        
        # Check if tensor already tracked
        if tensor_id in self._get_all_tracked_tensor_ids():
            return tensor_id
            
        # Add tensor to storage mapping
        self._add_tensor_to_storage_mapping(storage_id, tensor_id)
        
        # Decide whether to create empty tensor or view
        if len(self._storage_to_tensor_ids[storage_id]) == 1:
            # First tensor for this storage - create empty tensor
            log.info(f"Creating first tensor {tensor_id} for storage {storage_id}")
            self.create_empty_tensor(
                tensor_id=tensor_id,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype).replace("torch.", "")  # Remove torch. prefix
            )
        else:
            # Additional tensor for existing storage - create as view
            log.info(f"Creating view tensor {tensor_id} for existing storage {storage_id}")
            self.create_tensor_view(
                tensor_id=tensor_id,
                base_storage_id=storage_id,
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                offset=tensor.storage_offset(),
                dtype=str(tensor.dtype).replace("torch.", "")  # Remove torch. prefix
            )
        
        return tensor_id

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


# PytorchServer and app creation logic is in _mycelya_torch_modal.modal_app
