# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Abstract client interface for mycelya_torch cloud providers.

This module defines the base interface that all cloud provider clients must implement,
ensuring consistent API across different backends (Modal, AWS, GCP, Azure, etc.).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypedDict

if TYPE_CHECKING:
    import torch


class BatchCall(TypedDict):
    """Structure for a single batched RPC call.

    This TypedDict defines the structure used for batching multiple operations
    into a single RPC call for performance optimization.
    """

    method_name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


class Client(ABC):
    """
    Abstract base class defining the interface for cloud provider clients.

    All cloud provider clients (ModalClient, MockClient, etc.) must inherit from this
    class and implement all abstract methods to ensure consistent API across providers.
    """

    def __init__(self, gpu_type: str, machine_id: str):
        """
        Initialize the client with GPU type, machine ID, and configuration.

        Args:
            gpu_type: The GPU type (e.g., "T4", "A100-40GB")
            machine_id: Unique machine identifier
        """
        self.gpu_type = gpu_type
        self.machine_id = machine_id


    @abstractmethod
    def start(self) -> None:
        """
        Start the cloud provider's compute resources.

        This method should initialize and start the remote client,
        making it ready to accept operations.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the cloud provider's compute resources.

        This method should cleanly shutdown the remote client
        and release any associated resources.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the machine is currently running and ready.

        Returns:
            True if the machine is running and can accept operations, False otherwise
        """
        pass

    # Tensor management methods
    @abstractmethod
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
        pass

    def create_empty_tensor(
        self,
        tensor_id: int,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
        nbytes: int,
    ) -> None:
        """
        Create an empty tensor on the remote machine with proper storage layout.

        Args:
            tensor_id: Unique tensor ID (metadata hash)
            shape: Shape of the tensor
            stride: Stride of the tensor
            storage_offset: Storage offset for the tensor
            dtype: Data type of the tensor (e.g., "float32", "int64")
            nbytes: Number of bytes for the underlying storage (from client allocator)

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        self._create_empty_tensor_impl(
            tensor_id, shape, stride, storage_offset, dtype, nbytes
        )

    @abstractmethod
    def _create_tensor_view_impl(
        self,
        new_tensor_id: int,
        base_tensor_id: int,
        shape: List[int],
        stride: List[int],
        offset: int,
    ) -> None:
        """Implementation: Create a tensor view from an existing tensor."""
        pass

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

        self._create_tensor_view_impl(
            new_tensor_id, base_tensor_id, shape, stride, offset
        )

    @abstractmethod
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
        pass

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

        self._update_tensor_impl(
            tensor_id,
            raw_data,
            source_shape,
            source_stride,
            source_storage_offset,
            source_dtype,
        )

    @abstractmethod
    def _get_storage_data_impl(self, tensor_id: int) -> bytes:
        """Implementation: Get raw storage data by tensor ID."""
        pass

    def get_storage_data(self, tensor_id: int) -> bytes:
        """
        Get raw storage data by tensor ID.

        Args:
            tensor_id: The tensor ID to retrieve storage data from

        Returns:
            Raw storage data as bytes
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        return self._get_storage_data_impl(tensor_id)

    @abstractmethod
    def _remove_tensors_impl(self, tensor_ids: List[int]) -> None:
        """Implementation: Remove multiple tensors from the remote machine."""
        pass

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

        self._remove_tensors_impl(tensor_ids)

    @abstractmethod
    def _resize_storage_impl(self, tensor_id: int, nbytes: int) -> None:
        """Implementation: Resize the underlying storage for a tensor."""
        pass

    def resize_storage(self, tensor_id: int, nbytes: int) -> None:
        """
        Resize the underlying storage for a tensor.

        Args:
            tensor_id: Tensor ID whose storage to resize
            nbytes: New size in bytes

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        self._resize_storage_impl(tensor_id, nbytes)

    # Operation execution methods
    @abstractmethod
    def _execute_aten_operation_impl(
        self,
        op_name: str,
        input_tensor_ids: List[int],
        output_tensor_ids: List[int],
        args: List[Any],
        kwargs: Dict[str, Any],
        tensor_mask: List[bool],
        return_metadata: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """Implementation: Execute an aten operation on the remote machine with tensor IDs."""
        pass

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

        return self._execute_aten_operation_impl(
            op_name,
            input_tensor_ids,
            output_tensor_ids,
            args,
            kwargs,
            tensor_mask,
            return_metadata,
        )

    # HuggingFace model loading methods
    @abstractmethod
    def _prepare_huggingface_model_impl(
        self,
        checkpoint: str,
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> Dict[str, Any]:
        """Implementation: Download and prepare a HuggingFace model directly on the remote machine."""
        pass

    def prepare_huggingface_model(
        self,
        checkpoint: str,
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> Dict[str, Any]:
        """
        Download and prepare a HuggingFace model directly on the remote machine.

        This method downloads the model weights directly on the remote GPU,
        loads them into GPU memory, and returns metadata needed to create
        local tensor stubs.

        Args:
            checkpoint: HuggingFace model checkpoint (e.g., "gpt2", "bert-base-uncased")
            torch_dtype: Data type for model weights ("auto", "float32", "float16", etc.)
            trust_remote_code: Whether to trust remote code for custom models

        Returns:
            Dict containing:
            - state_dict_metadata: Dict[str, Dict] mapping parameter names to tensor metadata
            - config: Model configuration dictionary
            - model_type: Model class name string
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        return self._prepare_huggingface_model_impl(
            checkpoint, torch_dtype, trust_remote_code
        )

    @abstractmethod
    def _link_model_tensors_impl(
        self,
        local_tensor_ids: List[int],
        parameter_names: List[str],
    ) -> None:
        """Implementation: Link local mycelya tensor IDs to remote model parameter tensors."""
        pass

    def link_model_tensors(
        self,
        local_tensor_ids: List[int],
        parameter_names: List[str],
    ) -> None:
        """
        Link local mycelya tensor IDs to remote model parameter tensors.

        This method is used after HuggingFace model loading to connect the locally
        created mycelya tensors to the corresponding remote model parameter tensors.

        Args:
            local_tensor_ids: List of local tensor IDs from created mycelya tensors
            parameter_names: List of parameter names corresponding to each tensor ID

        Returns:
            None

        Example:
            # After model preparation and local tensor creation
            local_tensor_ids = [tensor._get_tensor_id() for tensor in model.parameters()]
            parameter_names = ["model.embed_tokens.weight", "layer.0.weight", ...]
            client.link_model_tensors(local_tensor_ids, parameter_names)
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        self._link_model_tensors_impl(local_tensor_ids, parameter_names)

    # Context manager methods (optional to override, but provide default behavior)
    def __enter__(self):
        """Context manager entry - starts the machine."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the machine."""
        self.stop()

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation of the client.

        Returns:
            Human-readable string describing the client state
        """
        pass
