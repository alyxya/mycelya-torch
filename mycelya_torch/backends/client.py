# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Abstract client interface for mycelya_torch cloud providers.

This module defines the base interface that all cloud provider clients must implement,
ensuring consistent API across different backends (Modal, AWS, GCP, Azure, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .._batching import RPCBatchQueue


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

        # Note: Tensor cache moved to orchestrator level for centralized management

        # RPC batching queue
        self._batch_queue = RPCBatchQueue(client_id=machine_id)

        # Register with orchestrator for batching (will be done in subclass start())
        self._registered_for_batching = False

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

    def health_check(self) -> bool:
        """
        Perform a health check on the client connection.

        Default implementation delegates to is_running(). Providers can override
        for more sophisticated health checking.

        Returns:
            True if client is healthy, False otherwise
        """
        return self.is_running()

    def reconnect(self) -> bool:
        """
        Attempt to reconnect the client.

        Default implementation stops and starts the client. Providers can override
        for more sophisticated reconnection logic.

        Returns:
            True if reconnection succeeded, False otherwise
        """
        try:
            self.stop()
            self.start()
            return self.is_running()
        except Exception:
            return False

    # Tensor management methods
    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_tensor_data(self, tensor_id: int) -> bytes:
        """
        Get raw tensor data by tensor ID.

        Args:
            tensor_id: The tensor ID to retrieve

        Returns:
            Raw tensor data as bytes
        """
        pass

    @abstractmethod
    def remove_tensors(self, tensor_ids: List[int]) -> None:
        """
        Remove multiple tensors from the remote machine.

        Args:
            tensor_ids: List of tensor IDs to remove

        Returns:
            None
        """
        pass

    @abstractmethod
    def resize_tensor_storage(self, tensor_id: int, nbytes: int) -> None:
        """
        Resize the underlying storage for a tensor.

        Args:
            tensor_id: Tensor ID whose storage to resize
            nbytes: New size in bytes

        Returns:
            None
        """
        pass

    # Operation execution methods
    @abstractmethod
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
        pass

    # HuggingFace model loading methods
    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    # RPC batching helper methods
    def _queue_rpc(
        self,
        method_name: str,
        call_type: str,
        args: tuple,
        kwargs: dict,
        return_future: bool = False,
        invalidate_tensor_ids: Optional[List[int]] = None,
    ) -> Optional[Any]:
        """
        Helper method to queue an RPC for batching.

        Args:
            method_name: Name of the RPC method to call
            call_type: "spawn" for fire-and-forget, "remote" for blocking
            args: Arguments for the RPC method
            kwargs: Keyword arguments for the RPC method
            return_future: Whether to return a Future for this call
            invalidate_tensor_ids: Tensor IDs to invalidate immediately (at queue time)

        Returns:
            Future object if return_future=True or call_type="remote", None otherwise
        """
        # Note: Cache invalidation now handled at orchestrator level

        # Queue the RPC for batching
        future = self._batch_queue.enqueue_call(
            call_type=call_type,
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            return_future=return_future,
        )

        # Wake up background thread immediately for blocking calls to reduce latency
        if call_type == "remote" or return_future:
            from .._orchestrator import orchestrator

            orchestrator.wake_batch_thread_for_blocking_rpc()

        return future

    def _register_for_batching(self) -> None:
        """Register this client with the orchestrator for batching."""
        if not self._registered_for_batching:
            from .._orchestrator import orchestrator

            orchestrator.register_client_for_batching(self)
            self._registered_for_batching = True

    def _unregister_for_batching(self) -> None:
        """Unregister this client from the orchestrator for batching."""
        if self._registered_for_batching:
            from .._orchestrator import orchestrator

            orchestrator.unregister_client_for_batching(self)
            self._registered_for_batching = False

    # Note: Cache methods removed - caching now handled at orchestrator level

    # Context manager methods (optional to override, but provide default behavior)
    def __enter__(self):
        """Context manager entry - starts the machine."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the machine."""
        self.stop()
        # Note: Cache cleanup removed - handled at orchestrator level
        self._unregister_for_batching()

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation of the client.

        Returns:
            Human-readable string describing the client state
        """
        pass
