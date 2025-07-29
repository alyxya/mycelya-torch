# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Local client implementation for mycelya_torch.

This module provides the LocalClient class that uses Modal's .local() execution
for development and testing without requiring remote cloud resources.
"""

from typing import Any, Dict, List, Optional, Union

import torch

from .local_modal_app import create_local_modal_app_for_gpu

from ..._logging import get_logger
from ..client_interface import ClientInterface

log = get_logger(__name__)


class LocalClient(ClientInterface):
    """
    Client interface for local execution using Modal's .local() calls.

    This class provides a local execution environment that mimics the Modal cloud GPU
    interface but runs everything locally. It uses Modal's .local() method to execute
    the same operations that would normally run on remote GPUs.
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

        # Initialize the Modal app and server for local execution
        self._initialize()

    def _initialize(self):
        """Initialize the local app and server class for local execution."""
        self._app, self._server_class = create_local_modal_app_for_gpu(
            self.gpu_type, self.machine_id, self.timeout, self.retries
        )

    def start(self):
        """Start the local execution environment."""
        if not self._is_running:
            # Create server instance for local execution
            self._server_instance = self._server_class()
            self._is_running = True
            
            # Register for RPC batching
            self._register_for_batching()
            
            log.info(f"Started local client: {self.machine_id}")

    def stop(self):
        """Stop the local execution environment."""
        # Unregister from RPC batching first
        self._unregister_for_batching()
        
        if self._is_running:
            self._server_instance = None
            self._is_running = False
            log.info(f"Stopped local client: {self.machine_id}")

    def is_running(self) -> bool:
        """Check if the local client is currently running."""
        return self._is_running

    # Storage management methods
    def create_storage(self, storage_id: int, nbytes: int) -> None:
        """
        Create a storage locally.

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
            # Execute locally using .local() method
            self._queue_rpc_call(
                method_name="create_storage",
                call_type="spawn",
                args=(storage_id, nbytes),
                kwargs={}
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
        target_dtype: str
    ) -> None:
        """
        Update an existing storage with storage tensor data locally.

        Args:
            storage_id: Storage ID to update
            storage_tensor: CPU tensor wrapping the storage data
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

        # Serialize storage tensor using torch.save (same as Modal client)
        from ..._tensor_utils import cpu_tensor_to_torch_bytes

        torch_bytes = cpu_tensor_to_torch_bytes(storage_tensor)

        # Queue the RPC call for batching (fire-and-forget)
        # Invalidate cache immediately since this modifies storage
        self._queue_rpc_call(
            method_name="update_storage",
            call_type="spawn",
            args=(storage_id, torch_bytes,
                  source_shape, source_stride, source_storage_offset, source_dtype,
                  target_shape, target_stride, target_storage_offset, target_dtype),
            kwargs={},
            invalidate_storage_ids=[storage_id]
        )

    def _get_storage_data(
        self,
        storage_id: int,
    ) -> bytes:
        """
        Get raw storage data by ID locally.

        Args:
            storage_id: The storage ID

        Returns:
            Raw untyped storage bytes
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Execute locally using .local() method
        torch_bytes = self._queue_rpc_call(
            method_name="get_storage_data",
            call_type="remote",
            args=(storage_id,),
            kwargs={}
        )

        # Deserialize tensor and extract storage bytes properly
        import ctypes

        from ..._tensor_utils import torch_bytes_to_cpu_tensor

        storage_tensor = torch_bytes_to_cpu_tensor(torch_bytes)
        untyped_storage = storage_tensor.untyped_storage()
        nbytes = untyped_storage.nbytes()
        data_ptr = untyped_storage.data_ptr()

        # Extract bytes using ctypes from the data pointer
        raw_bytes = (ctypes.c_uint8 * nbytes).from_address(data_ptr)
        return bytes(raw_bytes)

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

        # Execute locally using .local() method
        future = self._queue_rpc_call(
            method_name="get_storage_data",
            call_type="remote",
            args=(storage_id,),
            kwargs={}
        )

        torch_bytes = future.result() if future else None
        if torch_bytes is None:
            raise RuntimeError(f"Failed to retrieve storage data for storage {storage_id}")

        # Deserialize tensor directly for caching (should be 1D uint8)
        from ..._tensor_utils import torch_bytes_to_cpu_tensor

        return torch_bytes_to_cpu_tensor(torch_bytes)

    def resize_storage(self, storage_id: int, nbytes: int) -> None:
        """
        Resize a storage locally.

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

        # Execute locally using .local() method
        self._queue_rpc_call(
            method_name="resize_storage",
            call_type="spawn",
            args=(storage_id, nbytes),
            kwargs={},
            invalidate_storage_ids=[storage_id]
        )

    def remove_storage(self, storage_id: int) -> None:
        """
        Remove a storage locally.

        Args:
            storage_id: The storage ID

        Returns:
            None
        """
        if not self.is_running():
            raise RuntimeError(
                f"Machine {self.machine_id} is not running. Call start() first."
            )

        # Execute locally using .local() method
        self._queue_rpc_call(
            method_name="remove_storage",
            call_type="spawn",
            args=(storage_id,),
            kwargs={},
            invalidate_storage_ids=[storage_id]
        )

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
        Execute an aten operation locally.

        Args:
            op_name: The aten operation name
            input_tensor_metadata: Metadata for reconstructing input tensors only
            output_storage_ids: List of storage IDs to update with results
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
        log.info(f"📡 Local Client sending Input Storage IDs: {input_storage_ids}")
        log.info(f"📡 Local Client sending Output Storage IDs: {output_storage_ids}")

        # Determine which storage IDs will be modified by this operation
        modified_storage_ids = [sid for sid in output_storage_ids if sid is not None]

        if return_metadata:
            # Use remote call type to get return value when metadata is needed
            log.info(f"📡 Local Client requesting metadata for {op_name}")
            future = self._queue_rpc_call(
                method_name="execute_aten_operation",
                call_type="remote",
                args=(op_name, input_tensor_metadata, output_storage_ids, args, kwargs),
                kwargs={"return_metadata": True},
                invalidate_storage_ids=modified_storage_ids
            )
            return future.result() if future else None
        else:
            # Use spawn call type for fire-and-forget execution
            self._queue_rpc_call(
                method_name="execute_aten_operation",
                call_type="spawn",
                args=(op_name, input_tensor_metadata, output_storage_ids, args, kwargs),
                kwargs={},
                invalidate_storage_ids=modified_storage_ids
            )
            return None

    def _execute_local_call(self, method_name: str, args: tuple, kwargs: dict) -> Any:
        """
        Execute a method call locally using Modal's .local() functionality.
        
        This method maps RPC calls to local method execution on the server instance.
        """
        if not self._server_instance:
            raise RuntimeError("Server instance not initialized")

        # Get the actual method from the server instance and call it locally
        if method_name == "create_storage":
            return self._server_instance._create_storage_impl(*args, **kwargs)
        elif method_name == "update_storage":
            return self._server_instance._update_storage_impl(*args, **kwargs)
        elif method_name == "get_storage_data":
            return self._server_instance._get_storage_data_impl(*args, **kwargs)
        elif method_name == "resize_storage":
            return self._server_instance._resize_storage_impl(*args, **kwargs)
        elif method_name == "remove_storage":
            return self._server_instance._remove_storage_impl(*args, **kwargs)
        elif method_name == "execute_aten_operation":
            return self._server_instance._execute_aten_operation_impl(*args, **kwargs)
        else:
            raise AttributeError(f"Unknown method: {method_name}")

    def _queue_rpc_call(
        self,
        method_name: str,
        call_type: str,
        args: tuple,
        kwargs: dict,
        return_future: bool = False,
        invalidate_storage_ids: Optional[List[int]] = None
    ) -> Optional[Any]:
        """
        Override the base class method to execute calls locally instead of queuing.
        
        For the local client, we execute immediately rather than batching,
        but maintain the same interface for compatibility.
        """
        # Invalidate cache immediately for storage-modifying operations
        if invalidate_storage_ids:
            for storage_id in invalidate_storage_ids:
                if storage_id in self._storage_cache:
                    del self._storage_cache[storage_id]

        # Execute locally instead of queuing for remote execution
        try:
            result = self._execute_local_call(method_name, args, kwargs)
            
            if call_type == "spawn":
                # Fire-and-forget calls return None
                return None
            else:
                # Remote calls return the actual result
                # For compatibility with the Future interface, we'll wrap in a simple object
                class LocalResult:
                    def __init__(self, value):
                        self._value = value
                    def result(self):
                        return self._value
                
                return LocalResult(result)
        except Exception as e:
            log.error(f"Local execution failed for {method_name}: {e}")
            raise

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return (
            f'LocalClient(gpu_type="{self.gpu_type}", '
            f'machine_id="{self.machine_id}", status="{status}")'
        )