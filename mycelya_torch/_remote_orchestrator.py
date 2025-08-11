# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""

import atexit
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from ._batching import BatchProcessor
from ._logging import get_logger
from ._storage import get_machine_for_storage
from ._tensor_utils import MycelyaTensorMetadata
from .backends.client_interface import ClientInterface
from ._device import RemoteMachine

log = get_logger(__name__)


# Exception handling is done through standard RuntimeError
# Custom exceptions removed as they were not used elsewhere in the codebase


class RemoteOrchestrator:
    """Orchestrates remote execution of aten operations across remote machines.

    This class coordinates operation execution between local tensors and remote
    machines, handling tensor transfers, device communication, and distributed
    execution flow. Currently supports Modal as the primary provider.

    Also manages background thread for batching RPCs to improve performance.
    """

    def __init__(self):
        # Simple utility-based architecture - no service objects needed

        # RPC Batching System
        self._batch_clients: Set[ClientInterface] = set()
        self._batch_lock = threading.RLock()
        self._batch_thread: Optional[threading.Thread] = None
        self._batch_shutdown = threading.Event()
        self._batch_wakeup = (
            threading.Event()
        )  # Wake up thread immediately for blocking calls
        self._batch_interval = 0.1  # Process batches every 100ms

        # Start background thread for batch processing
        self._start_batch_thread()

        # Register cleanup on exit
        atexit.register(self._cleanup_batch_thread)

    def _get_device_client(self, machine: "RemoteMachine"):
        """Get the active client for a specific machine."""
        return machine._client

    # Background thread management for RPC batching
    def _start_batch_thread(self) -> None:
        """Start the background thread for processing RPC batches."""
        if self._batch_thread is None or not self._batch_thread.is_alive():
            self._batch_shutdown.clear()
            self._batch_thread = threading.Thread(
                target=self._batch_processing_loop,
                name="RPCBatchProcessor",
                daemon=True,
            )
            self._batch_thread.start()
            log.info("🧵 Started RPC batch processing thread")

    def _cleanup_batch_thread(self) -> None:
        """Clean up the background batch processing thread."""
        if self._batch_thread and self._batch_thread.is_alive():
            log.info("🛑 Shutting down RPC batch processing thread")
            self._batch_shutdown.set()
            self._batch_thread.join(timeout=2.0)
            if self._batch_thread.is_alive():
                log.warning("⚠️ Batch processing thread did not shutdown cleanly")

    def _batch_processing_loop(self) -> None:
        """Main loop for the background batch processing thread."""
        log.info("🚀 RPC batch processing loop started")

        while not self._batch_shutdown.is_set():
            try:
                # Process batches for all registered clients
                with self._batch_lock:
                    clients_to_process = list(self._batch_clients)

                for client in clients_to_process:
                    try:
                        self._process_client_batch(client)
                    except Exception as e:
                        log.error(f"❌ Error processing batch for client {client}: {e}")

                # Wait for next batch interval OR immediate wakeup for blocking calls
                woken_early = self._batch_wakeup.wait(self._batch_interval)
                if woken_early:
                    # Clear the event for next time and process immediately
                    self._batch_wakeup.clear()
                    log.debug("🚀 Background thread woken early for blocking RPC")

            except Exception as e:
                log.error(f"❌ Error in batch processing loop: {e}")

        log.info("🏁 RPC batch processing loop terminated")

    def _process_client_batch(self, client: ClientInterface) -> None:
        """Process a batch of RPCs for a specific client."""
        if not hasattr(client, "_batch_queue"):
            return

        batch = client._batch_queue.get_batch()
        if not batch:
            return

        try:
            # Execute the batch
            result = BatchProcessor.execute_batch(client._server_instance, batch)

            log.debug(
                f"📊 Batch processed for {client}: "
                f"{result.success_count} success, {result.error_count} errors, "
                f"{result.execution_time:.3f}s"
            )

        except Exception as e:
            log.error(f"❌ Batch execution failed for client {client}: {e}")

            # Cancel all futures in the batch
            for call in batch:
                if call.future and not call.future.done():
                    call.future.set_exception(e)

    def register_client_for_batching(self, client: ClientInterface) -> None:
        """Register a client for RPC batching."""
        with self._batch_lock:
            self._batch_clients.add(client)
            log.info(f"📝 Registered client for batching: {client}")

    def unregister_client_for_batching(self, client: ClientInterface) -> None:
        """Unregister a client from RPC batching."""
        with self._batch_lock:
            self._batch_clients.discard(client)
            log.info(f"🗑️ Unregistered client from batching: {client}")

    def wake_batch_thread_for_blocking_rpc(self) -> None:
        """Wake up the background thread immediately for processing blocking RPCs."""
        if self._batch_thread and self._batch_thread.is_alive():
            self._batch_wakeup.set()
            log.debug("💨 Signaled batch thread to wake up for blocking RPC")

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get statistics about RPC batching across all clients."""
        with self._batch_lock:
            stats = {
                "registered_clients": len(self._batch_clients),
                "batch_interval": self._batch_interval,
                "thread_alive": self._batch_thread.is_alive()
                if self._batch_thread
                else False,
                "clients": [],
            }

            for client in self._batch_clients:
                if hasattr(client, "_batch_queue"):
                    client_stats = client._batch_queue.get_stats()
                    stats["clients"].append(client_stats)

            return stats

    def _get_client_for_storage(self, storage_id: int) -> ClientInterface:
        """Get the client for a specific storage ID with validation.

        Args:
            storage_id: Storage ID to resolve to client

        Returns:
            ClientInterface: The client managing this storage

        Raises:
            RuntimeError: If storage, machine, or client not found/available
        """
        try:
            machine = get_machine_for_storage(storage_id)
            return self._get_validated_client(machine)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve client for storage {storage_id}: {e}"
            ) from e

    def _get_client_for_machine(self, machine: RemoteMachine) -> ClientInterface:
        """Get the client for a specific machine with validation.

        Args:
            machine: RemoteMachine to get client for

        Returns:
            ClientInterface: The validated client for this machine

        Raises:
            RuntimeError: If client not available or not running
        """
        return self._get_validated_client(machine)

    def _get_validated_client(self, machine: RemoteMachine) -> ClientInterface:
        """Get a validated client for a machine, ensuring it's running.

        Args:
            machine: RemoteMachine to get client for

        Returns:
            ClientInterface: The validated, running client

        Raises:
            RuntimeError: If client is None or not running
        """
        client = machine._client
        if client is None:
            raise RuntimeError(f"No client available for machine {machine.machine_id}")

        if not client.is_running():
            raise RuntimeError(
                f"Client for machine {machine.machine_id} is not running"
            )

        return client

    def _ensure_client_running(self, client: ClientInterface) -> None:
        """Ensure a client is running, with basic retry logic.

        Args:
            client: Client to validate

        Raises:
            RuntimeError: If client cannot be started or validated
        """
        if not client.is_running():
            log.warning(f"Client not running, attempting to start: {client}")
            try:
                client.start()
                if not client.is_running():
                    raise RuntimeError(f"Failed to start client: {client}")
                log.info(f"Successfully started client: {client}")
            except Exception as e:
                raise RuntimeError(f"Failed to start client {client}: {e}") from e

    # Storage management methods - mirroring ClientInterface
    def create_storage(self, storage_id: int, nbytes: int, device_index: int) -> None:
        """Create storage on remote machine using device index routing.

        Args:
            storage_id: Specific ID to use for the storage
            nbytes: Number of bytes to allocate
            device_index: Device index to create storage on

        Raises:
            RuntimeError: If device or client not available
        """
        from ._device import get_device_registry

        registry = get_device_registry()
        machine = registry.get_device_by_index(device_index)
        if machine is None:
            raise RuntimeError(f"No machine found for device index {device_index}")

        client = self._get_validated_client(machine)
        client.create_storage(storage_id, nbytes)
        log.info(
            f"✅ ORCHESTRATOR: Created storage {storage_id} on device {device_index}"
        )

        # Note: No cache invalidation needed for create_storage - new storage has no cached data

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
        """Update existing storage with storage tensor data.

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

        Raises:
            RuntimeError: If storage or client not available
        """
        client = self._get_client_for_storage(storage_id)
        client.update_storage(
            storage_id,
            storage_tensor,
            source_shape,
            source_stride,
            source_storage_offset,
            source_dtype,
            target_shape,
            target_stride,
            target_storage_offset,
            target_dtype,
        )

        # Note: Cache invalidation now happens at queue time in batching system
        log.info(f"✅ ORCHESTRATOR: Updated storage {storage_id}")

    def get_storage_tensor(
        self,
        storage_id: int,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str,
    ) -> "torch.Tensor":
        """Get storage data as a tensor with specified view parameters.

        This is a convenience method that combines _get_storage_data() with tensor
        reconstruction.

        Args:
            storage_id: The storage ID to retrieve
            shape: Tensor shape for view
            stride: Tensor stride for view
            storage_offset: Storage offset for view
            dtype: Tensor data type

        Returns:
            CPU tensor reconstructed from storage with specified view

        Raises:
            RuntimeError: If storage or client not available
        """
        client = self._get_client_for_storage(storage_id)
        result = client.get_storage_tensor(
            storage_id, shape, stride, storage_offset, dtype
        )
        log.info(f"✅ ORCHESTRATOR: Retrieved tensor for storage {storage_id}")
        return result

    def resize_storage(self, storage_id: int, nbytes: int) -> None:
        """Resize storage to accommodate new byte size.

        Args:
            storage_id: The storage ID to resize
            nbytes: The number of bytes needed for the new storage size

        Raises:
            RuntimeError: If storage or client not available
        """
        client = self._get_client_for_storage(storage_id)
        client.resize_storage(storage_id, nbytes)

        # Note: Cache invalidation now happens at queue time in batching system
        log.info(f"✅ ORCHESTRATOR: Resized storage {storage_id} to {nbytes} bytes")

    def remove_storage(self, storage_id: int) -> None:
        """Remove storage from remote machine.

        Args:
            storage_id: The storage ID to remove

        Raises:
            RuntimeError: If storage or client not available
        """
        client = self._get_client_for_storage(storage_id)
        client.remove_storage(storage_id)

        # Note: Cache invalidation now happens at queue time in batching system
        log.info(f"✅ ORCHESTRATOR: Removed storage {storage_id}")

    # New tensor-based methods for the refactored architecture
    def update_tensor(self, tensor_id: int, tensor: torch.Tensor) -> None:
        """Update a tensor with new data using tensor-based approach.

        Args:
            tensor_id: The tensor ID (metadata hash)
            tensor: CPU tensor containing the data to upload

        Raises:
            RuntimeError: If tensor or client not available
        """
        # Get client by first finding the storage ID for the tensor
        storage_id = tensor.untyped_storage().data_ptr()
        client = self._get_client_for_storage(storage_id)
        client.update_tensor(tensor_id, tensor)

        log.info(f"✅ ORCHESTRATOR: Updated tensor {tensor_id}")

    def get_tensor_data(self, tensor_id: int, tensor: torch.Tensor) -> torch.Tensor:
        """Get tensor data from remote using tensor-based approach.

        Args:
            tensor_id: The tensor ID (metadata hash)
            tensor: Reference tensor to get client routing information

        Returns:
            CPU tensor with the retrieved data

        Raises:
            RuntimeError: If tensor or client not available
        """
        # Get client by first finding the storage ID for the tensor
        storage_id = tensor.untyped_storage().data_ptr()
        client = self._get_client_for_storage(storage_id)
        
        # Get raw bytes from client
        raw_bytes = client.get_tensor_data(tensor_id)
        
        # Reconstruct CPU tensor from raw bytes
        from ._tensor_utils import numpy_bytes_to_cpu_tensor
        result = numpy_bytes_to_cpu_tensor(
            raw_bytes, tensor.shape, tensor.dtype
        )
        
        log.info(f"✅ ORCHESTRATOR: Retrieved tensor data for tensor {tensor_id}")
        return result

    def execute_aten_operation(
        self,
        op_name: str,
        input_metadata: List[MycelyaTensorMetadata],
        output_storage_ids: List[Optional[int]],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        return_metadata: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute remote operation with pure metadata (early conversion boundary).

        This method represents the clean boundary where all tensors have been
        converted to metadata at the PyTorch integration layer. No raw tensors
        should be passed to this method.

        Args:
            op_name: Name of the operation to execute
            input_metadata: Metadata for remote input tensors (always have storage_id)
            output_storage_ids: Storage IDs for all output tensors (both new and reused)
            args: Processed args with tensor placeholders
            kwargs: Processed kwargs with tensor placeholders
            return_metadata: If True, return output tensor metadata instead of None

        Returns:
            None for normal operations, or List[Dict] of output tensor metadata if return_metadata=True
        """
        log.info(
            f"🎯 ORCHESTRATOR: Executing {op_name} with separated input/output interface"
        )

        # Convert input metadata to serializable dictionaries
        input_tensor_metadata_dicts = []
        for metadata in input_metadata:
            meta_dict = metadata.to_dict()
            input_tensor_metadata_dicts.append(meta_dict)

        # output_storage_ids is now passed directly from _create_output_tensors
        # Contains storage_id for all output tensors (both new and reused)

        # Validate that we have input metadata
        if not input_metadata:
            raise RuntimeError(f"No input metadata provided for operation {op_name}")

        # Collect all storage IDs for cross-device validation
        all_storage_ids = []
        for metadata in input_metadata:
            if metadata.storage_id is not None:
                all_storage_ids.append(metadata.storage_id)
        for storage_id in output_storage_ids:
            if storage_id is not None:
                all_storage_ids.append(storage_id)

        # Validate all storage IDs are on the same device
        if all_storage_ids:
            from ._storage import validate_cross_device_operation

            validate_cross_device_operation(all_storage_ids)

        # Get the client using the first input tensor's storage ID
        storage_id = input_metadata[0].storage_id
        client = self._get_client_for_storage(storage_id)

        # Execute with separated input/output interface
        result = client.execute_aten_operation(
            op_name,
            input_tensor_metadata_dicts,
            output_storage_ids,
            args,
            kwargs,
            return_metadata,
        )

        # Note: With batching, cache invalidation for aten operations happens at queue time
        # when the operation is queued, not when it's executed

        if return_metadata:
            log.info(f"✅ ORCHESTRATOR: Completed {op_name} with metadata return")
            return result
        else:
            log.info(f"✅ ORCHESTRATOR: Completed {op_name} with separated interface")
            return None

    def _remote_tensor_to_cpu(self, remote_tensor: torch.Tensor) -> torch.Tensor:
        """Convert remote tensor to CPU tensor by retrieving data from remote GPU."""
        if remote_tensor.device.type != "mycelya":
            raise ValueError(
                f"Expected remote tensor, got device: {remote_tensor.device}"
            )

        # Get device registry to find the machine
        from ._device import get_device_registry

        registry = get_device_registry()
        machine = registry.get_device_by_index(remote_tensor.device.index)

        if machine is None:
            raise RuntimeError(
                f"No RemoteMachine found for remote device index {remote_tensor.device.index}"
            )

        # Get tensor data using storage ID with internal client resolution
        storage_id = remote_tensor.untyped_storage().data_ptr()

        # Create metadata for the remote tensor
        metadata = MycelyaTensorMetadata.from_mycelya_tensor(remote_tensor)

        # Get tensor data from remote storage using new interface
        return self.get_storage_tensor(
            storage_id,
            shape=list(metadata.shape),
            stride=list(metadata.stride),
            storage_offset=metadata.storage_offset,
            dtype=str(metadata.dtype),
        )

    def remove_tensor_from_remote(
        self, storage_id: int, machine: "RemoteMachine"
    ) -> bool:
        """Remove a tensor from remote storage."""
        try:
            # Use internal client resolution for consistent error handling
            client = self._get_validated_client(machine)
            client.remove_storage(storage_id)
            log.info(f"✅ ORCHESTRATOR: Removed storage {storage_id} from remote")
            return True
        except Exception as e:
            log.warning(f"Failed to remove storage {storage_id}: {e}")
            return False


# Global orchestrator instance (Modal provider implementation)
remote_orchestrator = RemoteOrchestrator()
