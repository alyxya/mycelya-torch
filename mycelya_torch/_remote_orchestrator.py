# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from ._logging import get_logger
from ._storage import get_machine_for_storage
from ._tensor_utils import RemoteTensorMetadata
from .backends.client_interface import ClientInterface
from .device import RemoteMachine

log = get_logger(__name__)


# Exception handling is done through standard RuntimeError
# Custom exceptions removed as they were not used elsewhere in the codebase


class RemoteOrchestrator:
    """Orchestrates remote execution of aten operations across remote machines.

    This class coordinates operation execution between local tensors and remote
    machines, handling tensor transfers, device communication, and distributed
    execution flow. Currently supports Modal as the primary provider.
    """

    def __init__(self):
        # Simple utility-based architecture - no service objects needed
        pass

    def _get_device_client(self, machine: "RemoteMachine"):
        """Get the active client for a specific machine."""
        return machine._client

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
            raise RuntimeError(f"Failed to resolve client for storage {storage_id}: {e}") from e

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
            raise RuntimeError(f"Client for machine {machine.machine_id} is not running")

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
        from .device import get_device_registry

        registry = get_device_registry()
        machine = registry.get_device_by_index(device_index)
        if machine is None:
            raise RuntimeError(f"No machine found for device index {device_index}")

        client = self._get_validated_client(machine)
        client.create_storage(storage_id, nbytes)
        log.info(f"✅ ORCHESTRATOR: Created storage {storage_id} on device {device_index}")

    def update_storage(
        self,
        storage_id: int,
        tensor_data: bytes,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str
    ) -> None:
        """Update existing storage with tensor data.

        Args:
            storage_id: Storage ID to update
            tensor_data: Serialized tensor data to store
            shape: Shape of the target view
            stride: Stride of the target view
            storage_offset: Storage offset of the target view
            dtype: Data type of the target view

        Raises:
            RuntimeError: If storage or client not available
        """
        client = self._get_client_for_storage(storage_id)
        client.update_storage(storage_id, tensor_data, shape, stride, storage_offset, dtype)
        log.info(f"✅ ORCHESTRATOR: Updated storage {storage_id}")

    def get_storage_data(
        self,
        storage_id: int,
        shape: List[int],
        stride: List[int],
        storage_offset: int,
        dtype: str
    ) -> bytes:
        """Get storage data by ID as a specific view.

        Args:
            storage_id: The storage ID to retrieve
            shape: Tensor shape for view
            stride: Tensor stride for view
            storage_offset: Storage offset for view
            dtype: Tensor data type

        Returns:
            Serialized tensor data (contiguous representation of the view)

        Raises:
            RuntimeError: If storage or client not available
        """
        client = self._get_client_for_storage(storage_id)
        result = client.get_storage_data(storage_id, shape, stride, storage_offset, dtype)
        log.info(f"✅ ORCHESTRATOR: Retrieved data for storage {storage_id}")
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
        log.info(f"✅ ORCHESTRATOR: Removed storage {storage_id}")

    def execute_aten_operation(
        self,
        op_name: str,
        input_metadata: List[RemoteTensorMetadata],
        output_storage_ids: List[Optional[int]],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        """Execute remote operation with pure metadata (early conversion boundary).

        This method represents the new clean boundary where all tensors have been
        converted to metadata at the PyTorch integration layer. No raw tensors
        should be passed to this method.

        Args:
            op_name: Name of the operation to execute
            input_metadata: Metadata for remote input tensors (always have storage_id)
            output_metadata: Metadata for remote output tensors (always have storage_id)
            args: Processed args with tensor placeholders
            kwargs: Processed kwargs with tensor placeholders
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
        # Contains storage_id for new tensors, None for reused tensors

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
        client.execute_aten_operation(
            op_name, input_tensor_metadata_dicts, output_storage_ids, args, kwargs
        )

        log.info(f"✅ ORCHESTRATOR: Completed {op_name} with separated interface")

    def _remote_tensor_to_cpu(self, remote_tensor: torch.Tensor) -> torch.Tensor:
        """Convert remote tensor to CPU tensor by retrieving data from remote GPU."""
        if remote_tensor.device.type != "mycelya":
            raise ValueError(
                f"Expected mycelya tensor, got device: {remote_tensor.device}"
            )

        # Get device registry to find the machine
        from .device import get_device_registry

        registry = get_device_registry()
        machine = registry.get_device_by_index(remote_tensor.device.index)

        if machine is None:
            raise RuntimeError(
                f"No RemoteMachine found for mycelya device index {remote_tensor.device.index}"
            )

        # Get tensor data using storage ID with internal client resolution
        storage_id = remote_tensor.untyped_storage().data_ptr()

        # Create metadata for the remote tensor
        metadata = RemoteTensorMetadata.from_remote_tensor(remote_tensor)

        # Get serialized data from remote storage using internal method
        tensor_data = self.get_storage_data(
            storage_id,
            shape=list(metadata.shape),
            stride=list(metadata.stride),
            storage_offset=metadata.storage_offset,
            dtype=str(metadata.dtype),
        )

        # Convert bytes back to CPU tensor using metadata
        return metadata.to_cpu_tensor_from_bytes(tensor_data)

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
