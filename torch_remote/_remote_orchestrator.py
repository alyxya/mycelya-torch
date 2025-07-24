# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""

from typing import Any, Dict, List, Tuple

import torch

from ._logging import get_logger
from ._storage import get_machine_for_storage
from ._tensor_utils import TensorMetadata
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


    def execute_remote_aten_operation(
        self,
        op_name: str,
        input_metadata: List[TensorMetadata],
        output_metadata: List[TensorMetadata],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        """Execute remote operation with pure metadata (early conversion boundary).

        This method represents the new clean boundary where all tensors have been
        converted to metadata at the PyTorch integration layer. No raw tensors
        should be passed to this method.

        Args:
            op_name: Name of the operation to execute
            input_metadata: Metadata for input tensors
            output_metadata: Metadata for output tensors
            args: Processed args with tensor placeholders
            kwargs: Processed kwargs with tensor placeholders
        """
        log.info(f"🎯 ORCHESTRATOR: Executing {op_name} with pure metadata boundary")

        # Determine input/output indices for metadata list
        total_metadata = input_metadata + output_metadata
        input_indices = set(range(len(input_metadata)))
        output_indices = set(range(len(input_metadata), len(total_metadata)))

        # Convert metadata to serializable dictionaries with operation flags
        tensor_metadata_dicts = []
        for i, metadata in enumerate(total_metadata):
            meta_dict = metadata.to_dict()
            if i in input_indices:
                meta_dict["is_input"] = True
            if i in output_indices:
                meta_dict["is_output"] = True
            tensor_metadata_dicts.append(meta_dict)

        # Get the machine from first input tensor's storage ID
        if not input_metadata:
            raise RuntimeError(f"No input metadata provided for operation {op_name}")

        storage_id = input_metadata[0].storage_id
        machine = get_machine_for_storage(storage_id)

        # Execute with pure metadata interface
        client = self._get_device_client(machine)
        client.execute_aten_operation(op_name, tensor_metadata_dicts, args, kwargs)

        log.info(f"✅ ORCHESTRATOR: Completed {op_name} with metadata boundary")


    def _remote_tensor_to_cpu(self, remote_tensor: torch.Tensor) -> torch.Tensor:
        """Convert remote tensor to CPU tensor by retrieving data from remote GPU."""
        if remote_tensor.device.type != "remote":
            raise ValueError(
                f"Expected remote tensor, got device: {remote_tensor.device}"
            )

        # Get device registry to find the machine
        from .device import get_device_registry

        registry = get_device_registry()
        machine = registry.get_device_by_index(remote_tensor.device.index)

        if machine is None:
            raise RuntimeError(
                f"No RemoteMachine found for remote device index {remote_tensor.device.index}"
            )

        # Get the client for this machine
        client = machine._client
        if client is None or not client.is_running():
            raise RuntimeError(f"Client not available for machine {machine.machine_id}")

        # Get tensor data using storage ID
        storage_id = remote_tensor.untyped_storage().data_ptr()

        # Create metadata for the remote tensor
        metadata = TensorMetadata.from_remote_tensor(remote_tensor)

        # Get serialized data from remote storage
        tensor_data = client.get_storage_data(
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
            client = self._get_device_client(machine)
            if client and client.is_running():
                return client.remove_storage(storage_id)
            return False
        except Exception as e:
            log.warning(f"Failed to remove storage {storage_id}: {e}")
            return False


# Global orchestrator instance (Modal provider implementation)
remote_orchestrator = RemoteOrchestrator()
