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
from ._tensor_utils import RemoteTensorMetadata
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

    def execute_aten_operation(
        self,
        op_name: str,
        input_metadata: List[RemoteTensorMetadata],
        output_metadata: List[RemoteTensorMetadata],
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

        # Extract output storage IDs (None for outputs that should be ignored)
        output_storage_ids = []
        for metadata in output_metadata:
            # All output metadata should have storage_id since they're pre-allocated
            output_storage_ids.append(metadata.storage_id)

        # Get the machine from first input tensor's storage ID
        if not input_metadata:
            raise RuntimeError(f"No input metadata provided for operation {op_name}")

        storage_id = input_metadata[0].storage_id
        machine = get_machine_for_storage(storage_id)

        # Execute with separated input/output interface
        client = self._get_device_client(machine)
        client.execute_aten_operation(
            op_name, input_tensor_metadata_dicts, output_storage_ids, args, kwargs
        )

        log.info(f"✅ ORCHESTRATOR: Completed {op_name} with separated interface")

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
        metadata = RemoteTensorMetadata.from_remote_tensor(remote_tensor)

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
                client.remove_storage(storage_id)
                return True
            return False
        except Exception as e:
            log.warning(f"Failed to remove storage {storage_id}: {e}")
            return False


# Global orchestrator instance (Modal provider implementation)
remote_orchestrator = RemoteOrchestrator()
