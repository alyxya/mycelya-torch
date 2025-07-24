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
from ._tensor_utils import (
    TensorMetadata,
    remote_tensor_to_cpu,
    cpu_tensor_to_bytes,
)
from ._storage import get_machine_for_storage
from .device import RemoteMachine

log = get_logger(__name__)


# Exception handling is done through standard RuntimeError
# Custom exceptions removed as they were not used elsewhere in the codebase


def with_error_handling(func):
    """Decorator to add error handling to remote operations."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check for specific error patterns and provide helpful context
            error_msg = str(e).lower()

            if "storage id" in error_msg and "not found" in error_msg:
                raise RuntimeError(f"Storage reference is stale: {e}") from e
            elif "machine" in error_msg and "not running" in error_msg:
                raise RuntimeError(f"Remote machine connection lost: {e}") from e
            elif "remote execution failed" in error_msg:
                raise RuntimeError(f"Remote execution failed: {e}") from e
            else:
                # Re-raise with context about remote operation failure
                raise RuntimeError(f"Remote operation failed: {e}") from e

    return wrapper


# Try to load the remote execution module (Modal provider implementation)
try:
    log.info("Loaded modal client")
except Exception as e:
    log.warning(f"Modal client not available: {e}")


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

    def _get_machine_for_storage(self, storage_id: int) -> "RemoteMachine":
        """Get the machine that owns a specific storage ID."""
        return get_machine_for_storage(storage_id)

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

    def cleanup(self):
        """Clean up the remote orchestrator."""
        # Note: Individual machine cleanup is handled by RemoteMachine instances
        pass

    def _remote_tensor_to_cpu(self, remote_tensor: torch.Tensor) -> torch.Tensor:
        """Convert remote tensor to CPU tensor by retrieving data from remote GPU."""
        return remote_tensor_to_cpu(remote_tensor)


    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """Serialize tensor to bytes, ensuring view data is contiguous."""
        return cpu_tensor_to_bytes(tensor)



    def remove_tensor_from_remote(self, storage_id: int, machine: "RemoteMachine") -> bool:
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
