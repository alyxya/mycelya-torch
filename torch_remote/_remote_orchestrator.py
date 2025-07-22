# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from ._meta_parser import RemoteTensorMeta, TensorMetadataConverter
from .core.container import get_service
from .device import RemoteMachine
from .services.storage_resolver import StorageMachineResolver
from .services.tensor_transfer import TensorTransferService

log = logging.getLogger(__name__)


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
        # Use dependency injection for services - clean architecture with no deprecated fields
        self._tensor_transfer = get_service(TensorTransferService)
        self._storage_resolver = get_service(StorageMachineResolver)

    def _get_device_client(self, machine: "RemoteMachine"):
        """Get the active client for a specific machine."""
        return machine.get_client()

    def _get_machine_for_storage(self, storage_id: int) -> "RemoteMachine":
        """Get the machine that owns a specific storage ID using storage resolver."""
        return self._storage_resolver.get_machine_for_storage(storage_id)

    def execute_remote_aten_operation(
        self,
        op_name: str,
        input_metadata: List[RemoteTensorMeta],
        output_metadata: List[RemoteTensorMeta],
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
        tensor_metadata_dicts = TensorMetadataConverter.metadata_list_to_dicts(
            total_metadata, input_indices, output_indices
        )

        # Get the machine from first input tensor's storage ID
        if not input_metadata:
            raise RuntimeError(f"No input metadata provided for operation {op_name}")

        storage_id = input_metadata[0].storage_id
        machine = self._get_machine_for_storage(storage_id)

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
        return self._tensor_transfer.remote_tensor_to_cpu(remote_tensor)

    def _cpu_tensor_to_remote(
        self, cpu_tensor: torch.Tensor, machine: "RemoteMachine"
    ) -> torch.Tensor:
        """Convert CPU tensor to remote tensor."""
        return self._tensor_transfer.cpu_tensor_to_remote(cpu_tensor, machine)

    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """Serialize tensor to bytes, ensuring view data is contiguous."""
        return self._tensor_transfer.serialize_tensor(tensor)

    def _deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """Deserialize tensor from bytes as a contiguous tensor."""
        return self._tensor_transfer.deserialize_tensor(data)

    def _get_tensor_metadata(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Get tensor metadata."""
        return self._tensor_transfer.get_tensor_metadata(tensor)


# Global orchestrator instance (Modal provider implementation)
remote_orchestrator = RemoteOrchestrator()
