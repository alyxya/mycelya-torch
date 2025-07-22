# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""

import io
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch

from ._meta_parser import RemoteTensorMeta, TensorMetadataConverter
from .device import RemoteMachine, get_device_registry

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
        self._device_apps: Dict[str, Any] = {}  # Cache for device-specific clients
        self._last_heartbeat: Dict[
            str, float
        ] = {}  # Track last successful communication per device

    def _get_device_client(self, machine: "RemoteMachine"):
        """Get the active ModalClient for a specific machine."""
        # Get the pre-started client from the machine
        client = machine.get_client()

        if client is None:
            raise RuntimeError(f"No client available for machine {machine.machine_id}")

        if not client.is_running():
            raise RuntimeError(
                f"Client for machine {machine.machine_id} is not running"
            )

        return client

    def _get_machine_for_storage(self, storage_id: int) -> "RemoteMachine":
        """Get the machine that owns a specific storage ID."""
        # Import here to avoid circular imports
        from . import driver

        # Get device index for this storage
        device_idx = driver.exec("get_storage_device", storage_id)
        if device_idx is None:
            raise RuntimeError(f"No device found for storage {storage_id}")

        # Get machine for device index
        registry = get_device_registry()
        machine = registry.get_device_by_index(device_idx)
        if machine is None:
            raise RuntimeError(f"No machine found for device index {device_idx}")

        return machine

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

    def _update_heartbeat(self, machine_id: str) -> None:
        """Update the last successful communication timestamp for a device."""
        self._last_heartbeat[machine_id] = time.time()

    def get_last_heartbeat(self, machine_id: str) -> Optional[float]:
        """Get the timestamp of last successful communication with a device."""
        return self._last_heartbeat.get(machine_id)

    def reconnect_device(self, machine: "RemoteMachine") -> bool:
        """
        Attempt to reconnect to a device.

        Args:
            device: The device to reconnect

        Returns:
            True if reconnection successful, False otherwise
        """
        try:
            client = machine.get_client()
            if client:
                # Stop and restart the client
                client.stop()
                client.start()

                # Test connection
                if self.is_device_connected(machine):
                    log.info(f"Successfully reconnected to device {machine.machine_id}")
                    return True
                else:
                    log.warning(f"Failed to reconnected to device {machine.machine_id}")
                    return False
            return False
        except Exception as e:
            log.error(f"Error during reconnection to device {machine.machine_id}: {e}")
            return False

    def cleanup(self):
        """Clean up the remote orchestrator."""
        # No longer needed since machines are managed by devices
        self._device_apps.clear()
        self._last_heartbeat.clear()

    def _remote_tensor_to_cpu(self, remote_tensor: torch.Tensor) -> torch.Tensor:
        """Convert remote tensor to CPU tensor by retrieving data from remote GPU."""
        from .device import get_device_registry

        # Get the machine backend
        registry = get_device_registry()
        machine = registry.get_device_by_index(remote_tensor.device.index)

        if machine is None:
            raise RuntimeError(
                f"No RemoteMachine found for remote device index {remote_tensor.device.index}"
            )

        # Get the client for this machine
        client = machine.get_client()
        if client is None or not client.is_running():
            raise RuntimeError(f"Client not available for machine {machine.machine_id}")

        # Get tensor data using storage ID
        storage_id = remote_tensor.untyped_storage().data_ptr()

        # Use client to get tensor data by ID with view information
        # Pass tensor metadata so remote side can serialize just the view's data
        tensor_data = client.get_storage_data(
            storage_id,
            shape=list(remote_tensor.shape),
            stride=list(remote_tensor.stride()),
            storage_offset=remote_tensor.storage_offset(),
            dtype=str(remote_tensor.dtype),
        )

        # Deserialize the tensor
        return self._deserialize_tensor(tensor_data)

    def _cpu_tensor_to_remote(
        self, cpu_tensor: torch.Tensor, machine: "RemoteMachine"
    ) -> torch.Tensor:
        """Convert CPU tensor to remote tensor."""
        # Create a new remote tensor from the CPU tensor using the RemoteMachine
        if machine is None:
            raise RuntimeError(
                "RemoteMachine must be specified for remote tensor creation"
            )

        # Convert to remote device - no need to manually attach _device_id anymore
        result = cpu_tensor.to(machine.device())
        return result

    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """Serialize tensor to bytes, ensuring view data is contiguous."""
        buffer = io.BytesIO()
        # Convert to pure CPU tensor and make contiguous to serialize only the view's data
        # This ensures views are serialized as their actual data, not the full underlying storage
        cpu_tensor = tensor.cpu().detach().contiguous()
        torch.save(cpu_tensor, buffer)
        return buffer.getvalue()

    def _deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """Deserialize tensor from bytes as a contiguous tensor."""
        buffer = io.BytesIO(data)
        tensor = torch.load(buffer, map_location="cpu")

        # Since we serialize with .contiguous(), the deserialized tensor should already be contiguous
        # with storage_offset=0 and optimal stride. No view reconstruction needed.
        # Just ensure we have untyped storage for consistency with the remote tensor system.
        if hasattr(tensor, "untyped_storage"):
            untyped_storage = tensor.untyped_storage()
            # The tensor should already be contiguous, so we preserve its natural layout
            tensor = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
                untyped_storage,
                0,  # storage_offset should be 0 for contiguous tensors
                tensor.shape,
                tensor.stride(),
            )
        return tensor

    def _get_tensor_metadata(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Get tensor metadata."""
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "size": tensor.numel(),
            "element_size": tensor.element_size(),
        }



# Global orchestrator instance (Modal provider implementation)
remote_orchestrator = RemoteOrchestrator()
