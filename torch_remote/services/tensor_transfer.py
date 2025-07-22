# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tensor transfer service for handling serialization and device transfers.

This service handles all tensor data movement operations:
- Serialization/deserialization of tensors
- Remote-to-CPU tensor transfers
- CPU-to-remote tensor transfers
- Tensor metadata extraction

Extracted from RemoteOrchestrator to provide better separation of concerns.
"""

import io
import logging
from typing import Any, Dict

import torch

log = logging.getLogger(__name__)


class TensorTransferService:
    """Service for handling tensor transfers between devices and serialization.
    
    This class encapsulates all tensor data movement logic, providing
    a clean interface for tensor transfers and serialization operations.
    """

    def serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """Serialize tensor to bytes, ensuring view data is contiguous.
        
        Args:
            tensor: Tensor to serialize
            
        Returns:
            Serialized tensor data as bytes
        """
        buffer = io.BytesIO()
        # Convert to pure CPU tensor and make contiguous to serialize only the view's data
        # This ensures views are serialized as their actual data, not the full underlying storage
        cpu_tensor = tensor.cpu().detach().contiguous()
        torch.save(cpu_tensor, buffer)
        return buffer.getvalue()

    def deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """Deserialize tensor from bytes as a contiguous tensor.
        
        Args:
            data: Serialized tensor data
            
        Returns:
            Deserialized tensor on CPU
        """
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

    def remote_tensor_to_cpu(self, remote_tensor: torch.Tensor) -> torch.Tensor:
        """Convert remote tensor to CPU tensor by retrieving data from remote GPU.
        
        Args:
            remote_tensor: Tensor on remote device to transfer
            
        Returns:
            Tensor on CPU with same data
            
        Raises:
            RuntimeError: If no machine found for device or client not available
        """
        from ..device import get_device_registry

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
        return self.deserialize_tensor(tensor_data)

    def cpu_tensor_to_remote(
        self, cpu_tensor: torch.Tensor, machine: "RemoteMachine"
    ) -> torch.Tensor:
        """Convert CPU tensor to remote tensor.
        
        Args:
            cpu_tensor: Tensor on CPU to transfer
            machine: Target remote machine
            
        Returns:
            Tensor on remote device
            
        Raises:
            RuntimeError: If machine is None
        """
        if machine is None:
            raise RuntimeError(
                "RemoteMachine must be specified for remote tensor creation"
            )

        # Convert to remote device - delegates to PyTorch's device transfer system
        result = cpu_tensor.to(machine.device())
        return result

    def get_tensor_metadata(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Get tensor metadata as a dictionary.
        
        Args:
            tensor: Tensor to extract metadata from
            
        Returns:
            Dictionary containing tensor metadata
        """
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "size": tensor.numel(),
            "element_size": tensor.element_size(),
        }
