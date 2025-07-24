# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tensor utilities for metadata handling, serialization, and device transfers.

This module provides a clean, minimal API for tensor conversions:
- TensorMetadata class for unified tensor metadata representation
- Methods for converting between CPU, remote, and meta tensors
- Serialization utilities for data transfer
"""

import io
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from ._logging import get_logger

if TYPE_CHECKING:
    from .device import RemoteMachine

log = get_logger(__name__)


class TensorMetadata:
    """
    Metadata representation of tensors (CPU, remote, or meta).
    
    Stores tensor shape, layout, and type information. For remote tensors,
    also includes the storage_id for referencing data on remote devices.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        stride: Tuple[int, ...],
        storage_offset: int,
        dtype: torch.dtype,
        storage_id: Optional[int] = None
    ):
        """
        Initialize tensor metadata.
        
        Args:
            shape: Tensor dimensions
            stride: Tensor stride for memory layout
            storage_offset: Offset into storage
            dtype: PyTorch data type
            storage_id: Storage ID for remote tensors (None for CPU/meta)
        """
        self.shape = shape
        self.stride = stride
        self.storage_offset = storage_offset
        self.dtype = dtype
        self.storage_id = storage_id
        
        # Calculate derived properties
        self.nelem_in_bytes = self._calculate_nelem_in_bytes()

    def _calculate_nelem_in_bytes(self) -> int:
        """Calculate the number of elements needed in storage."""
        if not self.shape:
            return self.dtype.itemsize  # scalar
        
        # Calculate the highest address accessed
        max_index = self.storage_offset
        for dim_size, dim_stride in zip(self.shape, self.stride):
            if dim_size > 1:
                max_index += (dim_size - 1) * abs(dim_stride)
        
        return (max_index + 1) * self.dtype.itemsize

    @classmethod
    def from_remote_tensor(cls, tensor: torch.Tensor) -> "TensorMetadata":
        """Create metadata from a remote tensor."""
        if tensor.device.type != "remote":
            raise ValueError(f"Expected remote tensor, got device: {tensor.device}")
        
        storage_id = tensor.untyped_storage().data_ptr()
        return cls(
            shape=tuple(tensor.size()),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            storage_id=storage_id
        )

    @classmethod
    def from_cpu_tensor(cls, tensor: torch.Tensor) -> "TensorMetadata":
        """Create metadata from a CPU tensor."""
        if tensor.device.type != "cpu":
            raise ValueError(f"Expected CPU tensor, got device: {tensor.device}")
        
        return cls(
            shape=tuple(tensor.size()),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            storage_id=None
        )

    @classmethod
    def from_meta_tensor(cls, tensor: torch.Tensor) -> "TensorMetadata":
        """Create metadata from a meta tensor."""
        if tensor.device.type != "meta":
            raise ValueError(f"Expected meta tensor, got device: {tensor.device}")
        
        return cls(
            shape=tuple(tensor.size()),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            storage_id=None
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorMetadata":
        """Create metadata from a dictionary representation."""
        return cls(
            shape=tuple(data["shape"]),
            stride=tuple(data["stride"]),
            storage_offset=data["storage_offset"],
            dtype=getattr(torch, data["dtype"]),
            storage_id=data.get("storage_id")
        )

    def to_remote_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Create a remote tensor from this metadata.
        
        Args:
            device: Remote device to create tensor on
            
        Returns:
            Remote tensor with this metadata
        """
        if device.type != "remote":
            raise ValueError(f"Expected remote device, got: {device}")
        
        if self.storage_id is None:
            raise ValueError("Cannot create remote tensor without storage_id")
        
        # Create remote tensor using storage ID as data pointer
        from ._device_daemon import driver
        
        # Create storage on device first
        storage_bytes = self.nelem_in_bytes
        driver.exec("create_storage_with_id", self.storage_id, storage_bytes)
        
        # Create tensor with the storage ID as data pointer
        storage = torch.UntypedStorage.from_buffer(
            bytearray(storage_bytes), dtype=torch.uint8
        )
        storage._set_cdata(self.storage_id)  # Use storage_id as data pointer
        
        # Create tensor from storage
        tensor = torch.tensor([], dtype=self.dtype, device=device)
        tensor = tensor.set_(storage, self.storage_offset, self.shape, self.stride)
        
        return tensor

    def to_meta_tensor(self) -> torch.Tensor:
        """Create a meta tensor from this metadata."""
        # Create meta tensor with same shape and dtype
        return torch.empty(
            self.shape,
            dtype=self.dtype,
            device="meta"
        ).as_strided(self.shape, self.stride, self.storage_offset)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        result = {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "dtype": str(self.dtype).split(".")[-1],  # e.g., "float32"
            "nelem_in_bytes": self.nelem_in_bytes
        }
        
        if self.storage_id is not None:
            result["storage_id"] = self.storage_id
            
        return result

    def to_cpu_tensor_from_bytes(self, data: bytes) -> torch.Tensor:
        """
        Create a CPU tensor from bytes using this metadata.
        
        Args:
            data: Serialized tensor data
            
        Returns:
            CPU tensor reconstructed from bytes
        """
        # Deserialize the tensor data
        buffer = io.BytesIO(data)
        tensor = torch.load(buffer, map_location="cpu", weights_only=False)
        
        # Verify the tensor matches our metadata
        if tensor.dtype != self.dtype:
            raise ValueError(f"Dtype mismatch: expected {self.dtype}, got {tensor.dtype}")
        
        # Apply the correct view if needed
        if (tuple(tensor.size()) != self.shape or 
            tuple(tensor.stride()) != self.stride or 
            tensor.storage_offset() != self.storage_offset):
            
            tensor = tensor.as_strided(self.shape, self.stride, self.storage_offset)
        
        return tensor

    def __repr__(self) -> str:
        storage_info = f", storage_id={self.storage_id}" if self.storage_id else ""
        return (f"TensorMetadata(shape={self.shape}, stride={self.stride}, "
                f"storage_offset={self.storage_offset}, dtype={self.dtype}{storage_info})")


def cpu_tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """
    Convert a CPU tensor to bytes for data transfer.
    
    Args:
        tensor: CPU tensor to serialize
        
    Returns:
        Serialized tensor data
    """
    if tensor.device.type != "cpu":
        raise ValueError(f"Expected CPU tensor, got device: {tensor.device}")
    
    # Make tensor contiguous for efficient serialization
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # Serialize to bytes
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def remote_tensor_to_cpu(remote_tensor: torch.Tensor) -> torch.Tensor:
    """
    Transfer a remote tensor to CPU.
    
    Args:
        remote_tensor: Remote tensor to transfer
        
    Returns:
        CPU tensor with the same data
    """
    if remote_tensor.device.type != "remote":
        raise ValueError(f"Expected remote tensor, got device: {remote_tensor.device}")
    
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
        dtype=str(metadata.dtype)
    )
    
    # Convert bytes back to CPU tensor using metadata
    return metadata.to_cpu_tensor_from_bytes(tensor_data)


# ============================================================================
# Legacy functions that still need updating throughout the codebase
# ============================================================================

def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Legacy function - use cpu_tensor_to_bytes() instead."""
    return cpu_tensor_to_bytes(tensor)


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """Legacy function - use TensorMetadata.to_cpu_tensor_from_bytes() instead."""
    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def get_tensor_metadata(tensor: torch.Tensor) -> Dict[str, Any]:
    """Legacy function - use TensorMetadata.from_*().to_dict() instead."""
    if tensor.device.type == "remote":
        return TensorMetadata.from_remote_tensor(tensor).to_dict()
    elif tensor.device.type == "cpu":
        return TensorMetadata.from_cpu_tensor(tensor).to_dict()
    elif tensor.device.type == "meta":
        return TensorMetadata.from_meta_tensor(tensor).to_dict()
    else:
        raise ValueError(f"Unsupported device type: {tensor.device.type}")


def cpu_tensor_to_remote(cpu_tensor: torch.Tensor, machine) -> torch.Tensor:
    """Legacy function - this is complex and should be refactored."""
    raise NotImplementedError("cpu_tensor_to_remote needs to be refactored to use new API")


class TensorMetadataConverter:
    """Legacy compatibility class - use TensorMetadata methods instead."""
    
    @staticmethod
    def tensor_to_metadata(tensor: torch.Tensor, name: str) -> TensorMetadata:
        """Convert tensor to metadata."""
        if tensor.device.type == "remote":
            return TensorMetadata.from_remote_tensor(tensor)
        elif tensor.device.type == "cpu":
            return TensorMetadata.from_cpu_tensor(tensor)
        elif tensor.device.type == "meta":
            return TensorMetadata.from_meta_tensor(tensor)
        else:
            raise ValueError(f"Unsupported device type: {tensor.device.type}")
    
    @staticmethod
    def metadata_list_to_dicts(
        metadata_list: list, 
        include_set=None, 
        exclude_set=None
    ) -> list:
        """Convert list of metadata to list of dicts."""
        return [meta.to_dict() for meta in metadata_list]
    
    @staticmethod
    def args_to_metadata_with_placeholders(args, kwargs, op_name=None, operation_context=None):
        """Legacy function - needs refactoring."""
        metadata_list = []
        processed_args = []
        processed_kwargs = {}
        
        # Process args
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
                metadata = TensorMetadata.from_remote_tensor(arg)
                tensor_index = len(metadata_list)
                metadata_list.append(metadata)
                processed_args.append(f"__TENSOR_{tensor_index}")
            else:
                processed_args.append(arg)
        
        # Process kwargs
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.device.type == "remote":
                metadata = TensorMetadata.from_remote_tensor(value)
                tensor_index = len(metadata_list)
                metadata_list.append(metadata)
                processed_kwargs[key] = f"__TENSOR_{tensor_index}"
            else:
                processed_kwargs[key] = value
        
        return tuple(processed_args), processed_kwargs, metadata_list