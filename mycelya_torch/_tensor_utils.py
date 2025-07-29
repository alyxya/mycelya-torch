# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tensor utilities for metadata handling, serialization, and device transfers.

This module provides a clean, type-safe API for tensor conversions:
- LocalTensorMetadata for CPU/meta tensors (no storage_id)
- RemoteTensorMetadata for remote tensors (always has storage_id)
- Methods for converting between CPU, remote, and meta tensors
- Serialization utilities for data transfer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from ._logging import get_logger

log = get_logger(__name__)


@dataclass
class BaseTensorMetadata(ABC):
    """Common interface for all tensor metadata."""

    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        pass

    @abstractmethod
    def is_remote(self) -> bool:
        """Check if this represents a remote tensor."""
        pass

    def to_meta_tensor(self) -> torch.Tensor:
        """Create a meta tensor from this metadata."""
        return torch.empty(self.shape, dtype=self.dtype, device="meta").as_strided(
            self.shape, self.stride, self.storage_offset
        )


@dataclass
class LocalTensorMetadata(BaseTensorMetadata):
    """Metadata for CPU/meta tensors - no storage_id."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "dtype": str(self.dtype).split(".")[-1],
        }

    def is_remote(self) -> bool:
        return False

    @classmethod
    def from_cpu_tensor(cls, tensor: torch.Tensor) -> "LocalTensorMetadata":
        """Create metadata from a CPU tensor."""
        if tensor.device.type != "cpu":
            raise ValueError(f"Expected CPU tensor, got device: {tensor.device}")
        return cls(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
        )

    @classmethod
    def from_meta_tensor(cls, tensor: torch.Tensor) -> "LocalTensorMetadata":
        """Create metadata from a meta tensor."""
        if tensor.device.type != "meta":
            raise ValueError(f"Expected meta tensor, got device: {tensor.device}")
        return cls(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
        )

    def __repr__(self) -> str:
        return (
            f"LocalTensorMetadata(shape={self.shape}, stride={self.stride}, "
            f"storage_offset={self.storage_offset}, dtype={self.dtype})"
        )


@dataclass
class RemoteTensorMetadata(BaseTensorMetadata):
    """Metadata for remote tensors - always has storage_id."""

    storage_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "dtype": str(self.dtype).split(".")[-1],
            "storage_id": self.storage_id,
        }

    def is_remote(self) -> bool:
        return True

    @classmethod
    def from_remote_tensor(cls, tensor: torch.Tensor) -> "RemoteTensorMetadata":
        """Create metadata from a remote tensor."""
        if tensor.device.type != "mycelya":
            raise ValueError(f"Expected remote tensor, got device: {tensor.device}")
        storage_id = tensor.untyped_storage().data_ptr()
        return cls(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            storage_id=storage_id,
        )

    def __repr__(self) -> str:
        return (
            f"RemoteTensorMetadata(shape={self.shape}, stride={self.stride}, "
            f"storage_offset={self.storage_offset}, dtype={self.dtype}, "
            f"storage_id={self.storage_id})"
        )


# Union type for contexts that can handle either
TensorMetadata = Union[LocalTensorMetadata, RemoteTensorMetadata]


# cpu_tensor_to_storage_bytes function removed - pass storage tensors directly instead
# This eliminates the problematic bytes(untyped_storage) call


def storage_bytes_to_cpu_tensor(
    raw_bytes: bytes, shape: list, stride: list, storage_offset: int, dtype: str
) -> torch.Tensor:
    """
    Convert raw storage bytes to a CPU tensor with specified view parameters.

    Args:
        raw_bytes: Raw untyped storage bytes
        shape: Tensor shape for view
        stride: Tensor stride for view
        storage_offset: Storage offset for view
        dtype: Tensor data type string (e.g., "float32")

    Returns:
        CPU tensor reconstructed from raw bytes with specified view
    """
    # Convert dtype string to torch.dtype
    dtype_name = dtype.replace("torch.", "")
    torch_dtype = getattr(torch, dtype_name)

    # Create untyped storage from raw bytes
    untyped_storage = torch.UntypedStorage.from_buffer(raw_bytes, dtype=torch.uint8)

    # Create tensor from storage with specified view parameters
    tensor = torch.empty(0, dtype=torch_dtype, device="cpu")
    tensor.set_(untyped_storage, storage_offset, shape, stride)

    return tensor


def cpu_tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """
    Convert a CPU tensor to bytes for data transfer.

    DEPRECATED: Use cpu_tensor_to_torch_bytes() instead for optimized serialization.

    Args:
        tensor: CPU tensor to serialize

    Returns:
        Serialized tensor data
    """
    import warnings

    warnings.warn(
        "cpu_tensor_to_bytes() is deprecated. Use cpu_tensor_to_torch_bytes() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Use the optimized implementation
    return cpu_tensor_to_torch_bytes(tensor)


def bytes_to_cpu_tensor(data: bytes) -> torch.Tensor:
    """
    Convert bytes to a CPU tensor.

    DEPRECATED: Use torch_bytes_to_cpu_tensor() instead for optimized deserialization.

    Args:
        data: Serialized tensor data

    Returns:
        CPU tensor reconstructed from bytes (always contiguous and packed)
    """
    import warnings

    warnings.warn(
        "bytes_to_cpu_tensor() is deprecated. Use torch_bytes_to_cpu_tensor() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Use the optimized implementation
    return torch_bytes_to_cpu_tensor(data)


def cpu_tensor_to_torch_bytes(tensor: torch.Tensor) -> bytes:
    """
    Convert a CPU tensor to optimized bytes using a custom serialization format.

    This function uses an optimized approach that stores tensor metadata and raw data
    separately for better performance than torch.save. The format includes:
    - Shape, stride, storage_offset, dtype as JSON metadata
    - Raw tensor data using numpy.tobytes()

    Args:
        tensor: CPU tensor to serialize

    Returns:
        Serialized tensor data in optimized format

    Raises:
        ValueError: If tensor is not on CPU device
    """
    if tensor.device.type != "cpu":
        raise ValueError(f"Expected CPU tensor, got device: {tensor.device}")

    import json

    # Create metadata dict
    metadata = {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype).split(".")[-1],  # e.g., "float32"
    }

    # Serialize metadata to JSON bytes
    metadata_json = json.dumps(metadata).encode("utf-8")
    metadata_length = len(metadata_json)

    # Get raw data using optimized numpy approach
    raw_data = tensor.numpy().tobytes()

    # Pack format: [metadata_length:4 bytes][metadata:N bytes][raw_data:remaining bytes]
    return metadata_length.to_bytes(4, byteorder="little") + metadata_json + raw_data


def torch_bytes_to_cpu_tensor(data: bytes) -> torch.Tensor:
    """
    Convert optimized serialized bytes back to a CPU tensor.

    This function deserializes bytes created by cpu_tensor_to_torch_bytes()
    using the custom format that includes metadata and raw data.

    Args:
        data: Serialized tensor data in optimized format

    Returns:
        CPU tensor reconstructed from serialized bytes
    """
    import json

    # Unpack format: [metadata_length:4 bytes][metadata:N bytes][raw_data:remaining bytes]
    if len(data) < 4:
        raise ValueError("Invalid serialized data: too short")

    # Extract metadata length
    metadata_length = int.from_bytes(data[:4], byteorder="little")

    if len(data) < 4 + metadata_length:
        raise ValueError("Invalid serialized data: insufficient metadata")

    # Extract and parse metadata
    metadata_json = data[4 : 4 + metadata_length].decode("utf-8")
    metadata = json.loads(metadata_json)

    # Extract raw data
    raw_data = data[4 + metadata_length :]

    # Convert dtype string back to torch.dtype
    dtype_name = metadata["dtype"]
    torch_dtype = getattr(torch, dtype_name)

    # Create writable buffer to avoid PyTorch warnings
    writable_data = bytearray(raw_data)

    # Reconstruct tensor using torch.frombuffer
    flat_tensor = torch.frombuffer(writable_data, dtype=torch_dtype)

    # Reshape and apply stride/offset
    tensor = flat_tensor.clone()  # Clone to ensure independent memory
    tensor = tensor.reshape(metadata["shape"])

    # If the tensor has custom stride or offset, we need to use as_strided
    if list(tensor.stride()) != metadata["stride"] or metadata["storage_offset"] != 0:
        tensor = tensor.as_strided(
            metadata["shape"], metadata["stride"], metadata["storage_offset"]
        )

    return tensor


def numpy_bytes_to_cpu_tensor(
    data: bytes, shape: Tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """
    Convert numpy serialized bytes back to a CPU tensor with writable buffer handling.

    This function deserializes bytes created by tensor.numpy().tobytes()
    back into a CPU tensor, ensuring the buffer is writable to avoid PyTorch warnings.

    Args:
        data: Serialized tensor data from numpy serialization
        shape: Original tensor shape
        dtype: Original tensor dtype

    Returns:
        CPU tensor reconstructed from serialized bytes
    """
    # Create a writable bytearray to avoid PyTorch warnings about non-writable buffers
    writable_data = bytearray(data)

    # Use torch.frombuffer with the writable buffer
    tensor = torch.frombuffer(writable_data, dtype=dtype).reshape(shape)

    # Clone to ensure we have our own memory (not a view into the bytearray)
    return tensor.clone()
