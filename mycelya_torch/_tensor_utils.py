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

import io
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


def cpu_tensor_to_storage_bytes(tensor: torch.Tensor) -> bytes:
    """
    Convert a CPU tensor to raw untyped storage bytes.

    Args:
        tensor: CPU tensor to extract storage bytes from

    Returns:
        Raw untyped storage bytes
    """
    if tensor.device.type != "cpu":
        raise ValueError(f"Expected CPU tensor, got device: {tensor.device}")

    # Get raw untyped storage bytes
    untyped_storage = tensor.untyped_storage()
    return bytes(untyped_storage)


def storage_bytes_to_cpu_tensor(
    raw_bytes: bytes,
    shape: list,
    stride: list,
    storage_offset: int,
    dtype: str
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

    DEPRECATED: Use cpu_tensor_to_storage_bytes() instead.

    Args:
        tensor: CPU tensor to serialize

    Returns:
        Serialized tensor data
    """
    if tensor.device.type != "cpu":
        raise ValueError(f"Expected CPU tensor, got device: {tensor.device}")

    # Serialize to bytes
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def bytes_to_cpu_tensor(data: bytes) -> torch.Tensor:
    """
    Convert bytes to a CPU tensor.

    DEPRECATED: Use storage_bytes_to_cpu_tensor() instead.

    Args:
        data: Serialized tensor data

    Returns:
        CPU tensor reconstructed from bytes (always contiguous and packed)
    """
    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location="cpu", weights_only=False)
