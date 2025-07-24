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
from typing import Any, Dict, Optional, Tuple

import torch

from ._logging import get_logger

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
        storage_id: Optional[int] = None,
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
            storage_id=storage_id,
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
            storage_id=None,
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
            storage_id=None,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorMetadata":
        """Create metadata from a dictionary representation."""
        return cls(
            shape=tuple(data["shape"]),
            stride=tuple(data["stride"]),
            storage_offset=data["storage_offset"],
            dtype=getattr(torch, data["dtype"]),
            storage_id=data.get("storage_id"),
        )

    def to_meta_tensor(self) -> torch.Tensor:
        """Create a meta tensor from this metadata."""
        # Create meta tensor with same shape and dtype
        return torch.empty(self.shape, dtype=self.dtype, device="meta").as_strided(
            self.shape, self.stride, self.storage_offset
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        result = {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "dtype": str(self.dtype).split(".")[-1],  # e.g., "float32"
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
            raise ValueError(
                f"Dtype mismatch: expected {self.dtype}, got {tensor.dtype}"
            )

        # Apply the correct view if needed
        if (
            tuple(tensor.size()) != self.shape
            or tuple(tensor.stride()) != self.stride
            or tensor.storage_offset() != self.storage_offset
        ):
            tensor = tensor.as_strided(self.shape, self.stride, self.storage_offset)

        return tensor

    def __repr__(self) -> str:
        storage_info = f", storage_id={self.storage_id}" if self.storage_id else ""
        return (
            f"TensorMetadata(shape={self.shape}, stride={self.stride}, "
            f"storage_offset={self.storage_offset}, dtype={self.dtype}{storage_info})"
        )


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
