# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Internal utility functions for mycelya tensor operations.

This module provides internal utility functions for getting tensor and storage IDs
from mycelya tensors. These functions are for internal use only and should not be
used by external users of mycelya_torch.
"""

from typing import Any, Dict, List, Tuple

import torch
from typing_extensions import TypedDict


class TensorMetadata(TypedDict):
    """Structure for tensor metadata with temp key.

    This TypedDict defines the structure returned by dynamic operations
    that need to pass tensor metadata along with a temporary key for
    linking local tensors to remote tensors.
    """

    shape: List[int]
    dtype: str
    stride: List[int]
    storage_offset: int
    nbytes: int
    requires_grad: bool
    temp_key: str


def get_tensor_id(tensor: torch.Tensor) -> int:
    """Get tensor metadata hash and ensure tensor ID is registered.

    This function computes a metadata hash for a mycelya tensor and automatically
    registers the tensor ID in the storage registry when first accessed.

    Args:
        tensor: Mycelya tensor to get metadata hash for

    Returns:
        64-bit integer hash of the tensor's metadata

    Raises:
        RuntimeError: If tensor is not a mycelya tensor
    """
    if tensor.device.type != "mycelya":
        raise RuntimeError(
            f"get_tensor_id() can only be called on mycelya tensors, got {tensor.device.type}"
        )

    from mycelya_torch._C import _get_metadata_hash

    return _get_metadata_hash(tensor)


def get_storage_id(tensor: torch.Tensor) -> int:
    """Get storage ID from tensor's data pointer.

    This function extracts the storage ID from a mycelya tensor's data pointer.
    The storage ID is used for memory management and storage-level operations.

    Args:
        tensor: Mycelya tensor to get storage ID for

    Returns:
        Storage ID as integer

    Raises:
        RuntimeError: If tensor is not a mycelya tensor
    """
    if tensor.device.type != "mycelya":
        raise RuntimeError(
            f"get_storage_id() can only be called on mycelya tensors, got {tensor.device.type}"
        )

    # Get storage ID as integer from data pointer
    data_ptr = tensor.untyped_storage().data_ptr()
    return data_ptr  # data_ptr is the storage ID cast to void*


def dtype_to_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to string without 'torch.' prefix.

    Args:
        dtype: PyTorch dtype (e.g., torch.float32)

    Returns:
        String representation without prefix (e.g., "float32")
    """
    return str(dtype).replace("torch.", "")


def map_args_kwargs(
    func, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Lightweight function to apply func to all elements in args/kwargs, recursing into lists/tuples."""

    def map_container(container):
        if isinstance(container, (list, tuple)):
            return type(container)(func(item) for item in container)
        return func(container)

    return tuple(map_container(arg) for arg in args), {
        k: map_container(v) for k, v in kwargs.items()
    }
