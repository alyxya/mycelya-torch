# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Internal utility functions for mycelya tensor operations.

This module provides internal utility functions for getting tensor and storage IDs
from mycelya tensors. These functions are for internal use only and should not be
used by external users of mycelya_torch.
"""

import torch

import mycelya_torch._C


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

    tensor_id = mycelya_torch._C._get_metadata_hash(tensor)

    # Register tensor ID for device tracking (needed for cross-device validation)
    from ._storage import get_tensor_device, register_tensor_id

    if get_tensor_device(tensor_id) is None:
        device_index = tensor.device.index
        register_tensor_id(tensor_id, device_index)

    return tensor_id


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
