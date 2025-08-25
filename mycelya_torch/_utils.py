# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Internal utility functions for mycelya tensor operations.

This module provides internal utility functions for getting tensor and storage IDs
from mycelya tensors. These functions are for internal use only and should not be
used by external users of mycelya_torch.
"""

from typing import Any, Dict, List, Tuple

import torch


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


def args_to_tensors_with_ids_and_mask(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[Tuple[Any, ...], Dict[str, Any], List[torch.Tensor], List[bool]]:
    """Convert args/kwargs, replacing remote tensors with tensor IDs and collecting tensors."""
    tensor_list: List[torch.Tensor] = []
    tensor_mask: List[bool] = []

    def replace_remote_tensor_with_id(obj):
        """Replace remote tensors with tensor IDs and collect tensors."""
        if isinstance(obj, torch.Tensor):
            tensor_list.append(obj)
            tensor_mask.append(True)
            return get_tensor_id(obj)
        tensor_mask.append(False)
        return obj

    # Iterate over args
    processed_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            processed_arg = []
            for item in arg:
                processed_arg.append(replace_remote_tensor_with_id(item))
            processed_args.append(type(arg)(processed_arg))
        else:
            processed_args.append(replace_remote_tensor_with_id(arg))

    # Iterate over kwargs values
    processed_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple)):
            processed_value = []
            for item in value:
                processed_value.append(replace_remote_tensor_with_id(item))
            processed_kwargs[key] = type(value)(processed_value)
        else:
            processed_kwargs[key] = replace_remote_tensor_with_id(value)

    return tuple(processed_args), processed_kwargs, tensor_list, tensor_mask
