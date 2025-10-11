# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Internal utility functions for mycelya tensor operations.

This module provides internal utility functions for getting tensor and storage IDs
from mycelya tensors. These functions are for internal use only and should not be
used by external users of mycelya_torch.
"""

from typing import TYPE_CHECKING, Any, TypedDict

import torch

if TYPE_CHECKING:
    from ._storage import StorageManager


class TensorMetadata(TypedDict):
    """Structure for tensor metadata with ID.

    This TypedDict defines the structure returned by dynamic operations
    that need to pass tensor metadata along with an ID for
    linking local tensors to remote tensors.
    """

    shape: list[int]
    stride: list[int]
    dtype: str
    storage_offset: int
    nbytes: int
    device_type: str
    device_index: int
    id: str | int
    alias_id: str | int | None


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


def create_mycelya_tensor_from_metadata(
    metadata: TensorMetadata,
    device: torch.device,
    storage_manager: "StorageManager",
) -> torch.Tensor:
    """Create a mycelya tensor from metadata with proper alias handling.

    This function handles three cases for tensor creation:
    1. No alias_id (None): Creates new storage and registers it under tensor_id if it's a temp ID
    2. Integer alias_id: Looks up existing storage from tensor_id_to_storage mapping
    3. String alias_id (temp ID): Looks up storage from temp_id_to_storage, or creates new
       storage if not found (edge case: alias created before source)

    Args:
        metadata: Tensor metadata containing shape, dtype, stride, storage_offset, nbytes, id, alias_id
        device: Mycelya device where the tensor should appear to be located
        storage_manager: StorageManager instance for handling storage lookups and registrations

    Returns:
        Mycelya tensor with properly resolved storage (either new or aliased)
    """
    tensor_id = metadata["id"]
    alias_id = metadata["alias_id"]

    storage = None

    # First check if this tensor_id already has a storage registered (edge case: alias created first)
    # tensor_id should always be a string (temp ID) since metadata comes from remote execution
    if tensor_id in storage_manager._temp_id_to_storage:
        storage = storage_manager._temp_id_to_storage[tensor_id]

    # If no storage found yet, check alias_id to determine what to do
    if storage is None:
        if alias_id is None:
            # Case 1: No alias, create new storage and register under tensor_id
            storage = torch.UntypedStorage(metadata["nbytes"], device=device)
            storage_manager._temp_id_to_storage[tensor_id] = storage
        elif isinstance(alias_id, int):
            # Case 2: Integer alias ID, look up in tensor_id_to_storage
            storage = storage_manager._tensor_id_to_storage.get(alias_id)
            if storage is None:
                raise RuntimeError(
                    f"Cannot create tensor alias: storage for alias_id {alias_id} not found in tensor_id_to_storage mapping"
                )
        elif isinstance(alias_id, str):
            # Case 3: String (temp) alias ID
            if alias_id in storage_manager._temp_id_to_storage:
                # Normal case: source tensor already created
                storage = storage_manager._temp_id_to_storage[alias_id]
            else:
                # Edge case: alias tensor created before source
                # Create new storage and register under alias_id
                storage = torch.UntypedStorage(metadata["nbytes"], device=device)
                storage_manager._temp_id_to_storage[alias_id] = storage
        else:
            raise ValueError(
                f"Invalid alias_id type: {type(alias_id)}. Expected int, str, or None"
            )

    # Create tensor using the resolved storage
    tensor = torch.empty(0, dtype=getattr(torch, metadata["dtype"]), device=device)
    tensor.set_(
        storage, metadata["storage_offset"], metadata["shape"], metadata["stride"]
    )

    return tensor


def map_args_kwargs(
    func, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Lightweight function to apply func to all elements in args/kwargs, recursing into lists/tuples."""

    def map_container(container):
        if isinstance(container, (list, tuple)):
            return type(container)(func(item) for item in container)
        return func(container)

    return tuple(map_container(arg) for arg in args), {
        k: map_container(v) for k, v in kwargs.items()
    }
