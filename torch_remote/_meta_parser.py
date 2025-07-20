# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from pprint import pformat
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils._pytree import tree_map, tree_map_only


class RemoteTensorMeta:
    def __init__(self, data_ptr: int, size: torch.Size, stride: Tuple[int, ...], 
                 storage_offset: int, dtype: torch.dtype, nelem_in_bytes: int, 
                 storage_id: Optional[int] = None) -> None:
        """Create RemoteTensorMeta with explicit metadata.
        
        Args:
            data_ptr: Tensor ID/storage ID
            size: Tensor shape
            stride: Tensor stride
            storage_offset: Storage offset
            dtype: Data type
            nelem_in_bytes: Number of elements in bytes
            storage_id: Optional storage ID for view tracking
        """
        self.data_ptr = data_ptr
        self.size = size
        self.stride = stride
        self.storage_offset = storage_offset
        self.dtype = dtype
        self.nelem_in_bytes = nelem_in_bytes
        self.storage_id = storage_id

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, checked: bool = True) -> 'RemoteTensorMeta':
        """Create RemoteTensorMeta from existing tensor.
        
        Args:
            tensor: Source tensor to extract metadata from
            checked: Whether to validate tensor is on remote device
            
        Returns:
            RemoteTensorMeta instance with tensor's metadata
            
        Raises:
            RuntimeError: If checked=True and tensor is not on remote device
        """
        if checked and tensor.device.type != "remote":
            raise RuntimeError(
                "Creating RemoteTensorMeta is only for Tensors on remote device"
            )
        
        return cls(
            data_ptr=tensor.untyped_storage().data_ptr(),
            size=tensor.size(),
            stride=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            nelem_in_bytes=tensor.nelement() * tensor.element_size()
        )

    @classmethod
    def from_remote_tensor(cls, tensor: torch.Tensor) -> 'RemoteTensorMeta':
        """Create RemoteTensorMeta for remote tensor serialization.
        
        Args:
            tensor: Remote tensor to create metadata from
            
        Returns:
            RemoteTensorMeta instance with storage_id set for view tracking
        """
        storage_id = tensor.untyped_storage().data_ptr()
        return cls(
            data_ptr=storage_id,
            size=tensor.size(),
            stride=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            nelem_in_bytes=tensor.numel() * tensor.element_size(),
            storage_id=storage_id
        )

    def __repr__(self) -> str:
        return (
            f"RemoteTensorMeta({self.data_ptr=}, {self.size=}, {self.stride=}, "
            f"{self.storage_offset=}, {self.dtype=}, {self.nelem_in_bytes=})"
        )


# RemoteTensorData class removed - C++ implementations now return regular torch.Tensor objects
# with proper device handling via the enhanced RemoteAllocator and ID-based allocation system


VALID_QUEUE_TYPES_IN = {torch.Tensor, int, float, torch.dtype}

VALID_QUEUE_TYPES_OUT = {RemoteTensorMeta, int, float, str, torch.dtype}


def safe_str(args: Any) -> str:
    """Convert arguments to a safe string representation for logging.

    Converts torch.Tensor objects to RemoteTensorMeta strings to avoid
    potential issues with remote tensor string representation.

    Args:
        args: Arguments to convert to string

    Returns:
        Safe string representation of the arguments
    """
    def convert(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return str(RemoteTensorMeta.from_tensor(obj, checked=False))
        else:
            return obj

    new_args = tree_map(convert, args)
    return pformat(new_args)


def validate_send_queue_args(cmd: str, args: Any) -> None:
    """Validate that arguments are safe to send through the remote queue.

    Ensures that only supported object types are sent over the remote
    communication channel to prevent serialization errors.

    Args:
        cmd: Command name for context in error messages
        args: Arguments to validate

    Raises:
        RuntimeError: If invalid object types are found
    """
    def check(obj: Any) -> None:
        if type(obj) not in VALID_QUEUE_TYPES_OUT:
            if (
                cmd == "recv_data"
                and type(obj) in [torch.Tensor]
                and obj.device.type == "cpu"
            ):
                # Only HtoD copy command can send cpu Tensors over
                return
            raise RuntimeError(
                f"Trying to send invalid object through queue: {type(obj)}"
            )

    tree_map(check, args)


def prepare_for_sending(args: Any, kwargs: Any) -> Any:
    """Prepare arguments for sending to remote device.

    Converts torch.Tensor objects to RemoteTensorMeta for efficient
    transmission. Remote tensors are converted to metadata only,
    while CPU tensors include full tensor data.

    Args:
        args: Positional arguments to prepare
        kwargs: Keyword arguments to prepare

    Returns:
        Converted arguments ready for remote transmission

    Raises:
        RuntimeError: If unsupported object types are found
    """
    def convert(obj: Any) -> Any:
        if type(obj) not in VALID_QUEUE_TYPES_IN:
            raise RuntimeError(
                f"Cannot send object of type {type(obj)} over remote device pipe."
            )

        if isinstance(obj, torch.Tensor):
            if obj.device.type == "remote":
                # For remote tensors, send storage_id + metadata instead of full tensor data
                return RemoteTensorMeta.from_remote_tensor(obj)
            else:
                # For non-remote tensors, use original behavior
                return RemoteTensorMeta.from_tensor(obj)
        else:
            return obj

    return tree_map(convert, (args, kwargs))


def receive_after_sending(allocator: Any, args: Any, kwargs: Any) -> Any:
    """Process arguments received from remote device.

    Converts RemoteTensorMeta objects back to torch.Tensor using
    the provided allocator. Handles reconstruction of tensor objects
    from metadata.

    Args:
        allocator: Allocator to create tensors from metadata
        args: Received positional arguments
        kwargs: Received keyword arguments

    Returns:
        Reconstructed arguments with tensor objects

    Raises:
        RuntimeError: If invalid object types are received
    """
    def convert(obj: Any) -> Any:
        if type(obj) not in VALID_QUEUE_TYPES_OUT:
            raise RuntimeError(
                f"Received invalid object of type {type(obj)} over remote device pipe."
            )

        if isinstance(obj, RemoteTensorMeta):
            return allocator.tensor_from_meta(obj)
        else:
            return obj

    return tree_map(convert, (args, kwargs))


