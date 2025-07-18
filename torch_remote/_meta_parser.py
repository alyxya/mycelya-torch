# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from pprint import pformat
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils._pytree import tree_map, tree_map_only


class RemoteTensorMeta:
    def __init__(self, tensor: Optional[torch.Tensor] = None, checked: bool = True, data_ptr: Optional[int] = None, size: Optional[torch.Size] = None, stride: Optional[Tuple[int, ...]] = None, 
                 storage_offset: Optional[int] = None, dtype: Optional[torch.dtype] = None, nelem_in_bytes: Optional[int] = None, storage_id: Optional[int] = None) -> None:
        """
        Create RemoteTensorMeta from either a tensor or explicit metadata.
        
        Args:
            tensor: Extract metadata from this tensor (original behavior)
            checked: Whether to check tensor is on remote device
            data_ptr: Explicit tensor ID (for view operations)
            size: Explicit size (for view operations)
            stride: Explicit stride (for view operations)
            storage_offset: Explicit storage offset (for view operations)
            dtype: Explicit dtype (for view operations)
            nelem_in_bytes: Explicit element count in bytes (for view operations)
            storage_id: Storage ID for shared storage tracking (for view operations)
        """
        if tensor is not None:
            # Original behavior - extract from tensor
            if checked and not tensor.device.type == "remote":
                raise RuntimeError(
                    "Creating RemoteTensorMeta is only for Tensors on remote device"
                )
            self.data_ptr = tensor.untyped_storage().data_ptr()
            self.size = tensor.size()
            self.stride = tensor.stride()
            self.storage_offset = tensor.storage_offset()
            self.dtype = tensor.dtype
            self.nelem_in_bytes = tensor.nelement() * tensor.element_size()
            self.storage_id = storage_id  # Optional storage ID for view tracking
        else:
            # Explicit metadata - for view operations
            if any(param is None for param in [data_ptr, size, stride, storage_offset, dtype, nelem_in_bytes]):
                raise ValueError("When not providing tensor, all metadata parameters must be specified")
            self.data_ptr = data_ptr  # type: ignore
            self.size = size  # type: ignore
            self.stride = stride  # type: ignore
            self.storage_offset = storage_offset  # type: ignore
            self.dtype = dtype  # type: ignore
            self.nelem_in_bytes = nelem_in_bytes  # type: ignore
            self.storage_id = storage_id

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
            return str(RemoteTensorMeta(obj, checked=False))
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
                storage_id = obj.untyped_storage().data_ptr()
                return RemoteTensorMeta(
                    data_ptr=storage_id,
                    size=obj.size(),
                    stride=obj.stride(), 
                    storage_offset=obj.storage_offset(),
                    dtype=obj.dtype,
                    nelem_in_bytes=obj.numel() * obj.element_size(),
                    storage_id=storage_id
                )
            else:
                # For non-remote tensors, use original behavior
                return RemoteTensorMeta(obj)
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


def to_device_no_copy(device: Union[torch.device, str], args: Any, kwargs: Any) -> Any:
    """Create empty tensors on target device without copying data.
    
    Creates tensor placeholders with the same shape and properties
    as input tensors but on the specified device. Useful for setting
    up tensor structures before actual data transfer.
    
    Args:
        device: Target device for new tensors
        args: Input arguments containing tensors
        kwargs: Input keyword arguments containing tensors
        
    Returns:
        Arguments with empty tensors on target device
    """
    def safe_to(t: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(t, device=device)

    return tree_map_only(torch.Tensor, safe_to, (args, kwargs))