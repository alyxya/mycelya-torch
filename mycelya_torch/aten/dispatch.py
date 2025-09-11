# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Dict, List, Tuple

import torch

from .._device import device_manager
from .._logging import get_logger
from .._orchestrator import orchestrator
from .._utils import (
    create_mycelya_tensor_from_metadata,
    get_storage_id,
    map_args_kwargs,
)

log = get_logger(__name__)


def _create_meta_tensor_from_mycelya(
    mycelya_tensor: torch.Tensor, meta_storage_cache: Dict[int, torch.UntypedStorage]
) -> torch.Tensor:
    """Create a meta tensor that closely mirrors a mycelya tensor, including storage sharing."""
    storage_id = get_storage_id(mycelya_tensor)

    # Create or reuse meta storage to preserve storage sharing relationships
    if storage_id not in meta_storage_cache:
        # Create meta storage with same nbytes as the original
        nbytes = mycelya_tensor.untyped_storage().nbytes()
        meta_storage_cache[storage_id] = torch.UntypedStorage(nbytes, device="meta")

    meta_storage = meta_storage_cache[storage_id]

    # Create meta tensor with same metadata as mycelya tensor
    meta_tensor = torch.empty(0, dtype=mycelya_tensor.dtype, device="meta")
    meta_tensor.set_(
        meta_storage,
        mycelya_tensor.storage_offset(),
        mycelya_tensor.shape,
        mycelya_tensor.stride(),
    )

    return meta_tensor


def _execute_meta_operation(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    device_container: List[torch.device],
) -> tuple[Any, Dict]:
    """Execute operation on meta tensors for shape inference and device resolution."""
    # Map from meta storage to original tensor for preserving storage relationships
    original_tensors = {}  # meta_storage -> original_tensor
    meta_storage_cache = {}  # storage_id -> meta storage for preserving sharing

    if "device" in kwargs:
        device_container.append(kwargs["device"])

    def to_meta_tensor(obj):
        # Check tensor device if container still empty
        if not device_container and isinstance(obj, torch.Tensor):
            if obj.device.type == "mycelya":
                # Equivalent to device_container.append(obj.device) except when virtual devices are a thing
                device_container.append(
                    device_manager.get_mycelya_device(
                        *orchestrator.storage.get_remote_device_info(
                            get_storage_id(obj)
                        )
                    )
                )

        # Convert tensor to meta for shape inference
        if isinstance(obj, torch.Tensor):
            meta_tensor = _create_meta_tensor_from_mycelya(obj, meta_storage_cache)
            original_tensors[meta_tensor.untyped_storage()] = obj
            return meta_tensor
        
        # Convert device arguments to meta device
        if isinstance(obj, torch.device):
            return torch.device("meta")

        return obj

    meta_args, meta_kwargs = map_args_kwargs(to_meta_tensor, args, kwargs)
    meta_result = op(*meta_args, **meta_kwargs)

    return meta_result, original_tensors


def _create_output_tensors(
    meta_outputs: List, original_tensors: Dict, remote_device: torch.device
) -> List[torch.Tensor]:
    """Create output tensors based on meta execution results with proper alias detection."""
    output_tensors = []

    for meta_output in meta_outputs:
        meta_storage = meta_output.untyped_storage()

        if meta_storage in original_tensors:
            # This output uses storage from an existing tensor
            original_tensor = original_tensors[meta_storage]

            # Resize if the original tensor is uninitialized (has 0 elements)
            if original_tensor.numel() == 0:
                original_tensor.resize_(meta_output.shape)

            tensor = original_tensor.as_strided(
                meta_output.shape,
                meta_output.stride(),
                meta_output.storage_offset(),
            )
            output_tensors.append(tensor)
        else:
            # Create new tensor with new storage
            tensor = torch.empty(
                meta_output.shape, dtype=meta_output.dtype, device=remote_device
            )
            # Record the storage mapping for future outputs that might alias
            original_tensors[meta_storage] = tensor
            output_tensors.append(tensor)

    return output_tensors


def _execute_with_static_outputs(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    remote_device: torch.device,
    meta_result: Any,
    original_tensors: Dict,
) -> Any:
    """Execute operation using meta tensors for shape inference."""
    # Normalize meta_result to list
    meta_outputs = (
        [meta_result]
        if isinstance(meta_result, torch.Tensor)
        else list(meta_result)
        if isinstance(meta_result, (tuple, list))
        else []
    )

    # Create output tensors
    output_tensors = (
        _create_output_tensors(meta_outputs, original_tensors, remote_device)
        if meta_outputs
        else []
    )

    orchestrator.execute_aten_operation(str(op), args, kwargs, output_tensors)

    return (
        tuple(output_tensors)
        if len(output_tensors) > 1
        else output_tensors[0]
        if output_tensors
        else None
    )


def _execute_with_dynamic_outputs(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    remote_device: torch.device,
) -> Any:
    """Execute operation with dynamic output shapes."""
    # Execute remotely and get metadata
    result = orchestrator.execute_aten_operation(
        str(op), args, kwargs, output_tensors=None
    )

    # TODO: Handle operations that return tensors aliasing to existing storage
    # This is a niche edge case that makes the code messy - should be implemented
    # in a cleaner way in future design

    # Create output tensors from metadata
    output_tensors = []
    temp_keys = []
    for metadata in result:
        output_tensor = create_mycelya_tensor_from_metadata(metadata, remote_device)
        output_tensors.append(output_tensor)
        temp_keys.append(metadata["temp_key"])

    # Link all tensors to remote data
    orchestrator.link_tensors(output_tensors, temp_keys)

    return output_tensors[0] if len(output_tensors) == 1 else tuple(output_tensors)


def _remote_kernel_fallback(
    op: torch._ops.OpOverload, *args: Any, **kwargs: Any
) -> Any:
    """Execute PyTorch operations on remote devices using simple dispatch logic."""

    device_container = []

    # Try meta tensor execution first, fall back to dynamic if not implemented
    try:
        meta_result, original_tensors = _execute_meta_operation(
            op, args, kwargs, device_container
        )
        return _execute_with_static_outputs(
            op, args, kwargs, device_container[0], meta_result, original_tensors
        )
    except NotImplementedError:
        return _execute_with_dynamic_outputs(op, args, kwargs, device_container[0])
