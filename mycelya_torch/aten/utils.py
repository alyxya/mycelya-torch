# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Dict, List, Tuple

import torch

from .._logging import get_logger
from .._utils import get_tensor_id

log = get_logger(__name__)


def _validate_cross_device_operation(
    op_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> torch.device:
    """Validate that all tensors are remote and on the same device. Returns the remote device."""
    remote_device = None

    def check_tensor_device(obj):
        nonlocal remote_device
        if isinstance(obj, torch.Tensor):
            if obj.device.type != "mycelya":
                raise RuntimeError(
                    f'Remote kernel fallback called for operation "{op_name}" with non-remote tensor '
                    f'on device "{obj.device}".'
                )

            if remote_device is None:
                remote_device = obj.device
            elif remote_device != obj.device:
                raise RuntimeError(
                    f'Cannot perform operation "{op_name}" between tensors on different remote devices '
                    f"({remote_device} and {obj.device}). "
                    f"Transfer tensors to the same device first."
                )
        return obj

    # Iterate over args
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                check_tensor_device(item)
        else:
            check_tensor_device(arg)

    # Iterate over kwargs values
    for value in kwargs.values():
        if isinstance(value, (list, tuple)):
            for item in value:
                check_tensor_device(item)
        else:
            check_tensor_device(value)

    if remote_device is None:
        raise RuntimeError(f'No remote tensors found for operation "{op_name}"')

    return remote_device


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


def _has_static_output_shape(
    op_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> bool:
    """Determine if operation has predictable output shape for meta tensor inference."""

    # Always dynamic operations (output shape depends on data)
    ALWAYS_DYNAMIC = {
        "aten::masked_select",
        "aten::nonzero",
        "aten::unique",
        "aten::_unique2",
    }
    if op_name in ALWAYS_DYNAMIC:
        return False

    # Special case: aten::index can be static (tensor indexing) or dynamic (boolean indexing)
    if op_name == "aten::index":
        if len(args) >= 2 and isinstance(args[1], (tuple, list)):
            # Boolean indexing is dynamic, tensor indexing is static
            return not any(
                isinstance(idx, torch.Tensor) and idx.dtype == torch.bool
                for idx in args[1]
                if idx is not None
            )

    # Add more conditional operations here as needed:
    # if op_name == "aten::where":
    #     return len(args) != 1  # 1-arg form is dynamic, 3-arg form is static

    return True
