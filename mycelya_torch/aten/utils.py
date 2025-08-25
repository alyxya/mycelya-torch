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
                from .._device import device_manager

                try:
                    remote_info = device_manager.get_remote_device_info(
                        remote_device.index
                    )
                    current_info = device_manager.get_remote_device_info(
                        obj.device.index
                    )

                    if remote_info[0] != current_info[0]:  # machine_id
                        raise RuntimeError(
                            f'Cannot perform operation "{op_name}" between different machines'
                        )
                    elif remote_info[1:] != current_info[1:]:  # type and index
                        raise RuntimeError(
                            f'Cannot perform operation "{op_name}" between different devices'
                        )
                except Exception:
                    pass
                raise RuntimeError(
                    f'Cannot perform operation "{op_name}" between tensors on different devices '
                    f"({remote_device} and {obj.device})"
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



def _has_static_output_shape(
    op_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> bool:
    """Determine if operation has predictable output shape for meta tensor inference."""

    # Always dynamic operations (output shape depends on data)
    ALWAYS_DYNAMIC = {
        "aten.masked_select",
        "aten.masked_select.default",
        "aten.nonzero",
        "aten.nonzero.default",
        "aten.unique",
        "aten.unique.default",
        "aten._unique2",
        "aten._unique2.default",
    }
    if op_name in ALWAYS_DYNAMIC:
        return False

    # Special case: aten.index can be static (tensor indexing) or dynamic (boolean indexing)
    if op_name == "aten.index":
        if len(args) >= 2 and isinstance(args[1], (tuple, list)):
            # Boolean indexing is dynamic, tensor indexing is static
            return not any(
                isinstance(idx, torch.Tensor) and idx.dtype == torch.bool
                for idx in args[1]
                if idx is not None
            )

    # Add more conditional operations here as needed:
    # if op_name == "aten.where":
    #     return len(args) != 1  # 1-arg form is dynamic, 3-arg form is static

    return True
