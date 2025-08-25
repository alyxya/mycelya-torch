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
    """Validate that all tensors are remote and on the same machine. Returns the remote device."""
    remote_machine_id = None
    remote_device = None

    def check_tensor_device(obj):
        nonlocal remote_machine_id, remote_device
        if isinstance(obj, torch.Tensor):
            if obj.device.type != "mycelya":
                raise RuntimeError(
                    f'Remote kernel fallback called for operation "{op_name}" with non-remote tensor '
                    f'on device "{obj.device}".'
                )

            # Get machine info through storage, not device index
            from .._utils import get_storage_id
            from .._orchestrator import orchestrator
            
            storage_id = get_storage_id(obj)
            machine_id, _, _ = orchestrator.storage.get_remote_device_info(storage_id)

            if remote_machine_id is None:
                remote_machine_id = machine_id
                remote_device = obj.device
            elif remote_machine_id != machine_id:
                raise RuntimeError(
                    f'Cannot perform operation "{op_name}" between different machines'
                )
        return obj

    # Use map_args_kwargs to check all tensors
    from .._utils import map_args_kwargs
    map_args_kwargs(check_tensor_device, args, kwargs)

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
