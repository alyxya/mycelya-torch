# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Dict, Tuple

import torch

from .._logging import get_logger
from .meta import _execute_with_dynamic_outputs, _execute_with_static_outputs
from .utils import _has_static_output_shape, _validate_cross_device_operation

log = get_logger(__name__)


def _execute_aten_operation(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    remote_device: torch.device,
) -> Any:
    """Execute operation on remote device - simplified from complex strategy pattern."""

    op_name = str(op)

    # Check for unsupported operations before meta execution
    if op_name == "aten.repeat_interleave":
        raise RuntimeError(
            "repeat_interleave is not supported on remote devices due to a PyTorch bug where tensor repeats "
            "are incorrectly dispatched to single-argument overload. "
            "Use tensor.repeat() or other alternatives instead."
        )

    # Note: aten::view is now handled directly in C++ (view_mycelya) and won't reach this fallback

    # Step 1: Check if operation requires dynamic output handling
    has_static_output = _has_static_output_shape(op_name, args, kwargs)

    if has_static_output:
        # Standard path: Use meta tensors for shape inference
        return _execute_with_static_outputs(op, args, kwargs, remote_device, op_name)
    else:
        # Dynamic path: Create placeholder outputs and get metadata from remote execution
        return _execute_with_dynamic_outputs(op, args, kwargs, remote_device, op_name)


def _remote_kernel_fallback(
    op: torch._ops.OpOverload, *args: Any, **kwargs: Any
) -> Any:
    """Execute PyTorch operations on remote devices using simple dispatch logic."""
    op_name = str(op)

    # Validate cross-device operations upfront and get the remote device
    remote_device = _validate_cross_device_operation(op_name, args, kwargs)

    # All operations go through standard aten dispatch
    # View operations are now handled by PyTorch's built-in mechanisms
    return _execute_aten_operation(op, args, kwargs, remote_device)
