# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Dict, Tuple

import torch

from .._logging import get_logger
from .meta import _execute_with_dynamic_outputs, _execute_with_static_outputs
from .utils import _has_static_output_shape, _validate_cross_device_operation
from .views import _execute_view_operation

log = get_logger(__name__)


def _execute_aten_operation(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    remote_device: torch.device,
) -> Any:
    """Execute operation on remote device - simplified from complex strategy pattern."""

    op_name = op.overloadpacket._qualified_op_name

    # Check for unsupported operations before meta execution
    if op_name == "aten::repeat_interleave":
        raise RuntimeError(
            "repeat_interleave is not supported on remote devices due to a PyTorch bug where tensor repeats "
            "are incorrectly dispatched to single-argument overload. "
            "Use tensor.repeat() or other alternatives instead."
        )

    # Note: aten::view is now handled directly in C++ (view_mycelya) and won't reach this fallback

    # Step 1: Check if operation requires dynamic output handling
    has_static_output = _has_static_output_shape(op_name, args, kwargs)
    log.debug(f"🔍 Operation {op_name} has static output shape: {has_static_output}")

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
    op_name = op.overloadpacket._qualified_op_name

    # Validate cross-device operations upfront and get the remote device
    remote_device = _validate_cross_device_operation(op_name, args, kwargs)

    # Check if operation is a view operation using schema alias information
    # View operations alias their input for reading (is_write=False)
    # In-place/out operations also have alias_info but with is_write=True
    # Note: aten::view is excluded as it's handled directly in C++ (view_mycelya)
    schema = op._schema
    is_view_op = (
        schema.returns
        and len(schema.returns) > 0
        and hasattr(schema.returns[0], "alias_info")
        and schema.returns[0].alias_info is not None
        and not schema.returns[0].alias_info.is_write
        and op_name != "aten::view"  # Exclude view - handled in C++
    )

    if is_view_op:
        return _execute_view_operation(op, *args, **kwargs)
    else:
        return _execute_aten_operation(op, args, kwargs, remote_device)
