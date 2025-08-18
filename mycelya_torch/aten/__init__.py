# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Dict, Tuple

import torch

from .._logging import get_logger
from .copy import _copy_from
from .meta import _execute_with_dynamic_outputs, _execute_with_static_outputs
from .scalar import _equal, _local_scalar_dense
from .utils import _has_static_output_shape, _validate_cross_device_operation
from .views import _execute_view_operation, _set_source_tensor

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
    schema = op._schema
    is_view_op = (
        schema.returns
        and len(schema.returns) > 0
        and hasattr(schema.returns[0], "alias_info")
        and schema.returns[0].alias_info is not None
        and not schema.returns[0].alias_info.is_write
    )

    if is_view_op:
        return _execute_view_operation(op, *args, **kwargs)
    else:
        return _execute_aten_operation(op, args, kwargs, remote_device)


# Register all the operations
_remote_lib = torch.library.Library("_", "IMPL")
_remote_lib.fallback(_remote_kernel_fallback, dispatch_key="PrivateUse1")

_remote_lib_aten = torch.library.Library("aten", "IMPL")
_remote_lib_aten.impl("_copy_from", _copy_from, dispatch_key="PrivateUse1")
_remote_lib_aten.impl(
    "set_.source_Tensor", _set_source_tensor, dispatch_key="PrivateUse1"
)
_remote_lib_aten.impl(
    "_local_scalar_dense", _local_scalar_dense, dispatch_key="PrivateUse1"
)
_remote_lib_aten.impl("equal", _equal, dispatch_key="PrivateUse1")
