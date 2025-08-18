# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import torch

from .._logging import get_logger
from .meta import _execute_meta_operation

log = get_logger(__name__)


def _execute_view_operation(
    op: torch._ops.OpOverload, *args: Any, **kwargs: Any
) -> torch.Tensor:
    """Handle view operations as local-only operations that share storage."""
    op_name = op.overloadpacket._qualified_op_name

    # Get the base tensor (first argument for most view operations)
    base_tensor = args[0]

    # Execute on meta tensors for shape inference to create the view locally
    meta_result, _ = _execute_meta_operation(op, args, kwargs, track_originals=False)

    # Create the local view tensor using PyTorch's native as_strided (shares storage)
    view_tensor = torch.as_strided(
        base_tensor,
        meta_result.shape,
        meta_result.stride(),
        meta_result.storage_offset(),
    )

    log.debug(f"View operation {op_name} executed locally only (no remote propagation)")
    return view_tensor


def _set_source_tensor(ten1: torch.Tensor, ten2: torch.Tensor) -> torch.Tensor:
    """Set one tensor to point to another tensor's storage.

    This creates a view relationship where ten1 shares ten2's storage,
    shape, stride, and offset. Used for tensor aliasing operations.

    Args:
        ten1: Tensor to modify
        ten2: Source tensor to point to

    Returns:
        Modified tensor ten1 pointing to ten2's data
    """
    return torch.ops.aten.set_.source_Storage_storage_offset(
        ten1,
        ten2.untyped_storage(),
        ten2.storage_offset(),
        ten2.shape,
        ten2.stride(),
    )
