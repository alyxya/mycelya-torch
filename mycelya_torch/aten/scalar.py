# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from .._logging import get_logger
from .._utils import dtype_to_str, get_tensor_id
from .utils import _get_orchestrator

log = get_logger(__name__)


def _local_scalar_dense(self: torch.Tensor):
    """Custom implementation of _local_scalar_dense for remote tensors."""
    # Check that tensor is scalar (replicate PyTorch's exact behavior)
    if self.numel() != 1:
        raise RuntimeError(
            f"a Tensor with {self.numel()} elements cannot be converted to Scalar"
        )

    # Get scalar value from remote device
    log.info(
        "🔢 _local_scalar_dense operation: retrieving scalar value from remote device"
    )

    # Get tensor ID and tensor data using tensor-based approach
    tensor_id = get_tensor_id(self)

    # Get tensor data for this scalar using orchestrator
    orchestrator = _get_orchestrator()

    cpu_tensor = orchestrator.get_tensor_by_id(
        tensor_id,
        shape=list(self.shape),
        stride=list(self.stride()),
        storage_offset=self.storage_offset(),
        dtype=dtype_to_str(self.dtype),
    )

    # Call item() on the CPU tensor to get the Python scalar
    return cpu_tensor.item()


def _equal(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Custom implementation of torch.equal for remote tensors."""
    log.info("⚖️ torch.equal operation: comparing tensors on remote device")

    # Both tensors should be remote (validated by caller)
    # Check basic compatibility first
    if self.shape != other.shape:
        return False
    if self.dtype != other.dtype:
        return False

    # Perform element-wise comparison on remote device, then reduce to scalar
    log.info("🔄 Executing element-wise comparison on remote device")

    # Do element-wise equality comparison on remote device
    eq_tensor = torch.eq(self, other)

    # Reduce to single boolean using torch.all() on remote device
    all_equal_tensor = torch.all(eq_tensor)

    # Get scalar result using .item() which will copy single value to CPU
    result = all_equal_tensor.item()

    log.info(f"✅ torch.equal completed on remote device: {result}")
    return result
