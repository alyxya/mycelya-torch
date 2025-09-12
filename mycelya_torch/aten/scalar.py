# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from .._orchestrator import orchestrator


def _local_scalar_dense(self: torch.Tensor):
    """Custom implementation of _local_scalar_dense for mycelya tensors."""
    # Check that tensor is scalar (replicate PyTorch's exact behavior)
    if self.numel() != 1:
        raise RuntimeError(
            f"a Tensor with {self.numel()} elements cannot be converted to Scalar"
        )

    # Get scalar value from mycelya device

    # Use orchestrator's synchronous copy method
    cpu_tensor = orchestrator.copy_tensor_to_cpu(self)

    # Call item() on the CPU tensor to get the Python scalar
    return cpu_tensor.item()


def _equal(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Custom implementation of torch.equal for mycelya tensors."""

    # Both tensors should be mycelya (validated by caller)
    # Check basic compatibility first
    if self.shape != other.shape:
        return False
    if self.dtype != other.dtype:
        return False

    # Perform element-wise comparison on mycelya device, then reduce to scalar

    # Do element-wise equality comparison on mycelya device
    eq_tensor = torch.eq(self, other)

    # Reduce to single boolean using torch.all() on mycelya device
    all_equal_tensor = torch.all(eq_tensor)

    # Get scalar result using .item() which will copy single value to CPU
    result = all_equal_tensor.item()

    return result
