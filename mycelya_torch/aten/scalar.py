# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


def _local_scalar_dense(self: torch.Tensor):
    """Custom implementation of _local_scalar_dense for remote tensors."""
    # Check that tensor is scalar (replicate PyTorch's exact behavior)
    if self.numel() != 1:
        raise RuntimeError(
            f"a Tensor with {self.numel()} elements cannot be converted to Scalar"
        )

    # Get scalar value from remote device
    # Import orchestrator lazily to avoid circular imports
    from .._orchestrator import orchestrator

    # Use orchestrator's new async copy method
    cpu_future = orchestrator.copy_tensor_to_cpu(self)
    cpu_tensor = cpu_future.result()

    # Call item() on the CPU tensor to get the Python scalar
    return cpu_tensor.item()


def _equal(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Custom implementation of torch.equal for remote tensors."""

    # Both tensors should be remote (validated by caller)
    # Check basic compatibility first
    if self.shape != other.shape:
        return False
    if self.dtype != other.dtype:
        return False

    # Perform element-wise comparison on remote device, then reduce to scalar

    # Do element-wise equality comparison on remote device
    eq_tensor = torch.eq(self, other)

    # Reduce to single boolean using torch.all() on remote device
    all_equal_tensor = torch.all(eq_tensor)

    # Get scalar result using .item() which will copy single value to CPU
    result = all_equal_tensor.item()

    return result
