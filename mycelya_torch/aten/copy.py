# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from .._orchestrator import orchestrator


def copy_from_device(from_: torch.Tensor) -> torch.Tensor:
    """Copy data from mycelya tensor to CPU tensor using tensor-based execution"""
    if from_.device.type != "mycelya":
        raise ValueError("copy_from_device requires a mycelya tensor")

    # Use orchestrator's synchronous copy method
    return orchestrator.copy_tensor_to_cpu(from_)


def copy_from_host_to_device(from_: torch.Tensor, to_: torch.Tensor) -> torch.Tensor:
    """Copy data from CPU tensor to mycelya tensor using tensor-based execution"""
    if to_.device.type != "mycelya":
        raise ValueError("copy_from_host_to_device requires a mycelya target tensor")
    if from_.device.type != "cpu":
        raise ValueError("copy_from_host_to_device requires a CPU source tensor")

    # Ensure tensor exists and update with data in one operation
    orchestrator.update_tensor(to_, from_)
    return to_


def _copy_from(
    from_: torch.Tensor, to_: torch.Tensor, non_blocking: bool = False
) -> torch.Tensor:
    """Copy data from one tensor to another, handling mycelya device transfers.

    This function implements the core copy operation for mycelya tensors,
    supporting CPU↔mycelya transfers and same-machine mycelya copies.
    Cross-machine mycelya transfers and non-mycelya device copies are blocked.

    Args:
        from_: Source tensor to copy from
        to_: Target tensor to copy to
        non_blocking: Whether to perform the copy asynchronously (currently ignored)

    Returns:
        Target tensor with copied data

    Raises:
        RuntimeError: If attempting unsupported copy operations
    """
    # Support CPU ↔ mycelya transfers and same-machine mycelya copies

    if from_.device.type == "mycelya" and to_.device.type == "cpu":
        # Mycelya to CPU - supported
        host_mem = copy_from_device(from_)
        result = to_.copy_(host_mem)
    elif from_.device.type == "cpu" and to_.device.type == "mycelya":
        # CPU to mycelya - supported
        result = copy_from_host_to_device(from_, to_)
    elif from_.device.type == "mycelya" and to_.device.type == "mycelya":
        # Same-machine mycelya transfers - use orchestrator for validation and execution
        orchestrator.copy_tensor(from_, to_)
        result = to_
    else:
        # All other cases (non-mycelya device copies) - blocked
        raise RuntimeError(
            f"Copy operation from {from_.device.type} to {to_.device.type} is not supported. "
            f"Only CPU↔mycelya transfers and same-machine mycelya copies are allowed."
        )

    return result
