# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from .._orchestrator import orchestrator


def copy_from_device(from_: torch.Tensor) -> torch.Tensor:
    """Copy data from remote tensor to CPU tensor using tensor-based execution"""
    if from_.device.type != "mycelya":
        raise ValueError("copy_from_device requires a remote tensor")

    # Use orchestrator's new async copy method
    cpu_future = orchestrator.copy_tensor_to_cpu(from_)
    return cpu_future.result()


def copy_from_host_to_device(from_: torch.Tensor, to_: torch.Tensor) -> torch.Tensor:
    """Copy data from CPU tensor to remote tensor using tensor-based execution"""
    if to_.device.type != "mycelya":
        raise ValueError("copy_from_host_to_device requires a remote target tensor")
    if from_.device.type != "cpu":
        raise ValueError("copy_from_host_to_device requires a CPU source tensor")

    # Ensure tensor exists and update with data in one operation
    orchestrator.ensure_tensor_exists_and_update(to_, from_)
    return to_


def _copy_from(
    from_: torch.Tensor, to_: torch.Tensor, non_blocking: bool = False
) -> torch.Tensor:
    """Copy data from one tensor to another, handling remote device transfers.

    This function implements the core copy operation for remote tensors,
    supporting CPU↔remote transfers and same-device remote copies.
    Cross-device remote transfers and non-remote device copies are blocked.

    Args:
        from_: Source tensor to copy from
        to_: Target tensor to copy to
        non_blocking: Whether to perform the copy asynchronously (currently ignored)

    Returns:
        Target tensor with copied data

    Raises:
        RuntimeError: If attempting unsupported copy operations
    """
    # Only support CPU ↔ remote transfers

    if from_.device.type == "mycelya" and to_.device.type == "cpu":
        # Remote to CPU - supported
        host_mem = copy_from_device(from_)
        result = to_.copy_(host_mem)
    elif from_.device.type == "cpu" and to_.device.type == "mycelya":
        # CPU to remote - supported
        result = copy_from_host_to_device(from_, to_)
    elif from_.device.type == "mycelya" and to_.device.type == "mycelya":
        # Remote to remote transfers
        if from_.device.index == to_.device.index:
            # Same remote device - allowed (needed for gradients and internal operations)
            # We can't import _remote_kernel_fallback here due to circular imports,
            # so we'll use the orchestrator directly for same-device copies
            from .utils import args_to_tensors_with_ids_and_mask

            processed_args, processed_kwargs, input_tensors, tensor_mask = (
                args_to_tensors_with_ids_and_mask((to_, from_), {})
            )

            orchestrator.execute_aten_operation(
                "aten::copy_.default",
                input_tensors,
                [to_],  # output tensors
                processed_args,
                processed_kwargs,
                tensor_mask,
            )
            result = to_
        else:
            # Different remote devices - not supported
            raise RuntimeError(
                f"Cross-device remote transfers are not supported. "
                f"Source device index: {from_.device.index}, "
                f"Target device index: {to_.device.index}. "
                f"Only CPU↔remote and same-device transfers are allowed. Use CPU as intermediate."
            )
    else:
        # All other cases (non-remote device copies) - blocked
        raise RuntimeError(
            f"Copy operation from {from_.device.type} to {to_.device.type} is not supported. "
            f"Only CPU↔remote transfers are allowed."
        )

    return result
