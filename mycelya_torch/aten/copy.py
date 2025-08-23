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
    orchestrator.update_tensor(to_, from_)
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
        # Remote to remote transfers - check if they're on the same machine
        from .._utils import get_storage_id

        from_storage_id = get_storage_id(from_)
        to_storage_id = get_storage_id(to_)

        # Get machine info for both tensors
        from_machine_id, from_device_type, from_device_index = (
            orchestrator.storage.get_remote_device_info(from_storage_id)
        )
        to_machine_id, to_device_type, to_device_index = (
            orchestrator.storage.get_remote_device_info(to_storage_id)
        )

        if from_machine_id == to_machine_id:
            # Same remote machine - use new copy operation
            client = orchestrator._clients[from_machine_id]
            client.copy_tensor(from_, to_)
            result = to_
        else:
            # Different remote machines - not supported
            raise RuntimeError(
                f"Cross-machine remote transfers are not supported. "
                f"Source machine: {from_machine_id}, Target machine: {to_machine_id}. "
                f"Only CPU↔remote and same-machine transfers are allowed. Use CPU as intermediate."
            )
    else:
        # All other cases (non-remote device copies) - blocked
        raise RuntimeError(
            f"Copy operation from {from_.device.type} to {to_.device.type} is not supported. "
            f"Only CPU↔remote transfers are allowed."
        )

    return result
