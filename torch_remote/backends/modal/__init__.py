# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal backend implementation for torch_remote.

This backend provides remote execution capabilities using Modal's cloud GPUs.
Supports: T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200.
"""

from typing import Optional, Any

import torch
import torch_remote._C

def device_count() -> int:
    """Return the number of remote devices available via Modal."""
    return 1

def current_device() -> int:
    """Return the current remote device index."""
    return 0

def set_device(device: int) -> None:
    """Set the current remote device."""
    if device != 0:
        raise RuntimeError(f"Remote device {device} not available. Only device 0 is supported.")

def is_available() -> bool:
    """Return True if remote device is available via Modal."""
    return True

def get_device_name(device: Optional[int] = None) -> str:
    """Get the name of a remote device."""
    return "Modal GPU"

def get_device_properties(device: Optional[int] = None) -> Any:
    """Get properties of a remote device."""
    class Properties:
        def __init__(self) -> None:
            self.name = "Modal GPU"
            self.total_memory = 0  # Remote memory not directly accessible
            self.major = 8  # A100 compute capability
            self.minor = 0

    return Properties()

def synchronize() -> None:
    """Synchronize remote device operations."""
    pass

def get_rng_state() -> torch.Tensor:
    """Get random number generator state."""
    return torch.get_rng_state()

def set_rng_state(new_state: torch.Tensor) -> None:
    """Set random number generator state."""
    torch.set_rng_state(new_state)

def manual_seed(seed: int) -> None:
    """Set manual seed for remote device."""
    torch.manual_seed(seed)

# Tensor types for remote device (all map to base Tensor for now)
FloatTensor = torch.Tensor
DoubleTensor = torch.Tensor
HalfTensor = torch.Tensor
BFloat16Tensor = torch.Tensor
ByteTensor = torch.Tensor
CharTensor = torch.Tensor
ShortTensor = torch.Tensor
IntTensor = torch.Tensor
LongTensor = torch.Tensor
BoolTensor = torch.Tensor
