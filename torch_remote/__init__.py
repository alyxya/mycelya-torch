# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import types
from typing import Any, Optional, Union

import torch

# Direct driver access for C++ via factory pattern
from ._device_daemon import driver

# Factory pattern for C++ method access with caching
_IMPL_REGISTRY = {}


def impl_factory(name: str):
    """Factory function that returns cached method implementations.

    This follows the pytorch-openreg-2 pattern for cleaner C++ integration.

    Args:
        name: Method name to get implementation for

    Returns:
        Callable that executes the named method
    """
    if name in _IMPL_REGISTRY:
        return _IMPL_REGISTRY[name]

    def _method_impl(*args, **kwargs):
        return driver.exec(name, *args, **kwargs)

    _IMPL_REGISTRY[name] = _method_impl
    return _method_impl


# Load the C++ Module
import torch_remote._C  # isort:skip # type: ignore[import] # noqa: F401


def _create_module() -> types.ModuleType:
    """Create the remote device module for PyTorch backend registration.

    This function creates a module that implements the PyTorch accelerator
    backend interface for remote devices. It provides device context
    management, RNG state handling, and other core device operations.

    Returns:
        Module implementing the remote device backend interface
    """
    module = types.ModuleType("_RemoteMod")

    class device:
        r"""Context-manager that changes the selected device.

        Args:
            device (torch.device or int): device index to select. It's a no-op if
                this argument is a negative integer or ``None``.
        """

        def __init__(self, device: Union[torch.device, int, str]) -> None:
            self.idx = torch.accelerator._get_device_index(device, optional=True)
            self.prev_idx = -1

        def __enter__(self) -> None:
            self.prev_idx = driver.exec("exchange_device", self.idx)

        def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
            self.idx = driver.exec("unchecked_set_device", self.prev_idx)

    def device_count() -> int:
        """Get the number of available remote devices.

        Returns:
            Number of remote devices available
        """
        return driver.exec("device_count")

    def is_available() -> bool:
        """Check if remote device support is available.

        Returns:
            True if remote devices are available, False otherwise
        """
        return True

    def current_device() -> int:
        """Get the current remote device index.

        Returns:
            Index of the currently selected remote device
        """
        return torch.accelerator.current_device_index()

    def get_rng_state(device: Union[str, int, torch.device] = "remote") -> torch.Tensor:
        """Get the random number generator state for a remote device.

        Args:
            device: Remote device to get RNG state from

        Returns:
            Tensor containing the RNG state
        """
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("remote", device)
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = torch_remote._C._get_default_generator(idx)
        return default_generator.get_state()

    def set_rng_state(
        new_state: torch.Tensor, device: Union[str, int, torch.device] = "remote"
    ) -> None:
        """Set the random number generator state for a remote device.

        Args:
            new_state: Tensor containing the new RNG state
            device: Remote device to set RNG state for
        """
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("remote", device)
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = torch_remote._C._get_default_generator(idx)
        default_generator.set_state(new_state)

    def initial_seed() -> int:
        _lazy_init()
        idx = current_device()
        default_generator = torch_remote._C._get_default_generator(idx)
        return default_generator.initial_seed()

    def manual_seed(seed: int) -> None:
        """Set the random seed for the current remote device.

        Args:
            seed: Random seed value
        """
        seed = int(seed)

        idx = current_device()
        default_generator = torch_remote._C._get_default_generator(idx)
        default_generator.manual_seed(seed)

    def manual_seed_all(seed: int) -> None:
        """Set the random seed for all remote devices.

        Args:
            seed: Random seed value
        """
        seed = int(seed)

        for idx in range(device_count()):
            default_generator = torch_remote._C._get_default_generator(idx)
            default_generator.manual_seed(seed)

    def is_initialized() -> bool:
        return module._initialized

    def _is_in_bad_fork() -> bool:
        return False

    def _lazy_init() -> None:
        if is_initialized():
            return
        torch_remote._C._init()
        module._initialized = True

    module.is_available = is_available  # type: ignore[assignment]

    module._initialized = False  # type: ignore[assignment]
    module._lazy_init = _lazy_init  # type: ignore[assignment]
    module.is_initialized = is_initialized  # type: ignore[assignment]

    module.device = device  # type: ignore[assignment]
    module.device_count = device_count  # type: ignore[assignment]
    module.current_device = current_device  # type: ignore[assignment]
    module.get_rng_state = get_rng_state  # type: ignore[assignment]
    module.set_rng_state = set_rng_state  # type: ignore[assignment]
    module._is_in_bad_fork = _is_in_bad_fork  # type: ignore[assignment]
    module.initial_seed = initial_seed  # type: ignore[assignment]
    module.manual_seed = manual_seed  # type: ignore[assignment]
    module.manual_seed_all = manual_seed_all  # type: ignore[assignment]

    return module


# Set all the appropriate state on PyTorch
torch.utils.rename_privateuse1_backend("remote")
torch._register_device_module("remote", _create_module())


# Import device management
# Import ATen implementations to ensure PyTorch registrations are executed
import torch_remote._aten_impl  # noqa: F401

from .device import (
    CloudProvider,
    GPUType,
    RemoteMachine,
    create_modal_machine,
    get_device_registry,
)

# Import logging utilities
from ._logging import (
    set_logging_level,
    get_logging_level,
    disable_logging,
    enable_debug_logging,
    enable_info_logging,
    reset_logging,
)
