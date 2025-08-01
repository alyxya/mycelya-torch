# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import types
from typing import Any, Callable, Dict, Optional, Union

import torch

# Direct driver access for C++ via factory pattern
from ._device_daemon import driver

# Factory pattern for C++ method access with caching
_IMPL_REGISTRY: Dict[str, Callable] = {}


def impl_factory(name: str) -> Callable:
    """Factory function that returns cached method implementations.

    This follows the pytorch-openreg-2 pattern for cleaner C++ integration.

    Args:
        name: Method name to get implementation for

    Returns:
        Callable that executes the named method
    """
    if name in _IMPL_REGISTRY:
        return _IMPL_REGISTRY[name]

    def _method_impl(*args: Any, **kwargs: Any) -> Any:
        return driver.exec(name, *args, **kwargs)

    _IMPL_REGISTRY[name] = _method_impl
    return _method_impl


# Load the C++ Module (must come after impl_factory definition)
import mycelya_torch._C  # noqa: E402


def _create_module() -> types.ModuleType:
    """Create the remote device module for PyTorch backend registration.

    This function creates a module that implements the PyTorch accelerator
    backend interface for remote devices. It provides device context
    management, RNG state handling, and other core device operations.

    Returns:
        Module implementing the remote device backend interface
    """
    module = types.ModuleType("_RemoteMod")

    def device_count() -> int:
        """Get the number of available remote devices.

        Returns:
            Number of remote devices available
        """
        return driver.device_count()

    def is_available() -> bool:
        """Check if remote device support is available.

        Returns:
            True if remote devices are available, False otherwise
        """
        return True

    def get_rng_state(device: Union[int, torch.device]) -> torch.Tensor:
        """Get the random number generator state for a remote device.

        Args:
            device: Remote device index or torch.device to get RNG state from

        Returns:
            Tensor containing the RNG state
        """
        if isinstance(device, int):
            idx = device
        elif isinstance(device, torch.device):
            if device.index is None:
                raise ValueError("Device index must be specified for remote devices")
            idx = device.index
        else:
            raise TypeError("Device must be int index or torch.device with index")

        default_generator = mycelya_torch._C._get_default_generator(idx)
        return default_generator.get_state()

    def set_rng_state(
        new_state: torch.Tensor, device: Union[int, torch.device]
    ) -> None:
        """Set the random number generator state for a remote device.

        Args:
            new_state: Tensor containing the new RNG state
            device: Remote device index or torch.device to set RNG state for
        """
        if isinstance(device, int):
            idx = device
        elif isinstance(device, torch.device):
            if device.index is None:
                raise ValueError("Device index must be specified for remote devices")
            idx = device.index
        else:
            raise TypeError("Device must be int index or torch.device with index")

        default_generator = mycelya_torch._C._get_default_generator(idx)
        default_generator.set_state(new_state)

    def initial_seed(device: Union[int, torch.device]) -> int:
        """Get the initial seed for a remote device.

        Args:
            device: Remote device index or torch.device to get initial seed from

        Returns:
            Initial seed value
        """
        _lazy_init()
        if isinstance(device, int):
            idx = device
        elif isinstance(device, torch.device):
            if device.index is None:
                raise ValueError("Device index must be specified for remote devices")
            idx = device.index
        else:
            raise TypeError("Device must be int index or torch.device with index")

        default_generator = mycelya_torch._C._get_default_generator(idx)
        return default_generator.initial_seed()

    def manual_seed(seed: int, device: Union[int, torch.device]) -> None:
        """Set the random seed for a remote device.

        Args:
            seed: Random seed value
            device: Remote device index or torch.device to set seed for
        """
        seed = int(seed)

        if isinstance(device, int):
            idx = device
        elif isinstance(device, torch.device):
            if device.index is None:
                raise ValueError("Device index must be specified for remote devices")
            idx = device.index
        else:
            raise TypeError("Device must be int index or torch.device with index")

        default_generator = mycelya_torch._C._get_default_generator(idx)
        default_generator.manual_seed(seed)

    def manual_seed_all(seed: int) -> None:
        """Set the random seed for all remote devices.

        Args:
            seed: Random seed value
        """
        seed = int(seed)

        for idx in range(device_count()):
            default_generator = mycelya_torch._C._get_default_generator(idx)
            default_generator.manual_seed(seed)

    def is_initialized() -> bool:
        return module._initialized

    def _lazy_init() -> None:
        if is_initialized():
            return
        mycelya_torch._C._init()
        module._initialized = True

    def get_amp_supported_dtype():
        """Get the list of supported dtypes for AMP (Automatic Mixed Precision).

        Returns:
            List of torch.dtype objects supported for AMP operations
        """
        return [torch.float16, torch.bfloat16]

    module.is_available = is_available  # type: ignore[assignment]

    module._initialized = False  # type: ignore[assignment]
    module._lazy_init = _lazy_init  # type: ignore[assignment]
    module.is_initialized = is_initialized  # type: ignore[assignment]

    module.device_count = device_count  # type: ignore[assignment]
    module.get_rng_state = get_rng_state  # type: ignore[assignment]
    module.set_rng_state = set_rng_state  # type: ignore[assignment]
    module.initial_seed = initial_seed  # type: ignore[assignment]
    module.manual_seed = manual_seed  # type: ignore[assignment]
    module.manual_seed_all = manual_seed_all  # type: ignore[assignment]
    module.get_amp_supported_dtype = get_amp_supported_dtype  # type: ignore[assignment]

    return module


# Set all the appropriate state on PyTorch
torch.utils.rename_privateuse1_backend("mycelya")
torch._register_device_module("mycelya", _create_module())

# Import ATen implementations to ensure PyTorch registrations are executed
import mycelya_torch._aten_impl  # noqa: E402

# Import public API components
from ._logging import (  # noqa: E402
    disable_logging,
    enable_debug_logging,
    enable_info_logging,
    get_logging_level,
    reset_logging,
    set_logging_level,
)
from .device import (  # noqa: E402
    CloudProvider,
    GPUType,
    RemoteMachine,
    create_mock_machine,
    create_modal_machine,
    get_all_machines,
    get_device_registry,
)
