# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import types
from typing import Any, Callable

import torch

# Direct driver access for C++ via factory pattern
from ._backend_hooks import driver

# Factory pattern for C++ method access with caching
_IMPL_REGISTRY: dict[str, Callable] = {}


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
    """Create the mycelya device module for PyTorch backend registration.

    This function creates a module that implements the PyTorch accelerator
    backend interface for mycelya devices. It provides device context
    management, RNG state handling, and other core device operations.

    Returns:
        Module implementing the mycelya device backend interface
    """
    module = types.ModuleType("_MycelyaMod")

    def device_count() -> int:
        """Get the number of available mycelya devices.

        Returns:
            Number of mycelya devices available
        """
        return driver.device_count()

    def is_available() -> bool:
        """Check if mycelya device support is available.

        Returns:
            True if mycelya devices are available, False otherwise
        """
        return True

    def get_rng_state(device: int | torch.device) -> torch.Tensor:
        """Get the random number generator state for a mycelya device.

        Args:
            device: Mycelya device index or torch.device to get RNG state from

        Returns:
            Tensor containing the RNG state
        """
        if isinstance(device, int):
            idx = device
        elif isinstance(device, torch.device):
            if device.index is None:
                raise ValueError("Device index must be specified for mycelya devices")
            idx = device.index
        else:
            raise TypeError("Device must be int index or torch.device with index")

        default_generator = mycelya_torch._C._get_default_generator(idx)
        return default_generator.get_state()

    def set_rng_state(new_state: torch.Tensor, device: int | torch.device) -> None:
        """Set the random number generator state for a mycelya device.

        Args:
            new_state: Tensor containing the new RNG state
            device: Mycelya device index or torch.device to set RNG state for
        """
        if isinstance(device, int):
            idx = device
        elif isinstance(device, torch.device):
            if device.index is None:
                raise ValueError("Device index must be specified for mycelya devices")
            idx = device.index
        else:
            raise TypeError("Device must be int index or torch.device with index")

        default_generator = mycelya_torch._C._get_default_generator(idx)
        default_generator.set_state(new_state)

    def initial_seed(device: int | torch.device) -> int:
        """Get the initial seed for a mycelya device.

        Args:
            device: Mycelya device index or torch.device to get initial seed from

        Returns:
            Initial seed value
        """
        _lazy_init()
        if isinstance(device, int):
            idx = device
        elif isinstance(device, torch.device):
            if device.index is None:
                raise ValueError("Device index must be specified for mycelya devices")
            idx = device.index
        else:
            raise TypeError("Device must be int index or torch.device with index")

        default_generator = mycelya_torch._C._get_default_generator(idx)
        return default_generator.initial_seed()

    def manual_seed(seed: int, device: int | torch.device) -> None:
        """Set the random seed for a mycelya device.

        Args:
            seed: Random seed value
            device: Mycelya device index or torch.device to set seed for
        """
        seed = int(seed)

        if isinstance(device, int):
            idx = device
        elif isinstance(device, torch.device):
            if device.index is None:
                raise ValueError("Device index must be specified for mycelya devices")
            idx = device.index
        else:
            raise TypeError("Device must be int index or torch.device with index")

        default_generator = mycelya_torch._C._get_default_generator(idx)
        default_generator.manual_seed(seed)

    def manual_seed_all(seed: int) -> None:
        """Set the random seed for all mycelya devices.

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

    def _is_in_bad_fork() -> bool:
        """Check if we're in a bad fork state for multiprocessing.

        Returns:
            False
        """
        return False

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
    module._is_in_bad_fork = _is_in_bad_fork  # type: ignore[assignment]
    module.get_amp_supported_dtype = get_amp_supported_dtype  # type: ignore[assignment]

    return module


# Set all the appropriate state on PyTorch
torch.utils.rename_privateuse1_backend("mycelya")
torch._register_device_module("mycelya", _create_module())

# Import ATen implementations to ensure PyTorch registrations are executed
import mycelya_torch.aten  # noqa: E402

# Import public API components
from ._logging import (  # noqa: E402
    disable_logging,
    enable_debug_logging,
    enable_info_logging,
    get_logging_level,
    reset_logging,
    set_logging_level,
)
from ._machine import (  # noqa: E402
    RemoteMachine,
    get_all_machines,
)

# Remote execution utilities
from ._remote import remote  # noqa: E402


# Monkeypatch torch.Tensor with cpu_future() method
def _tensor_cpu_future(self: torch.Tensor):
    """Copy mycelya tensor to CPU asynchronously, returning a Future.

    This is a monkeypatch method added to torch.Tensor that provides
    asynchronous CPU transfer for mycelya tensors only.

    Returns:
        Future[torch.Tensor]: Future that will resolve to the CPU tensor

    Raises:
        RuntimeError: If tensor is not a mycelya tensor

    Example:
        >>> import torch
        >>> import mycelya_torch
        >>> machine = mycelya_torch.RemoteMachine("modal", "A100")
        >>> device = machine.device("cuda")
        >>> x = torch.randn(1000, 1000, device=device)
        >>> cpu_future = x.cpu_future()  # Returns immediately
        >>> cpu_tensor = cpu_future.result()  # Wait for completion
    """
    if self.device.type != "mycelya":
        raise RuntimeError(
            f"cpu_future() can only be called on mycelya tensors, got {self.device.type}. "
            f"Use .cpu() for synchronous transfer or .cpu_future() only with mycelya tensors."
        )

    # Import orchestrator locally to avoid circular import issues
    from ._orchestrator import orchestrator

    # Use orchestrator's async copy method for mycelya tensors
    return orchestrator.copy_tensor_to_cpu_future(self)


# Add the monkeypatch to torch.Tensor
torch.Tensor.cpu_future = _tensor_cpu_future  # type: ignore[assignment]


# Define the public API
__all__ = [
    # Core machine and device classes
    "RemoteMachine",
    "get_all_machines",
    # Remote execution utilities
    "remote",
    # Logging utilities
    "enable_debug_logging",
    "enable_info_logging",
    "disable_logging",
    "get_logging_level",
    "set_logging_level",
    "reset_logging",
]
