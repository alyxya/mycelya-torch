# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Custom pickle system for mycelya tensors and remote execution.

This module provides custom pickler/unpickler classes and remote execution decorator that handle
mycelya tensors and devices properly during serialization for remote execution. It includes:

- Pickler: Converts mycelya tensors to tensor IDs and devices to remote info
- Unpickler: Reconstructs remote execution results back to local mycelya tensors
- remote decorator: Main decorator for remote function execution
"""

import functools
import io
import pickle
from typing import Any, Callable, Optional, Tuple

import torch
import cloudpickle

from ._device import device_manager
from ._logging import get_logger
from ._machine import RemoteMachine
from ._orchestrator import orchestrator
from ._utils import (
    get_storage_id,
    get_tensor_id,
)

log = get_logger(__name__)


class Pickler(cloudpickle.Pickler):
    """
    Custom Pickler that handles mycelya tensors and devices for remote execution.

    This pickler converts:
    - Mycelya tensors -> tensor IDs for remote lookup
    - Mycelya devices -> device info tuples for remote mapping

    It maintains internal state about the remote machine being serialized for,
    and validates that all tensors/devices belong to the same machine.
    Uses cloudpickle.Pickler for proper function serialization.
    """

    def __init__(self, file: io.BytesIO, protocol: int = None, buffer_callback: Any = None):
        super().__init__(file, protocol=protocol, buffer_callback=buffer_callback)
        self.machine_id: Optional[str] = None

    def persistent_id(self, obj: Any) -> Optional[Tuple[str, Any]]:
        """
        Handle mycelya tensors and devices during pickling.

        Args:
            obj: Object being pickled

        Returns:
            Tuple of (type_tag, data) for mycelya objects, None for regular objects

        Raises:
            RuntimeError: If tensors/devices from different machines are mixed
        """
        # Handle mycelya tensors
        if isinstance(obj, torch.Tensor) and obj.device.type == "mycelya":
            # Get tensor's machine information
            storage_id = get_storage_id(obj)
            machine_id, remote_type, remote_index = orchestrator.storage.get_remote_device_info(storage_id)

            # Validate machine consistency
            if self.machine_id is None:
                self.machine_id = machine_id
            elif self.machine_id != machine_id:
                raise RuntimeError(
                    f"Cannot serialize tensors from different machines: "
                    f"current machine {self.machine_id}, tensor machine {machine_id}"
                )

            # Return tensor ID for remote lookup
            # First ensure the tensor exists remotely
            orchestrator._maybe_create_tensor(obj)
            tensor_id = get_tensor_id(obj)  # Use metadata hash as tensor ID
            return ("mycelya_tensor", tensor_id)

        # Handle mycelya devices
        elif isinstance(obj, torch.device) and obj.type == "mycelya":
            if obj.index is None:
                raise ValueError("Mycelya device must have an index")

            # Get device's machine information
            machine_id, remote_type, remote_index = device_manager.get_remote_device_info(obj.index)

            # Validate machine consistency
            if self.machine_id is None:
                self.machine_id = machine_id
            elif self.machine_id != machine_id:
                raise RuntimeError(
                    f"Cannot serialize devices from different machines: "
                    f"current machine {self.machine_id}, device machine {machine_id}"
                )

            # Return device info for remote mapping
            return ("mycelya_device", (remote_type, remote_index))

        # Not a mycelya object - use normal pickling
        return None


class Unpickler(pickle.Unpickler):
    """
    Unpickler for remote function execution results.

    This unpickler handles the results returned from remote function execution,
    converting remote_tensor metadata back into local mycelya tensors and
    remote_device info back into local mycelya devices.
    """

    def __init__(self, file: io.BytesIO, machine_id: str, client):
        super().__init__(file)
        self.machine_id = machine_id
        self.client = client

    def persistent_load(self, pid: Tuple[str, Any]) -> Any:
        """
        Handle reconstruction of remote execution results.

        Args:
            pid: Persistent ID tuple from remote pickler

        Returns:
            Reconstructed mycelya tensor or device
        """
        type_tag, data = pid

        if type_tag == "remote_tensor":
            metadata = data
            # Import here to avoid circular imports
            from ._machine import RemoteMachine
            from ._utils import (
                create_mycelya_tensor_from_metadata,
                get_tensor_id,
            )

            # Find the appropriate machine for device mapping
            for machine in RemoteMachine._all_machines:
                if machine.machine_id == self.machine_id:
                    device = machine.device()
                    break
            else:
                raise RuntimeError(f"No RemoteMachine found for machine_id {self.machine_id}")

            # Create mycelya tensor from metadata
            tensor = create_mycelya_tensor_from_metadata(metadata, device)

            # Link the tensor to the remote tensor in temp registry
            temp_key = metadata["temp_key"]
            tensor_id = get_tensor_id(tensor)
            self.client.link_tensors([tensor_id], [temp_key])

            return tensor

        elif type_tag == "remote_device":
            device_type, device_index = data
            # Import here to avoid circular imports
            from ._machine import RemoteMachine

            # Find the appropriate machine for device mapping
            for machine in RemoteMachine._all_machines:
                if machine.machine_id == self.machine_id:
                    return machine.device(type=device_type, index=device_index)

            raise RuntimeError(f"No RemoteMachine found for machine_id {self.machine_id}")

        else:
            raise pickle.PicklingError(f"Unknown persistent ID type: {type_tag}")


def remote(_func: Optional[Callable[..., Any]] = None, *, run_async: bool = False):
    """
    Dual-mode decorator that converts a function to execute remotely on mycelya tensors.

    Can be used either as @remote or @remote() with identical behavior.

    This decorator:
    1. Analyzes function arguments to determine target remote machine
    2. Serializes function and arguments using cloudpickle.Pickler-based MycelyaPickler
    3. Executes function remotely via orchestrator coordination
    4. Deserializes results back to local mycelya tensors with proper linking

    Args:
        _func: Function to decorate (when used as @remote) or None (when used as @remote())
        run_async: Whether to run the function asynchronously (unused for now, defaults to False)

    Returns:
        Decorated function (when used as @remote) or decorator function (when used as @remote())

    Examples:
        # Both of these work identically:

        @remote
        def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b

        @remote()
        def matrix_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        # Future async support:
        @remote(run_async=True)
        def async_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        machine = RemoteMachine("modal", "A100")
        x = torch.randn(100, 100, device=machine.device())
        y = torch.randn(100, 100, device=machine.device())
        result1 = matrix_multiply(x, y)  # Executes remotely
        result2 = matrix_add(x, y)       # Executes remotely
    """

    def create_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(func):
            raise TypeError(f"@remote decorator expected a callable function, got {type(func).__name__}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the function remotely via orchestrator
            # Machine inference happens during pickling via Pickler.machine_id
            return orchestrator.execute_pickled_function(func, args, kwargs)

        return wrapper

    # Dual-mode logic: detect if used as @remote or @remote()
    if _func is None:
        # Called as @remote() - return decorator function
        return create_wrapper
    else:
        # Called as @remote - directly decorate the function
        return create_wrapper(_func)
