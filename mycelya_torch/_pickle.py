# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Custom pickle system for mycelya tensors and remote execution.

This module provides custom pickler/unpickler classes that handle mycelya tensors
and devices properly during serialization for remote execution. It includes:

- MycelyaPickler: Converts mycelya tensors to tensor IDs and devices to remote info
- MycelyaUnpickler: Reconstructs mycelya tensors from IDs and maps devices back
- RemotePickler: Server-side pickler that converts tensors to metadata
- RemoteUnpickler: Server-side unpickler that reconstructs tensors from registry
- remote decorator: Combines all functionality for remote function execution
"""

import io
import pickle
import uuid
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import torch

try:
    import cloudpickle
except ImportError:
    raise ImportError(
        "cloudpickle is required for remote function execution. "
        "Install with: pip install cloudpickle"
    )

from ._device import device_manager
from ._logging import get_logger
from ._machine import RemoteMachine
from ._orchestrator import orchestrator
from ._utils import (
    TensorMetadata,
    get_storage_id,
    get_tensor_id,
)

log = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class MycelyaPickler(cloudpickle.CloudPickler):
    """
    Custom CloudPickler that handles mycelya tensors and devices for remote execution.

    This pickler converts:
    - Mycelya tensors -> tensor IDs for remote lookup
    - Mycelya devices -> device info tuples for remote mapping

    It maintains internal state about the remote machine being serialized for,
    and validates that all tensors/devices belong to the same machine.
    Uses CloudPickle for proper function serialization.
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


class MycelyaUnpickler(pickle.Unpickler):
    """
    Custom unpickler that reconstructs mycelya tensors and devices from remote info.

    This unpickler converts:
    - Tensor IDs -> mycelya tensor stubs that will be linked later
    - Device info tuples -> mycelya device objects mapped to local indices
    """

    def __init__(self, file: io.BytesIO, machine: RemoteMachine):
        super().__init__(file)
        self.machine = machine
        self._tensor_metadata_map: Dict[int, TensorMetadata] = {}

    def persistent_load(self, pid: Tuple[str, Any]) -> Any:
        """
        Handle reconstruction of mycelya objects during unpickling.

        Args:
            pid: Persistent ID tuple from pickler

        Returns:
            Reconstructed mycelya tensor or device
        """
        type_tag, data = pid

        if type_tag == "mycelya_tensor":
            # Note: This method is primarily for completeness.
            # In practice, remote execution uses the ResultUnpickler above
            # which handles tensor metadata properly.
            raise NotImplementedError(
                "Direct tensor unpickling from tensor ID is not supported. "
                "Use the remote decorator for proper remote function execution."
            )

        elif type_tag == "mycelya_device":
            remote_type, remote_index = data

            # Map remote device info back to local mycelya device
            return self.machine.device(type=remote_type, index=remote_index)

        else:
            raise pickle.PicklingError(f"Unknown persistent ID type: {type_tag}")


class RemotePickler(cloudpickle.CloudPickler):
    """
    Server-side CloudPickler that converts torch tensors to metadata for return to client.

    This pickler runs on the remote server and converts:
    - Torch tensors -> tensor metadata with temp registry keys
    - Regular devices -> device type/index info for local reconstruction
    """

    def __init__(self, file: io.BytesIO, temp_registry: Dict[str, torch.Tensor],
                 protocol: int = None, buffer_callback: Any = None):
        super().__init__(file, protocol=protocol, buffer_callback=buffer_callback)
        self.temp_registry = temp_registry

    def persistent_id(self, obj: Any) -> Optional[Tuple[str, Any]]:
        """
        Handle torch tensors and devices during server-side pickling.

        Args:
            obj: Object being pickled on remote server

        Returns:
            Tuple of (type_tag, data) for special objects, None for regular objects
        """
        # Handle torch tensors - convert to metadata and register in temp registry
        if isinstance(obj, torch.Tensor):
            # Generate unique temp key
            temp_key = f"remote_result_{uuid.uuid4().hex[:8]}"

            # Register tensor in temp registry
            self.temp_registry[temp_key] = obj

            # Create metadata for client reconstruction
            metadata: TensorMetadata = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype).replace("torch.", ""),
                "stride": list(obj.stride()),
                "storage_offset": obj.storage_offset(),
                "nbytes": obj.untyped_storage().nbytes(),
                "temp_key": temp_key,
            }

            if obj.requires_grad:
                metadata["requires_grad"] = True

            return ("remote_tensor", metadata)

        # Handle regular devices - convert to type/index info
        elif isinstance(obj, torch.device):
            return ("remote_device", (obj.type, obj.index))

        # Regular object - use normal pickling
        return None


class RemoteUnpickler(pickle.Unpickler):
    """
    Server-side unpickler that reconstructs tensors from IDs for function execution.

    This unpickler runs on the remote server and converts:
    - Tensor IDs -> actual torch tensors from tensor registry
    - Device info -> actual torch device objects
    """

    def __init__(self, file: io.BytesIO, tensor_registry: Dict[int, torch.Tensor]):
        super().__init__(file)
        self.tensor_registry = tensor_registry

    def persistent_load(self, pid: Tuple[str, Any]) -> Any:
        """
        Handle reconstruction of tensors and devices on remote server.

        Args:
            pid: Persistent ID tuple from client pickler

        Returns:
            Reconstructed torch tensor or device
        """
        type_tag, data = pid

        if type_tag == "mycelya_tensor":
            tensor_id = data

            # Look up tensor in registry
            if tensor_id not in self.tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} not found in remote registry")

            return self.tensor_registry[tensor_id]

        elif type_tag == "mycelya_device":
            remote_type, remote_index = data

            # Reconstruct torch device
            return torch.device(remote_type, remote_index)

        else:
            raise pickle.PicklingError(f"Unknown persistent ID type: {type_tag}")


def mycelya_pickle(obj: Any, machine: Optional[RemoteMachine] = None) -> bytes:
    """
    Pickle an object using MycelyaPickler (based on CloudPickle).

    Args:
        obj: Object to pickle
        machine: Optional RemoteMachine for validation (inferred if None)

    Returns:
        Pickled bytes
    """
    buffer = io.BytesIO()
    pickler = MycelyaPickler(buffer)
    pickler.dump(obj)

    # Validate machine if provided
    if machine is not None and pickler.machine_id != machine.machine_id:
        raise RuntimeError(
            f"Object contains tensors/devices from machine {pickler.machine_id}, "
            f"but expected machine {machine.machine_id}"
        )

    return buffer.getvalue()


def mycelya_unpickle(data: bytes, machine: RemoteMachine) -> Any:
    """
    Unpickle an object using MycelyaUnpickler.

    Args:
        data: Pickled bytes
        machine: RemoteMachine for tensor/device reconstruction

    Returns:
        Unpickled object
    """
    buffer = io.BytesIO(data)
    unpickler = MycelyaUnpickler(buffer, machine)
    return unpickler.load()


def remote_pickle(obj: Any, temp_registry: Dict[str, torch.Tensor]) -> bytes:
    """
    Server-side pickle using RemotePickler (based on CloudPickle).

    Args:
        obj: Object to pickle on remote server
        temp_registry: Temporary tensor registry for metadata keys

    Returns:
        Pickled bytes
    """
    buffer = io.BytesIO()
    pickler = RemotePickler(buffer, temp_registry)
    pickler.dump(obj)
    return buffer.getvalue()


def remote_unpickle(data: bytes, tensor_registry: Dict[int, torch.Tensor]) -> Any:
    """
    Server-side unpickle using RemoteUnpickler.

    Args:
        data: Pickled bytes from client
        tensor_registry: Tensor registry for ID lookup

    Returns:
        Unpickled object
    """
    buffer = io.BytesIO(data)
    unpickler = RemoteUnpickler(buffer, tensor_registry)
    return unpickler.load()


def remote(func: F) -> F:
    """
    Decorator that converts a function to execute remotely on mycelya tensors.

    This decorator:
    1. Analyzes function arguments to determine target remote machine
    2. Serializes function and arguments using CloudPickle-based MycelyaPickler
    3. Executes function remotely via orchestrator coordination
    4. Deserializes results back to local mycelya tensors with proper linking

    Args:
        func: Function to make remotely executable

    Returns:
        Wrapped function that executes remotely

    Example:
        @remote
        def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b

        machine = RemoteMachine("modal", "A100")
        x = torch.randn(100, 100, device=machine.device())
        y = torch.randn(100, 100, device=machine.device())
        result = matrix_multiply(x, y)  # Executes remotely
    """

    def wrapper(*args, **kwargs):
        # Find mycelya tensors/devices to infer target machine
        machine_id = None

        def check_for_machine(obj):
            nonlocal machine_id

            if isinstance(obj, torch.Tensor) and obj.device.type == "mycelya":
                storage_id = get_storage_id(obj)
                obj_machine_id = orchestrator.storage.get_remote_device_info(storage_id)[0]

                if machine_id is None:
                    machine_id = obj_machine_id
                elif machine_id != obj_machine_id:
                    raise RuntimeError(
                        f"Function arguments contain tensors from different machines: "
                        f"{machine_id} and {obj_machine_id}"
                    )

            elif isinstance(obj, torch.device) and obj.type == "mycelya":
                if obj.index is None:
                    raise ValueError("Mycelya device must have an index")
                obj_machine_id = device_manager.get_remote_device_info(obj.index)[0]

                if machine_id is None:
                    machine_id = obj_machine_id
                elif machine_id != obj_machine_id:
                    raise RuntimeError(
                        f"Function arguments contain devices from different machines: "
                        f"{machine_id} and {obj_machine_id}"
                    )

        # Scan all arguments for mycelya objects
        for arg in args:
            if isinstance(arg, (list, tuple)):
                for item in arg:
                    check_for_machine(item)
            else:
                check_for_machine(arg)

        for kwarg_value in kwargs.values():
            if isinstance(kwarg_value, (list, tuple)):
                for item in kwarg_value:
                    check_for_machine(item)
            else:
                check_for_machine(kwarg_value)

        if machine_id is None:
            raise RuntimeError(
                "No mycelya tensors or devices found in function arguments. "
                "Remote execution requires at least one mycelya object to determine target machine."
            )

        # Execute the function remotely via orchestrator (proper architecture)
        return orchestrator.execute_pickled_function(func, args, kwargs, machine_id)

    # Preserve function metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = getattr(func, '__annotations__', {})

    return wrapper

