# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Custom pickle system for mycelya tensors and remote execution.

This module provides custom pickler/unpickler classes that handle mycelya tensors and devices
properly during serialization for remote execution. It includes:

- Pickler: Converts mycelya tensors to tensor IDs and devices to remote info
- Unpickler: Reconstructs remote execution results back to local mycelya tensors
"""

import dis
import io
import pickle
import types
from typing import Any

import cloudpickle
import torch

from ._device import device_manager
from ._logging import get_logger
from ._utils import (
    create_mycelya_tensor_from_metadata,
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

    def __init__(
        self,
        file: io.BytesIO,
        storage_manager,
        protocol: int = None,
        buffer_callback: Any = None,
    ):
        super().__init__(file, protocol=protocol, buffer_callback=buffer_callback)
        self.storage_manager = storage_manager
        self.machine_id: str | None = None
        # Collect tensors that need _materialize_tensor called (mapping from object_id to tensor)
        self.tensors: dict[int, torch.Tensor] = {}
        # Collect module dependencies
        self.module_dependencies: set[str] = set()

    def _extract_module_from_globals(self, globals_dict: dict) -> None:
        """
        Extract module dependencies from function globals.

        Args:
            globals_dict: Function's __globals__ dictionary
        """
        for name, value in globals_dict.items():
            # Skip dunder attributes and None values
            if name.startswith("__") or value is None:
                continue

            # Check if it's a module
            if isinstance(value, types.ModuleType):
                module_name = getattr(value, "__name__", None)
                if module_name:
                    # Get base module name (e.g. 'torch' from 'torch.nn.functional')
                    base_module = module_name.split(".")[0]
                    self.module_dependencies.add(base_module)
            else:
                # Check if object has __module__ attribute
                module_name = getattr(value, "__module__", None)
                if module_name:
                    # Get base module name
                    base_module = module_name.split(".")[0]
                    self.module_dependencies.add(base_module)

    def _extract_modules_from_code(self, code_obj: types.CodeType) -> None:
        """
        Extract module dependencies from code object using bytecode analysis.

        Args:
            code_obj: Code object to analyze
        """
        # Analyze bytecode instructions
        for instruction in dis.get_instructions(code_obj):
            # Look for IMPORT_NAME instructions - these are the actual imports
            if instruction.opname == "IMPORT_NAME":
                # Direct import statement (import foo, from foo import bar)
                module_name = instruction.argval
                if module_name:
                    base_module = module_name.split(".")[0]
                    self.module_dependencies.add(base_module)

        # Recursively analyze nested code objects (functions, classes, etc.)
        for const in code_obj.co_consts:
            if isinstance(const, types.CodeType):
                self._extract_modules_from_code(const)

    def _analyze_dependencies(self, obj: Any) -> None:
        """
        Analyze object for package dependencies.

        Args:
            obj: Object to analyze for dependencies
        """
        # Handle dictionary objects with __globals__ key (function dictionaries)
        if isinstance(obj, dict) and "__globals__" in obj:
            globals_dict = obj["__globals__"]
            if isinstance(globals_dict, dict):
                self._extract_module_from_globals(globals_dict)

        # Handle code objects directly
        elif isinstance(obj, types.CodeType):
            self._extract_modules_from_code(obj)

    def persistent_id(self, obj: Any) -> tuple[str, Any] | None:
        """
        Handle mycelya tensors and devices during pickling, and analyze dependencies.

        Args:
            obj: Object being pickled

        Returns:
            Tuple of (type_tag, data) for mycelya objects, None for regular objects

        Raises:
            RuntimeError: If tensors/devices from different machines are mixed
        """
        # Analyze object for package dependencies
        self._analyze_dependencies(obj)
        # Handle mycelya tensors
        if isinstance(obj, torch.Tensor) and obj.device.type == "mycelya":
            # Get tensor's machine information
            machine_id, remote_type, remote_index = (
                self.storage_manager.get_remote_device_info(obj)
            )

            # Validate machine consistency
            if self.machine_id is None:
                self.machine_id = machine_id
            elif self.machine_id != machine_id:
                raise RuntimeError(
                    f"Cannot serialize tensors from different machines: "
                    f"current machine {self.machine_id}, tensor machine {machine_id}"
                )

            # Collect tensor for orchestrator to call _materialize_tensor on
            self.tensors[id(obj)] = obj
            tensor_id = get_tensor_id(obj)  # Use metadata hash as tensor ID

            # Include requires_grad and is_parameter metadata for proper autograd reconstruction
            # Include object_id for server-side deduplication
            # Include gradient (will be None or a tensor, both pickle correctly)
            return ("mycelya_tensor", {
                "id": tensor_id,
                "requires_grad": obj.requires_grad,
                "is_parameter": isinstance(obj, torch.nn.Parameter),
                "object_id": id(obj),
                "grad": obj.grad  # None or tensor, pickled recursively via persistent_id
            })

        # Handle mycelya devices
        elif isinstance(obj, torch.device) and obj.type == "mycelya":
            if obj.index is None:
                raise ValueError("Mycelya device must have an index")

            # Get device's machine information
            machine_id, remote_type, remote_index = (
                device_manager.get_remote_device_info(obj.index)
            )

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

    def __init__(self, file: io.BytesIO, machine_id: str, storage_manager, pickler_tensors: dict[int, torch.Tensor]):
        super().__init__(file)
        self.machine_id = machine_id
        self.storage_manager = storage_manager
        # Cache for tensor deduplication: object_id (int | str) -> tensor
        # Initialize from pickler's tensors
        self.tensor_cache: dict[int | str, torch.Tensor] = {}
        for object_id, tensor in pickler_tensors.items():
            self.tensor_cache[object_id] = tensor
        # Collect tensor linking info for orchestrator to handle
        self.tensors_to_link = []  # List of (tensor, temp_id) tuples

    def persistent_load(self, pid: tuple[str, Any]) -> Any:
        """
        Handle reconstruction of remote execution results.

        Args:
            pid: Persistent ID tuple from remote pickler

        Returns:
            Reconstructed mycelya tensor or device
        """
        type_tag, data = pid

        if type_tag == "remote_tensor":
            # Extract metadata, requires_grad, is_parameter, object_id, and grad
            metadata = data["metadata"]
            requires_grad = data["requires_grad"]
            is_parameter = data["is_parameter"]
            object_id = data["object_id"]

            # Check cache first for deduplication
            if object_id in self.tensor_cache:
                tensor = self.tensor_cache[object_id]

                # Resize storage if needed
                storage = tensor.untyped_storage()
                if metadata["nbytes"] > storage.nbytes():
                    storage.resize_(metadata["nbytes"])

                # Apply in-place metadata mutations if changed
                if (list(tensor.shape) != metadata["shape"]
                    or list(tensor.stride()) != metadata["stride"]
                    or tensor.storage_offset() != metadata["storage_offset"]):
                    tensor.as_strided_(metadata["shape"], metadata["stride"], metadata["storage_offset"])

                # Apply requires_grad change if needed
                if tensor.requires_grad != requires_grad:
                    tensor.requires_grad_(requires_grad)
            else:
                # Get device using device_manager
                device = device_manager.get_mycelya_device(
                    self.machine_id, metadata["device_type"], metadata["device_index"]
                )

                # Create mycelya tensor from metadata with storage_manager
                tensor = create_mycelya_tensor_from_metadata(
                    metadata, device, self.storage_manager
                )

                # Wrap as Parameter if it was originally a Parameter
                if is_parameter:
                    tensor = torch.nn.Parameter(tensor)

                # Apply requires_grad after wrapping
                tensor.requires_grad_(requires_grad)

                # Cache the tensor for future lookups
                self.tensor_cache[object_id] = tensor

            # Restore gradient (None or tensor, both work)
            tensor.grad = data["grad"]

            # Collect tensor linking info for orchestrator to handle
            temp_id = metadata["id"]
            self.tensors_to_link.append((tensor, temp_id))

            return tensor

        elif type_tag == "remote_device":
            device_type, device_index = data

            # Get device using device_manager
            return device_manager.get_mycelya_device(
                self.machine_id, device_type, device_index
            )

        else:
            raise pickle.PicklingError(f"Unknown persistent ID type: {type_tag}")
