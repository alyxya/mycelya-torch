# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Modal remote execution app for mycelya_torch extension.

This module handles all Modal-specific functionality including:
- Dynamic device-specific app creation for different GPU types
- Remote execution of PyTorch operations
- Dynamic GPU selection and configuration

Part of: mycelya_torch PyTorch extension
"""

import io
import os
import pickle
import subprocess
import uuid
import warnings
import weakref
from typing import Any, TypedDict

import cloudpickle
import modal
import torch


def create_modal_app_for_gpu(
    machine_id: str,
    gpu_type: str,
    packages: list[str],
    python_version: str,
    timeout: int | None = 3600,
) -> tuple[Any, Any]:
    """
    Create a Modal app and class for a specific GPU type.

    Args:
        machine_id: Unique machine ID used as the app name
        gpu_type: The GPU type (e.g., "T4", "A100", "local" for local execution)
        timeout: Function timeout in seconds (default: 3600)
        packages: List of versioned packages to install (e.g., ["torch==2.0.0", "numpy==1.24.0"])
        python_version: Python version for base image (e.g., "3.11")

    Returns:
        Tuple of (modal_app, server_class) for the specified GPU type
    """

    class BatchCall(TypedDict):
        """Structure for a single batched RPC call."""

        method_name: str
        args: tuple[Any, ...]
        kwargs: dict[str, Any]

    class TensorMetadata(TypedDict):
        """Structure for tensor metadata with ID.

        This TypedDict defines the structure returned by dynamic operations
        that need to pass tensor metadata along with an ID for
        linking local tensors to remote tensors.
        """

        shape: list[int]
        stride: list[int]
        dtype: str
        storage_offset: int
        nbytes: int
        device_type: str
        device_index: int
        id: str | int
        alias_id: str | int | None

    class TensorManager:
        """Manages tensor registries and storage-to-tensor ID mappings for server-side tensor operations."""

        def __init__(self):
            # tensor_id -> torch.Tensor (direct mapping from tensor ID to tensor)
            self.tensor_registry: dict[int, torch.Tensor] = {}

            # Temporary tensor registry: temp_id -> torch.Tensor (for operations that create tensors remotely first)
            self.temp_tensor_registry: dict[str, torch.Tensor] = {}

            # Storage-to-IDs mapping: untyped_storage -> set of tensor_ids/temp_ids
            self.storage_to_ids: weakref.WeakKeyDictionary[
                torch.UntypedStorage, set[str | int]
            ] = weakref.WeakKeyDictionary()

        def register_tensor(self, tensor_id: int, tensor: torch.Tensor) -> None:
            """Register a tensor in the main registry and update storage mapping."""
            self.tensor_registry[tensor_id] = tensor
            storage = tensor.untyped_storage()
            self.storage_to_ids.setdefault(storage, set()).add(tensor_id)

        def register_temp_tensor(self, tensor: torch.Tensor) -> TensorMetadata:
            """Register a temporary tensor and return its metadata."""
            # Generate unique temp ID
            temp_id = f"temp_{uuid.uuid4().hex}"

            # Get alias ID before registering (to see if storage already exists)
            storage = tensor.untyped_storage()
            existing_ids = self.storage_to_ids.get(storage, set())
            alias_id = next(iter(existing_ids), None) if existing_ids else None

            # Register tensor in temp registry
            self.temp_tensor_registry[temp_id] = tensor
            self.storage_to_ids.setdefault(storage, set()).add(temp_id)

            # Create and return metadata
            return TensorMetadata(
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                dtype=str(tensor.dtype).replace("torch.", ""),
                storage_offset=tensor.storage_offset(),
                nbytes=tensor.untyped_storage().nbytes(),
                device_type=tensor.device.type,
                device_index=tensor.device.index if tensor.device.index is not None else 0,
                id=temp_id,
                alias_id=alias_id,
            )

        def link_tensors(self, tensor_ids: list[int], temp_ids: list[str]) -> None:
            """Link local tensor IDs to remote tensors from temporary registry."""
            for tensor_id, temp_id in zip(tensor_ids, temp_ids, strict=True):
                if temp_id not in self.temp_tensor_registry:
                    raise KeyError(f"Temporary tensor ID '{temp_id}' not found in temporary registry")

                temp_tensor = self.temp_tensor_registry.pop(temp_id)
                self.register_tensor(tensor_id, temp_tensor)

                # Update storage mapping - remove temp_id, add tensor_id
                storage = temp_tensor.untyped_storage()
                if storage in self.storage_to_ids:
                    self.storage_to_ids[storage].discard(temp_id)
                    self.storage_to_ids[storage].add(tensor_id)

        def offload(self, filepath: str) -> None:
            """Save tensor registries to disk."""
            state = {
                "tensor_registry": self.tensor_registry,
                "temp_tensor_registry": self.temp_tensor_registry,
            }
            torch.save(state, filepath)

        def reload(self, filepath: str) -> None:
            """Load tensor registries from disk and reconstruct storage_to_ids mapping."""
            state = torch.load(filepath, weights_only=False)
            self.tensor_registry = state["tensor_registry"]
            self.temp_tensor_registry = state["temp_tensor_registry"]

            # Reconstruct storage_to_ids mapping from both registries
            self.storage_to_ids = weakref.WeakKeyDictionary()
            for tensor_id, tensor in self.tensor_registry.items():
                storage = tensor.untyped_storage()
                self.storage_to_ids.setdefault(storage, set()).add(tensor_id)
            for temp_id, tensor in self.temp_tensor_registry.items():
                storage = tensor.untyped_storage()
                self.storage_to_ids.setdefault(storage, set()).add(temp_id)

            # Delete the file after successful reload
            os.remove(filepath)

    class Unpickler(pickle.Unpickler):
        """Custom unpickler to reconstruct tensors from IDs with object deduplication."""

        def __init__(self, file: Any, tensor_manager: TensorManager) -> None:
            super().__init__(file)
            self.tensor_manager = tensor_manager
            # Cache for tensor deduplication: object_id -> tensor
            self.tensor_cache: dict[int, torch.Tensor] = {}

        def persistent_load(self, pid: tuple[str, Any]) -> Any:
            type_tag, data = pid

            if type_tag == "mycelya_tensor":
                # Check tensor cache first for deduplication
                object_id = data["object_id"]
                if object_id in self.tensor_cache:
                    return self.tensor_cache[object_id]

                tensor_id = data["id"]
                requires_grad = data["requires_grad"]
                is_parameter = data["is_parameter"]

                if not tensor_id in self.tensor_manager.tensor_registry:
                    raise ValueError(
                        f"Tensor ID {tensor_id} not found in remote registry"
                    )

                # Detach from registry tensor
                tensor = self.tensor_manager.tensor_registry[tensor_id].detach()

                # Wrap as Parameter if it was originally a Parameter
                if is_parameter:
                    tensor = torch.nn.Parameter(tensor)

                # Restore requires_grad after wrapping
                tensor.requires_grad_(requires_grad)

                # Restore gradient (None or tensor, both work)
                tensor.grad = data["grad"]

                # Cache the tensor for future lookups
                self.tensor_cache[object_id] = tensor

                return tensor

            elif type_tag == "mycelya_device":
                remote_type, remote_index = data
                return torch.device(remote_type, remote_index)

            else:
                raise pickle.PicklingError(f"Unknown persistent ID type: {type_tag}")

    class Pickler(cloudpickle.Pickler):
        """Custom pickler to convert results back to metadata."""

        def __init__(
            self,
            file: Any,
            tensor_manager: TensorManager,
            unpickler_tensor_cache: dict[int, torch.Tensor],
        ) -> None:
            super().__init__(file)
            self.tensor_manager = tensor_manager
            # Create reverse mapping from tensors to object_ids (int from client | str from server)
            self.tensor_to_object_id: dict[torch.Tensor, int | str] = {}
            for object_id, tensor in unpickler_tensor_cache.items():
                self.tensor_to_object_id[tensor] = object_id

        def persistent_id(self, obj: Any) -> tuple[str, Any] | None:
            if isinstance(obj, torch.Tensor):
                # Register tensor and get metadata
                metadata = self.tensor_manager.register_temp_tensor(obj)

                # Determine object_id for deduplication
                if obj in self.tensor_to_object_id:
                    # Tensor from client - use client's object_id (int)
                    object_id = self.tensor_to_object_id[obj]
                else:
                    # New server-side tensor - create server object_id (str)
                    object_id = str(id(obj))
                    self.tensor_to_object_id[obj] = object_id

                # Get gradient (None or tensor, both pickle correctly)
                # Suppress warning about accessing .grad on non-leaf tensors
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*non-leaf Tensor.*")
                    grad = obj.grad

                # Return metadata, requires_grad, is_parameter, object_id, and grad
                return ("remote_tensor", {
                    "metadata": metadata,
                    "requires_grad": obj.requires_grad,
                    "is_parameter": isinstance(obj, torch.nn.Parameter),
                    "object_id": object_id,
                    "grad": grad  # None or tensor, pickled recursively via persistent_id
                })

            elif isinstance(obj, torch.device):
                return ("remote_device", (obj.type, obj.index))

            return None

    def _dtype_to_str(dtype) -> str:
        """Convert torch.dtype to string without 'torch.' prefix."""
        return str(dtype).replace("torch.", "")

    app = modal.App(f"mycelya-torch-{machine_id}")

    # Create image with synchronized packages and Python version
    image = modal.Image.debian_slim(python_version=python_version).uv_pip_install(
        *packages
    )

    cls_kwargs = {
        "image": image,
        "gpu": gpu_type,
        "retries": 0,
        "serialized": True,
        "max_containers": 1,
        "min_containers": 1,
    }
    if timeout:
        cls_kwargs["timeout"] = timeout

    # Only create volumes for remote execution
    if gpu_type != "local":
        # Create HuggingFace cache volume and mount at cache directory
        hf_cache_volume = modal.Volume.from_name(
            "mycelya-torch-huggingface-cache", create_if_missing=True
        )

        # Create data volume and mount at data directory
        data_volume = modal.Volume.from_name(
            "mycelya-torch-data", create_if_missing=True
        )

        # Create offload volume for model offloading
        offload_volume = modal.Volume.from_name(
            "mycelya-torch-offload", create_if_missing=True
        )

        cls_kwargs["volumes"] = {
            "/huggingface-cache": hf_cache_volume,
            "/data": data_volume,
            "/offload": offload_volume,
        }

    @app.cls(**cls_kwargs)
    class PytorchServer:
        @modal.enter()
        def setup(self) -> None:
            """Initialize the server when container starts."""
            # Use getattr to avoid pickling errors - torch.ops is an _Ops object that cannot be pickled
            self.torch_ops = getattr(torch, "ops")  # noqa: B009

            # Change to data directory and set HF cache if available (only when volumes are mounted)
            if gpu_type != "local":
                os.chdir("/data")
                # Set HuggingFace cache directory to mounted volume
                os.environ["HF_HOME"] = "/huggingface-cache"

            # Initialize tensor manager for all tensor-related operations
            self.tensor_manager = TensorManager()

            # Store machine_id for offload/reload operations
            self.machine_id = machine_id

            # Flag for offloading on exit (for preemption handling)
            self.offload_on_exit = True

            # Track all packages installed via pip_install
            self.installed_packages: list[str] = []

            # Check for preemption recovery files and reload if they exist
            if gpu_type != "local":
                state_filepath = f"/offload/{self.machine_id}_preempt.pt"
                packages_filepath = f"/offload/{self.machine_id}_preempt_packages.txt"

                # Reload packages first if available
                if os.path.exists(packages_filepath):
                    with open(packages_filepath, "r") as f:
                        packages_to_install = [line.strip() for line in f if line.strip()]
                    if packages_to_install:
                        self._pip_install_impl(packages_to_install)
                    # Delete packages file after installing
                    os.remove(packages_filepath)

                # Then reload tensors
                if os.path.exists(state_filepath):
                    self.tensor_manager.reload(state_filepath)

            # Method mapping for batch execution
            self._method_map = {
                "create_tensor": self._create_tensor_impl,
                "update_tensor": self._update_tensor_impl,
                "get_storage_data": self._get_storage_data_impl,
                "remove_tensors": self._remove_tensors_impl,
                "resize_storage": self._resize_storage_impl,
                "copy_tensor": self._copy_tensor_impl,
                "execute_aten_operation": self._execute_aten_operation_impl,
                "link_tensors": self._link_tensors_impl,
                "execute_function": self._execute_function_impl,
                "pip_install": self._pip_install_impl,
                "offload": self._offload_impl,
                "reload": self._reload_impl,
                "stop": self._stop_impl,
            }

        @modal.exit()
        def cleanup(self) -> None:
            """Cleanup when container shuts down, offload on preemption."""
            if self.offload_on_exit:
                # Preemption detected - offload tensors to disk
                state_filepath = f"/offload/{self.machine_id}_preempt.pt"
                self.tensor_manager.offload(state_filepath)

                # Save installed packages list if there are any
                if self.installed_packages:
                    packages_filepath = f"/offload/{self.machine_id}_preempt_packages.txt"
                    with open(packages_filepath, "w") as f:
                        f.write("\n".join(self.installed_packages))

        # Tensor ID-based methods
        def _create_tensor_impl(self, metadata: TensorMetadata) -> None:
            """Create a tensor based on metadata.

            Creates either a new empty tensor or a tensor view based on metadata.alias_id:
            - If alias_id is None: Creates new empty tensor
            - If alias_id is int: Creates tensor view using alias_id as base tensor
            """
            tensor_id = metadata["id"]
            alias_id = metadata.get("alias_id")

            if tensor_id in self.tensor_manager.tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} already exists")

            if alias_id is None:
                # Create empty tensor (equivalent to old _create_empty_tensor_impl)
                torch_dtype = getattr(torch, metadata["dtype"])

                # Use the explicit device type and index from the client
                device = torch.device(metadata["device_type"], metadata["device_index"])

                # Use the exact nbytes provided by the client allocator
                # The client has already calculated the correct storage size
                storage_nbytes = metadata["nbytes"]

                # Create untyped storage with the exact nbytes size
                untyped_storage = torch.UntypedStorage(storage_nbytes, device=device)

                # Create the tensor view with the specified layout
                tensor = torch.empty(0, dtype=torch_dtype, device=device).set_(
                    untyped_storage,
                    metadata["storage_offset"],
                    metadata["shape"],
                    metadata["stride"],
                )

                self.tensor_manager.register_tensor(tensor_id, tensor)
            else:
                # Create tensor view (equivalent to old _create_tensor_view_impl)
                if alias_id not in self.tensor_manager.tensor_registry:
                    raise ValueError(f"Base tensor ID {alias_id} does not exist")

                base_tensor = self.tensor_manager.tensor_registry[alias_id]

                # Get dtype from metadata and create view with correct dtype
                torch_dtype = getattr(torch, metadata["dtype"])

                # Create view with correct dtype using set_() for proper dtype handling
                view_tensor = torch.empty(0, dtype=torch_dtype, device=base_tensor.device).set_(
                    base_tensor.untyped_storage(),
                    metadata["storage_offset"],
                    metadata["shape"],
                    metadata["stride"],
                )

                self.tensor_manager.register_tensor(tensor_id, view_tensor)

        @modal.method()
        def create_tensor(self, metadata: TensorMetadata) -> None:
            """Create a tensor on the remote machine with proper storage layout."""
            self._create_tensor_impl(metadata)

        def _update_tensor_impl(
            self,
            tensor_id: int,
            raw_data: bytes,
            source_shape: list[int],
            source_stride: list[int],
            source_storage_offset: int,
            source_dtype: str,
        ) -> None:
            """Update an existing tensor with new data and source metadata."""

            # Use tensor manager instead of direct registry access

            if not tensor_id in self.tensor_manager.tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} does not exist")

            target_tensor = self.tensor_manager.tensor_registry[tensor_id]

            # Convert dtype string to torch.dtype
            torch_dtype = getattr(torch, source_dtype)

            # Create writable buffer to avoid PyTorch warnings
            writable_data = bytearray(raw_data)

            # Handle empty buffer as noop - no data to transfer
            if len(writable_data) == 0:
                # Empty buffer means no actual data to transfer, so this is a noop
                return

            # Reconstruct source tensor from raw data using provided metadata
            flat_tensor = torch.frombuffer(writable_data, dtype=torch_dtype)

            # Create source tensor with exact layout using as_strided
            source_tensor = flat_tensor.as_strided(
                source_shape, source_stride, source_storage_offset
            )

            # Move to target device and copy
            device_source = source_tensor.to(target_tensor.device)
            target_tensor.copy_(device_source)

        @modal.method()
        def update_tensor(
            self,
            tensor_id: int,
            raw_data: bytes,
            source_shape: list[int],
            source_stride: list[int],
            source_storage_offset: int,
            source_dtype: str,
        ) -> None:
            """Update an existing tensor with new data and source metadata."""
            self._update_tensor_impl(
                tensor_id,
                raw_data,
                source_shape,
                source_stride,
                source_storage_offset,
                source_dtype,
            )

        def _get_storage_data_impl(self, tensor_id: int) -> bytes:
            """Get raw storage data by tensor ID."""

            # Use tensor manager instead of direct registry access

            if not tensor_id in self.tensor_manager.tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} does not exist")

            tensor = self.tensor_manager.tensor_registry[tensor_id]
            # Get the underlying storage data, not just the tensor view
            storage = tensor.untyped_storage()
            # Create a tensor that views the entire storage as bytes (minimal allocation)
            full_tensor = torch.empty(0, dtype=torch.uint8, device=tensor.device)
            full_tensor.set_(
                storage, storage_offset=0, size=(storage.nbytes(),), stride=(1,)
            )
            result = full_tensor.cpu().detach().numpy().tobytes()

            return result

        @modal.method()
        def get_storage_data(self, tensor_id: int) -> bytes:
            """Get raw storage data by tensor ID."""
            return self._get_storage_data_impl(tensor_id)

        def _remove_tensors_impl(self, tensor_ids: list[int]) -> None:
            """Remove multiple tensors from the remote machine."""
            for tensor_id in tensor_ids:
                if tensor_id in self.tensor_manager.tensor_registry:
                    del self.tensor_manager.tensor_registry[tensor_id]

        @modal.method()
        def remove_tensors(self, tensor_ids: list[int]) -> None:
            """Remove multiple tensors from the remote machine."""
            self._remove_tensors_impl(tensor_ids)

        def _resize_storage_impl(self, tensor_id: int, nbytes: int) -> None:
            """Resize the underlying storage for a tensor."""

            # Use tensor manager instead of direct registry access

            if not tensor_id in self.tensor_manager.tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} does not exist")

            tensor = self.tensor_manager.tensor_registry[tensor_id]
            current_bytes = tensor.untyped_storage().nbytes()

            if nbytes <= current_bytes:
                return

            # Create temporary view and resize underlying storage
            temp_storage_tensor = torch.empty(
                0, dtype=torch.uint8, device=tensor.device
            )
            temp_storage_tensor.set_(tensor.untyped_storage(), 0, [current_bytes])
            temp_storage_tensor.resize_([nbytes])

        @modal.method()
        def resize_storage(self, tensor_id: int, nbytes: int) -> None:
            """Resize the underlying storage for a tensor."""
            self._resize_storage_impl(tensor_id, nbytes)

        def _copy_tensor_impl(
            self, source_tensor_id: int, target_tensor_id: int
        ) -> None:
            """Copy tensor data from source to target on the remote machine."""

            # Use tensor manager instead of direct registry access

            # Validate both tensors exist
            if source_tensor_id not in self.tensor_manager.tensor_registry:
                raise ValueError(f"Source tensor ID {source_tensor_id} does not exist")
            if target_tensor_id not in self.tensor_manager.tensor_registry:
                raise ValueError(f"Target tensor ID {target_tensor_id} does not exist")

            # Get tensors
            source_tensor = self.tensor_manager.tensor_registry[source_tensor_id]
            target_tensor = self.tensor_manager.tensor_registry[target_tensor_id]

            # Perform copy operation directly on the remote machine
            target_tensor.copy_(source_tensor)

        @modal.method()
        def copy_tensor(self, source_tensor_id: int, target_tensor_id: int) -> None:
            """Copy tensor data from source to target on the remote machine."""
            self._copy_tensor_impl(source_tensor_id, target_tensor_id)

        def _execute_aten_operation_impl(
            self,
            op_name: str,
            args: list[Any],
            kwargs: dict[str, Any],
            tensor_mask: list[bool],
            output_tensor_ids: list[int | None] | None = None,
        ) -> list[TensorMetadata] | None:
            """Implementation of execute_aten_operation without Modal decorators."""
            # Use tensor manager instead of direct registry access

            mask_iter = iter(tensor_mask)

            def process_item(obj: Any) -> Any:
                if isinstance(obj, (list, tuple)):
                    return type(obj)(
                        self.tensor_manager.tensor_registry[item].detach() if next(mask_iter) else item
                        for item in obj
                    )
                return self.tensor_manager.tensor_registry[obj].detach() if next(mask_iter) else obj

            processed_args = [process_item(arg) for arg in args]
            processed_kwargs = {k: process_item(v) for k, v in kwargs.items()}

            # Execute operation using cached torch_ops
            op = self.torch_ops
            op_parts = op_name.split(".")
            for part in op_parts:
                op = getattr(op, part)
            result = op(*processed_args, **processed_kwargs)

            # Normalize result to list
            result_tensors = (
                [result]
                if isinstance(result, torch.Tensor)
                else list(result)
                if isinstance(result, (list, tuple))
                else []
            )

            # Handle static vs dynamic operations
            if output_tensor_ids is None:
                # Dynamic: return metadata with IDs
                output_metadata = []
                for t in result_tensors:
                    # Register temp tensor and get metadata
                    metadata = self.tensor_manager.register_temp_tensor(t)
                    output_metadata.append(metadata)
                return output_metadata
            else:
                # Static: store in main registry
                # Skip None entries (common in backward operations where gradients aren't needed)
                for tid, tensor in zip(output_tensor_ids, result_tensors, strict=True):
                    if tid is not None and tensor is not None:
                        self.tensor_manager.register_tensor(tid, tensor)

        @modal.method()
        def execute_aten_operation(
            self,
            op_name: str,
            args: list[Any],
            kwargs: dict[str, Any],
            tensor_mask: list[bool],
            output_tensor_ids: list[int | None] | None = None,
        ) -> list[TensorMetadata] | None:
            """Execute an aten operation on the remote machine."""
            result = self._execute_aten_operation_impl(
                op_name,
                args,
                kwargs,
                tensor_mask,
                output_tensor_ids,
            )

            # Handle return format based on whether this is a dynamic operation
            if output_tensor_ids is None:
                # Dynamic operation: result is metadata list with id embedded
                return result
            else:
                # Static operation: no return value needed
                return None

        def _link_tensors_impl(
            self,
            tensor_ids: list[int],
            temp_ids: list[str],
        ) -> None:
            """Implementation of link_tensors without Modal decorators."""

            if len(tensor_ids) != len(temp_ids):
                raise ValueError(
                    f"Mismatch between tensor IDs ({len(tensor_ids)}) and temp IDs ({len(temp_ids)})"
                )

            # Use tensor manager for linking tensors
            self.tensor_manager.link_tensors(tensor_ids, temp_ids)

        @modal.method()
        def link_tensors(
            self,
            tensor_ids: list[int],
            temp_ids: list[str],
        ) -> None:
            """
            Link local mycelya tensor IDs to remote tensors from temporary registry.

            This method establishes linkage between local tensor IDs and remote tensors
            that were previously stored in the temporary registry.

            Args:
                tensor_ids: List of local tensor IDs from created mycelya tensors
                temp_ids: List of temporary registry IDs corresponding to each tensor ID
            """
            self._link_tensors_impl(tensor_ids, temp_ids)

        def _execute_function_impl(self, pickled_function: bytes) -> bytes:
            """Implementation of execute_function without Modal decorators."""
            # Unpickle the function bundle using tensor manager
            buffer = io.BytesIO(pickled_function)
            unpickler = Unpickler(buffer, self.tensor_manager)
            func_bundle = unpickler.load()

            # Extract function and arguments
            func = func_bundle["function"]
            args = func_bundle["args"]
            kwargs = func_bundle["kwargs"]

            # Execute the function directly (CloudPickle handles the function properly)
            result = func(*args, **kwargs)

            # Pickle the result bundle with args and kwargs
            result_bundle = {
                "result": result,
                "args": args,
                "kwargs": kwargs,
            }
            result_buffer = io.BytesIO()
            pickler = Pickler(result_buffer, self.tensor_manager, unpickler.tensor_cache)
            pickler.dump(result_bundle)

            return result_buffer.getvalue()

        @modal.method()
        def execute_function(self, pickled_function: bytes) -> bytes:
            """Execute a pickled function on the remote machine."""
            return self._execute_function_impl(pickled_function)

        def _pip_install_impl(self, packages: list[str]) -> None:
            """Install packages using uv pip on the remote machine."""
            # Skip pip install for local/mock execution
            if gpu_type == "local":
                return

            if not packages:
                return

            # Use uv pip install with --system flag
            cmd = ["uv", "pip", "install", "--system"] + packages
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            # Track installed packages
            self.installed_packages.extend(packages)

        @modal.method()
        def pip_install(self, packages: list[str]) -> None:
            """Install packages using pip on the remote machine."""
            self._pip_install_impl(packages)

        def _offload_impl(self) -> None:
            """Offload tensor registry to disk."""
            # Skip offload for local/mock execution
            if gpu_type == "local":
                return

            filepath = f"/offload/{self.machine_id}.pt"
            self.tensor_manager.offload(filepath)

        @modal.method()
        def offload(self) -> None:
            """Offload tensor registry to disk."""
            self._offload_impl()

        def _reload_impl(self) -> None:
            """Reload tensor registry from disk."""
            # Skip reload for local/mock execution
            if gpu_type == "local":
                return

            filepath = f"/offload/{self.machine_id}.pt"
            self.tensor_manager.reload(filepath)

        @modal.method()
        def reload(self) -> None:
            """Reload tensor registry from disk."""
            self._reload_impl()

        def _stop_impl(self) -> None:
            """Stop the server and disable offload on exit."""
            self.offload_on_exit = False

        @modal.method()
        def stop(self) -> None:
            """Stop the server and disable offload on exit."""
            self._stop_impl()

        @modal.method()
        def execute_batch(self, batch_calls: list[BatchCall]) -> list[Any]:
            """
            Execute a batch of RPCs in sequence.

            This method allows multiple operations to be batched together in a single
            RPC, reducing network overhead and improving performance.

            Args:
                batch_calls: List of BatchCall TypedDict objects, each containing:
                    - method_name: Name of the method to call
                    - args: Arguments for the method
                    - kwargs: Keyword arguments for the method

            Returns:
                List of non-None return values from the batched operations
            """
            results = []
            for call in batch_calls:
                method_name = call["method_name"]
                args = call.get("args", ())
                kwargs = call.get("kwargs", {})

                # Look up the method implementation
                method_impl = self._method_map[method_name]

                # Call the implementation and collect any return values
                result = method_impl(*args, **kwargs)
                if result is not None:
                    results.append(result)

            # Always return a list (empty if no results)
            return results

    return app, PytorchServer
