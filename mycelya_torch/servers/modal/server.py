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
import uuid
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import modal
import torch
import cloudpickle
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file as load_safetensors


def create_modal_app_for_gpu(
    gpu_type: str,
    packages: List[str],
    python_version: str,
    timeout: Optional[int] = None,
) -> Tuple[Any, Any]:
    """
    Create a Modal app and class for a specific GPU type.

    Args:
        gpu_type: The GPU type (e.g., "T4", "A100", "local" for local execution)
        timeout: Function timeout in seconds (defaults to 3600 if None)
        packages: List of versioned packages to install (e.g., ["torch==2.0.0", "numpy==1.24.0"])
        python_version: Python version for base image (e.g., "3.11")

    Returns:
        Tuple of (modal_app, server_class) for the specified GPU type
    """
    class BatchCall(TypedDict):
        """Structure for a single batched RPC call."""

        method_name: str
        args: Tuple[Any, ...]
        kwargs: Dict[str, Any]

    class TensorMetadata(TypedDict):
        """Structure for tensor metadata with temp key.

        This TypedDict defines the structure returned by dynamic operations
        that need to pass tensor metadata along with a temporary key for
        linking local tensors to remote tensors.
        """

        shape: List[int]
        stride: List[int]
        dtype: str
        storage_offset: int
        nbytes: int
        device_type: str
        device_index: int
        requires_grad: bool
        temp_key: str

    class Unpickler(pickle.Unpickler):
        """Custom unpickler to reconstruct tensors from IDs."""

        def __init__(self, file, tensor_registry):
            super().__init__(file)
            self.tensor_registry = tensor_registry

        def persistent_load(self, pid):
            type_tag, data = pid

            if type_tag == "mycelya_tensor":
                tensor_id = data
                if tensor_id not in self.tensor_registry:
                    raise ValueError(f"Tensor ID {tensor_id} not found in remote registry")
                return self.tensor_registry[tensor_id]

            elif type_tag == "mycelya_device":
                remote_type, remote_index = data
                return torch.device(remote_type, remote_index)

            else:
                raise pickle.PicklingError(f"Unknown persistent ID type: {type_tag}")

    class Pickler(cloudpickle.Pickler):
        """Custom pickler to convert results back to metadata."""

        def __init__(self, file, temp_tensor_registry):
            super().__init__(file)
            self.temp_tensor_registry = temp_tensor_registry

        def persistent_id(self, obj):
            if isinstance(obj, torch.Tensor):
                # Generate unique temp key
                temp_key = f"remote_result_{uuid.uuid4().hex[:8]}"

                # Register tensor in temp registry
                self.temp_tensor_registry[temp_key] = obj

                # Create metadata for client reconstruction
                metadata = {
                    "shape": list(obj.shape),
                    "stride": list(obj.stride()),
                    "dtype": str(obj.dtype).replace("torch.", ""),
                    "storage_offset": obj.storage_offset(),
                    "nbytes": obj.untyped_storage().nbytes(),
                    "device_type": obj.device.type,
                    "device_index": obj.device.index if obj.device.index is not None else 0,
                    "requires_grad": obj.requires_grad,
                    "temp_key": temp_key,
                }

                return ("remote_tensor", metadata)

            elif isinstance(obj, torch.device):
                return ("remote_device", (obj.type, obj.index))

            return None

    app = modal.App("mycelya-torch")

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

        cls_kwargs["volumes"] = {
            "/huggingface-cache": hf_cache_volume,
            "/data": data_volume,
        }

    @app.cls(**cls_kwargs)
    class PytorchServer:
        @staticmethod
        def _dtype_to_str(dtype) -> str:
            """Convert torch.dtype to string without 'torch.' prefix."""
            return str(dtype).replace("torch.", "")

        @modal.enter()
        def setup(self):
            """Initialize the server when container starts."""
            # Cache torch.ops using getattr to avoid serialization issues with _Ops object
            self.torch_ops = getattr(torch, "ops")

            # Change to data directory and set HF cache if available (only when volumes are mounted)
            if gpu_type != "local":
                os.chdir("/data")
                # Set HuggingFace cache directory to mounted volume
                os.environ["HF_HOME"] = "/huggingface-cache"

            # Initialize registries only (device detection done per-method to avoid serialization issues)
            # tensor_id -> torch.Tensor (direct mapping from tensor ID to tensor)
            self._tensor_registry = {}

            # checkpoint -> {model: nn.Module, parameter_tensor_map: Dict[str, int]}
            self._model_registry = {}

            # Temporary tensor registry: string_key -> torch.Tensor (for operations that create tensors remotely first)
            self._temp_tensor_registry = {}

            # Method mapping for batch execution
            self._method_map = {
                "create_empty_tensor": self._create_empty_tensor_impl,
                "create_tensor_view": self._create_tensor_view_impl,
                "update_tensor": self._update_tensor_impl,
                "get_storage_data": self._get_storage_data_impl,
                "remove_tensors": self._remove_tensors_impl,
                "resize_storage": self._resize_storage_impl,
                "copy_tensor": self._copy_tensor_impl,
                "execute_aten_operation": self._execute_aten_operation_impl,
                "load_huggingface_state_dicts": self._load_huggingface_state_dicts_impl,
                "link_tensors": self._link_tensors_impl,
                "execute_function": self._execute_function_impl,
            }

        @modal.exit()
        def cleanup(self):
            """Cleanup when container shuts down (no-op for now)."""
            pass

        # Tensor ID-based methods
        def _create_empty_tensor_impl(
            self,
            tensor_id: int,
            shape: List[int],
            stride: List[int],
            storage_offset: int,
            dtype: str,
            nbytes: int,
            device_type: str,
            device_index: int,
        ):
            """Create an empty tensor with given tensor_id and proper storage layout."""

            tensor_registry = self._tensor_registry

            if tensor_id in tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} already exists")

            torch_dtype = getattr(torch, dtype)

            # Use the explicit device type and index from the client
            device = torch.device(device_type, device_index)

            # Use the exact nbytes provided by the client allocator
            # The client has already calculated the correct storage size
            storage_nbytes = nbytes

            # Create untyped storage with the exact nbytes size
            untyped_storage = torch.UntypedStorage(storage_nbytes, device=device)

            # Create the tensor view with the specified layout
            tensor = torch.empty(0, dtype=torch_dtype, device=device).set_(
                untyped_storage, storage_offset, shape, stride
            )

            tensor_registry[tensor_id] = tensor

        @modal.method()
        def create_empty_tensor(
            self,
            tensor_id: int,
            shape: List[int],
            stride: List[int],
            storage_offset: int,
            dtype: str,
            nbytes: int,
            device_type: str,
            device_index: int,
        ):
            """Create an empty tensor on the remote machine with proper storage layout."""
            self._create_empty_tensor_impl(
                tensor_id,
                shape,
                stride,
                storage_offset,
                dtype,
                nbytes,
                device_type,
                device_index,
            )

        def _create_tensor_view_impl(
            self,
            new_tensor_id: int,
            base_tensor_id: int,
            shape: List[int],
            stride: List[int],
            offset: int,
        ):
            """Create a tensor view from existing tensor using as_strided."""

            tensor_registry = self._tensor_registry

            if new_tensor_id in tensor_registry:
                raise ValueError(f"New tensor ID {new_tensor_id} already exists")

            if base_tensor_id not in tensor_registry:
                raise ValueError(f"Base tensor ID {base_tensor_id} does not exist")

            base_tensor = tensor_registry[base_tensor_id]

            # Create view using as_strided directly on the base tensor
            view_tensor = torch.as_strided(base_tensor, shape, stride, offset)

            tensor_registry[new_tensor_id] = view_tensor

        @modal.method()
        def create_tensor_view(
            self,
            new_tensor_id: int,
            base_tensor_id: int,
            shape: List[int],
            stride: List[int],
            offset: int,
        ):
            """Create a tensor view from existing tensor using as_strided."""
            self._create_tensor_view_impl(
                new_tensor_id, base_tensor_id, shape, stride, offset
            )

        def _update_tensor_impl(
            self,
            tensor_id: int,
            raw_data: bytes,
            source_shape: List[int],
            source_stride: List[int],
            source_storage_offset: int,
            source_dtype: str,
        ):
            """Update an existing tensor with new data and source metadata."""

            tensor_registry = self._tensor_registry

            if tensor_id not in tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} does not exist")

            target_tensor = tensor_registry[tensor_id]

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
            source_shape: List[int],
            source_stride: List[int],
            source_storage_offset: int,
            source_dtype: str,
        ):
            """Update an existing tensor with new data and source metadata."""
            self._update_tensor_impl(
                tensor_id,
                raw_data,
                source_shape,
                source_stride,
                source_storage_offset,
                source_dtype,
            )

        def _get_storage_data_impl(self, tensor_id: int):
            """Get raw storage data by tensor ID."""

            tensor_registry = self._tensor_registry

            if tensor_id not in tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} does not exist")

            tensor = tensor_registry[tensor_id]
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
        def get_storage_data(self, tensor_id: int):
            """Get raw storage data by tensor ID."""
            return self._get_storage_data_impl(tensor_id)

        def _remove_tensors_impl(self, tensor_ids: List[int]):
            """Remove multiple tensors from the remote machine."""
            tensor_registry = self._tensor_registry

            for tensor_id in tensor_ids:
                if tensor_id in tensor_registry:
                    del tensor_registry[tensor_id]

        @modal.method()
        def remove_tensors(self, tensor_ids: List[int]):
            """Remove multiple tensors from the remote machine."""
            self._remove_tensors_impl(tensor_ids)

        def _resize_storage_impl(self, tensor_id: int, nbytes: int):
            """Resize the underlying storage for a tensor."""

            tensor_registry = self._tensor_registry

            if tensor_id not in tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} does not exist")

            tensor = tensor_registry[tensor_id]
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
        def resize_storage(self, tensor_id: int, nbytes: int):
            """Resize the underlying storage for a tensor."""
            self._resize_storage_impl(tensor_id, nbytes)

        def _copy_tensor_impl(self, source_tensor_id: int, target_tensor_id: int):
            """Copy tensor data from source to target on the remote machine."""

            tensor_registry = self._tensor_registry

            # Validate both tensors exist
            if source_tensor_id not in tensor_registry:
                raise ValueError(f"Source tensor ID {source_tensor_id} does not exist")
            if target_tensor_id not in tensor_registry:
                raise ValueError(f"Target tensor ID {target_tensor_id} does not exist")

            # Get tensors
            source_tensor = tensor_registry[source_tensor_id]
            target_tensor = tensor_registry[target_tensor_id]

            # Perform copy operation directly on the remote machine
            target_tensor.copy_(source_tensor)

        @modal.method()
        def copy_tensor(self, source_tensor_id: int, target_tensor_id: int):
            """Copy tensor data from source to target on the remote machine."""
            self._copy_tensor_impl(source_tensor_id, target_tensor_id)

        def _execute_aten_operation_impl(
            self,
            op_name: str,
            args: List[Any],
            kwargs: Dict[str, Any],
            tensor_mask: List[bool],
            output_tensor_ids: Optional[List[int]] = None,
        ):
            """Implementation of execute_aten_operation without Modal decorators."""
            tensor_registry = self._tensor_registry

            mask_iter = iter(tensor_mask)

            def process_item(obj):
                if isinstance(obj, (list, tuple)):
                    return type(obj)(
                        tensor_registry[item] if next(mask_iter) else item
                        for item in obj
                    )
                return tensor_registry[obj] if next(mask_iter) else obj

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
                # Dynamic: return metadata with temp keys
                temp_registry = self._temp_tensor_registry
                output_metadata = []
                for i, t in enumerate(result_tensors):
                    temp_key = f"temp_{op_name}_{uuid.uuid4().hex[:8]}_{i}"
                    temp_registry[temp_key] = t
                    output_metadata.append(
                        TensorMetadata(
                            shape=list(t.shape),
                            stride=list(t.stride()),
                            dtype=self._dtype_to_str(t.dtype),
                            storage_offset=t.storage_offset(),
                            nbytes=t.untyped_storage().nbytes(),
                            device_type=t.device.type,
                            device_index=t.device.index if t.device.index is not None else 0,
                            requires_grad=t.requires_grad,
                            temp_key=temp_key,
                        )
                    )
                return output_metadata
            else:
                # Static: store in main registry
                for tid, tensor in zip(output_tensor_ids, result_tensors, strict=True):
                    tensor_registry[tid] = tensor

        @modal.method()
        def execute_aten_operation(
            self,
            op_name: str,
            args: List[Any],
            kwargs: Dict[str, Any],
            tensor_mask: List[bool],
            output_tensor_ids: Optional[List[int]] = None,
        ):
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
                # Dynamic operation: result is metadata list with temp_key embedded
                return result
            else:
                # Static operation: no return value needed
                return None

        def _load_huggingface_state_dicts_impl(
            self, repo: str, path: str, device_type: str, device_index: int
        ):
            """Load HuggingFace model weights organized by directory and store with temporary keys."""

            device = torch.device(device_type, device_index)

            # Find weight files
            files = list_repo_files(repo)
            if path:
                files = [f for f in files if f.startswith(path)]

            safetensor_files = [f for f in files if f.endswith(".safetensors")]
            pytorch_files = [
                f for f in files if f.endswith(".bin") and "pytorch_model" in f
            ]

            # Group files by directory
            file_dirs = {}  # directory -> list of files
            all_weight_files = safetensor_files if safetensor_files else pytorch_files

            if not all_weight_files:
                path_info = f" in path '{path}'" if path else ""
                raise RuntimeError(f"No weight files found in {repo}{path_info}")

            for weight_file in all_weight_files:
                file_dir = os.path.dirname(weight_file) or ""  # Empty string for root
                if file_dir not in file_dirs:
                    file_dirs[file_dir] = []
                file_dirs[file_dir].append(weight_file)

            # Always return hierarchical structure for consistency
            hierarchical_metadata = {}

            for directory, dir_files in file_dirs.items():
                dir_state_dict = {}

                # Load all files in this directory
                for weight_file in dir_files:
                    file_path = hf_hub_download(repo, weight_file)
                    if weight_file.endswith(".safetensors"):
                        file_state_dict = load_safetensors(file_path, device="cpu")
                        file_state_dict = {
                            k: v.to(device) for k, v in file_state_dict.items()
                        }
                    else:  # pytorch files
                        file_state_dict = torch.load(file_path, map_location=device)
                    dir_state_dict.update(file_state_dict)

                # Create metadata for this directory's tensors
                dir_metadata = {}
                for name, tensor in dir_state_dict.items():
                    temp_key = f"hf_{repo}_{directory}_{name}_{uuid.uuid4().hex[:8]}"
                    self._temp_tensor_registry[temp_key] = tensor
                    dir_metadata[name] = {
                        "shape": list(tensor.shape),
                        "stride": list(tensor.stride()),
                        "dtype": self._dtype_to_str(tensor.dtype),
                        "storage_offset": tensor.storage_offset(),
                        "nbytes": tensor.untyped_storage().nbytes(),
                        "device_type": tensor.device.type,
                        "device_index": tensor.device.index if tensor.device.index is not None else 0,
                        "requires_grad": tensor.requires_grad,
                        "temp_key": temp_key,
                    }

                # Use directory name as key (empty string for root)
                hierarchical_metadata[directory] = dir_metadata

            return hierarchical_metadata

        @modal.method()
        def load_huggingface_state_dicts(
            self,
            repo: str,
            path: str,
            device_type: str,
            device_index: int,
        ):
            """Download and prepare HuggingFace model weights organized by directory."""
            return self._load_huggingface_state_dicts_impl(
                repo, path, device_type, device_index
            )

        def _link_tensors_impl(
            self,
            local_tensor_ids: List[int],
            temp_keys: List[str],
        ):
            """Implementation of link_tensors without Modal decorators."""

            if len(local_tensor_ids) != len(temp_keys):
                raise ValueError(
                    f"Mismatch between tensor IDs ({len(local_tensor_ids)}) and temp keys ({len(temp_keys)})"
                )

            tensor_registry = self._tensor_registry
            temp_tensor_registry = self._temp_tensor_registry

            for local_tensor_id, temp_key in zip(local_tensor_ids, temp_keys):
                if temp_key not in temp_tensor_registry:
                    raise KeyError(
                        f"Temporary tensor key '{temp_key}' not found in temporary registry"
                    )

                # Get the tensor from temporary registry
                remote_tensor = temp_tensor_registry[temp_key]

                # Link the local tensor ID to the remote tensor in the main registry
                tensor_registry[local_tensor_id] = remote_tensor

                # Remove from temporary registry after linking
                del temp_tensor_registry[temp_key]

        @modal.method()
        def link_tensors(
            self,
            local_tensor_ids: List[int],
            temp_keys: List[str],
        ):
            """
            Link local mycelya tensor IDs to remote tensors from temporary registry.

            This method establishes linkage between local tensor IDs and remote tensors
            that were previously stored in the temporary registry.

            Args:
                local_tensor_ids: List of local tensor IDs from created mycelya tensors
                temp_keys: List of temporary registry keys corresponding to each tensor ID
            """
            self._link_tensors_impl(local_tensor_ids, temp_keys)

        def _execute_function_impl(self, pickled_function: bytes) -> bytes:
            """Implementation of execute_function without Modal decorators."""
            tensor_registry = self._tensor_registry
            temp_tensor_registry = self._temp_tensor_registry

            # Unpickle the function bundle
            buffer = io.BytesIO(pickled_function)
            unpickler = Unpickler(buffer, tensor_registry)
            func_bundle = unpickler.load()

            # Extract function and arguments
            func = func_bundle["function"]
            args = func_bundle["args"]
            kwargs = func_bundle["kwargs"]

            # Execute the function directly (CloudPickle handles the function properly)
            result = func(*args, **kwargs)

            # Pickle the result
            result_buffer = io.BytesIO()
            pickler = Pickler(result_buffer, temp_tensor_registry)
            pickler.dump(result)

            return result_buffer.getvalue()

        @modal.method()
        def execute_function(self, pickled_function: bytes) -> bytes:
            """Execute a pickled function on the remote machine."""
            return self._execute_function_impl(pickled_function)

        @modal.method()
        def execute_batch(self, batch_calls: List[BatchCall]):
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
