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

from typing import Any, Dict, List, Optional, Tuple, TypedDict

import modal

# Create image with PyTorch, CUDA support, and transformers for HuggingFace models
image = modal.Image.debian_slim().pip_install(
    "numpy", "torch", "transformers", "huggingface_hub", "safetensors", "accelerate"
)


def create_modal_app_for_gpu(
    gpu_type: str,
    timeout: int,
) -> Tuple[modal.App, Any]:
    """
    Create a Modal app and class for a specific GPU type.

    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB")
        timeout: Function timeout in seconds

    Returns:
        Tuple of (modal_app, server_class) for the specified GPU type
    """
    app = modal.App("mycelya-torch")

    @app.cls(
        image=image,
        gpu=gpu_type,
        timeout=timeout,
        retries=0,
        serialized=True,
        max_containers=1,
        min_containers=1,
    )
    class PytorchServer:
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
            dtype: str
            stride: List[int]
            storage_offset: int
            nbytes: int
            temp_key: str

        @staticmethod
        def _dtype_to_str(dtype) -> str:
            """Convert torch.dtype to string without 'torch.' prefix."""
            return str(dtype).replace("torch.", "")

        @modal.enter()
        def setup(self):
            """Initialize the server when container starts."""
            import torch

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
                "prepare_huggingface_model": self._prepare_huggingface_model_impl,
                "link_tensors": self._link_tensors_impl,
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
            import torch

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
            import torch

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
            import torch

            tensor_registry = self._tensor_registry

            if tensor_id not in tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} does not exist")

            target_tensor = tensor_registry[tensor_id]

            # Convert dtype string to torch.dtype
            torch_dtype = getattr(torch, source_dtype)

            # Create writable buffer to avoid PyTorch warnings
            writable_data = bytearray(raw_data)

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
            import torch

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
            import torch

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
            import uuid

            import torch

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

            # Execute operation
            op = torch.ops
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
                        self.TensorMetadata(
                            shape=list(t.shape),
                            dtype=self._dtype_to_str(t.dtype),
                            stride=list(t.stride()),
                            storage_offset=t.storage_offset(),
                            nbytes=t.untyped_storage().nbytes(),
                            temp_key=temp_key,
                        )
                    )
                return output_metadata
            else:
                # Static: store in main registry
                if len(output_tensor_ids) != len(result_tensors):
                    raise RuntimeError(f"Output tensor count mismatch in {op_name}")
                for tid, tensor in zip(output_tensor_ids, result_tensors):
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

        def _prepare_huggingface_model_impl(
            self,
            checkpoint: str,
            torch_dtype: str = "auto",
            trust_remote_code: bool = False,
        ):
            """Implementation of prepare_huggingface_model without Modal decorators."""
            import torch

            try:
                from transformers import AutoModelForCausalLM
            except ImportError:
                raise ImportError(
                    "transformers library required for HuggingFace model loading. "
                    "Add 'transformers' to the Modal image dependencies."
                )

            # Get the appropriate device for tensor operations
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Handle torch_dtype parameter
            if torch_dtype == "auto" or torch_dtype is None:
                torch_dtype_obj = (
                    torch.float16 if device.type == "cuda" else torch.float32
                )
            elif torch_dtype.startswith("torch."):
                # Handle string like "torch.float32"
                dtype_name = torch_dtype.split(".")[
                    -1
                ]  # Get "float32" from "torch.float32"
                torch_dtype_obj = getattr(torch, dtype_name)
            else:
                # Handle string like "float32"
                torch_dtype_obj = getattr(torch, torch_dtype)

            # Load model directly to appropriate device
            if device.type == "cpu":
                # For CPU/mock execution, don't use device_map (requires accelerate)
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint,
                    torch_dtype=torch_dtype_obj,
                    trust_remote_code=trust_remote_code,
                )
                model = model.to(device)
            else:
                # For GPU execution, use device_map
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint,
                    torch_dtype=torch_dtype_obj,
                    device_map={"": device},  # Load directly to our device
                    trust_remote_code=trust_remote_code,
                )

            # Extract state dict metadata without transferring data
            state_dict_metadata = {}

            for name, param in model.named_parameters():
                # Collect metadata for client (no storage ID generation here)
                state_dict_metadata[name] = {
                    "shape": list(param.shape),
                    "stride": list(param.stride()),
                    "dtype": self._dtype_to_str(param.dtype),
                    "storage_offset": param.storage_offset(),
                    "requires_grad": param.requires_grad,
                    # Store actual tensor data for later linking
                    "_tensor_data": param.detach().contiguous().to(device),
                }

            # Also handle buffers (non-trainable parameters like batch norm running stats)
            buffer_metadata = {}
            for name, buffer in model.named_buffers():
                # Collect metadata for client (no storage ID generation here)
                buffer_metadata[name] = {
                    "shape": list(buffer.shape),
                    "stride": list(buffer.stride()),
                    "dtype": self._dtype_to_str(buffer.dtype),
                    "storage_offset": buffer.storage_offset(),
                    "requires_grad": False,  # Buffers don't require gradients
                    # Store actual tensor data for later linking
                    "_tensor_data": buffer.detach().contiguous().to(device),
                }

            # Store model and tensor data for later linking
            model_registry = self._model_registry
            model_registry[checkpoint] = {
                "model": model,
                "parameter_tensors": {
                    name: metadata["_tensor_data"]
                    for name, metadata in state_dict_metadata.items()
                },
                "buffer_tensors": {
                    name: metadata["_tensor_data"]
                    for name, metadata in buffer_metadata.items()
                },
            }

            # Clean metadata for return (remove internal tensor data)
            clean_state_dict = {
                name: {k: v for k, v in meta.items() if k != "_tensor_data"}
                for name, meta in state_dict_metadata.items()
            }
            clean_buffer_dict = {
                name: {k: v for k, v in meta.items() if k != "_tensor_data"}
                for name, meta in buffer_metadata.items()
            }

            result = {
                "state_dict_metadata": clean_state_dict,
                "buffer_metadata": clean_buffer_dict,
                "config": model.config.to_dict(),
                "model_type": type(model).__name__,
                "checkpoint": checkpoint,
            }

            return result

        @modal.method()
        def prepare_huggingface_model(
            self,
            checkpoint: str,
            torch_dtype: str = "auto",
            trust_remote_code: bool = False,
        ):
            """Download and prepare a HuggingFace model directly on the remote machine."""
            return self._prepare_huggingface_model_impl(
                checkpoint, torch_dtype, trust_remote_code
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

        @modal.method()
        def execute_batch(self, batch_calls: List["PytorchServer.BatchCall"]):
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
