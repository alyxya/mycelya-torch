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

from typing import Any, Dict, List, Tuple, TypedDict

import modal

# Create image with PyTorch, CUDA support, and transformers for HuggingFace models
image = modal.Image.debian_slim().pip_install(
    "numpy", "torch", "transformers", "huggingface_hub", "safetensors", "accelerate"
)


def create_modal_app_for_gpu(
    gpu_type: str,
    machine_id: str,
    timeout: int,
) -> Tuple[modal.App, Any, modal.Queue]:
    """
    Create a Modal app and class for a specific GPU type and device.

    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB")
        machine_id: The machine ID (e.g., "modal-t4-f3a7d67e")
        timeout: Function timeout in seconds

    Returns:
        Tuple of (modal_app, server_class, response_queue) for the specified device
    """
    app = modal.App(f"mycelya-torch-{machine_id}")

    # Create a Modal Queue for responses (for external access)
    response_queue = modal.Queue.from_name(
        f"responses-{machine_id}", create_if_missing=True
    )

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

        @staticmethod
        def _dtype_to_str(dtype) -> str:
            """Convert torch.dtype to string without 'torch.' prefix."""
            return str(dtype).replace("torch.", "")

        @modal.enter()
        def setup(self):
            """Initialize the server when container starts."""

            import modal

            # Create the response queue when container starts
            self.response_queue = modal.Queue.from_name(
                f"responses-{machine_id}", create_if_missing=True
            )

            # Initialize registries only (device detection done per-method to avoid serialization issues)
            # tensor_id -> torch.Tensor (direct mapping from tensor ID to tensor)
            self._tensor_registry = {}

            # checkpoint -> {model: nn.Module, parameter_tensor_map: Dict[str, int]}
            self._model_registry = {}

            # Method mapping for batch execution
            self._method_map = {
                "create_empty_tensor": self._create_empty_tensor_impl,
                "create_tensor_view": self._create_tensor_view_impl,
                "update_tensor": self._update_tensor_impl,
                "get_storage_data": self._get_storage_data_impl,
                "remove_tensors": self._remove_tensors_impl,
                "resize_storage": self._resize_storage_impl,
                "execute_aten_operation": self._execute_aten_operation_impl,
                "prepare_huggingface_model": self._prepare_huggingface_model_impl,
                "link_model_tensors": self._link_model_tensors_impl,
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

        def _execute_aten_operation_impl(
            self,
            op_name: str,
            input_tensor_ids: List[int],
            output_tensor_ids: List[int],
            args: List[Any],
            kwargs: Dict[str, Any],
            tensor_mask: List[bool],
            return_metadata: bool = False,
        ):
            """Implementation of execute_aten_operation without Modal decorators."""
            import torch

            # Get tensor registry
            tensor_registry = self._tensor_registry

            # Reconstruct input tensors from tensor registry
            input_tensors = []
            for tensor_id in input_tensor_ids:
                # Check if tensor exists in registry
                if tensor_id in tensor_registry:
                    tensor = tensor_registry[tensor_id]
                else:
                    # Tensor missing from registry - this shouldn't happen with proper orchestrator management
                    raise ValueError(
                        f"Input tensor ID {tensor_id} not found in registry"
                    )

                input_tensors.append(tensor)

            # Replace tensor IDs with actual tensors using tensor mask
            tensor_index = 0
            mask_index = 0

            def replace_tensor_id_with_tensor(obj):
                """Replace tensor ID with actual tensor, consuming mask once per call."""
                nonlocal tensor_index, mask_index

                if mask_index >= len(tensor_mask):
                    raise IndexError(
                        f"Mask index {mask_index} out of range (have {len(tensor_mask)} mask entries)"
                    )

                is_tensor = tensor_mask[mask_index]
                mask_index += 1

                if is_tensor:
                    # Expecting a tensor ID (integer), replace with actual tensor
                    if tensor_index < len(input_tensors):
                        result = input_tensors[tensor_index]
                        tensor_index += 1
                        return result
                    else:
                        raise IndexError(
                            f"Tensor index {tensor_index} out of range (have {len(input_tensors)} input tensors)"
                        )
                return obj

            # Iterate over args (matching client-side structure exactly)
            processed_args = []
            for arg in args:
                if isinstance(arg, (list, tuple)):
                    processed_arg = []
                    for item in arg:
                        processed_arg.append(replace_tensor_id_with_tensor(item))
                    processed_args.append(type(arg)(processed_arg))
                else:
                    processed_args.append(replace_tensor_id_with_tensor(arg))

            # Iterate over kwargs values (matching client-side structure exactly)
            processed_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, (list, tuple)):
                    processed_value = []
                    for item in value:
                        processed_value.append(replace_tensor_id_with_tensor(item))
                    processed_kwargs[key] = type(value)(processed_value)
                else:
                    processed_kwargs[key] = replace_tensor_id_with_tensor(value)

            # Get the operation
            op_parts = op_name.split(".")
            op = torch.ops
            for part in op_parts:
                op = getattr(op, part)

            # Execute the operation on input tensors - this will create result tensors
            result = op(*processed_args, **processed_kwargs)

            # Handle output tensors
            result_tensors = (
                [result]
                if isinstance(result, torch.Tensor)
                else list(result)
                if isinstance(result, (list, tuple))
                else []
            )

            # Enforce contract: output_tensor_ids and result_tensors must have same length
            if len(output_tensor_ids) != len(result_tensors):
                raise RuntimeError(
                    f"Contract violation in {op_name}: output_tensor_ids length ({len(output_tensor_ids)}) "
                    f"!= result_tensors length ({len(result_tensors)}). This indicates a bug in the client-side "
                    f"meta tensor execution or output tensor creation."
                )

            # Store result tensors in tensor registry
            for tensor_id, result_tensor in zip(output_tensor_ids, result_tensors):
                tensor_registry[tensor_id] = result_tensor

            # Return metadata if requested
            if return_metadata:
                output_metadata = []
                for i, result_tensor in enumerate(result_tensors):
                    if i < len(output_tensor_ids):
                        metadata = {
                            "shape": list(result_tensor.shape),
                            "dtype": self._dtype_to_str(result_tensor.dtype),
                            "stride": list(result_tensor.stride()),
                            "storage_offset": result_tensor.storage_offset(),
                            "storage_nelements": result_tensor.untyped_storage().nbytes()
                            // result_tensor.element_size(),
                        }
                        output_metadata.append(metadata)

                return output_metadata

        @modal.method()
        def execute_aten_operation(
            self,
            op_name: str,
            input_tensor_ids: List[int],
            output_tensor_ids: List[int],
            args: List[Any],
            kwargs: Dict[str, Any],
            tensor_mask: List[bool],
            return_metadata: bool = False,
        ):
            """Execute an aten operation on the remote machine with input tensor IDs."""
            return self._execute_aten_operation_impl(
                op_name,
                input_tensor_ids,
                output_tensor_ids,
                args,
                kwargs,
                tensor_mask,
                return_metadata,
            )

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

        def _link_model_tensors_impl(
            self,
            local_tensor_ids: List[int],
            parameter_names: List[str],
        ):
            """Implementation of link_model_tensors without Modal decorators."""

            if len(local_tensor_ids) != len(parameter_names):
                raise ValueError(
                    f"Mismatch between tensor IDs ({len(local_tensor_ids)}) and parameter names ({len(parameter_names)})"
                )

            tensor_registry = self._tensor_registry
            model_registry = self._model_registry

            if not model_registry:
                raise RuntimeError(
                    "No models found in registry. Call prepare_huggingface_model first."
                )

            # Find the model that contains these parameters
            model_data = None
            for _checkpoint, model_info in model_registry.items():
                # Check if this model has the requested parameters
                param_tensors = model_info.get("parameter_tensors", {})
                buffer_tensors = model_info.get("buffer_tensors", {})
                all_tensors = {**param_tensors, **buffer_tensors}

                missing_params = [p for p in parameter_names if p not in all_tensors]
                if len(missing_params) == 0:
                    model_data = model_info
                    break
                else:
                    continue

            if model_data is None:
                available_params = (
                    list(
                        next(iter(model_registry.values()))
                        .get("parameter_tensors", {})
                        .keys()
                    )[:10]
                    if model_registry
                    else []
                )
                raise RuntimeError(
                    f"Could not find model containing all parameters: {parameter_names[:5]}... "
                    f"Available parameters: {available_params}"
                )

            # Get all available tensors from the model
            param_tensors = model_data.get("parameter_tensors", {})
            buffer_tensors = model_data.get("buffer_tensors", {})
            all_tensors = {**param_tensors, **buffer_tensors}

            for local_tensor_id, param_name in zip(local_tensor_ids, parameter_names):
                if param_name not in all_tensors:
                    continue

                # Get the actual tensor data
                remote_tensor = all_tensors[param_name]

                # Link the local tensor ID to the remote tensor in the registry
                tensor_registry[local_tensor_id] = remote_tensor

        @modal.method()
        def link_model_tensors(
            self,
            local_tensor_ids: List[int],
            parameter_names: List[str],
        ):
            """
            Link local mycelya tensor IDs to remote model parameter tensors.

            This method establishes linkage between local tensor IDs and remote model parameters
            in the tensor registry.

            Args:
                local_tensor_ids: List of local tensor IDs from created mycelya tensors
                parameter_names: List of parameter names corresponding to each tensor ID
            """
            self._link_model_tensors_impl(local_tensor_ids, parameter_names)

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

    return app, PytorchServer, response_queue
