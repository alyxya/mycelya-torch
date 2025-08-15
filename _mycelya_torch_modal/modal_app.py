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

import logging
from typing import Any, Dict, List, Tuple

import modal

log = logging.getLogger(__name__)

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
        ):
            """Create an empty tensor with given tensor_id and proper storage layout."""
            import torch

            tensor_registry = self._tensor_registry

            if tensor_id in tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} already exists")

            torch_dtype = getattr(torch, dtype.replace("torch.", ""))

            # Detect device
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Calculate the required storage size based on shape, stride and offset
            # The storage size needs to accommodate the maximum element accessed
            numel = (
                sum((s - 1) * st for s, st in zip(shape, stride)) + storage_offset + 1
            )
            storage_nbytes = numel * torch.empty(0, dtype=torch_dtype).element_size()

            # Create a storage tensor that can hold the data
            storage_tensor = torch.empty(
                storage_nbytes, dtype=torch.uint8, device=device
            )

            # Create the tensor view with the specified layout
            tensor = torch.empty(0, dtype=torch_dtype, device=device).set_(
                storage_tensor.untyped_storage(), storage_offset, shape, stride
            )

            tensor_registry[tensor_id] = tensor

            log.info(
                f"✅ Created empty tensor {tensor_id} with shape {shape}, stride {stride}, offset {storage_offset}"
            )
            # No queue operation - this method has no return value

        @modal.method()
        def create_empty_tensor(
            self,
            tensor_id: int,
            shape: List[int],
            stride: List[int],
            storage_offset: int,
            dtype: str,
        ):
            """Create an empty tensor on the remote machine with proper storage layout."""
            self._create_empty_tensor_impl(
                tensor_id, shape, stride, storage_offset, dtype
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

            log.info(
                f"✅ Created tensor view {new_tensor_id} from tensor {base_tensor_id}"
            )
            # No queue operation - this method has no return value

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

            # Convert dtype string back to torch.dtype
            dtype_name = source_dtype.replace("torch.", "")
            torch_dtype = getattr(torch, dtype_name)

            # Create writable buffer to avoid PyTorch warnings
            writable_data = bytearray(raw_data)

            # Reconstruct source tensor from raw data using provided metadata
            flat_tensor = torch.frombuffer(writable_data, dtype=torch_dtype)

            # Reshape to source shape
            source_tensor = flat_tensor.reshape(source_shape)

            # Apply custom stride/offset if needed
            if (
                list(source_tensor.stride()) != source_stride
                or source_storage_offset != 0
            ):
                source_tensor = source_tensor.as_strided(
                    source_shape, source_stride, source_storage_offset
                )

            # Move to target device and copy
            device_source = source_tensor.to(target_tensor.device)
            target_tensor.copy_(device_source)

            log.info(f"✅ Updated tensor {tensor_id}")
            # No queue operation - this is fire-and-forget

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

            log.debug(f"get_storage_data called for tensor {tensor_id}")
            log.debug(f"Current registry has {len(tensor_registry)} tensors")

            if tensor_id not in tensor_registry:
                log.error(f"Tensor ID {tensor_id} does not exist in registry")
                raise ValueError(f"Tensor ID {tensor_id} does not exist")

            tensor = tensor_registry[tensor_id]
            log.info(f"📦 Retrieving storage data for tensor {tensor_id}")
            result = tensor.cpu().numpy().tobytes()

            # Put result in queue
            self.response_queue.put(result)

        @modal.method()
        def get_storage_data(self, tensor_id: int):
            """Get raw storage data by tensor ID."""
            self._get_storage_data_impl(tensor_id)

        def _remove_tensors_impl(self, tensor_ids: List[int]):
            """Remove multiple tensors from the remote machine."""
            tensor_registry = self._tensor_registry

            removed_count = 0
            for tensor_id in tensor_ids:
                if tensor_id in tensor_registry:
                    del tensor_registry[tensor_id]
                    removed_count += 1

            log.info(f"🗑️ Removed {removed_count}/{len(tensor_ids)} tensors")
            # No queue operation - this is fire-and-forget

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

            log.info(f"🔄 Resized storage for tensor {tensor_id}")
            # No queue operation - this is fire-and-forget

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

            log.info(f"🚀 Modal {gpu_type} executing: {op_name}")
            log.debug(f"Input tensor IDs: {input_tensor_ids}")
            log.debug(f"Output tensor IDs: {output_tensor_ids}")

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

            log.debug(f"📥 Retrieved {len(input_tensors)} input tensors from registry")

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
            op_name_fixed = op_name.replace("::", ".")
            op_parts = op_name_fixed.split(".")
            op = torch.ops
            for part in op_parts:
                op = getattr(op, part)

            log.debug(
                f"Executing operation with {len(processed_args)} args, "
                f"{len(input_tensors)} inputs, {len(output_tensor_ids)} outputs to update"
            )

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
                log.debug(
                    f"Stored output tensor {tensor_id} in registry (shape: {result_tensor.shape}, device: {result_tensor.device})"
                )

            log.debug(f"📦 Updated {len(output_tensor_ids)} output tensors in registry")

            # Return metadata if requested
            if return_metadata:
                output_metadata = []
                for i, result_tensor in enumerate(result_tensors):
                    if i < len(output_tensor_ids):
                        metadata = {
                            "shape": list(result_tensor.shape),
                            "dtype": str(result_tensor.dtype),
                            "stride": list(result_tensor.stride()),
                            "storage_offset": result_tensor.storage_offset(),
                            "storage_nelements": result_tensor.untyped_storage().nbytes()
                            // result_tensor.element_size(),
                        }
                        output_metadata.append(metadata)

                log.info(
                    f"✅ Completed: {op_name} (returning metadata for {len(output_metadata)} outputs)"
                )
                # Put metadata result in queue
                self.response_queue.put(output_metadata)
            else:
                log.info(f"✅ Completed: {op_name}")
                # No queue operation when not returning metadata

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
            self._execute_aten_operation_impl(
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

            log.info(
                f"🤗 Loading HuggingFace model {checkpoint} directly on {gpu_type}"
            )

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
            log.info(f"Loading model on device: {device}")

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
            log.info(
                f"Downloading and loading {checkpoint} with dtype {torch_dtype_obj}"
            )
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
            log.info(f"✅ Model {checkpoint} loaded successfully on {device}")

            # Extract state dict metadata without transferring data
            state_dict_metadata = {}
            param_count = 0
            total_params = sum(1 for _ in model.named_parameters())

            for name, param in model.named_parameters():
                param_count += 1
                log.debug(f"Processing parameter {param_count}/{total_params}: {name}")

                # Collect metadata for client (no storage ID generation here)
                state_dict_metadata[name] = {
                    "shape": list(param.shape),
                    "stride": list(param.stride()),
                    "dtype": str(param.dtype),
                    "storage_offset": param.storage_offset(),
                    "requires_grad": param.requires_grad,
                    # Store actual tensor data for later linking
                    "_tensor_data": param.detach().contiguous().to(device),
                }

                log.debug(
                    f"Cached parameter {name}: shape={param.shape}, dtype={param.dtype}"
                )

            # Also handle buffers (non-trainable parameters like batch norm running stats)
            buffer_metadata = {}
            for name, buffer in model.named_buffers():
                # Collect metadata for client (no storage ID generation here)
                buffer_metadata[name] = {
                    "shape": list(buffer.shape),
                    "stride": list(buffer.stride()),
                    "dtype": str(buffer.dtype),
                    "storage_offset": buffer.storage_offset(),
                    "requires_grad": False,  # Buffers don't require gradients
                    # Store actual tensor data for later linking
                    "_tensor_data": buffer.detach().contiguous().to(device),
                }

                log.debug(
                    f"Cached buffer {name}: shape={buffer.shape}, dtype={buffer.dtype}"
                )

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

            log.info(
                f"🎯 Model preparation complete: {len(state_dict_metadata)} parameters, {len(buffer_metadata)} buffers"
            )

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

            # Put result in queue
            self.response_queue.put(result)

        @modal.method()
        def prepare_huggingface_model(
            self,
            checkpoint: str,
            torch_dtype: str = "auto",
            trust_remote_code: bool = False,
        ):
            """Download and prepare a HuggingFace model directly on the remote machine."""
            self._prepare_huggingface_model_impl(
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

            log.info(
                f"🔗 Linking {len(local_tensor_ids)} local tensor IDs to remote model parameters"
            )
            log.debug(
                f"Parameter names to link: {parameter_names[:5]}..."
            )  # Show first 5

            tensor_registry = self._tensor_registry
            model_registry = self._model_registry

            log.debug(f"Model registry contains {len(model_registry)} models")
            if not model_registry:
                raise RuntimeError(
                    "No models found in registry. Call prepare_huggingface_model first."
                )

            # Find the model that contains these parameters
            model_data = None
            for checkpoint, model_info in model_registry.items():
                # Check if this model has the requested parameters
                param_tensors = model_info.get("parameter_tensors", {})
                buffer_tensors = model_info.get("buffer_tensors", {})
                all_tensors = {**param_tensors, **buffer_tensors}

                missing_params = [p for p in parameter_names if p not in all_tensors]
                if len(missing_params) == 0:
                    model_data = model_info
                    log.info(f"Found matching model: {checkpoint}")
                    break
                else:
                    log.debug(
                        f"Model {checkpoint} missing {len(missing_params)} parameters"
                    )

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

            linked_count = 0
            for local_tensor_id, param_name in zip(local_tensor_ids, parameter_names):
                if param_name not in all_tensors:
                    log.warning(f"Parameter {param_name} not found in model tensors")
                    continue

                # Get the actual tensor data
                remote_tensor = all_tensors[param_name]

                log.debug(
                    f"Linking local tensor {local_tensor_id} to parameter {param_name}"
                )
                log.debug(
                    f"Remote tensor type: {type(remote_tensor)}, shape: {remote_tensor.shape}"
                )
                log.debug(f"Remote tensor dtype: {remote_tensor.dtype}")

                # Link the local tensor ID to the remote tensor in the registry
                tensor_registry[local_tensor_id] = remote_tensor

                log.debug(f"Linked tensor {local_tensor_id} to parameter {param_name}")

                linked_count += 1

            log.info(
                f"✅ Model tensor linking complete: {linked_count}/{len(local_tensor_ids)} links successful"
            )
            # No queue operation - this is fire-and-forget

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
        def execute_batch(self, batch_calls: List[Dict[str, Any]]):
            """
            Execute a batch of RPCs in sequence.

            This method allows multiple operations to be batched together in a single
            RPC, reducing network overhead and improving performance.

            Args:
                batch_calls: List of dictionaries, each containing:
                    - method_name: Name of the method to call
                    - call_type: "spawn" or "remote"
                    - args: Arguments for the method
                    - kwargs: Keyword arguments for the method
                    - call_id: Unique identifier for debugging
            """
            log.info(f"🚀 BATCH EXECUTE: Processing {len(batch_calls)} batched RPCs")
            results = []

            for i, call in enumerate(batch_calls):
                call_id = call.get("call_id", f"batch_call_{i}")
                method_name = call["method_name"]
                call_type = call["call_type"]
                args = call.get("args", ())
                kwargs = call.get("kwargs", {})

                try:
                    log.debug(
                        f"📞 Executing batched RPC {call_id}: {method_name} ({call_type})"
                    )

                    # Call the same underlying implementations that the @modal.method() functions use
                    if method_name == "create_empty_tensor":
                        self._create_empty_tensor_impl(*args, **kwargs)
                        result = None  # No return value
                    elif method_name == "create_tensor_view":
                        self._create_tensor_view_impl(*args, **kwargs)
                        result = None  # No return value
                    elif method_name == "update_tensor":
                        self._update_tensor_impl(*args, **kwargs)
                        result = None  # Fire-and-forget, no return value
                    elif method_name == "get_storage_data":
                        self._get_storage_data_impl(*args, **kwargs)
                        result = self.response_queue.get()  # Always returns bytes
                    elif method_name == "remove_tensors":
                        self._remove_tensors_impl(*args, **kwargs)
                        result = None  # Fire-and-forget, no return value
                    elif method_name == "resize_storage":
                        self._resize_storage_impl(*args, **kwargs)
                        result = None  # Fire-and-forget, no return value
                    elif method_name == "execute_aten_operation":
                        self._execute_aten_operation_impl(*args, **kwargs)
                        # Conditionally get from queue based on return_metadata
                        return_metadata = kwargs.get("return_metadata", False)
                        if return_metadata:
                            result = self.response_queue.get()  # Get metadata
                        else:
                            result = None  # No return value
                    elif method_name == "prepare_huggingface_model":
                        self._prepare_huggingface_model_impl(*args, **kwargs)
                        result = self.response_queue.get()  # Always returns model metadata
                    elif method_name == "link_model_tensors":
                        self._link_model_tensors_impl(*args, **kwargs)
                        result = None  # Fire-and-forget, no return value
                    else:
                        raise AttributeError(f"Unknown method: {method_name}")

                    # For spawn calls, we return None
                    if call_type == "spawn":
                        results.append(None)
                    else:
                        results.append(result)

                except Exception as e:
                    log.error(f"❌ Batched RPC {call_id} failed: {method_name} - {e}")
                    results.append(e)

            log.info(
                f"✅ BATCH COMPLETE: Processed {len(batch_calls)} calls, "
                f"{sum(1 for r in results if not isinstance(r, Exception))} successful"
            )

            return results

    return app, PytorchServer, response_queue
