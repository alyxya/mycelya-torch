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
from typing import Any, Dict, List, Optional, Tuple, Union

import modal
import torch

log = logging.getLogger(__name__)

# Create image with PyTorch, CUDA support, and transformers for HuggingFace models
image = modal.Image.debian_slim().pip_install(
    "numpy", "torch", "transformers", "huggingface_hub", "safetensors", "accelerate"
)


def create_modal_app_for_gpu(
    gpu_type: str,
    machine_id: str,
    timeout: int,
    retries: int,
) -> Tuple[modal.App, Any]:
    """
    Create a Modal app and class for a specific GPU type and device.

    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB")
        machine_id: The machine ID (e.g., "modal-t4-f3a7d67e")
        timeout: Function timeout in seconds
        retries: Number of retries on failure

    Returns:
        Tuple of (modal_app, server_class) for the specified device
    """
    app = modal.App(f"mycelya-torch-{machine_id}")

    @app.cls(
        image=image,
        gpu=gpu_type,
        timeout=timeout,
        retries=retries,
        serialized=True,
        max_containers=1,
        min_containers=1,
    )
    class PytorchServer:
        def _get_device(self):
            """Get the appropriate device for tensor operations."""
            import torch

            # Detect if we're running in local mode by checking if CUDA is available
            # In local mode, we should use CPU even if CUDA is available
            try:
                # Try to check if we're running in Modal's local execution mode
                # This is a heuristic - if torch.cuda.is_available() is False, we're likely local
                if torch.cuda.is_available():
                    return torch.device("cuda")
                else:
                    return torch.device("cpu")
            except Exception:
                # Fall back to CPU if any issues
                return torch.device("cpu")

        def _get_tensor_registry(self):
            """Get or create tensor registry for this server instance.

            Maps tensor_id (metadata hash) directly to tensors.
            """
            if not hasattr(self, "_tensor_registry"):
                # tensor_id -> torch.Tensor (direct mapping from tensor ID to tensor)
                self._tensor_registry: Dict[int, torch.Tensor] = {}

            return self._tensor_registry

        def _get_model_registry(self):
            """Get or create model registry for this server instance."""
            if not hasattr(self, "_model_registry"):
                # checkpoint -> {model: nn.Module, parameter_tensor_map: Dict[str, int]}
                self._model_registry: Dict[str, Dict[str, Any]] = {}

            return self._model_registry

        def _find_base_tensor_with_same_storage(self, tensor: torch.Tensor) -> Optional[int]:
            """Find a tensor ID that shares the same underlying storage."""
            tensor_registry = self._get_tensor_registry()
            target_storage_ptr = tensor.untyped_storage().data_ptr()

            for tensor_id, existing_tensor in tensor_registry.items():
                if existing_tensor.untyped_storage().data_ptr() == target_storage_ptr:
                    return tensor_id

            return None

        # Tensor ID-based methods
        def _create_empty_tensor_impl(
            self,
            tensor_id: int,
            shape: List[int],
            stride: List[int],
            storage_offset: int,
            dtype: str
        ) -> None:
            """Create an empty tensor with given tensor_id and proper storage layout."""
            import torch

            tensor_registry = self._get_tensor_registry()

            if tensor_id in tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} already exists")

            torch_dtype = getattr(torch, dtype.replace("torch.", ""))
            device = self._get_device()

            # Calculate the required storage size based on shape, stride and offset
            # The storage size needs to accommodate the maximum element accessed
            numel = sum((s-1) * st for s, st in zip(shape, stride)) + storage_offset + 1
            storage_nbytes = numel * torch.empty(0, dtype=torch_dtype).element_size()

            # Create a storage tensor that can hold the data
            storage_tensor = torch.empty(storage_nbytes, dtype=torch.uint8, device=device)

            # Create the tensor view with the specified layout
            tensor = torch.empty(0, dtype=torch_dtype, device=device).set_(
                storage_tensor.untyped_storage(),
                storage_offset,
                shape,
                stride
            )

            tensor_registry[tensor_id] = tensor

            log.info(f"✅ Created empty tensor {tensor_id} with shape {shape}, stride {stride}, offset {storage_offset}")

        @modal.method()
        def create_empty_tensor(
            self,
            tensor_id: int,
            shape: List[int],
            stride: List[int],
            storage_offset: int,
            dtype: str
        ) -> None:
            """Create an empty tensor on the remote machine with proper storage layout."""
            return self._create_empty_tensor_impl(tensor_id, shape, stride, storage_offset, dtype)

        def _create_tensor_view_impl(
            self,
            new_tensor_id: int,
            base_tensor_id: int,
            shape: List[int],
            stride: List[int],
            offset: int,
        ) -> None:
            """Create a tensor view from existing tensor using as_strided."""
            import torch

            tensor_registry = self._get_tensor_registry()

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

        @modal.method()
        def create_tensor_view(
            self,
            new_tensor_id: int,
            base_tensor_id: int,
            shape: List[int],
            stride: List[int],
            offset: int,
        ) -> None:
            """Create a tensor view from existing tensor using as_strided."""
            return self._create_tensor_view_impl(
                new_tensor_id, base_tensor_id, shape, stride, offset
            )

        def _update_tensor_impl(
            self,
            tensor_id: int,
            raw_data: bytes,
            source_shape: List[int],
            source_stride: List[int],
            source_storage_offset: int,
            source_dtype: str
        ) -> None:
            """Update an existing tensor with new data and source metadata."""
            import torch

            tensor_registry = self._get_tensor_registry()

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

        @modal.method()
        def update_tensor(
            self,
            tensor_id: int,
            raw_data: bytes,
            source_shape: List[int],
            source_stride: List[int],
            source_storage_offset: int,
            source_dtype: str
        ) -> None:
            """Update an existing tensor with new data and source metadata."""
            return self._update_tensor_impl(
                tensor_id, raw_data, source_shape, source_stride,
                source_storage_offset, source_dtype
            )

        def _get_tensor_data_impl(self, tensor_id: int) -> bytes:
            """Get raw tensor data by tensor ID."""
            tensor_registry = self._get_tensor_registry()

            if tensor_id not in tensor_registry:
                raise ValueError(f"Tensor ID {tensor_id} does not exist")

            tensor = tensor_registry[tensor_id]
            log.info(f"📦 Retrieving tensor data for tensor {tensor_id}")
            return tensor.cpu().numpy().tobytes()

        @modal.method()
        def get_tensor_data(self, tensor_id: int) -> bytes:
            """Get raw tensor data by tensor ID."""
            return self._get_tensor_data_impl(tensor_id)

        def _remove_tensors_impl(self, tensor_ids: List[int]) -> None:
            """Remove multiple tensors from the remote machine."""
            tensor_registry = self._get_tensor_registry()

            removed_count = 0
            for tensor_id in tensor_ids:
                if tensor_id in tensor_registry:
                    del tensor_registry[tensor_id]
                    removed_count += 1

            log.info(f"🗑️ Removed {removed_count}/{len(tensor_ids)} tensors")

        @modal.method()
        def remove_tensors(self, tensor_ids: List[int]) -> None:
            """Remove multiple tensors from the remote machine."""
            return self._remove_tensors_impl(tensor_ids)

        def _resize_tensor_storage_impl(self, tensor_id: int, nbytes: int) -> None:
            """Resize the underlying storage for a tensor."""
            import torch

            tensor_registry = self._get_tensor_registry()

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

        @modal.method()
        def resize_tensor_storage(self, tensor_id: int, nbytes: int) -> None:
            """Resize the underlying storage for a tensor."""
            return self._resize_tensor_storage_impl(tensor_id, nbytes)

        def _prepare_huggingface_model_impl(
            self,
            checkpoint: str,
            torch_dtype: str = "auto",
            trust_remote_code: bool = False,
        ) -> Dict[str, Any]:
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
            device = self._get_device()
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
            model_registry = self._get_model_registry()
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

            return {
                "state_dict_metadata": clean_state_dict,
                "buffer_metadata": clean_buffer_dict,
                "config": model.config.to_dict(),
                "model_type": type(model).__name__,
                "checkpoint": checkpoint,
            }


        @modal.method()
        def prepare_huggingface_model(
            self,
            checkpoint: str,
            torch_dtype: str = "auto",
            trust_remote_code: bool = False,
        ) -> Dict[str, Any]:
            """
            Download and prepare a HuggingFace model directly on the remote machine.

            This method downloads the model weights directly on the remote GPU,
            loads them into GPU memory, and returns metadata needed to create
            local tensor stubs.

            Args:
                checkpoint: HuggingFace model checkpoint (e.g., "gpt2", "bert-base-uncased")
                torch_dtype: Data type for model weights ("auto", "float32", "float16", etc.)
                trust_remote_code: Whether to trust remote code for custom models

            Returns:
                Dict containing state_dict_metadata, config, and model_type
            """
            return self._prepare_huggingface_model_impl(
                checkpoint, torch_dtype, trust_remote_code
            )

        def _link_model_tensors_impl(
            self,
            local_tensor_ids: List[int],
            parameter_names: List[str],
        ) -> None:
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

            tensor_registry = self._get_tensor_registry()
            model_registry = self._get_model_registry()

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

                log.debug(
                    f"Linked tensor {local_tensor_id} to parameter {param_name}"
                )

                linked_count += 1

            log.info(
                f"✅ Model tensor linking complete: {linked_count}/{len(local_tensor_ids)} links successful"
            )

        @modal.method()
        def link_model_tensors(
            self,
            local_tensor_ids: List[int],
            parameter_names: List[str],
        ) -> None:
            """
            Link local mycelya tensor IDs to remote model parameter tensors.

            This method establishes linkage between local tensor IDs and remote model parameters
            in the tensor registry.

            Args:
                local_tensor_ids: List of local tensor IDs from created mycelya tensors
                parameter_names: List of parameter names corresponding to each tensor ID

            Returns:
                None
            """
            return self._link_model_tensors_impl(local_tensor_ids, parameter_names)

        def _execute_aten_operation_impl(
            self,
            op_name: str,
            input_tensor_metadata: List[Dict[str, Any]],
            output_tensor_ids: List[Union[int, None]],
            args: List[Any],
            kwargs: Dict[str, Any],
            return_metadata: bool = False,
        ) -> Union[None, List[Dict[str, Any]]]:
            """Implementation of execute_aten_operation without Modal decorators."""
            # Import torch and tree_map locally to avoid serialization issues
            import torch
            from torch.utils._pytree import tree_map

            # Extract tensor IDs from input metadata
            input_tensor_ids = [
                metadata["tensor_id"] for metadata in input_tensor_metadata
            ]

            log.info(f"🚀 Modal {gpu_type} executing: {op_name}")
            log.debug(f"Input tensor IDs: {input_tensor_ids}")
            log.debug(f"Output tensor IDs: {output_tensor_ids}")

            # Get tensor registry
            tensor_registry = self._get_tensor_registry()

            # Reconstruct input tensors from tensor registry
            input_tensors = []
            for metadata in input_tensor_metadata:
                tensor_id = metadata["tensor_id"]

                # Check if tensor exists in registry
                if tensor_id in tensor_registry:
                    tensor = tensor_registry[tensor_id]
                else:
                    # Create tensor if it doesn't exist (shouldn't happen in normal flow)
                    raise ValueError(f"Input tensor ID {tensor_id} not found in registry")

                input_tensors.append(tensor)

            log.debug(f"📥 Retrieved {len(input_tensors)} input tensors from registry")

            # Replace tensor placeholders with actual tensors using tree_map
            def replace_placeholder_with_tensor(obj):
                if isinstance(obj, str) and obj.startswith("__TENSOR_"):
                    idx = int(obj.split("_")[-1])
                    if idx < len(input_tensors):
                        return input_tensors[idx]
                    else:
                        raise IndexError(
                            f"Tensor placeholder index {idx} out of range (have {len(input_tensors)} input tensors)"
                        )
                return obj

            # Use tree_map to handle nested structure traversal automatically
            processed_args, processed_kwargs = tree_map(
                replace_placeholder_with_tensor, (args, kwargs)
            )

            # Get the operation
            op_name_fixed = op_name.replace("::", ".")
            op_parts = op_name_fixed.split(".")
            op = torch.ops
            for part in op_parts:
                op = getattr(op, part)

            log.debug(
                f"Executing operation with {len(processed_args)} args, "
                f"{len(input_tensors)} inputs, {len([t for t in output_tensor_ids if t is not None])} outputs to update"
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
                if tensor_id is not None:
                    tensor_registry[tensor_id] = result_tensor
                    log.debug(f"💾 Stored output tensor {tensor_id} in registry")

            log.debug(f"📦 Updated {len([t for t in output_tensor_ids if t is not None])} output tensors in registry")

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
                return output_metadata

            log.info(f"✅ Completed: {op_name}")
            return None

        @modal.method()
        def execute_aten_operation(
            self,
            op_name: str,
            input_tensor_metadata: List[Dict[str, Any]],
            output_tensor_ids: List[Union[int, None]],
            args: List[Any],
            kwargs: Dict[str, Any],
            return_metadata: bool = False,
        ) -> Union[None, List[Dict[str, Any]]]:
            """
            Execute an operation with separated input metadata and output tensor IDs.

            This method handles operations where input tensor metadata and output tensor IDs
            are explicitly separated. Tensors are stored in the tensor registry by their IDs.

            Args:
                op_name: The operation name to execute
                input_tensor_metadata: List of metadata for input tensors (including tensor_id)
                output_tensor_ids: List of tensor IDs to store results (all output tensors)
                args: Operation arguments (with tensor placeholders)
                kwargs: Operation keyword arguments (with tensor placeholders)
                return_metadata: If True, return output tensor metadata instead of None

            Returns:
                None for normal operations, or List[Dict] of output tensor metadata if return_metadata=True
            """
            return self._execute_aten_operation_impl(
                op_name,
                input_tensor_metadata,
                output_tensor_ids,
                args,
                kwargs,
                return_metadata,
            )

        @modal.method()
        def execute_batch(
            self, batch_calls: List[Dict[str, Any]]
        ) -> List[Union[None, Any]]:
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

            Returns:
                List of results in the same order as input calls.
                None for "spawn" calls, actual return value for "remote" calls.
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

                    # Call the underlying method implementations directly
                    # We need to bypass Modal decorators and call the actual Python methods
                    # Tensor-based methods
                    if method_name == "create_empty_tensor":
                        result = self._create_empty_tensor_impl(*args, **kwargs)
                    elif method_name == "create_tensor_view":
                        result = self._create_tensor_view_impl(*args, **kwargs)
                    elif method_name == "update_tensor":
                        result = self._update_tensor_impl(*args, **kwargs)
                    elif method_name == "get_tensor_data":
                        result = self._get_tensor_data_impl(*args, **kwargs)
                    elif method_name == "remove_tensors":
                        result = self._remove_tensors_impl(*args, **kwargs)
                    elif method_name == "resize_tensor_storage":
                        result = self._resize_tensor_storage_impl(*args, **kwargs)
                    # Legacy methods
                    elif method_name == "execute_aten_operation":
                        result = self._execute_aten_operation_impl(*args, **kwargs)
                    elif method_name == "prepare_huggingface_model":
                        result = self._prepare_huggingface_model_impl(*args, **kwargs)
                    elif method_name == "link_model_tensors":
                        result = self._link_model_tensors_impl(*args, **kwargs)
                    else:
                        raise AttributeError(f"Unknown method: {method_name}")

                    # For spawn calls, we return None
                    if call_type == "spawn":
                        results.append(None)
                    else:
                        results.append(result)

                except Exception as e:
                    log.error(f"❌ Batched RPC {call_id} failed: {method_name} - {e}")

                    # Store the exception as the result
                    results.append(e)

            log.info(
                f"✅ BATCH COMPLETE: Processed {len(batch_calls)} calls, "
                f"{sum(1 for r in results if not isinstance(r, Exception))} successful"
            )

            return results

    return app, PytorchServer