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
from typing import Any, Dict, List, Tuple, Union

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

        def _get_storages(self):
            """Get or create storage mapping for this server instance."""
            if not hasattr(self, "_storages"):
                # storage_id -> int (lazy allocation byte count) or torch.Tensor (realized storage)
                # Actual tensor data is stored in tensor_cache, accessed via storage_to_tensors mapping
                self._storages: Dict[int, Union[torch.Tensor, int]] = {}

            return self._storages

        def _get_tensor_cache(self):
            """Get or create tensor cache for this server instance."""
            if not hasattr(self, "_tensor_cache"):
                # storage_id -> {metadata_key -> torch.Tensor}
                # where metadata_key is a hash of (shape, stride, offset, dtype)
                self._tensor_cache: Dict[int, Dict[str, Any]] = {}

            return self._tensor_cache

        def _get_model_registry(self):
            """Get or create model registry for this server instance."""
            if not hasattr(self, "_model_registry"):
                # checkpoint -> {model: nn.Module, parameter_storage_map: Dict[str, int]}
                self._model_registry: Dict[str, Dict[str, Any]] = {}

            return self._model_registry

        def _make_tensor_metadata_key(
            self, shape: List[int], stride: List[int], storage_offset: int, dtype: str
        ) -> str:
            """Create a unique key for tensor metadata."""
            return f"{tuple(shape)}-{tuple(stride)}-{storage_offset}-{dtype}"

        def _cache_tensor(
            self,
            storage_id: int,
            shape: List[int],
            stride: List[int],
            storage_offset: int,
            dtype: str,
            tensor,
        ):
            """Cache a tensor with metadata-based key."""
            tensor_cache = self._get_tensor_cache()

            # Create metadata key
            metadata_key = self._make_tensor_metadata_key(
                shape, stride, storage_offset, dtype
            )

            # Ensure storage_id entry exists
            if storage_id not in tensor_cache:
                tensor_cache[storage_id] = {}

            # Cache the tensor
            tensor_cache[storage_id][metadata_key] = tensor

        def _get_cached_tensor(
            self,
            storage_id: int,
            shape: List[int],
            stride: List[int],
            storage_offset: int,
            dtype: str,
        ):
            """Get a cached tensor by storage ID and metadata."""
            tensor_cache = self._get_tensor_cache()

            if storage_id not in tensor_cache:
                return None

            metadata_key = self._make_tensor_metadata_key(
                shape, stride, storage_offset, dtype
            )
            return tensor_cache[storage_id].get(metadata_key)

        def _get_any_cached_tensor_for_storage(self, storage_id: int):
            """Get any cached tensor for a storage ID (for reconstruction purposes)."""
            tensor_cache = self._get_tensor_cache()

            if storage_id in tensor_cache and tensor_cache[storage_id]:
                # Return any cached tensor for this storage
                return next(iter(tensor_cache[storage_id].values()))
            return None

        def _cleanup_tensor_cache_for_storage(self, storage_id: int):
            """Clean up all cached tensors associated with a storage ID."""
            tensor_cache = self._get_tensor_cache()

            if storage_id in tensor_cache:
                tensor_cache.pop(storage_id, None)

        def _construct_tensor_from_storage(
            self,
            storage_id: int,
            shape: List[int],
            stride: List[int],
            storage_offset: int,
            dtype: str,
        ) -> Any:
            """
            Construct a tensor from storage ID and tensor parameters.

            Uses cached tensors when available, or creates empty tensors for lazy storage.

            Args:
                storage_id: The storage ID to look up
                shape: Tensor shape
                stride: Tensor stride
                storage_offset: Storage offset
                dtype: Data type as string

            Returns:
                Reconstructed tensor on appropriate device

            Raises:
                KeyError: If storage_id is not found
            """
            import torch

            storages = self._get_storages()

            # Validate storage exists
            if storage_id not in storages:
                available_ids = list(storages.keys())
                log.error(f"❌ MISSING Storage ID {storage_id}")
                log.error(f"📋 Available Storage IDs on Modal: {available_ids}")
                raise KeyError(f"Storage ID {storage_id} not found")

            # Parse dtype string back to torch.dtype
            dtype_str = dtype.replace("torch.", "")
            torch_dtype = getattr(torch, dtype_str)

            # Try to get a cached tensor with exact metadata first
            cached_tensor = self._get_cached_tensor(
                storage_id, shape, stride, storage_offset, dtype
            )

            if cached_tensor is not None:
                log.debug(f"🎯 Found exact cached tensor for storage {storage_id}")
                return cached_tensor

            # Try to get any cached tensor for this storage to reconstruct from
            cached_tensor = self._get_any_cached_tensor_for_storage(storage_id)

            if cached_tensor is not None:
                # Reconstruct from cached tensor's storage
                target_device = cached_tensor.device
                tensor = torch.empty(0, dtype=torch_dtype, device=target_device).set_(
                    cached_tensor.untyped_storage(), storage_offset, shape, stride
                )
                # Cache this new tensor view
                self._cache_tensor(
                    storage_id, shape, stride, storage_offset, dtype, tensor
                )
                log.debug(
                    f"🔄 Reconstructed and cached tensor from storage {storage_id}"
                )
                return tensor

            # No cached tensor - check if storage is lazy or realized
            storage_item = storages[storage_id]
            device = self._get_device()

            if isinstance(storage_item, int):
                # Lazy storage - need to realize it with actual storage tensor
                nbytes = storage_item
                
                # Create 1D uint8 tensor to hold the storage data
                storage_tensor = torch.empty(nbytes, dtype=torch.uint8, device=device)
                
                # Update the storages mapping with realized tensor
                storages[storage_id] = storage_tensor
                log.info(f"🔄 REALIZED lazy storage {storage_id} ({nbytes} bytes)")
                
                # Now reconstruct tensor from realized storage
                tensor = torch.empty(0, dtype=torch_dtype, device=device).set_(
                    storage_tensor.untyped_storage(), storage_offset, shape, stride
                )
                log.debug(
                    f"🔄 Reconstructed tensor from realized storage {storage_id}"
                )
            else:
                # Realized storage - reconstruct from actual data
                storage_tensor = storage_item  # This is a torch.Tensor
                tensor = torch.empty(0, dtype=torch_dtype, device=device).set_(
                    storage_tensor.untyped_storage(), storage_offset, shape, stride
                )
                log.debug(
                    f"🔄 Reconstructed tensor from realized storage {storage_id}"
                )

            return tensor

        def _create_storage_impl(self, storage_id: int, nbytes: int) -> None:
            """Implementation of create_storage without Modal decorators."""
            # Store storage as lazy allocation (just the byte count)
            storages = self._get_storages()

            # Check if storage already exists
            if storage_id in storages:
                raise RuntimeError(f"Storage ID {storage_id} already exists")

            # Always store as int for lazy allocation
            storages[storage_id] = nbytes
            log.info(f"📝 LAZY Storage ID {storage_id} registered ({nbytes} bytes)")

        @modal.method()
        def create_storage(self, storage_id: int, nbytes: int) -> None:
            """
            Create a new lazy storage on the remote machine.

            Storage is always created lazily - actual GPU memory allocation
            is deferred until first use.

            Args:
                storage_id: Specific ID to use for the storage (required)
                nbytes: Number of bytes to allocate for the storage

            Returns:
                None
            """
            return self._create_storage_impl(storage_id, nbytes)

        def _update_storage_impl(
            self,
            storage_id: int,
            raw_data: bytes,
            source_shape: List[int],
            source_stride: List[int],
            source_storage_offset: int,
            source_dtype: str,
            target_shape: List[int],
            target_stride: List[int],
            target_storage_offset: int,
            target_dtype: str,
        ) -> None:
            """Implementation of update_storage without Modal decorators."""
            import torch

            # Get storages
            storages = self._get_storages()

            if storage_id not in storages:
                raise RuntimeError(f"Storage ID {storage_id} not found")

            # Deserialize source tensor from raw numpy bytes using provided metadata
            # Convert dtype string back to torch.dtype
            dtype_name = source_dtype.replace("torch.", "")
            torch_dtype = getattr(torch, dtype_name)

            # Create writable buffer to avoid PyTorch warnings
            writable_data = bytearray(raw_data)

            # Reconstruct tensor using torch.frombuffer - no clone needed since temp tensor
            flat_tensor = torch.frombuffer(writable_data, dtype=torch_dtype)

            # Reshape and apply stride/offset using provided metadata
            source_tensor = flat_tensor.reshape(source_shape)

            # If the tensor has custom stride or offset, we need to use as_strided
            if (
                list(source_tensor.stride()) != source_stride
                or source_storage_offset != 0
            ):
                source_tensor = source_tensor.as_strided(
                    source_shape, source_stride, source_storage_offset
                )

            # Ensure tensor is on CPU and has the expected properties
            if source_tensor.device.type != "cpu":
                source_tensor = source_tensor.cpu()

            storage_item = storages[storage_id]

            # Check if storage is lazy (int) or realized (torch.Tensor)
            if isinstance(storage_item, int):
                # Lazy storage - realize it on first write
                expected_bytes = storage_item
                device = self._get_device()
                device_source = source_tensor.to(device)
                actual_bytes = device_source.untyped_storage().nbytes()

                if expected_bytes != actual_bytes:
                    raise RuntimeError(
                        f"Storage size mismatch for storage {storage_id}: "
                        f"expected {expected_bytes} bytes, got {actual_bytes} bytes"
                    )

                # Create 1D uint8 tensor to hold the storage data
                storage_tensor = torch.empty(
                    actual_bytes, dtype=torch.uint8, device=device
                )
                storage_tensor.untyped_storage().copy_(device_source.untyped_storage())
                
                # Update storage mapping to realized tensor
                storages[storage_id] = storage_tensor
                
                log.info(
                    f"📥 REALIZED lazy storage {storage_id} with {actual_bytes} bytes"
                )
                return

            # Storage is already realized - update in-place
            storage_tensor = storage_item
            device = storage_tensor.device
            device_source = source_tensor.to(device)
            
            # Create target view from storage tensor
            target_tensor = torch.empty(0, dtype=torch.float32, device=device).set_(
                storage_tensor.untyped_storage(),
                target_storage_offset,
                target_shape, 
                target_stride
            )
            
            # Copy source data to target view
            target_tensor.copy_(device_source)
            
            log.info(
                f"📥 Updated realized storage {storage_id} in-place (shape: {target_shape})"
            )

        @modal.method()
        def update_storage(
            self,
            storage_id: int,
            raw_data: bytes,
            source_shape: List[int],
            source_stride: List[int],
            source_storage_offset: int,
            source_dtype: str,
            target_shape: List[int],
            target_stride: List[int],
            target_storage_offset: int,
            target_dtype: str,
        ) -> None:
            """
            Update an existing storage with raw tensor data.

            Supports both full storage replacement and partial in-place updates using
            dual tensor metadata to specify source data layout and target storage view.

            Args:
                storage_id: Storage ID to update
                raw_data: Raw untyped storage bytes to store
                source_shape: Shape of the source data
                source_stride: Stride of the source data
                source_storage_offset: Storage offset of the source data
                source_dtype: Data type of the source data
                target_shape: Shape of the target view in storage
                target_stride: Stride of the target view in storage
                target_storage_offset: Storage offset of the target view in storage
                target_dtype: Data type of the target view in storage

            Returns:
                None
            """
            return self._update_storage_impl(
                storage_id,
                raw_data,
                source_shape,
                source_stride,
                source_storage_offset,
                source_dtype,
                target_shape,
                target_stride,
                target_storage_offset,
                target_dtype,
            )

        def _get_storage_data_impl(
            self,
            storage_id: int,
        ) -> bytes:
            """Implementation of get_storage_data without Modal decorators."""
            import torch
            
            storages = self._get_storages()

            if storage_id not in storages:
                raise RuntimeError(f"Storage ID {storage_id} not found")

            storage_item = storages[storage_id]

            # Handle lazy storage - check tensor cache as fallback
            if isinstance(storage_item, int):
                # Storage is lazy, but check if we have cached tensor data
                cached_tensor = self._get_any_cached_tensor_for_storage(storage_id)
                if cached_tensor is not None:
                    # We have cached tensor data - realize the storage and return data
                    nbytes = storage_item  # This is the byte count
                    device = cached_tensor.device
                    
                    # Create realized storage tensor
                    storage_tensor = torch.empty(
                        nbytes, dtype=torch.uint8, device=device
                    )
                    storage_tensor.untyped_storage().copy_(cached_tensor.untyped_storage())
                    
                    # Update storage mapping to realized tensor
                    storages[storage_id] = storage_tensor
                    
                    log.info(
                        f"📦 REALIZED lazy storage {storage_id} from cached tensor ({nbytes} bytes)"
                    )
                    
                    # Return data from cached tensor (more direct)
                    cpu_tensor = cached_tensor.cpu()
                    return cpu_tensor.numpy().tobytes()
                else:
                    raise RuntimeError(
                        f"Storage ID {storage_id} is lazy with no cached tensor data. Cannot retrieve."
                    )

            # Storage is realized (torch.Tensor)
            storage_tensor = storage_item
            log.info(
                f"📦 Retrieving tensor data for storage {storage_id} ({storage_tensor.untyped_storage().nbytes()} bytes)"
            )

            # Convert to CPU and serialize with pure numpy approach
            cpu_tensor = storage_tensor.cpu()
            return cpu_tensor.numpy().tobytes()

        @modal.method()
        def get_storage_data(
            self,
            storage_id: int,
        ) -> bytes:
            """
            Retrieve raw storage data by storage ID.

            Returns the complete raw untyped storage bytes. The client interface layer
            will handle tensor reconstruction from metadata and these raw bytes.

            Args:
                storage_id: The storage ID

            Returns:
                Raw untyped storage bytes
            """
            return self._get_storage_data_impl(storage_id)

        def _resize_storage_impl(self, storage_id: int, nbytes: int) -> None:
            """Implementation of resize_storage without Modal decorators."""
            storages = self._get_storages()
            if storage_id not in storages:
                log.warning(f"Storage ID {storage_id} not found for resize")
                return

            storage_item = storages[storage_id]

            # Handle lazy storage (int) - propagate laziness
            if isinstance(storage_item, int):
                current_bytes = storage_item

                # Check if resize is actually needed (should be bigger)
                if nbytes <= current_bytes:
                    log.debug(
                        f"Lazy storage {storage_id} resize skipped: "
                        f"nbytes ({nbytes}) <= current_bytes ({current_bytes})"
                    )
                    return  # No-op

                # Update lazy storage with new byte count
                storages[storage_id] = nbytes
                log.info(
                    f"🔄 LAZY Resized storage {storage_id} from {current_bytes} "
                    f"to {nbytes} bytes (kept lazy)"
                )
                return

            # Handle realized storage (torch.Tensor)
            elif isinstance(storage_item, torch.Tensor):
                current_bytes = storage_item.untyped_storage().nbytes()

                # Check if resize is actually needed (should be bigger)
                if nbytes <= current_bytes:
                    log.debug(
                        f"Realized storage {storage_id} resize skipped: "
                        f"nbytes ({nbytes}) <= current_bytes ({current_bytes})"
                    )
                    return  # No-op

                # Resize the existing 1D uint8 tensor
                storage_item.resize_([nbytes])

                log.info(
                    f"🔄 REALIZED Resized storage {storage_id} from {current_bytes} "
                    f"to {nbytes} bytes using tensor.resize_()"
                )
            else:
                raise RuntimeError(
                    f"Unexpected storage type {type(storage_item)} for storage {storage_id}"
                )

        @modal.method()
        def resize_storage(self, storage_id: int, nbytes: int) -> None:
            """
            Resize a storage to accommodate new byte size.

            Propagates laziness - if storage is lazy, keeps it lazy with new size.
            If storage is realized, uses tensor.resize_() for proper resizing.
            Only resizes if nbytes > current storage size.

            Args:
                storage_id: The storage ID to resize
                nbytes: The number of bytes needed for the new storage size

            Returns:
                None
            """
            return self._resize_storage_impl(storage_id, nbytes)

        def _remove_storage_impl(self, storage_id: int) -> None:
            """Implementation of remove_storage without Modal decorators."""
            storages = self._get_storages()
            if storage_id in storages:
                # Clean up tensor cache associated with this storage
                self._cleanup_tensor_cache_for_storage(storage_id)
                # Remove the storage itself
                del storages[storage_id]
                log.info(f"🗑️ Removed storage {storage_id} and associated tensor cache")
            else:
                log.debug(f"Storage {storage_id} not found for removal")

        @modal.method()
        def remove_storage(self, storage_id: int) -> None:
            """
            Remove a storage from the registry.

            Args:
                storage_id: The storage ID

            Returns:
                None
            """
            return self._remove_storage_impl(storage_id)

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

            # Note: storages not needed since metadata extraction doesn't require storage mapping

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

        def _create_storage_for_tensor(self, tensor, storages, device):
            """Helper method to create storage for a model parameter/buffer tensor."""
            import random
            import torch

            # Generate a unique storage ID (using existing pattern from storage system)
            MAX_STORAGE_ID = 2**63 - 1
            MIN_STORAGE_ID = 1

            # Simple ID generation for now - in production should use proper collision detection
            storage_id = random.randint(MIN_STORAGE_ID, MAX_STORAGE_ID)
            while storage_id in storages:
                storage_id = random.randint(MIN_STORAGE_ID, MAX_STORAGE_ID)

            # Ensure the tensor is contiguous and on the correct device
            contiguous_tensor = tensor.detach().contiguous().to(device)
            storage_nbytes = contiguous_tensor.untyped_storage().nbytes()

            log.debug(f"Creating storage for tensor: shape={contiguous_tensor.shape}, dtype={contiguous_tensor.dtype}")
            log.debug(f"Original tensor data range: [{contiguous_tensor.min().item():.6f}, {contiguous_tensor.max().item():.6f}]")
            log.debug(f"Storage size: {storage_nbytes} bytes")

            # Create 1D uint8 tensor with the same number of bytes
            storage_tensor = torch.empty(
                storage_nbytes, dtype=torch.uint8, device=device
            )

            # Perform a direct memory copy from the contiguous tensor's storage
            storage_tensor.untyped_storage().copy_(contiguous_tensor.untyped_storage())

            # Verify the copy worked by checking a reconstructed view
            try:
                # Reconstruct the original tensor from the uint8 storage to verify
                test_tensor = torch.empty(0, dtype=contiguous_tensor.dtype, device=device)
                test_tensor.set_(storage_tensor.untyped_storage(), 0, contiguous_tensor.shape, contiguous_tensor.stride())
                log.debug(f"Reconstructed tensor data range: [{test_tensor.min().item():.6f}, {test_tensor.max().item():.6f}]")

                # Check if data was preserved
                if torch.allclose(contiguous_tensor, test_tensor, atol=1e-6):
                    log.debug("✅ Storage copy verification successful")
                else:
                    log.error("❌ Storage copy verification failed - data was not preserved")
            except Exception as e:
                log.error(f"❌ Storage copy verification error: {e}")

            # Store in storage mapping as realized tensor
            storages[storage_id] = storage_tensor

            log.debug(f"✅ Created realized storage {storage_id}")

            return storage_id

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
            local_storage_ids: List[int],
            parameter_names: List[str],
        ) -> None:
            """Implementation of link_model_tensors without Modal decorators."""

            if len(local_storage_ids) != len(parameter_names):
                raise ValueError(
                    f"Mismatch between storage IDs ({len(local_storage_ids)}) and parameter names ({len(parameter_names)})"
                )

            log.info(
                f"🔗 Linking {len(local_storage_ids)} local storage IDs to remote model parameters"
            )
            log.debug(
                f"Parameter names to link: {parameter_names[:5]}..."
            )  # Show first 5

            storages = self._get_storages()
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
            for local_storage_id, param_name in zip(local_storage_ids, parameter_names):
                if param_name not in all_tensors:
                    log.warning(f"Parameter {param_name} not found in model tensors")
                    continue

                # Get the actual tensor data
                remote_tensor = all_tensors[param_name]

                log.debug(
                    f"Linking local storage {local_storage_id} to parameter {param_name}"
                )
                log.debug(
                    f"Remote tensor type: {type(remote_tensor)}, shape: {remote_tensor.shape}"
                )
                log.debug(f"Remote tensor dtype: {remote_tensor.dtype}")

                # Create realized storage with actual tensor data for model parameters
                device = remote_tensor.device
                remote_storage_id = self._create_storage_for_tensor(remote_tensor, storages, device)
                
                # Link local storage ID to the remote realized storage
                storages[local_storage_id] = storages[remote_storage_id]
                
                log.debug(f"Created realized storage for {param_name}: local={local_storage_id}, remote={remote_storage_id}")

                # Cache the tensor with its metadata
                tensor_shape = list(remote_tensor.shape)
                tensor_stride = list(remote_tensor.stride())
                tensor_offset = remote_tensor.storage_offset()
                tensor_dtype = str(remote_tensor.dtype)

                self._cache_tensor(
                    local_storage_id,
                    tensor_shape,
                    tensor_stride,
                    tensor_offset,
                    tensor_dtype,
                    remote_tensor,
                )
                log.debug(f"Cached parameter {param_name} with metadata key")

                linked_count += 1

            log.info(
                f"✅ Model tensor linking complete: {linked_count}/{len(local_storage_ids)} links successful"
            )

        @modal.method()
        def link_model_tensors(
            self,
            local_storage_ids: List[int],
            parameter_names: List[str],
        ) -> None:
            """
            Link local mycelya tensor storage IDs to remote model parameter tensors.

            This method establishes linkage between local storage IDs and remote model parameters,
            caching tensors with metadata-based keys for efficient lookup.

            Args:
                local_storage_ids: List of local storage IDs from created mycelya tensors
                parameter_names: List of parameter names corresponding to each storage ID

            Returns:
                None
            """
            return self._link_model_tensors_impl(local_storage_ids, parameter_names)

        def _execute_aten_operation_impl(
            self,
            op_name: str,
            input_tensor_metadata: List[Dict[str, Any]],
            output_storage_ids: List[Union[int, None]],
            args: List[Any],
            kwargs: Dict[str, Any],
            return_metadata: bool = False,
        ) -> Union[None, List[Dict[str, Any]]]:
            """Implementation of execute_aten_operation without Modal decorators."""
            # Import torch and tree_map locally to avoid serialization issues
            import torch
            from torch.utils._pytree import tree_map

            # Extract storage IDs from input metadata
            input_storage_ids = [
                metadata["storage_id"] for metadata in input_tensor_metadata
            ]

            log.info(f"🚀 Modal {gpu_type} executing: {op_name}")
            log.debug(f"Input storage IDs: {input_storage_ids}")
            log.debug(f"Output storage IDs: {output_storage_ids}")

            # Get storage mapping
            storages = self._get_storages()

            # Reconstruct input tensors - use cached tensors when available
            input_tensors = []
            for _i, metadata in enumerate(input_tensor_metadata):
                # Use the _construct_tensor_from_storage which already handles metadata-based caching
                tensor = self._construct_tensor_from_storage(
                    storage_id=metadata["storage_id"],
                    shape=metadata["shape"],
                    stride=metadata["stride"],
                    storage_offset=metadata["storage_offset"],
                    dtype=metadata["dtype"],
                )
                input_tensors.append(tensor)

            log.debug(f"📥 Reconstructed {len(input_tensors)} input tensors")

            # Replace tensor placeholders with actual reconstructed input tensors using tree_map
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
                f"{len(input_tensors)} inputs, {len([s for s in output_storage_ids if s is not None])} outputs to update"
            )

            # Execute the operation on input tensors - this will create result tensors
            result = op(*processed_args, **processed_kwargs)

            # Update storage mapping for output tensors
            result_tensors = (
                [result]
                if isinstance(result, torch.Tensor)
                else list(result)
                if isinstance(result, (list, tuple))
                else []
            )

            # Enforce contract: output_storage_ids and result_tensors must have same length
            if len(output_storage_ids) != len(result_tensors):
                raise RuntimeError(
                    f"Contract violation in {op_name}: output_storage_ids length ({len(output_storage_ids)}) "
                    f"!= result_tensors length ({len(result_tensors)}). This indicates a bug in the client-side "
                    f"meta tensor execution or output tensor creation."
                )

            for _i, (storage_id, result_tensor) in enumerate(
                zip(output_storage_ids, result_tensors)
            ):
                # Store just the byte count for the result tensor
                storage_nbytes = result_tensor.untyped_storage().nbytes()
                storages[storage_id] = storage_nbytes

                # Always cache result tensors with their metadata
                result_shape = list(result_tensor.shape)
                result_stride = list(result_tensor.stride())
                result_offset = result_tensor.storage_offset()
                result_dtype = str(result_tensor.dtype)

                self._cache_tensor(
                    storage_id,
                    result_shape,
                    result_stride,
                    result_offset,
                    result_dtype,
                    result_tensor,
                )
                log.debug("💾 Cached output tensor with metadata")

            log.debug(f"📦 Updated {len(output_storage_ids)} output storage mappings")

            # Return metadata if requested
            if return_metadata:
                output_metadata = []
                for i, result_tensor in enumerate(result_tensors):
                    if i < len(output_storage_ids):
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
            output_storage_ids: List[Union[int, None]],
            args: List[Any],
            kwargs: Dict[str, Any],
            return_metadata: bool = False,
        ) -> Union[None, List[Dict[str, Any]]]:
            """
            Execute an operation with separated input metadata and output storage IDs.

            This method handles operations where input tensor metadata and output storage IDs
            are explicitly separated. Tensors are cached automatically based on their metadata.

            Args:
                op_name: The operation name to execute
                input_tensor_metadata: List of metadata for input tensors only
                output_storage_ids: List of storage IDs to update with results (all output tensors)
                args: Operation arguments (with tensor placeholders)
                kwargs: Operation keyword arguments (with tensor placeholders)
                return_metadata: If True, return output tensor metadata instead of None

            Returns:
                None for normal operations, or List[Dict] of output tensor metadata if return_metadata=True
            """
            return self._execute_aten_operation_impl(
                op_name,
                input_tensor_metadata,
                output_storage_ids,
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
                    if method_name == "create_storage":
                        result = self._create_storage_impl(*args, **kwargs)
                    elif method_name == "update_storage":
                        result = self._update_storage_impl(*args, **kwargs)
                    elif method_name == "get_storage_data":
                        result = self._get_storage_data_impl(*args, **kwargs)
                    elif method_name == "resize_storage":
                        result = self._resize_storage_impl(*args, **kwargs)
                    elif method_name == "remove_storage":
                        result = self._remove_storage_impl(*args, **kwargs)
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
