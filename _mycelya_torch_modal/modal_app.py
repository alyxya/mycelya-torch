# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Modal remote execution app for torch_remote extension.

This module handles all Modal-specific functionality including:
- Dynamic device-specific app creation for different GPU types
- Remote execution of PyTorch operations
- Dynamic GPU selection and configuration

Part of: torch_remote PyTorch extension
"""

import io
import logging
from typing import Any, Dict, List, Tuple, Union

import modal

log = logging.getLogger(__name__)

# Create simplified image with just PyTorch and CUDA support
image = modal.Image.debian_slim().pip_install("numpy", "torch")


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
    )
    class PytorchServer:
        def _get_storages(self):
            """Get or create storage mapping for this server instance."""
            import torch

            if not hasattr(self, "_storages"):
                # storage_id -> torch.Storage or int (for lazy allocation)
                self._storages: Dict[int, Union[torch.Storage, int]] = {}

            return self._storages

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

            Args:
                storage_id: The storage ID to look up
                shape: Tensor shape
                stride: Tensor stride
                storage_offset: Storage offset
                dtype: Data type as string

            Returns:
                Reconstructed tensor on CUDA device

            Raises:
                KeyError: If storage_id is not found
                RuntimeError: If storage is lazy-allocated
            """
            import torch

            storages = self._get_storages()

            # Validate storage exists
            if storage_id not in storages:
                available_ids = list(storages.keys())
                log.error(f"❌ MISSING Storage ID {storage_id}")
                log.error(f"📋 Available Storage IDs on Modal: {available_ids}")
                raise KeyError(f"Storage ID {storage_id} not found")

            storage = storages[storage_id]

            # Parse dtype string back to torch.dtype
            dtype_str = dtype.replace("torch.", "")
            torch_dtype = getattr(torch, dtype_str)

            # Handle lazy storage (int) - realize it
            if isinstance(storage, int):
                nbytes = storage
                device = torch.device("cuda")
                tensor = torch.empty(nbytes, dtype=torch.uint8, device=device)
                realized_storage = tensor.untyped_storage()

                # Update the storages mapping with realized storage
                storages[storage_id] = realized_storage
                log.info(f"🔄 REALIZED lazy storage {storage_id} ({nbytes} bytes)")

                storage = realized_storage

            # Determine device to use (prefer storage device if available)
            target_device = storage.device if hasattr(storage, "device") else "cuda"

            # Reconstruct tensor using storage + parameters
            tensor = torch.empty(0, dtype=torch_dtype, device=target_device).set_(
                storage, storage_offset, shape, stride
            )

            return tensor


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
            # Store storage as lazy allocation (just the byte count)
            storages = self._get_storages()

            # Check if storage already exists
            if storage_id in storages:
                log.warning(f"⚠️ Storage ID {storage_id} already exists, overwriting")

            # Always store as int for lazy allocation
            storages[storage_id] = nbytes
            log.info(f"📝 LAZY Storage ID {storage_id} registered ({nbytes} bytes)")

        @modal.method()
        def update_storage(
            self,
            storage_id: int,
            tensor_data: bytes,
            shape: List[int],
            stride: List[int],
            storage_offset: int,
            dtype: str
        ) -> None:
            """
            Update an existing storage with tensor data.

            Optimized to use fast path when updating entire storage, slow path for views.

            Args:
                storage_id: Storage ID to update
                tensor_data: Serialized tensor data
                shape: Shape of the target view
                stride: Stride of the target view
                storage_offset: Storage offset of the target view
                dtype: Data type of the target view

            Returns:
                None
            """
            import math

            import torch

            # Parse dtype to get element size
            torch_dtype = getattr(torch, dtype.replace("torch.", ""))
            element_size = torch.empty(0, dtype=torch_dtype).element_size()

            # Calculate tensor properties
            tensor_numel = math.prod(shape)
            tensor_bytes = tensor_numel * element_size

            # Check if tensor is contiguous by comparing with expected contiguous stride
            expected_stride = []
            running_size = 1
            for dim_size in reversed(shape):
                expected_stride.insert(0, running_size)
                running_size *= dim_size
            is_contiguous = stride == expected_stride

            # Get storages
            storages = self._get_storages()

            # Fast path: view uses entire storage contiguously from offset 0
            if storage_offset == 0 and is_contiguous and storage_id in storages:
                storage = storages[storage_id]
                storage_matches = False

                if isinstance(storage, int):
                    # Lazy storage - check if size matches
                    storage_matches = (storage == tensor_bytes)
                elif isinstance(storage, torch.Storage):
                    # Realized storage - check if size matches
                    storage_matches = (storage.nbytes() == tensor_bytes)

                if storage_matches:
                    # Direct replacement (fast path)
                    buffer = io.BytesIO(tensor_data)
                    tensor = torch.load(buffer, map_location="cuda", weights_only=True)
                    storages[storage_id] = tensor.untyped_storage()
                    log.info(
                        f"📥 FAST Updated Storage ID {storage_id} on Modal (shape: {tensor.shape})"
                    )
                    return

            # Slow path: in-place view update
            log.info(f"📥 SLOW Updating view of Storage ID {storage_id} (offset: {storage_offset}, contiguous: {is_contiguous})")

            # 1. Construct the target view tensor directly using existing method
            view_tensor = self._construct_tensor_from_storage(
                storage_id=storage_id,
                shape=shape,
                stride=stride,
                storage_offset=storage_offset,
                dtype=dtype
            )

            # 2. Deserialize the incoming tensor and copy it in-place
            buffer = io.BytesIO(tensor_data)
            source_tensor = torch.load(buffer, map_location="cuda", weights_only=True)
            view_tensor.copy_(source_tensor)

            log.info(
                f"📥 SLOW Updated view of Storage ID {storage_id} on Modal (shape: {shape})"
            )

        @modal.method()
        def get_storage_data(
            self,
            storage_id: int,
            shape: List[int],
            stride: List[int],
            storage_offset: int,
            dtype: str,
        ) -> bytes:
            """
            Retrieve current storage data by storage ID for transfer to client.
            Returns the view's data as contiguous.

            Args:
                storage_id: The storage ID
                shape: Tensor shape for view
                stride: Tensor stride for view
                storage_offset: Storage offset for view
                dtype: Tensor data type

            Returns:
                Serialized tensor data (contiguous representation of the view)
            """
            import torch

            # Use helper method to construct tensor with specified view parameters
            tensor = self._construct_tensor_from_storage(
                storage_id=storage_id,
                shape=shape,
                stride=stride,
                storage_offset=storage_offset,
                dtype=dtype,
            )
            log.info(
                f"📦 Serializing view of storage {storage_id}: "
                f"shape={shape}, stride={stride}, offset={storage_offset}"
            )

            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            return buffer.getvalue()

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
            import torch

            storages = self._get_storages()
            if storage_id not in storages:
                log.warning(f"Storage ID {storage_id} not found for resize")
                return

            old_storage = storages[storage_id]

            # Handle lazy storage (int) - propagate laziness
            if isinstance(old_storage, int):
                current_bytes = old_storage

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

            # Handle realized storage (torch.Storage)
            elif isinstance(old_storage, torch.Storage):
                current_bytes = old_storage.nbytes()

                # Check if resize is actually needed (should be bigger)
                if nbytes <= current_bytes:
                    log.debug(
                        f"Realized storage {storage_id} resize skipped: "
                        f"nbytes ({nbytes}) <= current_bytes ({current_bytes})"
                    )
                    return  # No-op

                device = old_storage.device

                # Create a tensor that uses the existing storage to leverage tensor.resize_()
                # Use uint8 dtype for byte-level operations (1 byte = 1 element)
                temp_tensor = torch.empty(0, dtype=torch.uint8, device=device)
                temp_tensor.set_(old_storage, storage_offset=0, size=(current_bytes,))

                # Resize to new byte count (1 byte = 1 element for uint8)
                temp_tensor.resize_([nbytes])

                # Get the new storage after resize
                new_storage = temp_tensor.untyped_storage()

                # Update the storage mapping
                storages[storage_id] = new_storage
                log.info(
                    f"🔄 REALIZED Resized storage {storage_id} from {current_bytes} "
                    f"to {nbytes} bytes using tensor.resize_()"
                )
            else:
                raise RuntimeError(f"Unexpected storage type {type(old_storage)} for storage {storage_id}")

        @modal.method()
        def remove_storage(self, storage_id: int) -> None:
            """
            Remove a storage from the registry.

            Args:
                storage_id: The storage ID

            Returns:
                None
            """
            storages = self._get_storages()
            if storage_id in storages:
                del storages[storage_id]
                log.info(f"🗑️ Removed storage {storage_id}")
            else:
                log.debug(f"Storage {storage_id} not found for removal")

        @modal.method()
        def execute_aten_operation(
            self,
            op_name: str,
            input_tensor_metadata: List[Dict[str, Any]],
            output_storage_ids: List[Union[int, None]],
            args: List[Any],
            kwargs: Dict[str, Any],
        ) -> None:
            """
            Execute an operation with separated input metadata and output storage IDs.

            This method handles operations where input tensor metadata and output storage IDs
            are explicitly separated, making the interface cleaner and more explicit.

            Args:
                op_name: The operation name to execute
                input_tensor_metadata: List of metadata for input tensors only
                output_storage_ids: List of storage IDs to update with results (None for outputs to ignore)
                args: Operation arguments (with tensor placeholders)
                kwargs: Operation keyword arguments (with tensor placeholders)

            Returns:
                None (operation results are written to output tensors)
            """
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

            # Reconstruct input tensors from storage and metadata
            input_tensors = [
                self._construct_tensor_from_storage(
                    storage_id=metadata["storage_id"],
                    shape=metadata["shape"],
                    stride=metadata["stride"],
                    storage_offset=metadata["storage_offset"],
                    dtype=metadata["dtype"],
                )
                for metadata in input_tensor_metadata
            ]

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

            for i, storage_id in enumerate(output_storage_ids):
                if storage_id is not None and i < len(result_tensors):
                    # Check if output storage is not lazy (warn if overwriting realized storage)
                    if storage_id in storages and not isinstance(storages[storage_id], int):
                        log.warning(
                            f"⚠️ Output storage {storage_id} is not lazy (type: {type(storages[storage_id])}), "
                            f"overwriting with result from {op_name}"
                        )

                    storages[storage_id] = result_tensors[i].untyped_storage()

            log.debug(
                f"📦 Updated {len([s for s in output_storage_ids if s is not None])} output storage mappings"
            )

            log.info(f"✅ Completed: {op_name}")
            return

    return app, PytorchServer
