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
    app = modal.App(f"torch-remote-{machine_id}")

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
                # storage_id -> torch.Storage
                self._storages: Dict[int, torch.Storage] = {}

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

            # Determine device to use (prefer storage device if available)
            target_device = storage.device if hasattr(storage, "device") else "cuda"

            # Reconstruct tensor using storage + parameters
            tensor = torch.empty(0, dtype=torch_dtype, device=target_device).set_(
                storage, storage_offset, shape, stride
            )

            return tensor

        def _construct_tensor_from_metadata(
            self, storage_id: int, metadata: Dict[str, Any]
        ) -> Any:
            """
            Construct a tensor from storage ID and metadata dictionary.

            This is a convenience wrapper around _construct_tensor_from_storage
            that extracts parameters from a metadata dictionary.

            Args:
                storage_id: The storage ID to look up
                metadata: Tensor metadata containing shape, stride, storage_offset, dtype

            Returns:
                Reconstructed tensor on CUDA device
            """
            return self._construct_tensor_from_storage(
                storage_id=storage_id,
                shape=metadata["shape"],
                stride=metadata["stride"],
                storage_offset=metadata["storage_offset"],
                dtype=metadata["dtype"],
            )

        @modal.method()
        def create_storage(self, storage_id: int, nbytes: int) -> None:
            """
            Create a new storage on the remote machine.

            Args:
                storage_id: Specific ID to use for the storage (required)
                nbytes: Number of bytes to allocate for the storage

            Returns:
                None
            """
            import torch

            # Store storage and original tensor data
            storages = self._get_storages()
            storage_id = int(storage_id)

            # Create tensor directly on GPU with exact byte size
            device = torch.device("cuda")
            tensor = torch.empty(nbytes, dtype=torch.uint8, device=device)
            storages[storage_id] = tensor.untyped_storage()
            log.info(
                f"📥 CREATED Storage ID {storage_id} on Modal ({nbytes} bytes)"
            )

        @modal.method()
        def update_storage(self, storage_id: int, tensor_data: bytes) -> None:
            """
            Update an existing storage with tensor data.

            Args:
                storage_id: Storage ID to update
                tensor_data: Serialized tensor data

            Returns:
                None
            """
            import torch

            # Deserialize tensor
            buffer = io.BytesIO(tensor_data)
            tensor = torch.load(buffer, map_location="cpu", weights_only=True)

            # Move to GPU (Modal environment always has CUDA)
            device = torch.device("cuda")
            tensor = tensor.to(device)

            # Update existing storage
            storages = self._get_storages()
            storage_id = int(storage_id)

            storages[storage_id] = tensor.untyped_storage()
            log.info(
                f"📥 UPDATED Storage ID {storage_id} on Modal (shape: {tensor.shape})"
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

            storage_id = int(storage_id)

            # Use helper method to construct tensor with specified view parameters
            tensor = self._construct_tensor_from_storage(
                storage_id=storage_id,
                shape=shape,
                stride=stride,
                storage_offset=storage_offset,
                dtype=dtype,
            )
            # Make contiguous to get only the view's data
            tensor = tensor.contiguous()
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

            This handles the case where resize_ needs more storage space than currently allocated.
            Uses tensor.resize_() to properly handle storage resizing and updates the storage mapping.
            Only resizes if nbytes > current storage size.

            Args:
                storage_id: The storage ID to resize
                nbytes: The number of bytes needed for the new storage size

            Returns:
                None
            """
            import torch

            storages = self._get_storages()
            storage_id = int(storage_id)
            if storage_id not in storages:
                log.warning(f"Storage ID {storage_id} not found for resize")
                return False

            old_storage = storages[storage_id]
            current_bytes = old_storage.nbytes()

            # Check if resize is actually needed (should be bigger)
            if nbytes <= current_bytes:
                log.debug(
                    f"Storage {storage_id} resize skipped: "
                    f"nbytes ({nbytes}) <= current_bytes ({current_bytes})"
                )
                return  # No-op

            device = old_storage.device

            # Create a tensor that uses the existing storage to leverage tensor.resize_()
            # Use uint8 dtype for byte-level operations
            uint8_dtype = torch.uint8
            current_element_count = (
                current_bytes // torch.tensor([], dtype=uint8_dtype).element_size()
            )
            temp_tensor = torch.tensor([], dtype=uint8_dtype, device=device)
            temp_tensor.set_(
                old_storage, storage_offset=0, size=(current_element_count,)
            )

            # Calculate new element count for the resized tensor
            new_element_count = (
                nbytes // torch.tensor([], dtype=uint8_dtype).element_size()
            )

            # Use tensor.resize_() to handle the storage resize properly
            temp_tensor.resize_([new_element_count])

            # Get the new storage after resize
            new_storage = temp_tensor.untyped_storage()

            # Update the storage mapping
            storages[storage_id] = new_storage
            log.info(
                f"🔄 Resized storage {storage_id} from {current_bytes} "
                f"to {nbytes} bytes using tensor.resize_()"
            )

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
            storage_id = int(storage_id)
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
                self._construct_tensor_from_metadata(metadata["storage_id"], metadata)
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
                    storages[int(storage_id)] = result_tensors[i].untyped_storage()

            log.debug(
                f"📦 Updated {len([s for s in output_storage_ids if s is not None])} output storage mappings"
            )

            log.info(f"✅ Completed: {op_name}")
            return

    return app, PytorchServer
