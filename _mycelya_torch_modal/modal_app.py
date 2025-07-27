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
                # storage_id -> torch.UntypedStorage or int (for lazy allocation)
                self._storages: Dict[int, Union[torch.UntypedStorage, int]] = {}

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
                raise RuntimeError(f"Storage ID {storage_id} already exists")

            # Always store as int for lazy allocation
            storages[storage_id] = nbytes
            log.info(f"📝 LAZY Storage ID {storage_id} registered ({nbytes} bytes)")

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
            import math

            import torch

            # Get storages
            storages = self._get_storages()

            if storage_id not in storages:
                raise RuntimeError(f"Storage ID {storage_id} not found")

            # Create source tensor from raw data
            source_torch_dtype = getattr(torch, source_dtype.replace("torch.", ""))
            source_storage = torch.UntypedStorage.from_buffer(raw_data, dtype=torch.uint8)
            source_tensor = torch.empty(0, dtype=source_torch_dtype, device="cpu")
            source_tensor.set_(source_storage, source_storage_offset, source_shape, source_stride)

            storage_item = storages[storage_id]

            # Check if storage is lazy
            if isinstance(storage_item, int):
                source_tensor = source_tensor.to("cuda")
                expected_bytes = storage_item
                actual_bytes = source_tensor.untyped_storage().nbytes()

                if expected_bytes != actual_bytes:
                    raise RuntimeError(
                        f"Storage size mismatch for storage {storage_id}: "
                        f"expected {expected_bytes} bytes, got {actual_bytes} bytes"
                    )

                log.info(f"📥 LAZY Storage {storage_id} update triggering realization with {expected_bytes} bytes")
                # Force storage realization by using the source tensor's storage
                storages[storage_id] = source_tensor.untyped_storage()
                log.info(f"📥 LAZY Updated Storage ID {storage_id} on Modal (realized: shape: {source_shape})")
                return

            # Storage is realized (torch.UntypedStorage) - always use in-place view update
            log.info(f"📥 IN-PLACE Updating view of Storage ID {storage_id} (offset: {target_storage_offset})")

            # Construct the target view tensor directly using existing method
            target_tensor = self._construct_tensor_from_storage(
                storage_id=storage_id,
                shape=target_shape,
                stride=target_stride,
                storage_offset=target_storage_offset,
                dtype=target_dtype
            )

            # 2. Copy source tensor data to target view in-place
            target_tensor.copy_(source_tensor)

            log.info(
                f"📥 IN-PLACE Updated view of Storage ID {storage_id} on Modal (target_shape: {target_shape})"
            )

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

            # Get storages
            storages = self._get_storages()

            if storage_id not in storages:
                raise RuntimeError(f"Storage ID {storage_id} not found")

            storage_item = storages[storage_id]

            # Handle lazy storage
            if isinstance(storage_item, int):
                raise RuntimeError(f"Storage ID {storage_id} is lazy (not realized). Cannot retrieve data.")

            # Storage is realized (torch.UntypedStorage)
            storage = storage_item
            log.info(f"📦 Retrieving raw storage data for storage {storage_id} ({storage.nbytes()} bytes)")

            # Return raw untyped storage bytes
            return bytes(storage)

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

            # Handle realized storage (torch.UntypedStorage)
            elif isinstance(old_storage, torch.UntypedStorage):
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
            return_metadata: bool = False,
        ) -> Union[None, List[Dict[str, Any]]]:
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
                return_metadata: If True, return output tensor metadata instead of None

            Returns:
                None for normal operations, or List[Dict] of output tensor metadata if return_metadata=True
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

            # Return metadata if requested
            if return_metadata:
                output_metadata = []
                for i, result_tensor in enumerate(result_tensors):
                    if i < len(output_storage_ids) and output_storage_ids[i] is not None:
                        metadata = {
                            "shape": list(result_tensor.shape),
                            "dtype": str(result_tensor.dtype),
                            "stride": list(result_tensor.stride()),
                            "storage_offset": result_tensor.storage_offset(),
                            "storage_nelements": result_tensor.untyped_storage().nbytes() // result_tensor.element_size(),
                        }
                        output_metadata.append(metadata)

                log.info(f"✅ Completed: {op_name} (returning metadata for {len(output_metadata)} outputs)")
                return output_metadata

            log.info(f"✅ Completed: {op_name}")
            return None

    return app, PytorchServer
