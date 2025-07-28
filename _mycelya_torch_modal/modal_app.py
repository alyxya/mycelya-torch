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
        min_containers=1,
    )
    class PytorchServer:
        def _get_storages(self):
            """Get or create storage mapping for this server instance."""
            import torch

            if not hasattr(self, "_storages"):
                # storage_id -> torch.Tensor (1D uint8 CUDA) or int (for lazy allocation)
                self._storages: Dict[int, Union[torch.Tensor, int]] = {}

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
                # Create 1D uint8 CUDA tensor to hold the storage
                storage_tensor = torch.empty(nbytes, dtype=torch.uint8, device=device)

                # Update the storages mapping with realized tensor
                storages[storage_id] = storage_tensor
                log.info(f"🔄 REALIZED lazy storage {storage_id} ({nbytes} bytes)")

                storage = storage_tensor

            # Get untyped storage from the stored tensor
            if isinstance(storage, torch.Tensor):
                untyped_storage = storage.untyped_storage()
                target_device = storage.device
            else:
                raise RuntimeError(f"Unexpected storage type {type(storage)} for storage {storage_id}")

            # Reconstruct tensor using storage + parameters
            tensor = torch.empty(0, dtype=torch_dtype, device=target_device).set_(
                untyped_storage, storage_offset, shape, stride
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

            # Deserialize source tensor from torch.save bytes
            import io
            buffer = io.BytesIO(raw_data)
            source_tensor = torch.load(buffer, map_location="cpu", weights_only=False)

            # Ensure tensor is on CPU and has the expected properties
            if source_tensor.device.type != "cpu":
                source_tensor = source_tensor.cpu()

            storage_item = storages[storage_id]

            # Check if storage is lazy
            if isinstance(storage_item, int):
                # Move source tensor to CUDA for storage
                cuda_source = source_tensor.to("cuda")
                expected_bytes = storage_item
                actual_bytes = cuda_source.untyped_storage().nbytes()

                if expected_bytes != actual_bytes:
                    raise RuntimeError(
                        f"Storage size mismatch for storage {storage_id}: "
                        f"expected {expected_bytes} bytes, got {actual_bytes} bytes"
                    )

                log.info(f"📥 LAZY Storage {storage_id} update triggering realization with {expected_bytes} bytes")
                # Create 1D uint8 tensor from the CUDA source tensor's storage
                storage_tensor = torch.empty(actual_bytes, dtype=torch.uint8, device="cuda")
                storage_tensor.untyped_storage().copy_(cuda_source.untyped_storage())
                storages[storage_id] = storage_tensor
                log.info(f"📥 LAZY Updated Storage ID {storage_id} on Modal (realized: shape: {source_tensor.shape})")
                return

            # Storage is realized (torch.Tensor) - always use in-place view update
            log.info(f"📥 IN-PLACE Updating view of Storage ID {storage_id} (offset: {target_storage_offset})")

            # Construct the target view tensor directly using existing method
            target_tensor = self._construct_tensor_from_storage(
                storage_id=storage_id,
                shape=target_shape,
                stride=target_stride,
                storage_offset=target_storage_offset,
                dtype=target_dtype
            )

            # Copy source tensor data to target view in-place
            target_tensor.copy_(source_tensor.to(target_tensor.device))

            log.info(
                f"📥 IN-PLACE Updated view of Storage ID {storage_id} on Modal (target_shape: {target_shape})"
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
                storage_id, raw_data, source_shape, source_stride, source_storage_offset, source_dtype,
                target_shape, target_stride, target_storage_offset, target_dtype
            )

        def _get_storage_data_impl(
            self,
            storage_id: int,
        ) -> bytes:
            """Implementation of get_storage_data without Modal decorators."""
            # Get storages
            storages = self._get_storages()

            if storage_id not in storages:
                raise RuntimeError(f"Storage ID {storage_id} not found")

            storage_item = storages[storage_id]

            # Handle lazy storage
            if isinstance(storage_item, int):
                raise RuntimeError(f"Storage ID {storage_id} is lazy (not realized). Cannot retrieve data.")

            # Storage is realized (torch.Tensor)
            storage_tensor = storage_item
            log.info(f"📦 Retrieving tensor data for storage {storage_id} ({storage_tensor.untyped_storage().nbytes()} bytes)")

            # Convert CUDA tensor to CPU and serialize with torch.save
            import io

            import torch
            cpu_tensor = storage_tensor.cpu()
            buffer = io.BytesIO()
            torch.save(cpu_tensor, buffer)
            return buffer.getvalue()

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

            # Handle realized storage (torch.Tensor)
            elif isinstance(old_storage, torch.Tensor):
                current_bytes = old_storage.untyped_storage().nbytes()

                # Check if resize is actually needed (should be bigger)
                if nbytes <= current_bytes:
                    log.debug(
                        f"Realized storage {storage_id} resize skipped: "
                        f"nbytes ({nbytes}) <= current_bytes ({current_bytes})"
                    )
                    return  # No-op

                # Resize the existing 1D uint8 tensor
                old_storage.resize_([nbytes])

                log.info(
                    f"🔄 REALIZED Resized storage {storage_id} from {current_bytes} "
                    f"to {nbytes} bytes using tensor.resize_()"
                )
            else:
                raise RuntimeError(f"Unexpected storage type {type(old_storage)} for storage {storage_id}")

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
                del storages[storage_id]
                log.info(f"🗑️ Removed storage {storage_id}")
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
                if i < len(result_tensors):
                    # Check if output storage is not lazy (warn if overwriting realized storage)
                    if storage_id in storages and not isinstance(storages[storage_id], int):
                        log.warning(
                            f"⚠️ Output storage {storage_id} is not lazy (type: {type(storages[storage_id])}), "
                            f"overwriting with result from {op_name}"
                        )

                    # Store the result tensor as a 1D uint8 tensor
                    result_tensor = result_tensors[i]
                    storage_nbytes = result_tensor.untyped_storage().nbytes()
                    # Create 1D uint8 tensor with the same storage
                    storage_tensor = torch.empty(storage_nbytes, dtype=torch.uint8, device=result_tensor.device)
                    storage_tensor.untyped_storage().copy_(result_tensor.untyped_storage())
                    storages[storage_id] = storage_tensor

            log.debug(
                f"📦 Updated {len(output_storage_ids)} output storage mappings"
            )

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
                            "storage_nelements": result_tensor.untyped_storage().nbytes() // result_tensor.element_size(),
                        }
                        output_metadata.append(metadata)

                log.info(f"✅ Completed: {op_name} (returning metadata for {len(output_metadata)} outputs)")
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
            are explicitly separated, making the interface cleaner and more explicit.

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
                op_name, input_tensor_metadata, output_storage_ids, args, kwargs, return_metadata
            )

        @modal.method()
        def execute_batch(
            self,
            batch_calls: List[Dict[str, Any]]
        ) -> List[Union[None, Any]]:
            """
            Execute a batch of RPC calls in sequence.
            
            This method allows multiple operations to be batched together in a single
            RPC call, reducing network overhead and improving performance.
            
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
            import torch
            
            log.info(f"🚀 BATCH EXECUTE: Processing {len(batch_calls)} batched calls")
            results = []
            
            for i, call in enumerate(batch_calls):
                call_id = call.get("call_id", f"batch_call_{i}")
                method_name = call["method_name"]
                call_type = call["call_type"]
                args = call.get("args", ())
                kwargs = call.get("kwargs", {})
                
                try:
                    log.debug(f"📞 Executing batched call {call_id}: {method_name} ({call_type})")
                    
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
                    else:
                        raise AttributeError(f"Unknown method: {method_name}")
                    
                    # For spawn calls, we return None
                    if call_type == "spawn":
                        results.append(None)
                    else:
                        results.append(result)
                        
                except Exception as e:
                    log.error(f"❌ Batched call {call_id} failed: {method_name} - {e}")
                    
                    # Store the exception as the result
                    results.append(e)
            
            log.info(f"✅ BATCH COMPLETE: Processed {len(batch_calls)} calls, "
                    f"{sum(1 for r in results if not isinstance(r, Exception))} successful")
            
            return results

    return app, PytorchServer
