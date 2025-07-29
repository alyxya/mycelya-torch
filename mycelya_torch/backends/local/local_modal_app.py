# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Local execution app for mycelya_torch extension.

This module provides a local execution environment that mimics the Modal app
but runs everything on CPU instead of CUDA for development and testing.
"""

import logging
from typing import Any, Dict, List, Tuple, Union

log = logging.getLogger(__name__)


def create_local_modal_app_for_gpu(
    gpu_type: str,
    machine_id: str,
    timeout: int,
    retries: int,
) -> Tuple[Any, Any]:
    """
    Create a local execution app and class for a specific GPU type.

    This mimics the Modal app structure but executes everything locally on CPU.

    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB") - simulated
        machine_id: The machine ID (e.g., "local-t4-f3a7d67e")
        timeout: Function timeout in seconds (unused for local)
        retries: Number of retries on failure (unused for local)

    Returns:
        Tuple of (None, local_server_class) for local execution
    """

    class LocalPytorchServer:
        def _get_storages(self):
            """Get or create storage mapping for this server instance."""
            import torch

            if not hasattr(self, "_storages"):
                # storage_id -> torch.Tensor (1D uint8 CPU) or int (for lazy allocation)
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
                Reconstructed tensor on CPU device

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
                log.error(f"📋 Available Storage IDs on Local: {available_ids}")
                raise KeyError(f"Storage ID {storage_id} not found")

            storage = storages[storage_id]

            # Parse dtype string back to torch.dtype
            dtype_str = dtype.replace("torch.", "")
            torch_dtype = getattr(torch, dtype_str)

            # Handle lazy storage (int) - realize it on CPU
            if isinstance(storage, int):
                nbytes = storage
                device = torch.device("cpu")  # Use CPU instead of CUDA
                # Create 1D uint8 CPU tensor to hold the storage
                storage_tensor = torch.empty(nbytes, dtype=torch.uint8, device=device)

                # Update the storages mapping with realized tensor
                storages[storage_id] = storage_tensor
                log.info(f"🔄 REALIZED lazy storage {storage_id} ({nbytes} bytes) on CPU")

                storage = storage_tensor

            # Get untyped storage from the stored tensor
            if isinstance(storage, torch.Tensor):
                target_device = storage.device
                # Reconstruct tensor using storage + parameters
                tensor = torch.empty(0, dtype=torch_dtype, device=target_device).set_(
                    storage.untyped_storage(), storage_offset, shape, stride
                )
            else:
                raise RuntimeError(f"Unexpected storage type {type(storage)} for storage {storage_id}")

            return tensor

        def _create_storage_impl(self, storage_id: int, nbytes: int) -> None:
            """Implementation of create_storage for local execution."""
            # Store storage as lazy allocation (just the byte count)
            storages = self._get_storages()

            # Check if storage already exists
            if storage_id in storages:
                raise RuntimeError(f"Storage ID {storage_id} already exists")

            # Always store as int for lazy allocation
            storages[storage_id] = nbytes
            log.info(f"📝 LAZY Storage ID {storage_id} registered ({nbytes} bytes) for local execution")

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
            """Implementation of update_storage for local execution."""
            import torch

            # Get storages
            storages = self._get_storages()

            if storage_id not in storages:
                raise RuntimeError(f"Storage ID {storage_id} not found")

            # Deserialize source tensor from torch.save bytes
            import io
            buffer = io.BytesIO(raw_data)
            source_tensor = torch.load(buffer, map_location="cpu", weights_only=False)

            # Ensure tensor is on CPU
            if source_tensor.device.type != "cpu":
                source_tensor = source_tensor.cpu()

            storage_item = storages[storage_id]

            # Check if storage is lazy
            if isinstance(storage_item, int):
                # Keep source tensor on CPU for storage
                expected_bytes = storage_item
                source_storage = source_tensor.untyped_storage()
                actual_bytes = source_storage.nbytes()

                if expected_bytes != actual_bytes:
                    raise RuntimeError(
                        f"Storage size mismatch for storage {storage_id}: "
                        f"expected {expected_bytes} bytes, got {actual_bytes} bytes"
                    )

                log.info(f"📥 LAZY Storage {storage_id} update triggering realization with {expected_bytes} bytes on CPU")
                # Create 1D uint8 tensor from the CPU source tensor's storage
                storage_tensor = torch.empty(actual_bytes, dtype=torch.uint8, device="cpu")
                storage_tensor.untyped_storage().copy_(source_storage)
                storages[storage_id] = storage_tensor
                log.info(f"📥 LAZY Updated Storage ID {storage_id} on Local CPU (realized: shape: {source_tensor.shape})")
                return

            # Storage is realized (torch.Tensor) - always use in-place view update
            log.info(f"📥 IN-PLACE Updating view of Storage ID {storage_id} (offset: {target_storage_offset}) on CPU")

            # Construct the target view tensor directly using existing method
            target_tensor = self._construct_tensor_from_storage(
                storage_id=storage_id,
                shape=target_shape,
                stride=target_stride,
                storage_offset=target_storage_offset,
                dtype=target_dtype
            )

            # Copy source tensor data to target view in-place (both should be on CPU)
            target_tensor.copy_(source_tensor.to(target_tensor.device))

            log.info(
                f"📥 IN-PLACE Updated view of Storage ID {storage_id} on Local CPU (target_shape: {target_shape})"
            )

        def _get_storage_data_impl(
            self,
            storage_id: int,
        ) -> bytes:
            """Implementation of get_storage_data for local execution."""
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
            log.info(f"📦 Retrieving tensor data for storage {storage_id} ({storage_tensor.untyped_storage().nbytes()} bytes) from CPU")

            # Tensor is already on CPU, serialize with torch.save
            import io
            import torch
            
            buffer = io.BytesIO()
            torch.save(storage_tensor, buffer)
            return buffer.getvalue()

        def _resize_storage_impl(self, storage_id: int, nbytes: int) -> None:
            """Implementation of resize_storage for local execution."""
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
                    f"to {nbytes} bytes (kept lazy) on CPU"
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
                    f"to {nbytes} bytes using tensor.resize_() on CPU"
                )
            else:
                raise RuntimeError(f"Unexpected storage type {type(old_storage)} for storage {storage_id}")

        def _remove_storage_impl(self, storage_id: int) -> None:
            """Implementation of remove_storage for local execution."""
            storages = self._get_storages()
            if storage_id in storages:
                del storages[storage_id]
                log.info(f"🗑️ Removed storage {storage_id} from local CPU")
            else:
                log.debug(f"Storage {storage_id} not found for removal")

        def _execute_aten_operation_impl(
            self,
            op_name: str,
            input_tensor_metadata: List[Dict[str, Any]],
            output_storage_ids: List[Union[int, None]],
            args: List[Any],
            kwargs: Dict[str, Any],
            return_metadata: bool = False,
        ) -> Union[None, List[Dict[str, Any]]]:
            """Implementation of execute_aten_operation for local execution."""
            # Import torch and tree_map locally
            import torch
            from torch.utils._pytree import tree_map

            # Extract storage IDs from input metadata
            input_storage_ids = [
                metadata["storage_id"] for metadata in input_tensor_metadata
            ]

            log.info(f"🚀 Local CPU executing: {op_name}")
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

            log.debug(f"📥 Reconstructed {len(input_tensors)} input tensors on CPU")

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
                f"{len(input_tensors)} inputs, {len([s for s in output_storage_ids if s is not None])} outputs to update on CPU"
            )

            # Execute the operation on input tensors - this will create result tensors on CPU
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

                    # Store the result tensor as a 1D uint8 tensor (should already be on CPU)
                    result_tensor = result_tensors[i]
                    # Ensure result is on CPU
                    if result_tensor.device.type != "cpu":
                        result_tensor = result_tensor.cpu()
                    
                    result_storage = result_tensor.untyped_storage()
                    storage_nbytes = result_storage.nbytes()
                    # Create 1D uint8 tensor with the same storage on CPU
                    storage_tensor = torch.empty(storage_nbytes, dtype=torch.uint8, device="cpu")
                    storage_tensor.untyped_storage().copy_(result_storage)
                    storages[storage_id] = storage_tensor

            log.debug(
                f"📦 Updated {len(output_storage_ids)} output storage mappings on CPU"
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

                log.info(f"✅ Completed: {op_name} (returning metadata for {len(output_metadata)} outputs) on CPU")
                return output_metadata

            log.info(f"✅ Completed: {op_name} on CPU")
            return None

    return None, LocalPytorchServer