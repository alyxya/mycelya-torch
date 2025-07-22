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
import logging
import os
import random
import threading
import weakref
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import modal
import torch
from torch.utils._pytree import tree_map

log = logging.getLogger(__name__)

# Create simplified image with just PyTorch and CUDA support
image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "torch")
)

# Cache for GPU-specific apps and their functions
_gpu_apps: Dict[str, Tuple[modal.App, Any]] = {}

# GPU configuration mapping
GPU_CONFIG = {
    "T4": {"timeout": 300, "retries": 2},
    "L4": {"timeout": 300, "retries": 2},
    "A10G": {"timeout": 450, "retries": 2},
    "A100": {"timeout": 450, "retries": 2},
    "A100-40GB": {"timeout": 450, "retries": 2},
    "A100-80GB": {"timeout": 450, "retries": 2},
    "L40S": {"timeout": 450, "retries": 2},
    "H100": {"timeout": 450, "retries": 2},
    "H200": {"timeout": 450, "retries": 2},
    "B200": {"timeout": 450, "retries": 2},
}


def create_modal_app_for_gpu(
    gpu_type: str, machine_id: str
) -> Tuple[modal.App, Any]:
    """
    Create a Modal app and server class for a specific GPU type and machine.

    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB")
        machine_id: The machine ID (e.g., "modal-t4-f3a7d67e")

    Returns:
        Tuple of (modal_app, server_class) for the specified device
    """
    return _create_modal_app_for_gpu(gpu_type, machine_id)


def _create_modal_app_for_gpu(
    gpu_type: str, machine_id: str
) -> Tuple[modal.App, Any]:
    """
    Create a Modal app and class for a specific GPU type and device.

    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB")
        machine_id: The machine ID (e.g., "modal-t4-f3a7d67e")

    Returns:
        Tuple of (modal_app, server_class) for the specified device
    """
    if machine_id in _gpu_apps:
        return _gpu_apps[machine_id]
    
    if gpu_type not in GPU_CONFIG:
        raise ValueError(
            f'GPU type "{gpu_type}" is not supported. '
            f'Available types: {list(GPU_CONFIG.keys())}'
        )
    
    config = GPU_CONFIG[gpu_type]
    app = modal.App(f"torch-remote-{machine_id}")
    
    @app.cls(
        image=image,
        gpu=gpu_type,
        timeout=config["timeout"],
        retries=config["retries"],
        serialized=True
    )
    class PytorchServer:

        def _get_storages(self):
            """Get or create storage mapping for this server instance."""
            if not hasattr(self, "_storages"):
                import threading
                from typing import Dict, Any, Union, Tuple
                
                # storage_id -> torch.Storage
                self._storages: Dict[int, Any] = {}
                self._storage_lock = threading.RLock()
            
            return self._storages, self._storage_lock

        @modal.method()
        def create_storage(
            self,
            nbytes: int,
            storage_id: int,
            lazy: bool = False
        ) -> None:
            """
            Create a new storage on the remote machine.
            
            Args:
                nbytes: Number of bytes to allocate for the storage
                storage_id: Specific ID to use for the storage (required)
                lazy: If True, defer actual GPU allocation until first use
                
            Returns:
                None
            """
            import torch
            
            # Store storage and original tensor data
            storages, lock = self._get_storages()
            with lock:
                storage_id = int(storage_id)
                
                if lazy:
                    # Store only the byte count for lazy allocation
                    storages[storage_id] = nbytes
                    log.info(f"📝 LAZY Storage ID {storage_id} registered ({nbytes} bytes)")
                else:
                    # Create tensor directly on GPU with exact byte size
                    device = torch.device("cuda")
                    tensor = torch.empty(nbytes, dtype=torch.uint8, device=device)
                    storages[storage_id] = tensor.untyped_storage()
                    log.info(f"📥 CREATED Storage ID {storage_id} on Modal ({nbytes} bytes)")

        @modal.method()
        def update_storage(
            self,
            tensor_data: bytes,
            storage_id: int
        ) -> None:
            """
            Update an existing storage with tensor data.
            
            Args:
                tensor_data: Serialized tensor data
                storage_id: Storage ID to update
                
            Returns:
                None
            """
            import torch
            import io
            
            # Deserialize tensor
            buffer = io.BytesIO(tensor_data)
            tensor = torch.load(buffer, map_location="cpu", weights_only=True)
            
            # Move to GPU (Modal environment always has CUDA)
            device = torch.device("cuda")
            tensor = tensor.to(device)
            
            # Update existing storage
            storages, lock = self._get_storages()
            with lock:
                storage_id = int(storage_id)
                
                # Check if storage was lazy-allocated and warn about direct update
                if storage_id in storages and isinstance(storages[storage_id], int):
                    lazy_bytes = storages[storage_id]
                    log.warning(f"⚠️ Updating lazy storage ID {storage_id} ({lazy_bytes} bytes) with direct data")
                
                storages[storage_id] = tensor.untyped_storage()
                log.info(f"📥 UPDATED Storage ID {storage_id} on Modal (shape: {tensor.shape})")

        @modal.method()
        def get_storage_data(
            self,
            storage_id: int,
            shape=None,
            stride=None,
            storage_offset=0,
            dtype=None,
        ) -> bytes:
            """
            Retrieve current storage data by storage ID for transfer to client.
            If view parameters are provided, returns only the view's data as contiguous.

            Args:
                storage_id: The storage ID
                shape: Tensor shape for view (if None, returns full storage)
                stride: Tensor stride for view
                storage_offset: Storage offset for view  
                dtype: Tensor data type
                
            Returns:
                Serialized tensor data (contiguous representation of the view)
            """
            import io
            
            storages, lock = self._get_storages()
            
            with lock:
                storage_id = int(storage_id)
                if storage_id not in storages:
                    raise KeyError(f"Storage ID {storage_id} not found")
                
                storage = storages[storage_id]
                
                # Validate that storage is not a lazy allocation (int)
                if isinstance(storage, int):
                    raise RuntimeError(
                        f"Storage ID {storage_id} is lazy-allocated (not yet realized). "
                        f"This indicates a bug: lazy storages should be converted to real storage "
                        f"during operation execution before data retrieval."
                    )
                
                # If view parameters are provided, create the view and make it contiguous
                if shape is not None:
                    # Parse dtype string back to torch.dtype
                    dtype_str = dtype.replace("torch.", "") if dtype else "float32"
                    torch_dtype = getattr(torch, dtype_str)
                    
                    # Create tensor from storage with view parameters (with correct dtype!)
                    tensor = torch.empty(
                        0, dtype=torch_dtype, device=storage.device
                    ).set_(storage, storage_offset, shape, stride or [])
                    # Make contiguous to get only the view's data
                    tensor = tensor.contiguous()
                    log.info(
                        f"📦 Serializing view of storage {storage_id}: "
                        f"shape={shape}, stride={stride}, offset={storage_offset}"
                    )
                else:
                    # Return full storage as before (backward compatibility)
                    tensor = torch.empty(0, device=storage.device).set_(storage)
                    log.info(f"📦 Serializing full storage {storage_id}")
                
                buffer = io.BytesIO()
                torch.save(tensor, buffer)
                return buffer.getvalue()

        @modal.method()
        def resize_storage(self, storage_id: int, nbytes: int) -> bool:
            """
            Resize a storage to accommodate new byte size.
            
            This handles the case where resize_ needs more storage space than currently allocated.
            Uses tensor.resize_() to properly handle storage resizing and updates the storage mapping.
            Only resizes if nbytes > current storage size.
            
            Args:
                storage_id: The storage ID to resize
                nbytes: The number of bytes needed for the new storage size
                
            Returns:
                True if resize succeeded, False if storage not found or nbytes <= current size
            """
            import torch
            
            storages, lock = self._get_storages()
            
            with lock:
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
                    return True  # No-op, but success
                
                device = old_storage.device
                
                # Create a tensor that uses the existing storage to leverage tensor.resize_()
                # Use uint8 dtype for byte-level operations
                uint8_dtype = torch.uint8
                current_element_count = current_bytes // torch.tensor([], dtype=uint8_dtype).element_size()
                temp_tensor = torch.tensor([], dtype=uint8_dtype, device=device)
                temp_tensor.set_(old_storage, storage_offset=0, size=(current_element_count,))
                
                # Calculate new element count for the resized tensor
                new_element_count = nbytes // torch.tensor([], dtype=uint8_dtype).element_size()
                
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
                return True

        @modal.method()
        def remove_storage(self, storage_id: int) -> bool:
            """
            Remove a storage from the registry.
            
            Args:
                storage_id: The storage ID
                
            Returns:
                True if removed, False if not found
            """
            storages, lock = self._get_storages()
            
            with lock:
                storage_id = int(storage_id)
                removed = storage_id in storages
                if removed:
                    del storages[storage_id]
                    log.info(f"🗑️ Removed storage {storage_id}")
                return removed

        @modal.method()
        def execute_aten_operation(
            self,
            op_name: str,
            tensor_metadata: List[Dict[str, Any]],
            args: List[Any],
            kwargs: Dict[str, Any],
            machine_id: str
        ) -> None:
            """
            Execute an operation with explicit input/output tensor separation.
            
            This method handles operations where input and output tensors are explicitly
            separated based on the is_input/is_output flags in tensor metadata.
            
            Args:
                op_name: The operation name to execute
                tensor_metadata: List of tensor metadata with is_input/is_output flags and storage_id
                args: Operation arguments (with tensor placeholders)
                kwargs: Operation keyword arguments (with tensor placeholders)
                machine_id: Machine ID for logging
                
            Returns:
                None (operation results are written to output tensors)
            """
            import torch
            
            # Extract storage IDs from metadata
            storage_ids = [metadata["storage_id"] for metadata in tensor_metadata]
            
            log.info(f"🚀 Modal {gpu_type} (machine {machine_id}) executing with IO separation: {op_name}")
            log.debug(f"Using storage IDs: {storage_ids}")
            
            try:
                # Get storage mapping
                storages, lock = self._get_storages()
                
                # Reconstruct only input tensors from storage and metadata
                # Output tensors don't need reconstruction - we'll update their storage mapping after the operation
                tensors = []
                input_tensors = []
                output_storage_ids = []
                output_metadata_list = []
                
                for i, metadata in enumerate(tensor_metadata):
                    storage_id = metadata["storage_id"]
                    log.debug(f"Modal app processing storage_id={storage_id} (type={type(storage_id)})")
                    
                    # Classify tensor as input or output
                    is_input = metadata.get("is_input", True)  # Default to input for backward compatibility
                    is_output = metadata.get("is_output", False)
                    
                    # Always reconstruct tensor from storage + metadata (needed for operation execution)
                    with lock:
                        storage_id = int(storage_id)
                        if storage_id not in storages:
                            available_ids = list(storages.keys())
                            log.error(f"❌ MISSING Storage ID {storage_id}")
                            log.error(f"📋 Available Storage IDs on Modal: {available_ids}")
                            raise KeyError(f"Storage ID {storage_id} not found")
                        storage = storages[storage_id]
                    
                    # Validate that storage is not a lazy allocation (int) 
                    if isinstance(storage, int):
                        raise RuntimeError(
                            f"Storage ID {storage_id} is lazy-allocated (not yet realized). "
                            f"This indicates a bug: lazy storages should be converted to real storage "
                            f"during operation execution before tensor reconstruction."
                        )
                    
                    # Parse dtype string back to torch.dtype
                    dtype_str = metadata["dtype"].replace("torch.", "")
                    dtype = getattr(torch, dtype_str)
                    
                    # Reconstruct tensor using storage + metadata (on CUDA device)
                    tensor = torch.empty(0, dtype=dtype, device="cuda").set_(
                        storage,
                        metadata["storage_offset"],
                        metadata["shape"],
                        metadata["stride"]
                    )
                    
                    log.debug(f"📥 MODAL tensor[{i}] ({'input' if is_input else ''}{'output' if is_output else ''}): ID={storage_id}, shape={tensor.shape}")
                    tensors.append(tensor)
                    
                    # Keep track of which tensors are inputs vs outputs for post-operation processing
                    if is_input:
                        input_tensors.append(tensor)
                    if is_output:
                        output_storage_ids.append(storage_id)
                        output_metadata_list.append(metadata)
                
                # Replace tensor placeholders with actual reconstructed tensors using tree_map
                def replace_placeholder_with_tensor(obj):
                    if isinstance(obj, str) and obj.startswith("__TENSOR_"):
                        idx = int(obj.split("_")[-1])
                        if idx < len(tensors):
                            return tensors[idx]
                        else:
                            raise IndexError(f"Tensor placeholder index {idx} out of range (have {len(tensors)} tensors)")
                    return obj
                
                # Use tree_map to handle nested structure traversal automatically
                processed_args, processed_kwargs = tree_map(replace_placeholder_with_tensor, (args, kwargs))
                
                # Get the operation
                op_name_fixed = op_name.replace("::", ".")
                op_parts = op_name_fixed.split(".")
                op = torch.ops
                for part in op_parts:
                    op = getattr(op, part)
                
                log.debug(f"Executing operation with {len(processed_args)} args, "
                         f"{len(input_tensors)} inputs, {len(output_storage_ids)} outputs")
                
                # Execute the operation on input tensors - this will create result tensors
                result = op(*processed_args, **processed_kwargs)
                
                # Handle storage mapping updates for output tensors
                if output_storage_ids:
                    log.debug(f"Processing {len(output_storage_ids)} output tensors for storage updates")
                    
                    # Convert result to list if it's a single tensor
                    if isinstance(result, torch.Tensor):
                        result_tensors = [result]
                    elif isinstance(result, tuple):
                        result_tensors = list(result)
                    else:
                        log.warning(f"Unexpected result type: {type(result)}")
                        result_tensors = []
                    
                    # Update storage mapping for each output tensor
                    with lock:
                        for i, (storage_id, metadata) in enumerate(zip(output_storage_ids, output_metadata_list)):
                            if i < len(result_tensors):
                                result_tensor = result_tensors[i]
                                storage_id = int(storage_id)
                                
                                # Check if the storage has changed
                                if storage_id in storages:
                                    current_storage = storages[storage_id]
                                    result_storage = result_tensor.untyped_storage()
                                    
                                    # Handle lazy storage conversion
                                    if isinstance(current_storage, int):
                                        # Lazy storage being realized by operation execution
                                        log.debug(f"📦 Converting lazy storage ID {storage_id} ({current_storage} bytes) to real storage")
                                        storages[storage_id] = result_storage
                                    elif current_storage is not result_storage:
                                        # Storage changed - update the mapping
                                        log.debug(f"📦 Updating storage mapping for ID {storage_id}")
                                        storages[storage_id] = result_storage
                                    else:
                                        log.debug(f"📦 Storage ID {storage_id} unchanged")
                                else:
                                    # New storage ID - add to mapping
                                    log.debug(f"📦 Adding new storage mapping for ID {storage_id}")
                                    storages[storage_id] = result_tensor.untyped_storage()
                            else:
                                log.warning(f"No result tensor for output storage ID {storage_id}")
                
                log.info(f"✅ Completed: {op_name} with IO separation")
                return
                
            except Exception as e:
                log.error(f"❌ Error executing {op_name} with IO separation: {str(e)}")
                import traceback
                traceback.print_exc()
                raise


    _gpu_apps[machine_id] = (app, PytorchServer)
    return app, PytorchServer


def clear_app_cache():
    """Clear the app cache."""
    global _gpu_apps
    _gpu_apps.clear()
