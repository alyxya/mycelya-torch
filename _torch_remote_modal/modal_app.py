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
import modal
from typing import Any, Dict, List, Tuple, Optional
import os
import weakref
import threading
import random
import logging
from collections import defaultdict
import torch

# Constants moved from torch_remote to eliminate dependency
TENSOR_PLACEHOLDER_PREFIX = "__TENSOR_"
CPU_DEVICE_TYPE = "cpu"
CUDA_DEVICE_TYPE = "cuda"

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


def create_modal_app_for_gpu(gpu_type: str, machine_id: str) -> Tuple[modal.App, Any]:
    """
    Create a Modal app and server class for a specific GPU type and machine.
    
    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB")
        machine_id: The machine ID (e.g., "modal-t4-f3a7d67e")
        
    Returns:
        Tuple of (modal_app, server_class) for the specified device
    """
    return _create_modal_app_for_gpu(gpu_type, machine_id)


def _create_modal_app_for_gpu(gpu_type: str, machine_id: str) -> Tuple[modal.App, Any]:
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
        raise ValueError(f"GPU type \"{gpu_type}\" is not supported. Available types: {list(GPU_CONFIG.keys())}")
    
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
            tensor_data: bytes,
            storage_id: int
        ) -> None:
            """
            Create a new storage on the remote machine.
            
            Args:
                tensor_data: Serialized tensor data
                storage_id: Specific ID to use for the storage (required)
                
            Returns:
                None
            """
            import torch
            import io
            
            # Deserialize tensor
            buffer = io.BytesIO(tensor_data)
            tensor = torch.load(buffer, map_location=CPU_DEVICE_TYPE, weights_only=True)
            
            # Move to GPU (Modal environment always has CUDA)
            device = torch.device(CUDA_DEVICE_TYPE)
            tensor = tensor.to(device)
            
            # Store storage and original tensor data
            storages, lock = self._get_storages()
            with lock:
                # Store tensor storage for all tensors
                storage_id = int(storage_id)
                storages[storage_id] = tensor.untyped_storage()
                log.info(f"📥 RECEIVED Storage ID {storage_id} on Modal (shape: {tensor.shape})")

        @modal.method()
        def get_storage_data(self, storage_id: int, shape=None, stride=None, storage_offset=0, dtype=None) -> bytes:
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
                
                # If view parameters are provided, create the view and make it contiguous
                if shape is not None:
                    # Parse dtype string back to torch.dtype
                    dtype_str = dtype.replace("torch.", "") if dtype else "float32"
                    torch_dtype = getattr(torch, dtype_str)
                    
                    # Create tensor from storage with view parameters (with correct dtype!)
                    tensor = torch.empty(0, dtype=torch_dtype, device=storage.device).set_(
                        storage, storage_offset, shape, stride or []
                    )
                    # Make contiguous to get only the view's data
                    tensor = tensor.contiguous()
                    log.info(f"📦 Serializing view of storage {storage_id}: shape={shape}, stride={stride}, offset={storage_offset}")
                else:
                    # Return full storage as before (backward compatibility)
                    tensor = torch.empty(0, device=storage.device).set_(storage)
                    log.info(f"📦 Serializing full storage {storage_id}")
                
                buffer = io.BytesIO()
                torch.save(tensor, buffer)
                return buffer.getvalue()

        @modal.method()
        def resize_storage(self, storage_id: int, new_bytes: int) -> bool:
            """
            Resize a storage to accommodate new byte size.
            
            This handles the case where resize_ needs more storage space than currently allocated.
            Only resizes if new_bytes > current storage size.
            
            Args:
                storage_id: The storage ID to resize
                new_bytes: The new size in bytes
                
            Returns:
                True if resize succeeded, False if storage not found or new_bytes <= current size
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
                if new_bytes <= current_bytes:
                    log.debug(f"Storage {storage_id} resize skipped: new_bytes ({new_bytes}) <= current_bytes ({current_bytes})")
                    return True  # No-op, but success
                
                device = old_storage.device
                
                # Allocate new storage with the bigger size
                new_storage = torch.UntypedStorage(new_bytes, device=device)
                
                # Copy old storage bytes into the beginning of new storage
                if current_bytes > 0:
                    # Create byte views of both storages for copying
                    old_bytes = old_storage.data_ptr()
                    new_bytes_ptr = new_storage.data_ptr()
                    
                    # Use torch.from_buffer to create tensors from raw bytes for copying
                    old_byte_tensor = torch.frombuffer(
                        old_storage, dtype=torch.uint8, count=current_bytes
                    )
                    new_byte_tensor = torch.frombuffer(
                        new_storage, dtype=torch.uint8, count=new_bytes
                    )
                    
                    # Copy old data to beginning of new storage
                    new_byte_tensor[:current_bytes] = old_byte_tensor
                
                # Replace the storage
                storages[storage_id] = new_storage
                log.info(f"🔄 Resized storage {storage_id} from {current_bytes} to {new_bytes} bytes")
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
            storage_ids: List[int],
            tensor_metadata: List[Dict[str, Any]],
            args: List[Any],
            kwargs: Dict[str, Any],
            machine_id: str
        ) -> None:
            """
            Execute an operation using storage IDs and tensor metadata.
            All tensors (input and output) are pre-allocated and passed as arguments.
            
            Args:
                op_name: The operation name to execute
                storage_ids: List of all tensor storage IDs (both input and output tensors)
                tensor_metadata: List of tensor metadata for reconstruction (shape, stride, offset, storage_id)
                args: Operation arguments (with tensor placeholders)
                kwargs: Operation keyword arguments (with tensor placeholders)
                machine_id: Machine ID for logging
                
            Returns:
                None (operation is executed on pre-allocated tensors)
            """
            import torch
            
            log.info(f"🚀 Modal {gpu_type} (machine {machine_id}) executing: {op_name}")
            log.debug(f"Using storage IDs: {storage_ids}")
            
            try:
                # Get storage mapping
                storages, lock = self._get_storages()
                
                # Reconstruct all tensors from storage and metadata
                tensors = []
                for i, (storage_id, metadata) in enumerate(zip(storage_ids, tensor_metadata)):
                    # DEBUG: Log what storage ID is being requested
                    log.debug(f"Modal app looking for storage_id={storage_id} (type={type(storage_id)})")
                    
                    # Get storage object
                    with lock:
                        storage_id = int(storage_id)
                        if storage_id not in storages:
                            available_ids = list(storages.keys())
                            log.error(f"❌ MISSING Storage ID {storage_id}")
                            log.error(f"📋 Available Storage IDs on Modal: {available_ids}")
                            log.error(f"🔍 Looking for: {storage_id} (type: {type(storage_id)})")
                            log.error(f"🔍 Available types: {[type(sid) for sid in available_ids]}")
                            raise KeyError(f"Storage ID {storage_id} not found")
                        storage = storages[storage_id]
                    
                    # Parse dtype string back to torch.dtype
                    dtype_str = metadata["dtype"].replace("torch.", "")
                    dtype = getattr(torch, dtype_str)
                    
                    # Reconstruct tensor using storage + metadata (on CUDA device)
                    tensor = torch.empty(0, dtype=dtype, device=CUDA_DEVICE_TYPE).set_(
                        storage,
                        metadata["storage_offset"],
                        metadata["shape"],
                        metadata["stride"]
                    )
                    
                    # Log tensor details on modal side
                    log.debug(f"📥 MODAL tensor[{i}]: ID={storage_id}, shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, size={tensor.numel()}")
                    
                    # Log tensor data summary for debugging
                    if tensor.numel() > 0:
                        # Only compute mean for floating point tensors
                        if tensor.dtype.is_floating_point:
                            log.debug(f"   Data range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}], mean={tensor.mean().item():.6f}")
                        else:
                            log.debug(f"   Data range: [{tensor.min().item()}, {tensor.max().item()}], dtype={tensor.dtype}")
                    
                    tensors.append(tensor)
                
                # Replace tensor placeholders in args with actual tensors
                processed_args = []
                for arg in args:
                    if isinstance(arg, str) and arg.startswith(TENSOR_PLACEHOLDER_PREFIX):
                        idx = int(arg.split("_")[-1])
                        processed_args.append(tensors[idx])
                    else:
                        processed_args.append(arg)
                
                # Process kwargs similarly
                processed_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, str) and value.startswith(TENSOR_PLACEHOLDER_PREFIX):
                        idx = int(value.split("_")[-1])
                        processed_kwargs[key] = tensors[idx]
                    else:
                        processed_kwargs[key] = value
                
                # Get the operation
                op_name_fixed = op_name.replace("::", ".")
                op_parts = op_name_fixed.split(".")
                op = torch.ops
                for part in op_parts:
                    op = getattr(op, part)
                
                log.debug(f"Executing operation with {len(processed_args)} args and {len(tensors)} tensors")
                
                # Execute the operation - results are written directly to pre-allocated tensors
                result = op(*processed_args, **processed_kwargs)
                
                log.info(f"✅ Completed: {op_name} - operation executed on pre-allocated tensors")
                
                # Operation completed successfully - any output tensors have been modified in-place
                return
                
            except Exception as e:
                log.error(f"❌ Error executing {op_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise

    _gpu_apps[machine_id] = (app, PytorchServer)
    return app, PytorchServer


def clear_app_cache():
    """Clear the app cache."""
    global _gpu_apps
    _gpu_apps.clear()
