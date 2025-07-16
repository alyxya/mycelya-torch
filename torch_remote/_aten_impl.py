# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import torch
from torch.utils._pytree import tree_any


log = logging.getLogger(__name__)

from ._device_daemon import driver
from ._meta_parser import prepare_for_sending, to_device_no_copy


# Lazy import to avoid import errors if remote execution is not available
_remote_executor = None

def _get_remote_executor():
    """Get remote executor, importing lazily."""
    global _remote_executor
    if _remote_executor is None:
        try:
            from ._remote_executor import remote_executor
            _remote_executor = remote_executor
        except ImportError as e:
            log.warning(f"Remote execution not available: {e}")
            _remote_executor = None
    return _remote_executor


_IMPL_REGISTRY = {}


def impl_factory(name):
    if name in _IMPL_REGISTRY:
        return _IMPL_REGISTRY[name]

    # Handle operations that need special error handling
    if name in ["create_tensor_with_id", "free_tensor_with_id"]:
        def _(tensor_id, *args):
            """Wrapper for tensor ID operations with error handling."""
            result = driver.exec(name, tensor_id, *args)
            if not result:
                operation = name.replace("_", " ")
                if name == "create_tensor_with_id":
                    nbytes, device_index = args
                    raise RuntimeError(f"Failed to create tensor with ID {tensor_id} ({nbytes} bytes) on device {device_index}")
                elif name == "free_tensor_with_id":
                    raise RuntimeError(f"Failed to free tensor with ID {tensor_id}")
            return result
        
        _IMPL_REGISTRY[name] = _
        return _

    def _(*args, **kwargs):
        log.info("Calling hook %s", name)
        return driver.exec(name, *args, **kwargs)

    _IMPL_REGISTRY[name] = _
    return _



def _remote_kernel_fallback(op, *args, **kwargs):
    log.info("Calling kernel %s", op)
    
    op_name = op.overloadpacket._qualified_op_name
    
    # Collect all tensor arguments
    remote_tensors = []
    non_remote_tensors = []
    
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if arg.device.type == "remote":
                remote_tensors.append(arg)
            else:
                non_remote_tensors.append(arg)
    
    for value in kwargs.values():
        if isinstance(value, torch.Tensor):
            if value.device.type == "remote":
                remote_tensors.append(value)
            else:
                non_remote_tensors.append(value)
    
    # Validate device compatibility - mixed remote and non-remote tensors are disallowed
    if remote_tensors and non_remote_tensors:
        remote_devices = [t.device for t in remote_tensors]
        non_remote_devices = [t.device for t in non_remote_tensors]
        raise RuntimeError(
            f"Operation {op_name}: Mixed remote and non-remote tensor operation not supported. "
            f"Remote devices: {remote_devices}, Non-remote devices: {non_remote_devices}. "
            f"Use explicit .to() to move tensors to the same device first."
        )
    
    # For remote operations, execute remotely with standard execution model
    if remote_tensors:
        executor = _get_remote_executor()
        if executor is not None:
            log.info(f"🚀 Executing {op_name} remotely")
            return executor.execute_remote_operation(op_name, args, kwargs)
        else:
            # No fallback for remote operations - they must work remotely or fail
            log.error(f"❌ Remote operation failed: {op_name} - Remote execution not available")
            raise RuntimeError(
                f"Operation {op_name} cannot be executed on remote tensors: remote execution not available. "
                f"Remote tensors must use remote execution - no fallback to local execution is allowed."
            )

    # Operations that reach this point should not be handled by remote dispatch
    # This indicates a configuration or logic error in the dispatch system
    log.error(f"❌ Unexpected operation dispatch: {op_name} - should not reach remote fallback")
    raise RuntimeError(
        f"Operation {op_name} was dispatched to remote kernel fallback but should not be handled remotely. "
        f"This indicates an error in the dispatch system configuration."
    )


def _execute_local_operation(op, *args, **kwargs):
    """
    Handle operations that don't involve remote tensors.
    Remote tensors should only be processed through remote execution.
    """
    # For non-remote operations, just execute normally
    return op(*args, **kwargs)


def copy_from_device(from_):
    """Copy data from remote tensor to CPU tensor using remote execution"""
    if from_.device.type != "remote":
        raise ValueError("copy_from_device requires a remote tensor")
    
    # Use remote execution to get the tensor data
    executor = _get_remote_executor()
    if executor is not None:
        from .device import get_device_registry
        
        # Get the device backend
        registry = get_device_registry()
        device = registry.get_device_by_index(from_.device.index)
        
        if device is None:
            raise RuntimeError(f"No RemoteBackend found for remote device index {from_.device.index}")
        
        # Get the GPU machine for this device
        gpu_machine = device.get_gpu_machine()
        if gpu_machine is None or not gpu_machine.is_running():
            raise RuntimeError(f"GPU machine not available for device {device.machine_id}")
        
        # Get tensor data using tensor ID (convert int to string for GPU machine)
        tensor_id_int = from_.untyped_storage().data_ptr()
        tensor_id_str = str(tensor_id_int)
        log.info(f"Copying tensor ID {tensor_id_int} from remote to CPU")
        
        # Use GPU machine to get tensor data by ID
        tensor_data = gpu_machine.get_tensor_data(tensor_id_str)
        
        # Deserialize the tensor
        result = executor._deserialize_tensor(tensor_data)
        log.info(f"Successfully copied tensor ID {tensor_id_int} to CPU")
        return result
    else:
        raise RuntimeError("Cannot copy from remote device: remote execution not available")


def copy_from_host_to_device(from_, to_):
    """Copy data from CPU tensor to remote tensor using remote execution"""
    if to_.device.type != "remote":
        raise ValueError("copy_from_host_to_device requires a remote target tensor")
    if from_.device.type != "cpu":
        raise ValueError("copy_from_host_to_device requires a CPU source tensor")
    
    # Use remote execution to send the tensor data
    executor = _get_remote_executor()
    if executor is not None:
        from .device import get_device_registry
        
        # Get the device backend
        registry = get_device_registry()
        device = registry.get_device_by_index(to_.device.index)
        
        if device is None:
            raise RuntimeError(f"No RemoteBackend found for remote device index {to_.device.index}")
        
        # Get the GPU machine for this device
        gpu_machine = device.get_gpu_machine()
        if gpu_machine is None or not gpu_machine.is_running():
            raise RuntimeError(f"GPU machine not available for device {device.machine_id}")
        
        # Send tensor data using tensor ID (convert int to string for GPU machine)
        tensor_id_int = to_.untyped_storage().data_ptr()
        tensor_id_str = str(tensor_id_int)
        log.info(f"Copying CPU tensor to remote tensor ID {tensor_id_int}")
        
        # Serialize the CPU tensor
        tensor_data = executor._serialize_tensor(from_)
        
        # Use GPU machine to create/update tensor with specific ID
        # This will overwrite any existing empty tensor with the actual data
        created_id = gpu_machine.create_tensor(tensor_data, tensor_id_str)
        log.info(f"Successfully created/updated remote tensor with ID {created_id}")
        return to_
    else:
        raise RuntimeError("Cannot copy to remote device: remote execution not available")


def _copy_from(from_, to_):
    # Simplified copy implementation - remote tensors are now regular torch.Tensor
    # with proper device handling via C++ allocator
    
    # Preserve requires_grad property from source tensor
    should_preserve_grad = from_.requires_grad
    
    if from_.device.type == to_.device.type:
        if from_.device.type == "remote":
            if from_.device.index == to_.device.index:
                # Same remote device - use direct copy
                op = torch.ops.aten.copy_.default
                result = _remote_kernel_fallback(op, to_, from_)
            else:
                # Different remote devices: NOT ALLOWED
                from torch_remote.device import get_device_registry
                device_registry = get_device_registry()
                from_device = device_registry.get_device_by_index(from_.device.index)
                to_device = device_registry.get_device_by_index(to_.device.index)
                
                raise RuntimeError(
                    f"Cannot transfer tensor between different remote devices. "
                    f"Source device: \"{from_device.machine_id}\" (index {from_.device.index}), "
                    f"Target device: \"{to_device.machine_id}\" (index {to_.device.index}). "
                    f"Transfer tensors to CPU first: tensor.cpu().to(target_device)"
                )
        else:
            # Both tensors on same non-remote device
            result = to_.copy_(from_)
    elif from_.device.type == "remote":
        # Remote to non-remote
        host_mem = copy_from_device(from_)
        result = to_.copy_(host_mem)
    elif to_.device.type == "remote":
        # Non-remote to remote
        result = copy_from_host_to_device(from_, to_)
    else:
        # Both non-remote but different devices
        result = to_.copy_(from_)
    
    # Preserve autograd properties
    if should_preserve_grad and not result.requires_grad:
        result.requires_grad_(True)
    
    return result


def _to_copy(input, *, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False, memory_format=None):
    """Implementation of tensor.to() for remote tensors with cross-device transfer restriction."""
    
    # Handle device-specific transfers first
    if device is not None:
        target_device = torch.device(device) if not isinstance(device, torch.device) else device
        
        # Different device transfer - check if both are remote
        if input.device.type == "remote" and target_device.type == "remote" and input.device != target_device:
            # Cross-device remote transfer - NOT ALLOWED
            from torch_remote.device import get_device_registry
            device_registry = get_device_registry()
            from_device = device_registry.get_device_by_index(input.device.index)
            to_device = device_registry.get_device_by_index(target_device.index)
            
            raise RuntimeError(
                f"Cannot transfer tensor between different remote devices. "
                f"Source device: \"{from_device.machine_id}\" (index {input.device.index}), "
                f"Target device: \"{to_device.machine_id}\" (index {target_device.index}). "
                f"Transfer tensors to CPU first: tensor.cpu().to(target_device)"
            )
    
    # For all other cases, create a new tensor and use _copy_from if needed
    # This avoids infinite recursion by not calling back to the kernel fallback
    
    # Determine output parameters
    output_dtype = dtype if dtype is not None else input.dtype
    output_layout = layout if layout is not None else input.layout
    output_device = torch.device(device) if device is not None else input.device
    output_memory_format = memory_format if memory_format is not None else torch.contiguous_format
    
    # Create output tensor
    if output_device.type == "remote":
        # Create empty remote tensor - use contiguous format for remote tensors
        result = torch.empty(input.size(), dtype=output_dtype, layout=output_layout, 
                             device=output_device, memory_format=torch.contiguous_format)
    else:
        # Create empty tensor on target device
        result = torch.empty(input.size(), dtype=output_dtype, layout=output_layout, 
                             device=output_device, memory_format=output_memory_format)
    
    # Copy data if needed (different device or same device but different dtype)
    if input.device != output_device or input.dtype != output_dtype:
        # Use _copy_from to handle the actual data transfer
        result = _copy_from(input, result)
    else:
        # Same device and dtype - just return input (no copy needed)
        return input
    
    return result


def _set_source_tensor(ten1, ten2):
    return torch.ops.aten.set_.source_Storage_storage_offset(
        ten1,
        ten2.untyped_storage(),
        ten2.storage_offset(),
        ten2.size(),
        ten2.stride(),
    )


def _local_scalar_dense(ten):
    host_mem = copy_from_device(ten)
    return host_mem.item()


# cpu_tensor_to_persistent_remote function removed - no longer needed with C++ implementations
# Remote tensors are now handled directly by the C++ allocator with ID-based allocation




_remote_lib = torch.library.Library("_", "IMPL")
_remote_lib.fallback(_remote_kernel_fallback, dispatch_key="PrivateUse1")

_remote_lib_aten = torch.library.Library("aten", "IMPL")
_remote_lib_aten.impl("_copy_from", _copy_from, dispatch_key="PrivateUse1")
_remote_lib_aten.impl("_to_copy", _to_copy, dispatch_key="PrivateUse1")
_remote_lib_aten.impl(
    "set_.source_Tensor", _set_source_tensor, dispatch_key="PrivateUse1"
)
_remote_lib_aten.impl(
    "_local_scalar_dense", _local_scalar_dense, dispatch_key="PrivateUse1"
)

# Note: empty.memory_format and empty_strided are now implemented in C++
# via TORCH_LIBRARY_IMPL in RemoteMem.cpp, so we don't register Python implementations

# randn_remote, zeros_remote, ones_remote functions removed
# These factory functions are now handled by C++ implementations
# via the registered TORCH_LIBRARY_IMPL dispatch system

# Note: randn, zeros, ones registrations removed - they will use C++ implementations
# when we add them to TORCH_LIBRARY_IMPL in RemoteMem.cpp


def cleanup_library_registrations():
    """Clean up library registrations to prevent hanging."""
    global _remote_lib, _remote_lib_aten
    try:
        # PyTorch doesn't provide a clean way to unregister, but we can try
        # Calling this during cleanup might help
        if hasattr(_remote_lib, "_destroy"):
            _remote_lib._destroy()
        if hasattr(_remote_lib_aten, "_destroy"):
            _remote_lib_aten._destroy()
    except Exception:
        pass