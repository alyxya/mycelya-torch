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

    # Handle new ID-based tensor creation methods
    if name == "create_tensor_with_id":
        def _(tensor_id, nbytes, device_index):
            """Create a tensor mapping with the given ID on the specified device."""
            # Use the ID-based allocation system - no fallbacks
            success = driver.exec("create_tensor_with_id", tensor_id, nbytes, device_index)
            if not success:
                raise RuntimeError(f"Failed to create tensor with ID {tensor_id} ({nbytes} bytes) on device {device_index}")
            
            # Also register the tensor with the GPU machine immediately
            # Create empty tensor data for the allocation
            if nbytes > 0:
                executor = _get_remote_executor()
                if executor is not None:
                    from .device import get_device_registry
                    
                    registry = get_device_registry()
                    device = registry.get_device_by_index(device_index)
                    
                    if device is not None:
                        gpu_machine = device.get_gpu_machine()
                        if gpu_machine and gpu_machine.is_running():
                            # Create empty tensor data of the right size
                            import io
                            empty_tensor = torch.empty(nbytes // 4, dtype=torch.float32)  # Assume float32 for now
                            buffer = io.BytesIO()
                            torch.save(empty_tensor, buffer)
                            tensor_data = buffer.getvalue()
                            
                            tensor_id_str = str(tensor_id)
                            gpu_machine.create_tensor(tensor_data, tensor_id_str)
                            log.info(f"Pre-registered tensor ID {tensor_id} with GPU machine")
            
            return success
        
        _IMPL_REGISTRY[name] = _
        return _
    
    elif name == "free_tensor_with_id":
        def _(tensor_id):
            """Free the tensor associated with the given ID."""
            # Use the ID-based cleanup system - no fallbacks
            success = driver.exec("free_tensor_with_id", tensor_id)
            if not success:
                raise RuntimeError(f"Failed to free tensor with ID {tensor_id}")
            return success
        
        _IMPL_REGISTRY[name] = _
        return _
    
    elif name == "register_tensor_with_gpu":
        def _(tensor_id, tensor_data):
            """Register tensor data with GPU machine for immediate access"""
            # This ensures that newly created tensors (including outputs) 
            # are immediately available on the GPU machine
            executor = _get_remote_executor()
            if executor is not None:
                from .device import get_device_registry
                
                # Get the current remote device (assumes single device for now)
                registry = get_device_registry()
                device = registry.get_device_by_index(0)  # Use device 0
                
                if device is not None:
                    gpu_machine = device.get_gpu_machine()
                    if gpu_machine and gpu_machine.is_running():
                        tensor_id_str = str(tensor_id)
                        gpu_machine.create_tensor(tensor_data, tensor_id_str)
                        log.info(f"Registered tensor ID {tensor_id} with GPU machine")
                        return True
            return False
        
        _IMPL_REGISTRY[name] = _
        return _
    
    elif name == "generate_tensor_id":
        def _():
            """Generate a unique tensor ID with duplicate validation."""
            return driver.exec("generate_tensor_id")
        
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
    
    # Device transfer operations are always handled locally to prevent recursion
    device_transfer_ops = {
        "aten.to",
        "aten.to.device", 
        "aten.to.dtype",
        "aten.to.dtype_layout",
        "aten.cpu",
        "aten.cuda",
        "aten._copy_from",
    }
    if any(transfer_op in op_name for transfer_op in device_transfer_ops):
        # Operations that reach this point should not be handled by remote dispatch
        # This indicates a configuration or logic error in the dispatch system
        log.error(f"❌ Unexpected device transfer operation dispatch: {op_name} - should not reach remote fallback")
        raise RuntimeError(
            f"Device transfer operation {op_name} was dispatched to remote kernel fallback but should not be handled remotely. "
            f"This indicates an error in the dispatch system configuration."
        )
    
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
                # Different remote devices: transfer via CPU
                host_mem = copy_from_device(from_)
                result = copy_from_host_to_device(host_mem, to_)
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
_remote_lib_aten.impl(
    "set_.source_Tensor", _set_source_tensor, dispatch_key="PrivateUse1"
)
_remote_lib_aten.impl(
    "_local_scalar_dense", _local_scalar_dense, dispatch_key="PrivateUse1"
)

# Note: to.device and to.dtype_layout are now implemented in C++
# via TORCH_LIBRARY_IMPL in RemoteMem.cpp for proper .cpu() support
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