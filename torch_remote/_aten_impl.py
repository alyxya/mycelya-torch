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

    # Handle new ID-based tensor creation methods - simplified to use traditional malloc
    if name == "create_tensor_with_id":
        def _(tensor_id, nbytes, device_index):
            """Create a tensor mapping with the given ID on the specified device."""
            # For now, just fall back to traditional malloc approach
            # This ensures compatibility with tensor_from_meta
            try:
                # Use the traditional malloc method which properly populates allocated dict
                ptr = driver.exec("malloc", nbytes)
                return ptr  # Return the actual pointer instead of boolean
            except Exception as e:
                log.warning(f"Malloc fallback failed: {e}")
                return 0  # Return null pointer on failure
        
        _IMPL_REGISTRY[name] = _
        return _
    
    elif name == "free_tensor_with_id":
        def _(tensor_id):
            """Free the tensor associated with the given ID."""
            try:
                if hasattr(driver, "_tensor_id_mappings") and tensor_id in driver._tensor_id_mappings:
                    # Get the actual remote tensor ID
                    actual_id = driver._tensor_id_mappings[tensor_id]
                    
                    # Remove from mapping
                    del driver._tensor_id_mappings[tensor_id]
                    
                    # TODO: Add remote cleanup logic here if needed
                    return True
            except Exception as e:
                log.warning(f"ID-based tensor cleanup failed: {e}")
            return False
        
        _IMPL_REGISTRY[name] = _
        return _

    def _(*args, **kwargs):
        log.info("Calling hook %s", name)
        return driver.exec(name, *args, **kwargs)

    _IMPL_REGISTRY[name] = _
    return _


def _should_use_remote_execution(op, args, kwargs):
    """
    Determine whether to use remote GPU execution for this operation.
    
    Fundamental principles:
    1. Operations between remote tensors always execute remotely
    2. Operations between non-remote tensors happen locally as usual  
    3. Operations between remote and non-remote tensors are disallowed
    4. Device movement for tensors is always explicit (handled elsewhere)
    """
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
        return False
    
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
    
    # Apply fundamental principles
    if remote_tensors and non_remote_tensors:
        # Mixed remote and non-remote tensors are disallowed
        remote_devices = [t.device for t in remote_tensors]
        non_remote_devices = [t.device for t in non_remote_tensors]
        raise RuntimeError(
            f"Mixed remote and non-remote tensor operation not supported for {op_name}. "
            f"Remote devices: {remote_devices}, Non-remote devices: {non_remote_devices}. "
            f"Use explicit .to() to move tensors to the same device first."
        )
    
    if remote_tensors:
        # Operations between remote tensors always execute remotely
        return True
    
    # Operations between non-remote tensors happen locally as usual
    return False


def _remote_kernel_fallback(op, *args, **kwargs):
    def get_tensor_device(*args):
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
                return arg.device

    device = get_tensor_device(*args)
    if device is None:
        return _kernel_fallback(op, *args, **kwargs)

    # Check if we should use remote execution
    should_use_remote = _should_use_remote_execution(op, args, kwargs)
    
    if should_use_remote:
        executor = _get_remote_executor()
        if executor is not None:
            log.info(f"Using remote execution for {op}")
            return executor.execute_remote_operation(
                op.overloadpacket._qualified_op_name, args, kwargs
            )
        else:
            log.warning(f"Remote execution requested but not available for {op}, using local execution")

    # Mimicks the DeviceGuard system we have in aten
    with torch.remote.device(device):  # type: ignore[misc]
        return _kernel_fallback(op, *args, **kwargs)


def _kernel_fallback(op, *args, **kwargs):
    log.info("Calling kernel %s", op)

    # Check if we should use remote execution for this operation
    if _should_use_remote_execution(op, args, kwargs):
        executor = _get_remote_executor()
        if executor is not None:
            log.info(f"Using remote execution for {op}")
            return executor.execute_remote_operation(
                op.overloadpacket._qualified_op_name, args, kwargs
            )
        else:
            log.warning(f"Remote execution requested but not available for {op}, using local execution")

    op_name = None
    post_process = None
    if "out" in op._overloadname:
        # Note that all structured native op will call here
        if isinstance(kwargs["out"], tuple):
            raise RuntimeError(f"out= variant {op} with tuple out= not supported")
        if kwargs["out"].nelement() == 0:
            # Out variant that needs a resize, convert to an out of place
            # and handle generically below
            orig_out = kwargs["out"]
            del kwargs["out"]
            if op._overloadname != "out":
                raise RuntimeError(
                    "Cannot retranslate non-default out= variant form 0 size"
                )
            op = op.overloadpacket.default

            def _post_process():
                nonlocal real_res
                orig_out.set_(real_res)
                real_res = orig_out

            post_process = _post_process

        else:
            # No metadata update to do, just run the op on the device
            op_name = op.overloadpacket._qualified_op_name
            real_res = kwargs["out"]
    elif not tree_any(lambda obj: isinstance(obj, torch.Tensor), (args, kwargs)):
        # No Tensor argument means factory function
        # Check if this is a remote device factory function
        device_arg = kwargs.get("device", None)
        if device_arg is not None:
            if isinstance(device_arg, torch.device) and device_arg.type == "remote":
                # This is a remote device factory function - handle it
                pass  # Continue to normal processing below
            elif isinstance(device_arg, str) and (device_arg == "remote" or device_arg.startswith("remote")):
                # Reject string-based remote devices
                raise ValueError("Remote devices must be RemoteBackend objects. Use create_modal_device() or similar to create a RemoteBackend.")
            else:
                # Not a remote device factory function
                raise RuntimeError(f"{op} not handled yet.")
        else:
            # No device specified, not a remote factory function
            raise RuntimeError(f"{op} not handled yet.")
        
        # For remote device factory functions, we'll let them fall through to normal processing
        # The meta computation and device allocation will handle the rest
    elif op._schema.is_mutable or op is torch.ops.aten._copy_from.default:
        # Only handle inplace ops returning their first arg
        assert len(args) >= 1, f"Inplace {op} needs at least one arg"
        assert len(op._schema.returns) == 1, (
            f"NYI Inplace {op} with more than one return"
        )
        op_name = op.overloadpacket._qualified_op_name
        real_res = args[0]
    elif any(r.alias_info is not None for r in op._schema.returns):
        # View ops
        if op is torch.ops.aten.view.default:
            return torch.ops.aten._unsafe_view(*args, **kwargs)
        raise RuntimeError(f"{op} view op is not handled yet")

    if op_name is None:
        # 1. Compute updated metadata
        if torch.Tag.dynamic_output_shape not in op.tags:
            # Use CPU tensors with same shape/dtype for shape inference
            cpu_args, cpu_kwargs = to_device_no_copy("cpu", args, kwargs)
            cpu_res = op(*cpu_args, **cpu_kwargs)

            # 2. Allocate the output on remote device
            real_res, _ = to_device_no_copy("remote", cpu_res, {})
        else:
            # Slow version for data-dependent functions:
            # Run the op on the device just to get the output shape
            args_, kwargs_ = prepare_for_sending(args, kwargs)
            shape = driver.exec(
                "get_op_output_shape",
                op.overloadpacket._qualified_op_name,
                args_,
                kwargs_,
            )

            # 2. Allocate the output
            real_res = args[0].new(shape)

        # 3. Move to out variant
        kwargs["out"] = real_res
        # Let overload resolution find the out= overload
        op_name = op.overloadpacket._qualified_op_name

    # 4. Run the compute and populate the output on the device
    args, kwargs = prepare_for_sending(args, kwargs)
    driver.exec("run_op", op_name, args, kwargs)

    if post_process is not None:
        post_process()

    return real_res


def copy_from_device(from_):
    with torch.remote.device(from_.device):  # type: ignore[misc]
        args, _ = prepare_for_sending((from_,), {})
        return driver.exec("send_data", *args)


def copy_from_host_to_device(from_, to_):
    with torch.remote.device(to_.device):  # type: ignore[misc]
        args, _ = prepare_for_sending((to_,), {})
        driver.exec("recv_data", from_, *args)
    return to_


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