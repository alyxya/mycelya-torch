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

    def _(*args, **kwargs):
        log.info("Calling hook %s", name)
        return driver.exec(name, *args, **kwargs)

    _IMPL_REGISTRY[name] = _
    return _


def _should_use_remote_execution(op, args, kwargs):
    """
    Determine whether to use remote GPU execution for this operation.
    
    Currently uses remote execution for most tensor operations involving remote tensors,
    except for certain operations that are better handled locally.
    """
    # Skip remote execution for certain operations
    skip_ops = {
        # Memory operations
        "aten.copy_",
        "aten._copy_from",
        "aten.set_",
        "aten.resize_",
        "aten.storage_offset",
        "aten.stride",
        "aten.size",
        "aten.numel",
        "aten.dim",
        "aten.is_contiguous",
        # Factory functions
        "aten.empty",
        "aten.empty_like",
        "aten.zeros",
        "aten.zeros_like",
        "aten.ones",
        "aten.ones_like",
        "aten.normal_",
        "aten.uniform_",
        "aten.random_",
        # Scalar operations (low compute)
        "aten.item",
        "aten._local_scalar_dense",
        # View operations (should be fast locally)
        "aten.view",
        "aten.reshape",
        "aten.squeeze",
        "aten.unsqueeze",
        "aten.transpose",
        "aten.permute",
        # Device transfer operations (CRITICAL - prevents infinite recursion)
        "aten.to",
        "aten.to.device",
        "aten.to.dtype",
        "aten.to.dtype_layout",
        "aten.cpu",
        "aten.cuda",
    }
    
    op_name = op.overloadpacket._qualified_op_name
    
    # Skip if operation is in the skip list
    if any(skip_op in op_name for skip_op in skip_ops):
        return False
    
    # Use remote execution for compute-intensive operations
    compute_intensive_ops = {
        "aten.add",
        "aten.sub", 
        "aten.mul",
        "aten.div",
        "aten.mm",
        "aten.bmm",
        "aten.addmm",
        "aten.conv2d",
        "aten.linear",
        "aten.relu",
        "aten.sigmoid",
        "aten.tanh",
        "aten.softmax",
        "aten.log_softmax",
        "aten.cross_entropy",
        "aten.mse_loss",
        "aten.sum",
        "aten.mean",
        "aten.var",
        "aten.std",
        "aten.max",
        "aten.min",
        "aten.argmax",
        "aten.argmin",
        "aten.sort",
        "aten.topk",
        "aten.matmul",
        "aten.einsum",
    }
    
    # Check if this is a compute-intensive operation
    if any(compute_op in op_name for compute_op in compute_intensive_ops):
        return True
    
    # All operations on remote tensors should execute on the remote machine
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
            return True
    
    for value in kwargs.values():
        if isinstance(value, torch.Tensor) and value.device.type == "remote":
            return True
    
    return False


def _remote_kernel_fallback(op, *args, **kwargs):
    print(f"🔍 _remote_kernel_fallback called for {op}")
    def get_tensor_device(*args):
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
                return arg.device

    device = get_tensor_device(*args)
    if device is None:
        print(f"❌ No remote device found, using standard kernel fallback")
        return _kernel_fallback(op, *args, **kwargs)

    print(f"✅ Found remote device: {device}")
    
    # Check if we should use remote execution
    should_use_remote = _should_use_remote_execution(op, args, kwargs)
    print(f"🤔 Should use remote execution: {should_use_remote}")
    
    if should_use_remote:
        executor = _get_remote_executor()
        print(f"🔧 Remote executor: {executor}")
        if executor is not None:
            log.info(f"🚀 Using remote execution for {op}")
            print(f"🚀 Creating remote job for {op.overloadpacket._qualified_op_name}")
            return executor.execute_remote_operation(
                op.overloadpacket._qualified_op_name, args, kwargs
            )
        else:
            log.warning(f"Remote execution requested but not available for {op}, using local execution")

    # Mimicks the DeviceGuard system we have in aten
    print(f"🔄 Using local execution for {op}")
    with torch.remote.device(device):  # type: ignore[misc]
        return _kernel_fallback(op, *args, **kwargs)


def _kernel_fallback(op, *args, **kwargs):
    log.info("Calling kernel %s", op)

    # Check if we should use remote execution for this operation
    if _should_use_remote_execution(op, args, kwargs):
        executor = _get_remote_executor()
        if executor is not None:
            log.info(f"Using remote execution for {op}")
            print(f"🚀 Creating remote job for {op.overloadpacket._qualified_op_name}")
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
        device_arg = kwargs.get('device', None)
        if device_arg is not None:
            if isinstance(device_arg, torch.device) and device_arg.type == "remote":
                # This is a remote device factory function - handle it
                pass  # Continue to normal processing below
            elif isinstance(device_arg, str) and (device_arg == "remote" or device_arg.startswith("remote")):
                # Reject string-based remote devices
                raise ValueError("Remote devices must be BackendDevice objects. Use create_modal_device() or similar to create a BackendDevice.")
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
            # Usual case: run the meta op to see the output metadata
            meta_args, meta_kwargs = to_device_no_copy("meta", args, kwargs)
            meta_res = op(*meta_args, **meta_kwargs)

            # 2. Allocate the output
            real_res, _ = to_device_no_copy("remote", meta_res, {})
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
    # Handle persistent tensor transfers
    if hasattr(from_, 'is_persistent') and from_.is_persistent and from_.tensor_id:
        if to_.device.type == "cpu":
            # Remote persistent tensor -> CPU: fetch data from remote
            return from_.cpu()  # This will use the updated cpu() method
        elif to_.device.type == "remote":
            if from_.device.index == to_.device.index:
                # Same device: no actual transfer needed for persistent tensors
                return to_
            else:
                # Different remote devices: transfer via CPU
                cpu_data = from_.cpu()
                return to_.copy_(cpu_data)
    
    if hasattr(to_, 'is_persistent') and to_.is_persistent and to_.tensor_id:
        if from_.device.type == "cpu":
            # CPU -> Remote persistent tensor: update remote data
            try:
                from .device import get_device_registry
                from ._remote_executor import _get_remote_executor
                
                registry = get_device_registry() 
                device = registry.get_device_by_index(to_.device.index)
                executor = _get_remote_executor()
                
                if device and executor:
                    # Serialize CPU tensor and update remote
                    tensor_data = executor._serialize_tensor(from_)
                    gpu_machine = device.get_gpu_machine()
                    if gpu_machine and gpu_machine.is_running():
                        # Create new tensor on remote and update the tensor_id
                        new_tensor_id = gpu_machine.create_tensor(tensor_data, to_.tensor_id)
                        # Update metadata
                        to_._tensor_id = new_tensor_id
                        return to_
            except Exception as e:
                print(f"Warning: Failed to update persistent tensor, falling back: {e}")
    
    # Legacy behavior for non-persistent tensors
    if from_.device.type == to_.device.type:
        assert from_.device.type == "remote"
        if from_.device.index == to_.device.index:
            op = torch.ops.aten.copy_.default
            return _remote_kernel_fallback(op, to_, from_)
        else:
            host_mem = copy_from_device(from_)
            return copy_from_host_to_device(host_mem, to_)
    elif from_.device.type == "remote":
        host_mem = copy_from_device(from_)
        return to_.copy_(host_mem)
    elif to_.device.type == "remote":
        return copy_from_host_to_device(from_, to_)
    else:
        raise RuntimeError("Should not happen")


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


def cpu_tensor_to_persistent_remote(cpu_tensor, device):
    """
    Convert a CPU tensor to a persistent remote tensor.
    
    Args:
        cpu_tensor: The CPU tensor to convert
        device: Target remote device (torch.device)
        
    Returns:
        RemoteTensorData with persistent tensor ID
    """
    try:
        from .device import get_device_registry
        from ._remote_executor import _get_remote_executor
        from ._meta_parser import RemoteTensorData
        
        device_idx = device.index if device.index is not None else 0
        registry = get_device_registry()
        backend_device = registry.get_device_by_index(device_idx)
        
        if backend_device is not None:
            executor = _get_remote_executor()
            if executor is not None:
                # Serialize CPU tensor and send to remote
                tensor_data = executor._serialize_tensor(cpu_tensor)
                tensor_id = executor.create_tensor_on_remote(tensor_data, backend_device)
                
                # Get metadata from remote
                gpu_machine = backend_device.get_gpu_machine()
                if gpu_machine and gpu_machine.is_running():
                    metadata = gpu_machine.get_tensor_metadata(tensor_id)
                    
                    # Create persistent RemoteTensorData
                    return RemoteTensorData.from_tensor_id(tensor_id, metadata, device_idx)
        
        # Fallback to legacy conversion
        return cpu_tensor.to(device)
        
    except Exception as e:
        print(f"Warning: Failed to create persistent remote tensor: {e}")
        # Fallback to legacy conversion
        return cpu_tensor.to(device)


def empty_remote(
    size,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    if device is None:
        device = torch.device("remote", 0)
    elif isinstance(device, int):
        device = torch.device("remote", device)
    elif isinstance(device, str):
        device = torch.device(device)
    
    dtype = dtype or torch.get_default_dtype()
    
    if layout is not None and layout != torch.strided:
        raise RuntimeError("Non strided layout not supported")
    if pin_memory:
        raise RuntimeError("Pin memory can only be on CPU")
    
    device_idx = device.index if device.index is not None else 0
    
    # Try to use the new persistent tensor system first
    try:
        from .device import get_device_registry
        from ._remote_executor import _get_remote_executor
        from ._meta_parser import RemoteTensorData
        
        registry = get_device_registry()
        backend_device = registry.get_device_by_index(device_idx)
        
        if backend_device is not None:
            executor = _get_remote_executor()
            if executor is not None:
                # Use new persistent system: create tensor directly on remote machine
                tensor_id = executor.create_factory_tensor_on_remote(
                    "empty", [size], {"dtype": dtype}, backend_device
                )
                
                # Get metadata from remote machine
                gpu_machine = backend_device.get_gpu_machine()
                if gpu_machine and gpu_machine.is_running():
                    metadata = gpu_machine.get_tensor_metadata(tensor_id)
                    
                    # Create RemoteTensorData with tensor ID (no local data)
                    return RemoteTensorData.from_tensor_id(tensor_id, metadata, device_idx)
    
    except Exception as e:
        print(f"Warning: Failed to create persistent tensor, falling back to legacy mode: {e}")
    
    # Fallback to legacy system
    with torch.remote.device(device):  # type: ignore[misc]
        # Use driver to allocate empty tensor on remote device
        args, _ = prepare_for_sending((size, dtype), {})
        # Get the meta result and convert to RemoteTensorData
        meta_result = driver.exec("empty_tensor", *args)
        from ._meta_parser import RemoteTensorData
        driver._lazy_init()  # Ensure devices are initialized
        allocator = driver.devices[device_idx][3].allocator
        tensor = RemoteTensorData.from_meta(allocator, meta_result, device_idx)
        
        return tensor


def empty_strided_remote(
    size,
    stride,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    if device is None:
        device = torch.device("remote", 0)
    elif isinstance(device, int):
        device = torch.device("remote", device)
    elif isinstance(device, str):
        device = torch.device(device)
    
    dtype = dtype or torch.get_default_dtype()
    
    if layout is not None and layout != torch.strided:
        raise RuntimeError("Non strided layout not supported")
    if pin_memory:
        raise RuntimeError("Pin memory can only be on CPU")
    
    # Create empty strided tensor with proper device and dtype
    with torch.remote.device(device):  # type: ignore[misc]
        # Use driver to allocate empty strided tensor on remote device
        args, _ = prepare_for_sending((size, stride, dtype), {})
        # Get the meta result and convert to RemoteTensorData
        meta_result = driver.exec("empty_strided_tensor", *args)
        from ._meta_parser import RemoteTensorData
        device_idx = device.index if device.index is not None else 0
        driver._lazy_init()  # Ensure devices are initialized
        allocator = driver.devices[device_idx][3].allocator
        tensor = RemoteTensorData.from_meta(allocator, meta_result, device_idx)
        
        # No need to attach _device_id anymore - device.index provides the mapping
        return tensor


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
_remote_lib_aten.impl(
    "empty.memory_format", empty_remote, dispatch_key="PrivateUse1"
)
_remote_lib_aten.impl(
    "empty_strided", empty_strided_remote, dispatch_key="PrivateUse1"
)

def randn_remote(
    size,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    """Remote implementation for torch.randn"""
    return _create_factory_tensor("randn", size, dtype, layout, device, pin_memory, memory_format)

def zeros_remote(
    size,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    """Remote implementation for torch.zeros"""
    return _create_factory_tensor("zeros", size, dtype, layout, device, pin_memory, memory_format)

def ones_remote(
    size,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    """Remote implementation for torch.ones"""
    return _create_factory_tensor("ones", size, dtype, layout, device, pin_memory, memory_format)

def _create_factory_tensor(factory_op, size, dtype, layout, device, pin_memory, memory_format):
    """Common implementation for factory tensor operations"""
    if device is None:
        device = torch.device("remote", 0)
    elif isinstance(device, int):
        device = torch.device("remote", device)
    elif isinstance(device, str):
        device = torch.device(device)
    
    dtype = dtype or torch.get_default_dtype()
    
    if layout is not None and layout != torch.strided:
        raise RuntimeError("Non strided layout not supported")
    if pin_memory:
        raise RuntimeError("Pin memory can only be on CPU")
    
    device_idx = device.index if device.index is not None else 0
    
    # Try to use the new persistent tensor system
    try:
        from .device import get_device_registry
        from ._remote_executor import _get_remote_executor
        from ._meta_parser import RemoteTensorData
        
        registry = get_device_registry()
        backend_device = registry.get_device_by_index(device_idx)
        
        if backend_device is not None:
            executor = _get_remote_executor()
            if executor is not None:
                # Use new persistent system: create tensor directly on remote machine
                tensor_id = executor.create_factory_tensor_on_remote(
                    factory_op, [size], {"dtype": dtype}, backend_device
                )
                
                # Get metadata from remote machine
                gpu_machine = backend_device.get_gpu_machine()
                if gpu_machine and gpu_machine.is_running():
                    metadata = gpu_machine.get_tensor_metadata(tensor_id)
                    
                    # Create RemoteTensorData with tensor ID (no local data)
                    return RemoteTensorData.from_tensor_id(tensor_id, metadata, device_idx)
    
    except Exception as e:
        print(f"Warning: Factory tensor creation failed: {e}, falling back to legacy")
    
    # Fallback to legacy behavior if persistent system fails
    from ._meta_parser import RemoteTensorData
    
    # Use meta tensor to avoid local memory allocation
    # This creates only metadata without allocating actual memory
    meta_tensor = torch.empty(size, dtype=dtype, device='meta')
    
    # Convert to RemoteTensorData
    remote_tensor = RemoteTensorData(meta_tensor)
    remote_tensor._remote_device_index = device_idx
    remote_tensor._is_persistent = False
    
    return remote_tensor

# Register factory function implementations
_remote_lib_aten.impl(
    "randn", randn_remote, dispatch_key="PrivateUse1"
)
_remote_lib_aten.impl(
    "zeros", zeros_remote, dispatch_key="PrivateUse1"
)
_remote_lib_aten.impl(
    "ones", ones_remote, dispatch_key="PrivateUse1"
)


def cleanup_library_registrations():
    """Clean up library registrations to prevent hanging."""
    global _remote_lib, _remote_lib_aten
    try:
        # PyTorch doesn't provide a clean way to unregister, but we can try
        # Calling this during cleanup might help
        if hasattr(_remote_lib, '_destroy'):
            _remote_lib._destroy()
        if hasattr(_remote_lib_aten, '_destroy'):
            _remote_lib_aten._destroy()
    except Exception:
        pass