# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .dispatch import ExecutionStrategyFactory, OperationClassifier

log = logging.getLogger(__name__)

# Thread-local storage removed - meta execution tracking simplified


def _check_and_fix_empty_output_tensors(
    args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    """
    Simple check to resize empty output tensors based on input tensor shapes.
    Only handles the basic case where output tensor is empty and needs to match input shape.
    """
    if "out" not in kwargs:
        return

    output_tensor = kwargs["out"]
    if (
        not isinstance(output_tensor, torch.Tensor)
        or output_tensor.device.type != "remote"
    ):
        return

    # Only fix empty tensors
    if output_tensor.numel() != 0:
        return

    # For simple operations like abs, the output shape should match the first tensor input
    # Find the first tensor argument to use as a shape reference
    # This handles the common case without complex meta execution

    # Check args first (positional arguments)
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
            # Use storage ID to check if it's the same tensor
            storage_ptr = arg.untyped_storage().data_ptr()
            output_ptr = output_tensor.untyped_storage().data_ptr()
            if storage_ptr != output_ptr:
                if arg.numel() > 0:  # Use non-empty tensor as reference
                    output_tensor.resize_(arg.shape)
                    log.debug(
                        f"Resized empty output tensor to match input shape: {arg.shape}"
                    )
                    return

    # Check kwargs if no suitable arg found
    for value in kwargs.values():
        if isinstance(value, torch.Tensor) and value.device.type == "remote":
            # Use storage ID to check if it's the same tensor
            value_ptr = value.untyped_storage().data_ptr()
            output_ptr = output_tensor.untyped_storage().data_ptr()
            if value_ptr != output_ptr:
                if value.numel() > 0:  # Use non-empty tensor as reference
                    output_tensor.resize_(value.shape)
                    log.debug(
                        f"Resized empty output tensor to match input shape: {value.shape}"
                    )
                    return


# _check_and_apply_post_operation_resize function removed - no longer needed with simplified approach

from ._device_daemon import driver


# Lazy import to avoid import errors if remote execution is not available
def _get_remote_orchestrator() -> Optional[Any]:
    """Get the global remote orchestrator instance.

    The remote orchestrator handles communication with remote clients
    and coordinates tensor operations across devices. This function
    imports the orchestrator and gracefully handles cases where remote
    execution is not available.

    Returns:
        RemoteOrchestrator instance if available, None otherwise
    """
    try:
        from ._remote_orchestrator import remote_orchestrator

        return remote_orchestrator
    except ImportError as e:
        log.warning(f"Remote execution not available: {e}")
        return None


def _handle_view_operation(
    op: torch._ops.OpOverload, *args: Any, **kwargs: Any
) -> torch.Tensor:
    """
    Handle view operations locally with shared storage IDs.

    View operations create new tensor views that share the same remote storage,
    updating only local metadata (shape, stride, storage_offset).
    """
    op_name = op.overloadpacket._qualified_op_name
    log.info(f"🔍 Handling view operation: {op_name}")

    # Debug: Log details about this view operation
    log.info(f"   - arg count: {len(args)}, kwarg keys: {list(kwargs.keys())}")
    log.info(f"   - schema: {op._schema}")

    # Log tensor argument details
    tensor_args = []
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            device_index = arg.device.index if hasattr(arg.device, "index") else "N/A"
            tensor_args.append(
                f"arg[{i}]: {arg.device.type}:{device_index}, shape={arg.shape}"
            )
    if tensor_args:
        log.info(f"   - tensor args: {'; '.join(tensor_args)}")

    # Get the base tensor (first argument for most view operations)
    base_tensor = args[0]

    if base_tensor.device.type != "remote":
        # Not a remote tensor - execute normally
        return op(*args, **kwargs)

    # Validate that any other remote tensors in args/kwargs are on the same device
    base_device = base_tensor.device
    for i, arg in enumerate(args[1:], 1):  # Skip first arg (base tensor)
        if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
            if arg.device != base_device:
                from torch_remote.device import get_device_registry

                device_registry = get_device_registry()
                base_device_info = device_registry.get_device_by_index(
                    base_device.index
                )
                arg_device_info = device_registry.get_device_by_index(arg.device.index)

                base_name = (
                    base_device_info.machine_id
                    if base_device_info
                    else f"remote:{base_device.index}"
                )
                arg_name = (
                    arg_device_info.machine_id
                    if arg_device_info
                    else f"remote:{arg.device.index}"
                )

                raise RuntimeError(
                    f'Cannot perform view operation "{op_name}" between tensors '
                    f'on different remote devices. Base tensor on "{base_name}" '
                    f'(index {base_device.index}), argument {i} on "{arg_name}" '
                    f"(index {arg.device.index}). "
                    "Transfer tensors to the same device first."
                )

    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) and value.device.type == "remote":
            if value.device != base_device:
                from torch_remote.device import get_device_registry

                device_registry = get_device_registry()
                base_device_info = device_registry.get_device_by_index(
                    base_device.index
                )
                value_device_info = device_registry.get_device_by_index(
                    value.device.index
                )

                base_name = (
                    base_device_info.machine_id
                    if base_device_info
                    else f"remote:{base_device.index}"
                )
                value_name = (
                    value_device_info.machine_id
                    if value_device_info
                    else f"remote:{value.device.index}"
                )

                raise RuntimeError(
                    f'Cannot perform view operation "{op_name}" between tensors on different remote devices. '
                    f'Base tensor on "{base_name}" (index {base_device.index}), '
                    f"kwarg '{key}' on \"{value_name}\" (index {value.device.index}). "
                    f"Transfer tensors to the same device first."
                )

    # Get base tensor's storage ID (this is what data_ptr() returns)
    storage_id = base_tensor.untyped_storage().data_ptr()

    # Verify storage exists on a remote device
    device_idx = driver.exec("get_storage_device", storage_id)
    if device_idx is None:
        log.warning(
            f"No device found for storage {storage_id}, falling back to normal execution"
        )
        return op(*args, **kwargs)

    # Execute the view operation on meta tensors to get the new shape/stride
    # Convert ALL tensor arguments to meta tensors to avoid device mixing
    meta_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
            # Convert remote tensor to meta tensor
            meta_arg = torch.empty(arg.size(), dtype=arg.dtype, device="meta")
            if arg.stride() != meta_arg.stride():
                meta_arg = torch.as_strided(
                    meta_arg, arg.shape, arg.stride(), arg.storage_offset()
                )
            meta_args.append(meta_arg)
        else:
            meta_args.append(arg)

    meta_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) and value.device.type == "remote":
            # Convert remote tensor to meta tensor
            meta_value = torch.empty(value.size(), dtype=value.dtype, device="meta")
            if value.stride() != meta_value.stride():
                meta_value = torch.as_strided(
                    meta_value, value.shape, value.stride(), value.storage_offset()
                )
            meta_kwargs[key] = meta_value
        else:
            meta_kwargs[key] = value

    meta_result = op(*meta_args, **meta_kwargs)

    # Extract new tensor metadata
    new_shape = meta_result.size()
    new_stride = meta_result.stride()
    new_storage_offset = meta_result.storage_offset()

    # Now that we have C++ implementations for view operations,
    # we can use PyTorch's native as_strided which will call our C++ implementation
    result = torch.as_strided(base_tensor, new_shape, new_stride, new_storage_offset)

    # Verify the view was created correctly
    assert result.size() == new_shape, (
        f"View shape mismatch: expected {new_shape}, got {result.size()}"
    )
    assert result.stride() == new_stride, (
        f"View stride mismatch: expected {new_stride}, got {result.stride()}"
    )
    assert result.storage_offset() == new_storage_offset, (
        f"View offset mismatch: expected {new_storage_offset}, got {result.storage_offset()}"
    )
    assert (
        result.untyped_storage().data_ptr() == base_tensor.untyped_storage().data_ptr()
    ), "View should share storage"

    # PyTorch automatically manages storage reference counting
    log.info(f"✅ Created view tensor sharing storage {storage_id} for {op_name}")
    return result


def _remote_kernel_fallback(
    op: torch._ops.OpOverload, *args: Any, **kwargs: Any
) -> Any:
    import threading
    import traceback

    op_name = op.overloadpacket._qualified_op_name
    thread_id = threading.get_ident()

    # Debug: Log tensor devices to understand why we're being called
    tensor_devices = []
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            tensor_devices.append(f"args[{i}]: {arg.device.type}")
        elif (
            isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], torch.Tensor)
        ):
            tensor_devices.append(
                f"args[{i}]: [{', '.join(t.device.type for t in arg if isinstance(t, torch.Tensor))}]"
            )

    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            tensor_devices.append(f"{key}: {value.device.type}")
        elif (
            isinstance(value, (list, tuple))
            and value
            and isinstance(value[0], torch.Tensor)
        ):
            tensor_devices.append(
                f"{key}: [{', '.join(t.device.type for t in value if isinstance(t, torch.Tensor))}]"
            )

    log.info(f"📱 FALLBACK CALLED: {op_name} with tensors on devices: {tensor_devices}")
    log.info(f"📋 OP DETAILS: {op}")
    log.info(
        f"📋 ARGS: {[type(arg).__name__ + (f'[{len(arg)}]' if isinstance(arg, (list, tuple)) else '') for arg in args]}"
    )
    log.info(f"📋 KWARGS: {list(kwargs.keys())}")

    # Debug: Track call depth to detect recursion
    if not hasattr(_remote_kernel_fallback, "_call_stack"):
        _remote_kernel_fallback._call_stack = {}

    if thread_id not in _remote_kernel_fallback._call_stack:
        _remote_kernel_fallback._call_stack[thread_id] = []

    call_stack = _remote_kernel_fallback._call_stack[thread_id]
    call_depth = len(call_stack)

    log.info(f"🔄 ENTER [{call_depth}] {op_name} (thread {thread_id})")

    # Detect potential recursion
    if call_depth > 10:  # Arbitrary threshold
        log.error(f"🚨 RECURSION DETECTED! Call depth: {call_depth}")
        log.error(f"Call stack: {' -> '.join(call_stack)}")
        log.error("Current stack trace:")
        traceback.print_stack()
        raise RuntimeError(
            f"Recursion detected in remote kernel fallback: {call_stack}"
        )

    # Add current operation to call stack
    call_stack.append(op_name)

    try:
        log.info(
            f"📍 Processing {op_name}, Args: {len(args)}, Kwargs: {list(kwargs.keys())}"
        )

        result = _remote_kernel_fallback_impl(op, *args, **kwargs)

        log.info(f"🔄 EXIT [{call_depth}] {op_name} -> SUCCESS")
        return result

    except Exception as e:
        log.error(f"🔄 EXIT [{call_depth}] {op_name} -> ERROR: {e}")
        raise
    finally:
        # Remove current operation from call stack
        if call_stack and call_stack[-1] == op_name:
            call_stack.pop()


def _remote_kernel_fallback_impl(
    op: torch._ops.OpOverload, *args: Any, **kwargs: Any
) -> Any:
    # Get operation name
    op_name = op.overloadpacket._qualified_op_name

    # Use strategy pattern for operation execution
    operation_type = OperationClassifier.classify_operation(op)
    strategy = ExecutionStrategyFactory.get_strategy(operation_type)

    log.info(f"🎯 Executing {op_name} using {strategy.__class__.__name__} (type: {operation_type.value})")
    return strategy.execute(op, args, kwargs)



def copy_from_device(from_: torch.Tensor) -> torch.Tensor:
    """Copy data from remote tensor to CPU tensor using remote execution"""
    if from_.device.type != "remote":
        raise ValueError("copy_from_device requires a remote tensor")

    # Use remote execution to get the tensor data
    orchestrator = _get_remote_orchestrator()
    if orchestrator is not None:
        from .device import get_device_registry

        # Get the device backend
        registry = get_device_registry()
        device = registry.get_device_by_index(from_.device.index)

        if device is None:
            raise RuntimeError(
                f"No RemoteMachine found for remote device index {from_.device.index}"
            )

        # Get the client for this device
        client = device.get_client()
        if client is None or not client.is_running():
            raise RuntimeError(f"Client not available for device {device.machine_id}")

        # Get storage data using storage ID
        storage_id = from_.untyped_storage().data_ptr()
        log.info(f"Copying storage ID {storage_id} from remote to CPU")

        # Use client to get tensor data by storage ID with view information
        # Pass tensor metadata so remote side can serialize just the view's data
        tensor_data = client.get_storage_data(
            storage_id,
            shape=list(from_.shape),
            stride=list(from_.stride()),
            storage_offset=from_.storage_offset(),
            dtype=str(from_.dtype),
        )

        # Deserialize the tensor data as contiguous representation
        # Since we now serialize with .contiguous(), the deserialized tensor contains exactly
        # the data that should be in the result tensor - no view reconstruction needed
        result = orchestrator._deserialize_tensor(tensor_data)
        # Verify the result has the expected shape (it should match the remote tensor's shape)
        if result.size() != from_.size():
            log.warning(
                f"Deserialized tensor shape {result.size()} doesn't match remote tensor shape {from_.size()}"
            )

        log.info(
            f"Successfully copied contiguous tensor data for storage ID {storage_id} to CPU"
        )
        return result
    else:
        raise RuntimeError(
            "Cannot copy from remote device: remote execution not available"
        )


def copy_from_host_to_device(from_: torch.Tensor, to_: torch.Tensor) -> torch.Tensor:
    """Copy data from CPU tensor to remote tensor using remote execution"""
    if to_.device.type != "remote":
        raise ValueError("copy_from_host_to_device requires a remote target tensor")
    if from_.device.type != "cpu":
        raise ValueError("copy_from_host_to_device requires a CPU source tensor")

    # Use remote execution to send the tensor data
    orchestrator = _get_remote_orchestrator()
    if orchestrator is not None:
        from .device import get_device_registry

        # Get the device backend
        registry = get_device_registry()
        device = registry.get_device_by_index(to_.device.index)

        if device is None:
            raise RuntimeError(
                f"No RemoteMachine found for remote device index {to_.device.index}"
            )

        # Get the client for this device
        client = device.get_client()
        if client is None or not client.is_running():
            raise RuntimeError(f"Client not available for device {device.machine_id}")

        # Send tensor data using storage ID
        storage_id = to_.untyped_storage().data_ptr()
        log.info(f"Copying CPU tensor to remote storage ID {storage_id}")

        # Serialize the CPU tensor
        tensor_data = orchestrator._serialize_tensor(from_)
        # Use client to update tensor with specific ID
        # This will overwrite any existing empty tensor with the actual data
        client.update_storage(tensor_data, storage_id)
        log.info(f"Successfully created/updated remote tensor with ID {storage_id}")
        return to_
    else:
        raise RuntimeError(
            "Cannot copy to remote device: remote execution not available"
        )


def _copy_from(from_: torch.Tensor, to_: torch.Tensor) -> torch.Tensor:
    """Copy data from one tensor to another, handling remote device transfers.

    This function implements the core copy operation for remote tensors,
    supporting CPU↔remote transfers and preventing cross-device transfers
    between different remote devices.

    Args:
        from_: Source tensor to copy from
        to_: Target tensor to copy to

    Returns:
        Target tensor with copied data

    Raises:
        RuntimeError: If attempting to transfer between different remote devices
    """
    # Simplified copy implementation - remote tensors are now regular torch.Tensor
    # with proper device handling via C++ allocator

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
                    f'Source device: "{from_device.machine_id}" (index {from_.device.index}), '
                    f'Target device: "{to_device.machine_id}" (index {to_.device.index}). '
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

    return result


def _local_scalar_dense(self: torch.Tensor):
    """Custom implementation of _local_scalar_dense for remote tensors."""
    # Check that tensor is scalar (replicate PyTorch's exact behavior)
    if self.numel() != 1:
        raise RuntimeError(
            f"a Tensor with {self.numel()} elements cannot be converted to Scalar"
        )

    # Get scalar value from remote device
    log.info(
        "🔢 _local_scalar_dense operation: retrieving scalar value from remote device"
    )

    # Get storage ID and remote machine
    storage_id = self.untyped_storage().data_ptr()

    # Get the remote machine using device registry
    from .device import get_device_registry

    registry = get_device_registry()
    machine = registry.get_device_by_index(self.device.index)

    if machine is None:
        raise RuntimeError(
            f"No RemoteMachine found for remote device index {self.device.index}"
        )

    # Get the client for this device
    client = machine.get_client()
    if client is None or not client.is_running():
        raise RuntimeError(f"Client not available for device {machine.machine_id}")

    # Get serialized tensor data for this scalar
    tensor_data = client.get_storage_data(
        storage_id,
        shape=list(self.shape),
        stride=list(self.stride()),
        storage_offset=self.storage_offset(),
        dtype=str(self.dtype),
    )

    # Deserialize to CPU tensor and extract scalar value
    orchestrator = _get_remote_orchestrator()
    if orchestrator is None:
        raise RuntimeError(
            "Cannot retrieve scalar value: remote execution not available"
        )

    cpu_tensor = orchestrator._deserialize_tensor(tensor_data)

    # Call item() on the CPU tensor to get the Python scalar
    return cpu_tensor.item()


def _to_copy(
    input: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[Union[torch.device, str, int]] = None,
    pin_memory: Optional[bool] = None,
    non_blocking: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> torch.Tensor:
    """Implementation of tensor.to() for remote tensors with cross-device transfer restriction."""

    # Handle device-specific transfers first
    if device is not None:
        target_device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

        # Different device transfer - check if both are remote
        if (
            input.device.type == "remote"
            and target_device.type == "remote"
            and input.device != target_device
        ):
            # Cross-device remote transfer - NOT ALLOWED
            from torch_remote.device import get_device_registry

            device_registry = get_device_registry()
            from_device = device_registry.get_device_by_index(input.device.index)
            to_device = device_registry.get_device_by_index(target_device.index)

            raise RuntimeError(
                f"Cannot transfer tensor between different remote devices. "
                f'Source device: "{from_device.machine_id}" (index {input.device.index}), '
                f'Target device: "{to_device.machine_id}" (index {target_device.index}). '
                f"Transfer tensors to CPU first: tensor.cpu().to(target_device)"
            )

    # For all other cases, create a new tensor and use _copy_from if needed
    # This avoids infinite recursion by not calling back to the kernel fallback

    # Determine output parameters
    output_dtype = dtype if dtype is not None else input.dtype
    output_layout = layout if layout is not None else input.layout
    output_device = torch.device(device) if device is not None else input.device
    output_memory_format = (
        memory_format if memory_format is not None else torch.contiguous_format
    )

    # Create output tensor
    if output_device.type == "remote":
        # Create empty remote tensor - use contiguous format for remote tensors
        result = torch.empty(
            input.size(),
            dtype=output_dtype,
            layout=output_layout,
            device=output_device,
            memory_format=torch.contiguous_format,
        )
    else:
        # Create empty tensor on target device
        result = torch.empty(
            input.size(),
            dtype=output_dtype,
            layout=output_layout,
            device=output_device,
            memory_format=output_memory_format,
        )

    # Copy data if needed (different device or same device but different dtype)
    if input.device != output_device or input.dtype != output_dtype:
        # Use _copy_from to handle the actual data transfer
        result = _copy_from(input, result)
    else:
        # Same device and dtype - just return input (no copy needed)
        return input

    return result


# resize_ implementation has been moved to C++ (RemoteMem.cpp) following OpenReg pattern
# The C++ implementation calls PyTorch's default resize_ with resizePrivateUse1Bytes hook


def _set_source_tensor(ten1: torch.Tensor, ten2: torch.Tensor) -> torch.Tensor:
    """Set one tensor to point to another tensor's storage.

    This creates a view relationship where ten1 shares ten2's storage,
    shape, stride, and offset. Used for tensor aliasing operations.

    Args:
        ten1: Tensor to modify
        ten2: Source tensor to point to

    Returns:
        Modified tensor ten1 pointing to ten2's data
    """
    return torch.ops.aten.set_.source_Storage_storage_offset(
        ten1,
        ten2.untyped_storage(),
        ten2.storage_offset(),
        ten2.size(),
        ten2.stride(),
    )


# Note: set_.source_Storage_storage_offset is implemented in C++ (RemoteMem.cpp)
# The C++ implementation calls at::cpu::set_ which is exactly what we need


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

# AUTOGRAD_"PrivateUse1" registrations removed for copy operations:
# _copy_from and _to_copy were causing autograd issues because:
# 1. _copy_from breaks autograd chain via copy_from_device() -> _deserialize_tensor() -> .detach()
# 2. In-place operations on leaf variables with requires_grad=True are not allowed
# 3. No proper grad_fn creation for autograd graph connectivity
# 4. This remote tensor system doesn't need device-specific autograd behavior for data movement
# 5. PyTorch's default autograd handles remote tensors correctly when only "PrivateUse1" is registered


# Note: set_.source_Storage_storage_offset is already implemented in C++ (RemoteMem.cpp)
# so we don't register a Python version to avoid conflicts
# resize_ is now implemented in C++ (RemoteMem.cpp) following OpenReg pattern

# via TORCH_LIBRARY_IMPL in RemoteMem.cpp, so we don't register Python implementations

# These factory functions are now handled by C++ implementations
# via the registered TORCH_LIBRARY_IMPL dispatch system

# when we add them to TORCH_LIBRARY_IMPL in RemoteMem.cpp
