# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from .constants import REMOTE_DEVICE_TYPE, CPU_DEVICE_TYPE, META_DEVICE_TYPE, PRIVATEUSE1_DISPATCH_KEY


log = logging.getLogger(__name__)

# Thread-local storage to track if we're in meta execution
_thread_local = threading.local()

def is_in_meta_execution() -> bool:
    """Check if we're currently executing operations for meta tensor inference."""
    return getattr(_thread_local, 'in_meta_execution', False)

def set_meta_execution(value: bool) -> None:
    """Set the meta execution flag."""
    _thread_local.in_meta_execution = value


def _check_and_fix_output_tensor_metadata(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> bool:
    """
    Check if output tensor metadata matches expected metadata from meta execution.
    If not, resize the output tensor appropriately.
    
    Returns:
        True if a resize was performed, False otherwise
    """
    # Only handle operations with 'out' parameter
    if "out" not in kwargs:
        return False
        
    output_tensor = kwargs["out"]
    if not isinstance(output_tensor, torch.Tensor) or output_tensor.device.type != REMOTE_DEVICE_TYPE:
        return False
    
    op_name = op.overloadpacket._qualified_op_name
    
    try:
        # Run meta execution to determine expected output metadata
        meta_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                meta_tensor = torch.empty(
                    arg.shape,
                    dtype=arg.dtype,
                    device='meta'
                )
                if arg.stride() != meta_tensor.stride():
                    meta_tensor = torch.as_strided(
                        meta_tensor,
                        arg.shape,
                        arg.stride(),
                        arg.storage_offset()
                    )
                meta_args.append(meta_tensor)
            else:
                meta_args.append(arg)
        
        # Create meta kwargs without the 'out' parameter
        meta_kwargs = {}
        for key, value in kwargs.items():
            if key == "out":
                continue  # Skip 'out' parameter for meta execution
            elif isinstance(value, torch.Tensor):
                meta_tensor = torch.empty(
                    value.shape,
                    dtype=value.dtype,
                    device='meta'
                )
                if value.stride() != meta_tensor.stride():
                    meta_tensor = torch.as_strided(
                        meta_tensor,
                        value.shape,
                        value.stride(),
                        value.storage_offset()
                    )
                meta_kwargs[key] = meta_tensor
            else:
                meta_kwargs[key] = value
        
        # Execute on meta device to get expected output metadata
        # We need to call the non-out variant to get the expected shape
        set_meta_execution(True)
        try:
            with torch.device('meta'):
                # For .out variants, call the base operation without 'out' parameter
                if op._overloadname == "out":
                    # Get the base operation (without .out)
                    base_op = op.overloadpacket.default
                    expected_result = base_op(*meta_args, **meta_kwargs)
                else:
                    expected_result = op(*meta_args, **meta_kwargs)
        finally:
            set_meta_execution(False)
        
        # Compare actual vs expected metadata
        actual_shape = output_tensor.shape
        expected_shape = expected_result.shape
        actual_numel = output_tensor.numel()
        expected_numel = expected_result.numel()
        
        if tuple(actual_shape) == tuple(expected_shape):
            # Shapes match, no resize needed
            return False
        
        log.debug(f"Metadata mismatch for {op_name}: actual {actual_shape} vs expected {expected_shape}")
        
        # Decide resize strategy based on current tensor size
        if actual_numel == 0:
            # Option 1: Empty tensor - resize BEFORE operation
            output_tensor.resize_(expected_shape)
            log.debug(f"Pre-resized output tensor to {output_tensor.shape}")
            return True
        else:
            # Option 2: Non-empty tensor - will resize AFTER operation
            # Store expected metadata for post-operation resize
            output_tensor._expected_post_op_shape = expected_shape
            output_tensor._expected_post_op_numel = expected_numel
            return False  # Don't pre-resize, but mark for post-resize
            
    except Exception as e:
        log.debug(f"Error during metadata check for {op_name}: {e}")
        return False


def _check_and_apply_post_operation_resize(
    op: torch._ops.OpOverload,
    result: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Any:
    """
    Check if any output tensors need post-operation resizing and apply if needed.
    
    This handles cases where non-empty tensors had size mismatches and were marked
    for post-operation resize during the pre-operation metadata check.
    
    Args:
        op: The operation that was executed
        result: The result of the operation
        args: Original operation arguments
        kwargs: Original operation keyword arguments
        
    Returns:
        The result, potentially with resized tensors
    """
    op_name = op.overloadpacket._qualified_op_name
    
    # Check if any output tensor was marked for post-operation resize
    output_tensor = None
    if "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
        output_tensor = kwargs["out"]
    elif isinstance(result, torch.Tensor):
        output_tensor = result
    
    if output_tensor is None:
        return result
        
    # Check if this tensor was marked for post-operation resize
    if (hasattr(output_tensor, '_expected_post_op_shape') and 
        hasattr(output_tensor, '_expected_post_op_numel')):
        
        expected_shape = output_tensor._expected_post_op_shape
        expected_numel = output_tensor._expected_post_op_numel
        actual_shape = output_tensor.shape
        actual_numel = output_tensor.numel()
        
        # Clean up the temporary attributes
        delattr(output_tensor, '_expected_post_op_shape')
        delattr(output_tensor, '_expected_post_op_numel')
        
        # Check if resize is still needed
        if tuple(actual_shape) != tuple(expected_shape):
            log.debug(f"Post-op metadata mismatch for {op_name}: actual {actual_shape} vs expected {expected_shape}")
            
            # Apply post-operation resize 
            try:
                # Now that we have a registered resize_ implementation, this should
                # use our _resize_remote function and bypass the kernel fallback
                output_tensor.resize_(expected_shape)
                log.debug(f"Post-operation resize completed: {output_tensor.shape}")
            except Exception as resize_error:
                log.debug(f"Post-operation resize failed: {resize_error}")
                # Continue without resize rather than failing the entire operation
    
    return result

from ._device_daemon import driver
from ._meta_parser import prepare_for_sending


def _update_tensor_metadata_from_meta_execution(
    op: torch._ops.OpOverload,
    result_tensor: torch.Tensor,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> None:
    """
    Update result tensor metadata by executing the same operation on meta tensors.
    
    This handles cases where operations like resize_ change tensor metadata and we need
    to update the local tensor to match what happened on the remote device.
    
    Args:
        op: The operation that was executed
        result_tensor: The tensor whose metadata should be updated
        args: Original operation arguments
        kwargs: Original operation keyword arguments
    """
    try:
        # Convert all tensor args to meta tensors with same properties
        meta_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                meta_tensor = torch.empty(
                    arg.shape,
                    dtype=arg.dtype,
                    device='meta'
                )
                # Preserve stride if non-contiguous
                if arg.stride() != meta_tensor.stride():
                    meta_tensor = torch.as_strided(
                        meta_tensor,
                        arg.shape,
                        arg.stride(),
                        arg.storage_offset()
                    )
                meta_args.append(meta_tensor)
            else:
                meta_args.append(arg)
        
        # Convert kwargs to meta tensors
        meta_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                meta_tensor = torch.empty(
                    value.shape,
                    dtype=value.dtype,
                    device='meta'
                )
                if value.stride() != meta_tensor.stride():
                    meta_tensor = torch.as_strided(
                        meta_tensor,
                        value.shape,
                        value.stride(),
                        value.storage_offset()
                    )
                meta_kwargs[key] = meta_tensor
            else:
                meta_kwargs[key] = value
        
        # Store original metadata for comparison
        original_shape = result_tensor.shape
        original_stride = result_tensor.stride()
        original_offset = result_tensor.storage_offset()
        
        # Execute on meta device to see what the metadata should be
        try:
            # Mark that we're in meta execution to prevent remote device allocation
            _thread_local.in_meta_execution = True
            
            with torch.device('meta'):
                meta_result = op(*meta_args, **meta_kwargs)
                
        except Exception as meta_exec_error:
            log.debug(f"Meta execution failed for {op.overloadpacket._qualified_op_name}: {meta_exec_error}")
            return  # Skip metadata update if meta execution fails
        finally:
            # Clear the meta execution flag
            _thread_local.in_meta_execution = False
        
        # If it's an inplace operation, check the first argument for changes
        if op._schema.is_mutable and not kwargs.get("out"):
            if len(meta_args) > 0 and isinstance(meta_args[0], torch.Tensor):
                target_meta_tensor = meta_args[0]
            else:
                return
        # If it has an "out" parameter, check that tensor
        elif "out" in kwargs and isinstance(meta_kwargs["out"], torch.Tensor):
            target_meta_tensor = meta_kwargs["out"]
        # For regular operations, check the result
        elif isinstance(meta_result, torch.Tensor):
            target_meta_tensor = meta_result
        else:
            return
        
        # Check if metadata changed
        try:
            new_shape = target_meta_tensor.shape
            new_stride = target_meta_tensor.stride()
            new_offset = target_meta_tensor.storage_offset()
            
            # Compare metadata carefully to avoid boolean tensor ambiguity
            shape_changed = tuple(original_shape) != tuple(new_shape)
            stride_changed = tuple(original_stride) != tuple(new_stride)
            offset_changed = original_offset != new_offset
            
            metadata_changed = shape_changed or stride_changed or offset_changed
        except Exception as comparison_error:
            log.debug(f"Failed to compare tensor metadata for {op.overloadpacket._qualified_op_name}: {comparison_error}")
            return  # Skip metadata update if comparison fails
        
        if metadata_changed:
            op_name = op.overloadpacket._qualified_op_name
            
            # Log warning for non-resize operations
            if "resize" not in op_name:
                log.warning(f"⚠️ Operation {op_name} changed tensor metadata - shape: {original_shape} → {new_shape}, stride: {original_stride} → {new_stride}")
            else:
                log.debug(f"📏 Resize operation {op_name} changed tensor metadata - shape: {original_shape} → {new_shape}")
            
            # Update the result tensor's metadata to match what happened remotely
            # Use set_ to update the tensor's view of its storage
            result_tensor.set_(
                result_tensor.untyped_storage(),
                new_offset,
                new_shape,
                new_stride
            )
            
            log.debug(f"✅ Updated tensor metadata to match remote execution")
        
    except Exception as e:
        log.debug(f"Skipping metadata update for {op.overloadpacket._qualified_op_name}: {e}")




# View operations that should be handled locally with shared storage IDs
VIEW_OPERATIONS = {
    "aten.view.default",
    "aten.view",
    "aten::view",
    "aten.as_strided.default",
    "aten.as_strided",
}


# Lazy import to avoid import errors if remote execution is not available
def _get_remote_orchestrator() -> Optional[Any]:
    """Get the global remote orchestrator instance.

    The remote orchestrator handles communication with remote GPU machines
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


def _handle_view_operation(op: torch._ops.OpOverload, *args: Any, **kwargs: Any) -> torch.Tensor:
    """
    Handle view operations locally with shared storage IDs.

    View operations create new tensor views that share the same remote storage,
    updating only local metadata (shape, stride, storage_offset).
    """
    op_name = op.overloadpacket._qualified_op_name
    log.info(f"🔍 Handling view operation: {op_name}")

    # Get the base tensor (first argument for most view operations)
    base_tensor = args[0]

    if base_tensor.device.type != REMOTE_DEVICE_TYPE:
        # Not a remote tensor - execute normally
        return op(*args, **kwargs)

    # Get base tensor's storage ID (this is what data_ptr() returns)
    storage_id = base_tensor.untyped_storage().data_ptr()

    # Verify storage exists on a remote device
    device_idx = driver.exec("get_storage_device", storage_id)
    if device_idx is None:
        log.warning(f"No device found for storage {storage_id}, falling back to normal execution")
        return op(*args, **kwargs)

    # Execute the view operation on a CPU meta tensor to get the new shape/stride)
    # This gives us the correct metadata without doing actual computation
    meta_tensor = torch.empty(base_tensor.size(), dtype=base_tensor.dtype, device=META_DEVICE_TYPE)
    meta_result = op(meta_tensor, *args[1:], **kwargs)

    # Extract new tensor metadata
    new_shape = meta_result.size()
    new_stride = meta_result.stride()
    new_storage_offset = meta_result.storage_offset()

    # Now that we have C++ implementations for view operations,
    # we can use PyTorch's native as_strided which will call our C++ implementation
    result = torch.as_strided(base_tensor, new_shape, new_stride, new_storage_offset)

    # Verify the view was created correctly
    assert result.size() == new_shape, f"View shape mismatch: expected {new_shape}, got {result.size()}"
    assert result.stride() == new_stride, f"View stride mismatch: expected {new_stride}, got {result.stride()}"
    assert result.storage_offset() == new_storage_offset, f"View offset mismatch: expected {new_storage_offset}, got {result.storage_offset()}"
    assert result.untyped_storage().data_ptr() == base_tensor.untyped_storage().data_ptr(), "View should share storage"

    # PyTorch automatically manages storage reference counting
    log.info(f"✅ Created view tensor sharing storage {storage_id} for {op_name}")
    return result


def _remote_kernel_fallback(op: torch._ops.OpOverload, *args: Any, **kwargs: Any) -> Any:
    log.info("Calling kernel %s", op)
    # Kernel fallback for remote operations
    
    # Skip remote execution if we're in meta execution mode
    # Meta execution should only create meta tensors, never remote tensors
    if is_in_meta_execution():
        log.debug(f"Skipping remote execution for {op.overloadpacket._qualified_op_name} - in meta execution mode")
        # Fall back to CPU execution for meta operations
        return op.redispatch(torch._C.DispatchKey.CPU, *args, **kwargs)
    
    # Get operation name
    op_name = op.overloadpacket._qualified_op_name
    # Normal logging
    log.info(f"Operation: {op_name}, Args: {len(args)}, Kwargs: {list(kwargs.keys())}")
    
    # Check and fix output tensor metadata if needed
    metadata_fixed = _check_and_fix_output_tensor_metadata(op, args, kwargs)
    

    # Handle operations using pytorch-openreg-2 logic for operation classification
    # but with remote execution for actual computation

    # First check for inplace operations (mutable)
    if op._schema.is_mutable or op is torch.ops.aten._copy_from.default:
        # Inplace operations - execute remotely and return the mutated tensor
        op_name = op.overloadpacket._qualified_op_name
        log.info(f"🔄 Inplace operation: {op_name}")

        # Determine the result tensor:
        # - If there's an "out" parameter, the result is the "out" tensor
        # - Otherwise, for true inplace ops, the result is the first argument (mutated in place)
        if "out" in kwargs:
            result_tensor = kwargs["out"]
        else:
            result_tensor = args[0]

        # Execute remotely using efficient tensor ID system
        orchestrator = _get_remote_orchestrator()
        if orchestrator is not None:
            log.info(f"🚀 Executing inplace operation {op_name} remotely (efficient)")
            orchestrator.execute_remote_aten_operation_efficient(op_name, args, kwargs)
            
            # Update tensor metadata to match what happened on remote device
            _update_tensor_metadata_from_meta_execution(op, result_tensor, args, kwargs)
            
            # Check and apply post-operation resize if needed
            result_tensor = _check_and_apply_post_operation_resize(op, result_tensor, args, kwargs)
            
            return result_tensor
        else:
            raise RuntimeError(f"Cannot execute inplace operation {op_name}: remote execution not available")

    # Handle as_strided separately from other view operations
    elif op is torch.ops.aten.as_strided.default:
        # as_strided should be handled by C++ but if it reaches here,
        # treat it as a regular operation (not a view operation)
        log.warning(f"as_strided reached Python fallback, executing remotely")
        op_name = op.overloadpacket._qualified_op_name
        log.info(f"🔧 as_strided operation: {op_name}")

        # Execute remotely using efficient storage ID system
        orchestrator = _get_remote_orchestrator()
        if orchestrator is not None:
            log.info(f"🚀 Executing as_strided operation {op_name} remotely (efficient)")
            result = orchestrator.execute_remote_aten_operation_efficient(op_name, args, kwargs)
            
            # Check and apply post-operation resize if needed
            result = _check_and_apply_post_operation_resize(op, result, args, kwargs)
            
            return result
        else:
            raise RuntimeError(f"Cannot execute operation {op_name}: remote execution not available")

    # Second check for view operations (alias_info) - excluding as_strided
    elif any(r.alias_info is not None for r in op._schema.returns):
        # View ops - handle consistently using the view handler
        result = _handle_view_operation(op, *args, **kwargs)
        
        # Check and apply post-operation resize if needed
        result = _check_and_apply_post_operation_resize(op, result, args, kwargs)
        
        return result

    # Everything else is a regular operation - execute remotely
    else:
        op_name = op.overloadpacket._qualified_op_name
        log.info(f"🔧 Regular operation: {op_name}")

        # Execute remotely using efficient tensor ID system
        orchestrator = _get_remote_orchestrator()
        if orchestrator is not None:
            log.info(f"🚀 Executing regular operation {op_name} remotely (efficient)")
            result = orchestrator.execute_remote_aten_operation_efficient(op_name, args, kwargs)
            
            # Update tensor metadata for operations that return tensors
            if isinstance(result, torch.Tensor):
                _update_tensor_metadata_from_meta_execution(op, result, args, kwargs)
            elif isinstance(result, tuple) and all(isinstance(t, torch.Tensor) for t in result):
                # Handle multiple tensor outputs
                for tensor in result:
                    _update_tensor_metadata_from_meta_execution(op, tensor, args, kwargs)
            
            # Check and apply post-operation resize if needed
            result = _check_and_apply_post_operation_resize(op, result, args, kwargs)
            
            return result
        else:
            raise RuntimeError(f"Cannot execute operation {op_name}: remote execution not available")


def copy_from_device(from_: torch.Tensor) -> torch.Tensor:
    """Copy data from remote tensor to CPU tensor using remote execution"""
    if from_.device.type != REMOTE_DEVICE_TYPE:
        raise ValueError("copy_from_device requires a remote tensor")

    # Use remote execution to get the tensor data
    orchestrator = _get_remote_orchestrator()
    if orchestrator is not None:
        from .device import get_device_registry

        # Get the device backend
        registry = get_device_registry()
        device = registry.get_device_by_index(from_.device.index)

        if device is None:
            raise RuntimeError(f"No RemoteMachine found for remote device index {from_.device.index}")

        # Get the GPU machine for this device
        gpu_machine = device.get_gpu_machine()
        if gpu_machine is None or not gpu_machine.is_running():
            raise RuntimeError(f"GPU machine not available for device {device.machine_id}")

        # Get storage data using storage ID
        storage_id = from_.untyped_storage().data_ptr()
        log.info(f"Copying storage ID {storage_id} from remote to CPU")

        # Use GPU machine to get tensor data by storage ID with view information
        # Pass tensor metadata so remote side can serialize just the view's data
        tensor_data = gpu_machine.get_storage_data(
            storage_id,
            shape=list(from_.shape),
            stride=list(from_.stride()),
            storage_offset=from_.storage_offset(),
            dtype=str(from_.dtype)
        )

        # Deserialize the tensor data as contiguous representation
        # Since we now serialize with .contiguous(), the deserialized tensor contains exactly
        # the data that should be in the result tensor - no view reconstruction needed
        result = orchestrator._deserialize_tensor(tensor_data)
        # Verify the result has the expected shape (it should match the remote tensor's shape)
        if result.size() != from_.size():
            log.warning(f"Deserialized tensor shape {result.size()} doesn't match remote tensor shape {from_.size()}")

        log.info(f"Successfully copied contiguous tensor data for storage ID {storage_id} to CPU")
        return result
    else:
        raise RuntimeError("Cannot copy from remote device: remote execution not available")


def copy_from_host_to_device(from_: torch.Tensor, to_: torch.Tensor) -> torch.Tensor:
    """Copy data from CPU tensor to remote tensor using remote execution"""
    if to_.device.type != REMOTE_DEVICE_TYPE:
        raise ValueError("copy_from_host_to_device requires a remote target tensor")
    if from_.device.type != CPU_DEVICE_TYPE:
        raise ValueError("copy_from_host_to_device requires a CPU source tensor")

    # Use remote execution to send the tensor data
    orchestrator = _get_remote_orchestrator()
    if orchestrator is not None:
        from .device import get_device_registry

        # Get the device backend
        registry = get_device_registry()
        device = registry.get_device_by_index(to_.device.index)

        if device is None:
            raise RuntimeError(f"No RemoteMachine found for remote device index {to_.device.index}")

        # Get the GPU machine for this device
        gpu_machine = device.get_gpu_machine()
        if gpu_machine is None or not gpu_machine.is_running():
            raise RuntimeError(f"GPU machine not available for device {device.machine_id}")

        # Send tensor data using storage ID
        storage_id = to_.untyped_storage().data_ptr()
        log.info(f"Copying CPU tensor to remote storage ID {storage_id}")

        # Serialize the CPU tensor
        tensor_data = orchestrator._serialize_tensor(from_)
        # Use GPU machine to create/update tensor with specific ID
        # This will overwrite any existing empty tensor with the actual data
        created_id = gpu_machine.create_storage(tensor_data, storage_id)
        log.info(f"Successfully created/updated remote tensor with ID {created_id}")
        return to_
    else:
        raise RuntimeError("Cannot copy to remote device: remote execution not available")


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

    # Preserve requires_grad property from source tensor
    should_preserve_grad = from_.requires_grad

    if from_.device.type == to_.device.type:
        if from_.device.type == REMOTE_DEVICE_TYPE:
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
    elif from_.device.type == REMOTE_DEVICE_TYPE:
        # Remote to non-remote
        host_mem = copy_from_device(from_)
        result = to_.copy_(host_mem)
    elif to_.device.type == REMOTE_DEVICE_TYPE:
        # Non-remote to remote
        result = copy_from_host_to_device(from_, to_)
    else:
        # Both non-remote but different devices
        result = to_.copy_(from_)

    # Preserve autograd properties
    if should_preserve_grad and not result.requires_grad:
        result.requires_grad_(True)

    return result


def _to_copy(input: torch.Tensor, *, dtype: Optional[torch.dtype] = None, layout: Optional[torch.layout] = None, device: Optional[Union[torch.device, str, int]] = None, pin_memory: Optional[bool] = None, non_blocking: bool = False, memory_format: Optional[torch.memory_format] = None) -> torch.Tensor:
    """Implementation of tensor.to() for remote tensors with cross-device transfer restriction."""

    # Handle device-specific transfers first
    if device is not None:
        target_device = torch.device(device) if not isinstance(device, torch.device) else device

        # Different device transfer - check if both are remote
        if input.device.type == REMOTE_DEVICE_TYPE and target_device.type == REMOTE_DEVICE_TYPE and input.device != target_device:
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
    if output_device.type == REMOTE_DEVICE_TYPE:
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
_remote_lib.fallback(_remote_kernel_fallback, dispatch_key=PRIVATEUSE1_DISPATCH_KEY)

_remote_lib_aten = torch.library.Library("aten", "IMPL")
_remote_lib_aten.impl("_copy_from", _copy_from, dispatch_key=PRIVATEUSE1_DISPATCH_KEY)
_remote_lib_aten.impl("_to_copy", _to_copy, dispatch_key=PRIVATEUSE1_DISPATCH_KEY)
_remote_lib_aten.impl(
    "set_.source_Tensor", _set_source_tensor, dispatch_key=PRIVATEUSE1_DISPATCH_KEY
)
# Note: set_.source_Storage_storage_offset is already implemented in C++ (RemoteMem.cpp)
# so we don't register a Python version to avoid conflicts
# resize_ is now implemented in C++ (RemoteMem.cpp) following OpenReg pattern

# via TORCH_LIBRARY_IMPL in RemoteMem.cpp, so we don't register Python implementations

# These factory functions are now handled by C++ implementations
# via the registered TORCH_LIBRARY_IMPL dispatch system

# when we add them to TORCH_LIBRARY_IMPL in RemoteMem.cpp


def cleanup_library_registrations() -> None:
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

