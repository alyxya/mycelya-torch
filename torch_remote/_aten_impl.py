# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

# Direct string literals (no longer using separate constants)


log = logging.getLogger(__name__)

# Thread-local storage removed - meta execution tracking simplified


def _check_and_fix_empty_output_tensors(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    """
    Simple check to resize empty output tensors based on input tensor shapes.
    Only handles the basic case where output tensor is empty and needs to match input shape.
    """
    if "out" not in kwargs:
        return
        
    output_tensor = kwargs["out"]
    if not isinstance(output_tensor, torch.Tensor) or output_tensor.device.type != "remote":
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
            # Use storage ID to check if it's the same tensor (avoid shape comparison issues)
            if arg.untyped_storage().data_ptr() != output_tensor.untyped_storage().data_ptr():
                if arg.numel() > 0:  # Use non-empty tensor as reference
                    output_tensor.resize_(arg.shape)
                    log.debug(f"Resized empty output tensor to match input shape: {arg.shape}")
                    return
    
    # Check kwargs if no suitable arg found
    for value in kwargs.values():
        if isinstance(value, torch.Tensor) and value.device.type == "remote":
            # Use storage ID to check if it's the same tensor
            if value.untyped_storage().data_ptr() != output_tensor.untyped_storage().data_ptr():
                if value.numel() > 0:  # Use non-empty tensor as reference
                    output_tensor.resize_(value.shape)
                    log.debug(f"Resized empty output tensor to match input shape: {value.shape}")
                    return


# _check_and_apply_post_operation_resize function removed - no longer needed with simplified approach

from ._device_daemon import driver
from ._meta_parser import prepare_for_sending
import contextlib
import threading


# Thread-local storage for tracking whether we should disable remote fallback
_local = threading.local()


def _is_remote_fallback_disabled() -> bool:
    """Check if remote fallback is currently disabled for this thread."""
    return getattr(_local, 'disable_remote_fallback', False)


@contextlib.contextmanager
def _disable_remote_fallback():
    """Context manager to temporarily disable remote fallback during meta execution."""
    old_value = getattr(_local, 'disable_remote_fallback', False)
    _local.disable_remote_fallback = True
    try:
        yield
    finally:
        _local.disable_remote_fallback = old_value


def _fallback_to_old_approach(op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """
    Fallback to the old implementation approach when meta execution fails.
    This handles cases where meta tensor execution isn't supported for certain operations.
    """
    op_name = op.overloadpacket._qualified_op_name
    log.warning(f"Using fallback approach for {op_name}")
    
    # Simple check for empty output tensors (legacy approach)
    _check_and_fix_empty_output_tensors(args, kwargs)
    
    # Check for inplace operations
    if op._schema.is_mutable or op is torch.ops.aten._copy_from.default:
        # Inplace operations - execute remotely and return the mutated tensor
        if "out" in kwargs:
            result_tensor = kwargs["out"]
        else:
            result_tensor = args[0]

        # Execute remotely
        orchestrator = _get_remote_orchestrator()
        if orchestrator is not None:
            orchestrator.execute_remote_aten_operation_efficient(op_name, args, kwargs)
            return result_tensor
        else:
            raise RuntimeError(f"Cannot execute inplace operation {op_name}: remote execution not available")

    # Handle as_strided separately
    elif op is torch.ops.aten.as_strided.default:
        orchestrator = _get_remote_orchestrator()
        if orchestrator is not None:
            orchestrator.execute_remote_aten_operation_efficient(op_name, args, kwargs)
            # Return the first tensor argument as the result
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
                    return arg
            raise RuntimeError(f"No output tensor found for as_strided operation {op_name}")
        else:
            raise RuntimeError(f"Cannot execute operation {op_name}: remote execution not available")

    # Everything else is a regular operation
    else:
        orchestrator = _get_remote_orchestrator()
        if orchestrator is not None:
            orchestrator.execute_remote_aten_operation_efficient(op_name, args, kwargs)
            
            # Check for 'out' parameter first
            if "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
                return kwargs["out"]
                
            # Find output tensor in args (typically the last tensor argument)
            for arg in reversed(args):
                if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
                    return arg
            
            raise RuntimeError(f"No output tensor found for operation {op_name}")
        else:
            raise RuntimeError(f"Cannot execute operation {op_name}: remote execution not available")


# _update_tensor_metadata_from_meta_execution function removed for simplification




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


def _handle_view_operation(op: torch._ops.OpOverload, *args: Any, **kwargs: Any) -> torch.Tensor:
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
            tensor_args.append(f"arg[{i}]: {arg.device.type}:{arg.device.index if hasattr(arg.device, 'index') else 'N/A'}, shape={arg.shape}")
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
                base_device_info = device_registry.get_device_by_index(base_device.index)
                arg_device_info = device_registry.get_device_by_index(arg.device.index)
                
                base_name = base_device_info.machine_id if base_device_info else f"remote:{base_device.index}"
                arg_name = arg_device_info.machine_id if arg_device_info else f"remote:{arg.device.index}"
                
                raise RuntimeError(
                    f"Cannot perform view operation \"{op_name}\" between tensors on different remote devices. "
                    f"Base tensor on \"{base_name}\" (index {base_device.index}), "
                    f"argument {i} on \"{arg_name}\" (index {arg.device.index}). "
                    f"Transfer tensors to the same device first."
                )
    
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) and value.device.type == "remote":
            if value.device != base_device:
                from torch_remote.device import get_device_registry
                device_registry = get_device_registry()
                base_device_info = device_registry.get_device_by_index(base_device.index)
                value_device_info = device_registry.get_device_by_index(value.device.index)
                
                base_name = base_device_info.machine_id if base_device_info else f"remote:{base_device.index}"
                value_name = value_device_info.machine_id if value_device_info else f"remote:{value.device.index}"
                
                raise RuntimeError(
                    f"Cannot perform view operation \"{op_name}\" between tensors on different remote devices. "
                    f"Base tensor on \"{base_name}\" (index {base_device.index}), "
                    f"kwarg '{key}' on \"{value_name}\" (index {value.device.index}). "
                    f"Transfer tensors to the same device first."
                )

    # Get base tensor's storage ID (this is what data_ptr() returns)
    storage_id = base_tensor.untyped_storage().data_ptr()

    # Verify storage exists on a remote device
    device_idx = driver.exec("get_storage_device", storage_id)
    if device_idx is None:
        log.warning(f"No device found for storage {storage_id}, falling back to normal execution")
        return op(*args, **kwargs)

    # Execute the view operation on meta tensors to get the new shape/stride
    # Convert ALL tensor arguments to meta tensors to avoid device mixing
    meta_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
            # Convert remote tensor to meta tensor
            meta_arg = torch.empty(arg.size(), dtype=arg.dtype, device="meta")
            if arg.stride() != meta_arg.stride():
                meta_arg = torch.as_strided(meta_arg, arg.shape, arg.stride(), arg.storage_offset())
            meta_args.append(meta_arg)
        else:
            meta_args.append(arg)
    
    meta_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) and value.device.type == "remote":
            # Convert remote tensor to meta tensor
            meta_value = torch.empty(value.size(), dtype=value.dtype, device="meta")
            if value.stride() != meta_value.stride():
                meta_value = torch.as_strided(meta_value, value.shape, value.stride(), value.storage_offset())
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
    assert result.size() == new_shape, f"View shape mismatch: expected {new_shape}, got {result.size()}"
    assert result.stride() == new_stride, f"View stride mismatch: expected {new_stride}, got {result.stride()}"
    assert result.storage_offset() == new_storage_offset, f"View offset mismatch: expected {new_storage_offset}, got {result.storage_offset()}"
    assert result.untyped_storage().data_ptr() == base_tensor.untyped_storage().data_ptr(), "View should share storage"

    # PyTorch automatically manages storage reference counting
    log.info(f"✅ Created view tensor sharing storage {storage_id} for {op_name}")
    return result


def _remote_kernel_fallback(op: torch._ops.OpOverload, *args: Any, **kwargs: Any) -> Any:
    import traceback
    import threading
    
    op_name = op.overloadpacket._qualified_op_name
    thread_id = threading.get_ident()
    
    # Debug: Track call depth to detect recursion
    if not hasattr(_remote_kernel_fallback, '_call_stack'):
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
        raise RuntimeError(f"Recursion detected in remote kernel fallback: {call_stack}")
    
    # Add current operation to call stack
    call_stack.append(op_name)
    
    try:
        log.info(f"📍 Processing {op_name}, Args: {len(args)}, Kwargs: {list(kwargs.keys())}")
        
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


def _remote_kernel_fallback_impl(op: torch._ops.OpOverload, *args: Any, **kwargs: Any) -> Any:
    # Check if remote fallback is disabled (during meta execution)
    if _is_remote_fallback_disabled():
        log.debug(f"Remote fallback disabled, executing {op.overloadpacket._qualified_op_name} normally")
        # Execute the operation normally without remote fallback
        return op(*args, **kwargs)
    
    # Get operation name
    op_name = op.overloadpacket._qualified_op_name
    
    # Handle view operations first (they don't need meta execution)
    # View operations are handled locally without remote execution
    # Use a whitelist approach for known view operations to avoid false positives
    KNOWN_VIEW_OPERATIONS = {
        "aten.view", "aten.view.default",
        "aten.reshape", "aten.reshape.default", 
        "aten.transpose", "aten.transpose.default",
        "aten.permute", "aten.permute.default",
        "aten.squeeze", "aten.squeeze.default", "aten.squeeze.dim", "aten.squeeze.dims",
        "aten.unsqueeze", "aten.unsqueeze.default",
        "aten.flatten", "aten.flatten.default",
        "aten.unflatten", "aten.unflatten.default",
        "aten.select", "aten.select.default",
        "aten.slice", "aten.slice.default",
        "aten.narrow", "aten.narrow.default",
        "aten.expand", "aten.expand.default",
        "aten.t", "aten.t.default",
        # Add more known view operations as needed
    }
    
    is_view_operation = (
        op_name in KNOWN_VIEW_OPERATIONS and
        op is not torch.ops.aten.as_strided.default  # as_strided handled separately
    )
    
    if is_view_operation:
        log.info(f"🔍 View operation: {op_name}")
        return _handle_view_operation(op, *args, **kwargs)
    
    # Debug: Log operations that have alias_info but aren't treated as view operations
    if any(r.alias_info is not None for r in op._schema.returns):
        log.info(f"🚨 Operation with alias_info (NOT treated as view): {op_name}")
        log.info(f"   - is_mutable: {op._schema.is_mutable}")
        log.info(f"   - contains 'loss': {'loss' in op_name.lower()}")
        log.info(f"   - contains 'reduce': {'reduce' in op_name.lower()}")
        log.info(f"   - contains 'sum': {'sum' in op_name.lower()}")
        log.info(f"   - contains 'mean': {'mean' in op_name.lower()}")
        log.info(f"   - alias_info details: {[r.alias_info for r in op._schema.returns]}")
        log.info(f"   - arg count: {len(args)}, kwarg keys: {list(kwargs.keys())}")
        
        # Log tensor argument details
        tensor_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tensor_args.append(f"arg[{i}]: {arg.device.type}:{arg.device.index if hasattr(arg.device, 'index') else 'N/A'}, shape={arg.shape}")
        if tensor_args:
            log.info(f"   - tensor args: {'; '.join(tensor_args)}")
        
        tensor_kwargs = []
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                tensor_kwargs.append(f"{key}: {value.device.type}:{value.device.index if hasattr(value.device, 'index') else 'N/A'}, shape={value.shape}")
        if tensor_kwargs:
            log.info(f"   - tensor kwargs: {'; '.join(tensor_kwargs)}")
    
    # For all other operations, use meta tensor approach for proper output handling
    
    # Step 1: Validate all remote tensors are on the same device and convert to meta tensors
    remote_device = None
    meta_args = []
    meta_kwargs = {}
    original_tensors = {}  # Maps meta tensor id to original tensor
    
    def validate_and_convert_to_meta_tensor(tensor: torch.Tensor, tensor_id: int) -> torch.Tensor:
        """Validate device consistency and convert a remote tensor to a meta tensor."""
        nonlocal remote_device
        
        # Check device consistency for remote tensors
        if tensor.device.type == "remote":
            if remote_device is None:
                remote_device = tensor.device
                log.debug(f"Using remote device: {remote_device}")
            elif tensor.device != remote_device:
                # Different remote devices - NOT ALLOWED
                from torch_remote.device import get_device_registry
                device_registry = get_device_registry()
                first_device = device_registry.get_device_by_index(remote_device.index)
                current_device = device_registry.get_device_by_index(tensor.device.index)
                
                first_device_name = first_device.machine_id if first_device else f"remote:{remote_device.index}"
                current_device_name = current_device.machine_id if current_device else f"remote:{tensor.device.index}"
                
                raise RuntimeError(
                    f"Cannot perform operations between tensors on different remote devices. "
                    f"Operation \"{op_name}\" has tensors on: "
                    f"\"{first_device_name}\" (index {remote_device.index}) and "
                    f"\"{current_device_name}\" (index {tensor.device.index}). "
                    f"Transfer tensors to the same device first: tensor.cpu().to(target_device)"
                )
        
        # Convert to meta tensor while preserving properties
        meta_tensor = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="meta",
            requires_grad=tensor.requires_grad
        )
        if tensor.stride() != meta_tensor.stride():
            meta_tensor = torch.as_strided(meta_tensor, tensor.shape, tensor.stride(), tensor.storage_offset())
        
        original_tensors[id(meta_tensor)] = tensor
        return meta_tensor
    
    # Convert args with device validation
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
            meta_args.append(validate_and_convert_to_meta_tensor(arg, i))
        else:
            meta_args.append(arg)
    
    # Convert kwargs with device validation
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) and value.device.type == "remote":
            meta_kwargs[key] = validate_and_convert_to_meta_tensor(value, len(args) + len(meta_kwargs))
        else:
            meta_kwargs[key] = value
    
    # If no remote tensors found, this shouldn't have been called - but handle gracefully
    if remote_device is None:
        log.warning(f"No remote tensors found for operation {op_name}, using fallback approach")
        return _fallback_to_old_approach(op, args, kwargs)
    
    # Step 2: Execute the operation on meta tensors to determine outputs
    log.debug(f"🔧 STEP 2: Executing {op_name} on meta tensors for shape inference")
    log.debug(f"Meta args devices: {[arg.device.type if isinstance(arg, torch.Tensor) else type(arg).__name__ for arg in meta_args]}")
    log.debug(f"Meta kwargs devices: {[value.device.type if isinstance(value, torch.Tensor) else type(value).__name__ for value in meta_kwargs.values()]}")
    
    try:
        log.debug(f"🔧 About to execute meta operation: {op_name}")
        meta_result = op(*meta_args, **meta_kwargs)
        log.debug(f"✅ Meta execution completed successfully for {op_name}")
    except Exception as e:
        log.error(f"Meta execution failed for {op_name}: {e}")
        # Fallback to the old approach if meta execution fails
        return _fallback_to_old_approach(op, args, kwargs)
    
    # Handle both single tensor and tuple results
    if isinstance(meta_result, torch.Tensor):
        meta_outputs = [meta_result]
        return_single = True
    elif isinstance(meta_result, tuple):
        meta_outputs = list(meta_result)
        return_single = False
    else:
        # Non-tensor result, execute remotely and return as-is
        log.debug(f"Non-tensor result from {op_name}, executing remotely")
        orchestrator = _get_remote_orchestrator()
        if orchestrator is not None:
            orchestrator.execute_remote_aten_operation_efficient(op_name, args, kwargs)
            return meta_result
        else:
            raise RuntimeError(f"Cannot execute operation {op_name}: remote execution not available")
    
    # Step 3: Check if any output meta tensors are the same object as input meta tensors
    input_meta_tensor_ids = set(id(tensor) for tensor in meta_args if isinstance(tensor, torch.Tensor))
    input_meta_tensor_ids.update(id(tensor) for tensor in meta_kwargs.values() if isinstance(tensor, torch.Tensor))
    
    # Step 4: Create output tensors - reuse input tensors or create new ones
    output_tensors = []
    all_tensors = []  # All tensors to pass to remote execution (inputs + outputs)
    
    log.debug(f"🔧 STEP 4: Creating output tensors for {len(meta_outputs)} outputs")
    
    # First add all input tensors
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
            all_tensors.append(arg)
    for value in kwargs.values():
        if isinstance(value, torch.Tensor) and value.device.type == "remote":
            all_tensors.append(value)
    
    # Special handling for "out" parameter - ensure it's treated as an output tensor
    out_tensor = None
    if "out" in kwargs and isinstance(kwargs["out"], torch.Tensor) and kwargs["out"].device.type == "remote":
        out_tensor = kwargs["out"]
        log.debug(f"Found 'out' parameter tensor with shape {out_tensor.shape}")
    
    # Process each output tensor
    for i, meta_output in enumerate(meta_outputs):
        log.debug(f"🔧 Processing output tensor {i}: meta_output.shape={meta_output.shape}")
        
        if id(meta_output) in input_meta_tensor_ids:
            # Output shares storage with input - reuse the original input tensor
            original_tensor = original_tensors[id(meta_output)]
            output_tensors.append(original_tensor)
            log.debug(f"✅ Output tensor {i} reuses input tensor storage (in-place operation)")
        else:
            # Check if this meta output corresponds to the "out" parameter
            if out_tensor is not None and len(output_tensors) == i:
                # Use the existing "out" tensor as the output tensor
                output_tensors.append(out_tensor)
                log.debug(f"✅ Using existing 'out' tensor as output {i}")
            else:
                log.debug(f"🔧 Creating new output tensor {i} with shape {meta_output.shape}")
                
                # Output is new - create empty remote tensor with validated remote device
                # Use the remote_device that was already validated to be consistent across all tensors
                
                # Create empty remote tensor (this will invoke the allocator and create storage ID)
                log.debug(f"🔧 About to call torch.empty for output {i}")
                new_tensor = torch.empty(
                    meta_output.shape,
                    dtype=meta_output.dtype,
                    device=remote_device,
                    requires_grad=meta_output.requires_grad
                )
                log.debug(f"✅ torch.empty completed for output {i}")
                
                # Apply stride if different from default
                if meta_output.stride() != new_tensor.stride():
                    log.debug(f"🔧 About to call torch.as_strided for output {i}: {meta_output.shape}, {meta_output.stride()}, {meta_output.storage_offset()}")
                    new_tensor = torch.as_strided(new_tensor, meta_output.shape, meta_output.stride(), meta_output.storage_offset())
                    log.debug(f"✅ torch.as_strided completed for output {i}")
                
                output_tensors.append(new_tensor)
                all_tensors.append(new_tensor)
                log.debug(f"✅ Created new output tensor {i} with shape {meta_output.shape}")
    
    # Step 5: Handle pre-operation resizing for empty "out" tensors
    if out_tensor is not None and out_tensor.numel() == 0:
        # Find the corresponding meta output for the "out" tensor
        out_idx = -1
        for i, output_tensor in enumerate(output_tensors):
            if output_tensor is out_tensor:
                out_idx = i
                break
        
        if out_idx >= 0 and out_idx < len(meta_outputs):
            meta_out = meta_outputs[out_idx]
            if meta_out.numel() > 0:
                log.debug(f"Pre-operation: resizing empty 'out' tensor from {out_tensor.shape} to {meta_out.shape}")
                out_tensor.resize_(meta_out.shape)

    # Step 6: Execute remotely with explicit input/output separation
    orchestrator = _get_remote_orchestrator()
    if orchestrator is not None:
        log.debug(f"🔧 STEP 6: Executing {op_name} remotely with meta-based output handling")
        
        # Separate input and output tensors
        input_tensors = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
                input_tensors.append(arg)
        for value in kwargs.values():
            if isinstance(value, torch.Tensor) and value.device.type == "remote":
                input_tensors.append(value)
        
        log.debug(f"🔧 About to call orchestrator with {len(input_tensors)} input tensors, {len(output_tensors)} output tensors")
        
        # Use the new interface with explicit input/output separation
        orchestrator.execute_remote_aten_operation_with_outputs(
            op_name, input_tensors, output_tensors, args, kwargs
        )
        
        log.debug(f"✅ Remote orchestrator execution completed for {op_name}")
        
        # Step 7: Handle post-operation resizing for output tensors that don't match meta results
        for i, (output_tensor, meta_output) in enumerate(zip(output_tensors, meta_outputs)):
            # Check if tensor metadata differs from meta tensor metadata
            shape_differs = output_tensor.shape != meta_output.shape
            stride_differs = output_tensor.stride() != meta_output.stride()
            offset_differs = output_tensor.storage_offset() != meta_output.storage_offset()
            
            if shape_differs or stride_differs or offset_differs:
                log.debug(f"Post-operation: output tensor {i} metadata differs from meta result")
                log.debug(f"  Tensor: shape={output_tensor.shape}, stride={output_tensor.stride()}, offset={output_tensor.storage_offset()}")
                log.debug(f"  Meta:   shape={meta_output.shape}, stride={meta_output.stride()}, offset={meta_output.storage_offset()}")
                
                # Resize/reshape the tensor to match the meta result
                if shape_differs:
                    # Use resize_ for shape changes
                    output_tensor.resize_(meta_output.shape)
                    log.debug(f"  Resized tensor to shape {meta_output.shape}")
                
                # Apply stride and offset if they still differ after resize
                if output_tensor.stride() != meta_output.stride() or output_tensor.storage_offset() != meta_output.storage_offset():
                    # Use as_strided to apply the correct stride and offset
                    output_tensors[i] = torch.as_strided(
                        output_tensor, 
                        meta_output.shape, 
                        meta_output.stride(), 
                        meta_output.storage_offset()
                    )
                    log.debug(f"  Applied stride {meta_output.stride()} and offset {meta_output.storage_offset()}")
        
        # Step 8: Return the analogous remote tensors
        if return_single:
            return output_tensors[0]
        else:
            return tuple(output_tensors)
    else:
        raise RuntimeError(f"Cannot execute operation {op_name}: remote execution not available")


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
            raise RuntimeError(f"No RemoteMachine found for remote device index {from_.device.index}")

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
            raise RuntimeError(f"No RemoteMachine found for remote device index {to_.device.index}")

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


    return result


def _remote_item_impl(self: torch.Tensor):
    """Custom implementation of item() for remote tensors.
    
    This function efficiently retrieves only the scalar value from the remote device
    by getting raw bytes and directly converting to the appropriate Python type,
    avoiding the overhead of creating a full CPU tensor.
    
    Args:
        self: The remote tensor (must be a scalar)
        
    Returns:
        Python scalar value
    """
    if self.device.type != "remote":
        # Fallback to default implementation for non-remote tensors
        return torch.ops.aten.item.default(self)
    
    # Check that tensor is scalar (replicate PyTorch's exact behavior)
    if self.numel() != 1:
        raise RuntimeError(f"a Tensor with {self.numel()} elements cannot be converted to Scalar")
    
    # Get scalar value from remote device (optimized for single element)
    log.info("🔢 Item operation: retrieving scalar value from remote device")
    
    # Get storage ID and remote machine
    storage_id = self.untyped_storage().data_ptr()
    
    # Get the remote machine using device registry (same pattern as copy_from_device)
    from .device import get_device_registry
    registry = get_device_registry()
    machine = registry.get_device_by_index(self.device.index)
    
    if machine is None:
        raise RuntimeError(f"No RemoteMachine found for remote device index {self.device.index}")
    
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
        dtype=str(self.dtype)
    )
    
    # Deserialize to CPU tensor and extract scalar value
    orchestrator = _get_remote_orchestrator()
    if orchestrator is None:
        raise RuntimeError("Cannot retrieve scalar value: remote execution not available")
    
    cpu_tensor = orchestrator._deserialize_tensor(tensor_data)
    
    # Call item() on the CPU tensor to get the Python scalar
    return cpu_tensor.item()


def _to_copy(input: torch.Tensor, *, dtype: Optional[torch.dtype] = None, layout: Optional[torch.layout] = None, device: Optional[Union[torch.device, str, int]] = None, pin_memory: Optional[bool] = None, non_blocking: bool = False, memory_format: Optional[torch.memory_format] = None) -> torch.Tensor:
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
_remote_lib_aten.impl("item", _remote_item_impl, dispatch_key="PrivateUse1")

# AUTOGRAD_"PrivateUse1" registrations removed for copy operations:
# _copy_from and _to_copy were causing autograd issues because:
# 1. _copy_from breaks autograd chain via copy_from_device() -> _deserialize_tensor() -> .detach()
# 2. In-place operations on leaf variables with requires_grad=True are not allowed
# 3. No proper grad_fn creation for autograd graph connectivity
# 4. This remote tensor system doesn't need device-specific autograd behavior for data movement
# 5. PyTorch's default autograd handles remote tensors correctly when only "PrivateUse1" is registered

# However, item() is safe to register to AutogradPrivateUse1 because:
# - It's a terminal operation that extracts scalar values
# - It doesn't involve data movement or tensor creation
# - It has well-defined behavior for tensors with requires_grad=True
_remote_lib_aten_autograd = torch.library.Library("aten", "IMPL")
_remote_lib_aten_autograd.impl("item", _remote_item_impl, dispatch_key="AutogradPrivateUse1")

# Note: set_.source_Storage_storage_offset is already implemented in C++ (RemoteMem.cpp)
# so we don't register a Python version to avoid conflicts
# resize_ is now implemented in C++ (RemoteMem.cpp) following OpenReg pattern

# via TORCH_LIBRARY_IMPL in RemoteMem.cpp, so we don't register Python implementations

# These factory functions are now handled by C++ implementations
# via the registered TORCH_LIBRARY_IMPL dispatch system

# when we add them to TORCH_LIBRARY_IMPL in RemoteMem.cpp



