# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Dict, List, Tuple

import torch
from torch.utils._pytree import tree_map

# Simple operation dispatch - no complex patterns needed
from ._logging import get_logger
from ._tensor_utils import MycelyaTensorMetadata

log = get_logger(__name__)


def _get_remote_orchestrator():
    """Get the global remote orchestrator instance."""
    from ._remote_orchestrator import remote_orchestrator

    return remote_orchestrator


def _validate_cross_device_operation(
    op_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> torch.device:
    """Validate that all tensors are remote and on the same device. Returns the remote device."""
    remote_device = None

    def check_tensor_device(obj):
        nonlocal remote_device
        if isinstance(obj, torch.Tensor):
            if obj.device.type != "mycelya":
                raise RuntimeError(
                    f'Remote kernel fallback called for operation "{op_name}" with non-remote tensor '
                    f'on device "{obj.device}".'
                )

            if remote_device is None:
                remote_device = obj.device
            elif remote_device != obj.device:
                raise RuntimeError(
                    f'Cannot perform operation "{op_name}" between tensors on different remote devices '
                    f"({remote_device} and {obj.device}). "
                    f"Transfer tensors to the same device first."
                )
        return obj

    tree_map(check_tensor_device, (args, kwargs))

    if remote_device is None:
        raise RuntimeError(f'No remote tensors found for operation "{op_name}"')

    return remote_device


def args_to_metadata_with_placeholders(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[Tuple[Any, ...], Dict[str, Any], List[MycelyaTensorMetadata]]:
    """Convert args/kwargs, replacing remote tensors with placeholders and collecting metadata."""
    metadata_list: List[MycelyaTensorMetadata] = []

    def replace_remote_tensor_with_placeholder(obj):
        """Replace remote tensors with placeholders and collect metadata."""
        if isinstance(obj, torch.Tensor):
            metadata = MycelyaTensorMetadata.from_mycelya_tensor(obj)
            tensor_index = len(metadata_list)
            metadata_list.append(metadata)
            return f"__TENSOR_{tensor_index}"
        return obj

    # Use tree_map to handle nested structures automatically
    processed_args, processed_kwargs = tree_map(
        replace_remote_tensor_with_placeholder, (args, kwargs)
    )

    return processed_args, processed_kwargs, metadata_list


def _execute_meta_operation(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    track_originals: bool = False,
) -> tuple[Any, Dict]:
    """Execute operation on meta tensors for shape inference.

    Args:
        op: Operation to execute
        args: Operation arguments
        kwargs: Operation keyword arguments
        track_originals: If True, return mapping of meta tensors to original tensors

    Returns:
        tuple: (meta_result, original_tensors_map)
    """
    original_tensors = {}

    def to_meta_tensor(obj):
        if isinstance(obj, torch.Tensor):
            if track_originals:
                # Use .to("meta") for tracking originals (remote operations)
                meta_tensor = obj.to("meta")
                original_tensors[meta_tensor] = obj
            else:
                # Use torch.empty + as_strided for view operations
                meta_tensor = torch.empty(obj.shape, dtype=obj.dtype, device="meta")
                if obj.stride() != meta_tensor.stride():
                    meta_tensor = torch.as_strided(
                        meta_tensor, obj.shape, obj.stride(), obj.storage_offset()
                    )
            return meta_tensor
        return obj

    meta_args, meta_kwargs = tree_map(to_meta_tensor, (args, kwargs))
    meta_result = op(*meta_args, **meta_kwargs)

    # Special handling for "out" parameter: resize empty output tensors to match meta result
    if "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
        out_tensor = kwargs["out"]
        if out_tensor.numel() == 0:
            # Find the corresponding meta output tensor to get the expected shape
            if "out" in meta_kwargs and isinstance(meta_kwargs["out"], torch.Tensor):
                meta_out = meta_kwargs["out"]
                if meta_out.shape != out_tensor.shape:
                    log.debug(
                        f"Resizing empty 'out' tensor from {out_tensor.shape} to {meta_out.shape}"
                    )
                    out_tensor.resize_(meta_out.shape)

    return meta_result, original_tensors


def _create_output_tensors(
    meta_outputs: List, original_tensors: Dict, remote_device: torch.device
) -> tuple[List, List]:
    """Create output tensors based on meta execution results.

    Returns:
        tuple: (output_tensors, output_tensor_ids)
            - output_tensors: List of created/reused tensors
            - output_tensor_ids: List of tensor IDs for all output tensors (both new and reused)
    """
    output_tensors = []
    output_tensor_ids = []

    for meta_output in meta_outputs:
        if meta_output in original_tensors:
            # Reuse original tensor (in-place operation)
            tensor = original_tensors[meta_output]
            output_tensors.append(tensor)
            # Include tensor ID for reused tensors to track all modifications
            tensor_id = tensor.get_metadata_hash()
            output_tensor_ids.append(tensor_id)
            
            # Ensure tensor ID is registered with device tracking system
            from ._storage import register_tensor_id
            device_index = remote_device.index
            register_tensor_id(tensor_id, device_index)
        else:
            # Create new tensor
            new_tensor = torch.empty(
                meta_output.shape, dtype=meta_output.dtype, device=remote_device
            )

            # Apply stride if different from default
            if meta_output.stride() != new_tensor.stride():
                new_tensor = torch.as_strided(
                    new_tensor,
                    meta_output.shape,
                    meta_output.stride(),
                    meta_output.storage_offset(),
                )

            output_tensors.append(new_tensor)
            # Get tensor ID from the newly created tensor
            tensor_id = str(new_tensor.get_metadata_hash())
            output_tensor_ids.append(tensor_id)
            
            # Register tensor ID with device tracking system
            from ._storage import register_tensor_id
            device_index = remote_device.index
            register_tensor_id(tensor_id, device_index)

    return output_tensors, output_tensor_ids


def _execute_view_operation(
    op: torch._ops.OpOverload, *args: Any, **kwargs: Any
) -> torch.Tensor:
    """Handle view operations as local-only operations that share storage."""
    op_name = op.overloadpacket._qualified_op_name

    # Get the base tensor (first argument for most view operations)
    base_tensor = args[0]

    # Execute on meta tensors for shape inference to create the view locally
    meta_result, _ = _execute_meta_operation(op, args, kwargs, track_originals=False)

    # Create the local view tensor using PyTorch's native as_strided (shares storage)
    view_tensor = torch.as_strided(
        base_tensor,
        meta_result.shape,
        meta_result.stride(),
        meta_result.storage_offset(),
    )

    log.debug(f"View operation {op_name} executed locally only (no remote propagation)")
    return view_tensor


def _execute_with_static_outputs(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    remote_device: torch.device,
    op_name: str,
) -> Any:
    """Execute operation using meta tensors for shape inference (original path)."""

    # Step 2: Execute the operation on meta tensors to determine outputs
    log.debug(f"🔧 Executing {op_name} on meta tensors for shape inference")

    try:
        meta_result, original_tensors = _execute_meta_operation(
            op, args, kwargs, track_originals=True
        )
        log.debug(f"✅ Meta execution completed successfully for {op_name}")
    except Exception as e:
        log.error(f"Meta execution failed for {op_name}: {e}")
        raise RuntimeError(
            f"Meta tensor execution failed for {op_name}: {e}. "
            "This operation cannot be executed remotely without meta tensor support."
        )

    # Handle both single tensor and tuple results
    if isinstance(meta_result, torch.Tensor):
        meta_outputs = [meta_result]
    elif isinstance(meta_result, tuple):
        meta_outputs = list(meta_result)
    else:
        # Non-tensor result, no output tensors to create
        meta_outputs = []

    # Step 3: Create output tensors (empty list for non-tensor results)
    if meta_outputs:
        output_tensors, output_tensor_ids = _create_output_tensors(
            meta_outputs, original_tensors, remote_device
        )
    else:
        output_tensors, output_tensor_ids = [], []

    # Step 4: Execute remotely
    processed_args, processed_kwargs, input_metadata = (
        args_to_metadata_with_placeholders(args, kwargs)
    )

    orchestrator = _get_remote_orchestrator()
    orchestrator.execute_aten_operation(
        op_name,
        input_metadata,
        output_tensor_ids,
        processed_args,
        processed_kwargs,
    )

    # Step 5: Correct output tensor shapes to match meta tensor shapes
    for output_tensor, meta_output in zip(output_tensors, meta_outputs):
        if output_tensor.shape != meta_output.shape:
            # Resize tensor to match meta result shape
            output_tensor.resize_(meta_output.shape)

    # Step 6: Return results
    if len(output_tensors) > 1:
        return tuple(output_tensors)
    elif len(output_tensors) == 1:
        return output_tensors[0]


def _execute_with_dynamic_outputs(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    remote_device: torch.device,
    op_name: str,
) -> torch.Tensor:
    """Execute operation with dynamic output shapes - no meta tensor support.

    Assumes single tensor output for simplicity. Multi-output dynamic operations
    can be added later as needed.
    """

    log.info(f"🔄 Executing {op_name} with dynamic output shapes (no meta kernel)")

    # Check for "out" kwarg - special handling needed
    out_tensor = kwargs.get("out", None)
    has_out_kwarg = out_tensor is not None

    if has_out_kwarg:
        log.debug(f"Operation {op_name} has 'out' kwarg, using existing tensor")
        output_tensor = out_tensor
        # For out tensors, we need to get the tensor ID for remote execution
        output_tensor_ids = [output_tensor.get_metadata_hash()]

        # Ensure tensor ID is registered with device tracking system
        from ._storage import register_tensor_id
        device_index = remote_device.index
        register_tensor_id(output_tensor_ids[0], device_index)
    else:
        # Step 1: Infer output dtype based on operation type
        output_dtype = torch.int64 if op_name == "aten::nonzero" else args[0].dtype
        log.debug(f"Inferred output dtype: {output_dtype}")

        # Step 2: Create minimal placeholder tensor with 0 bytes
        output_tensor = torch.empty(0, dtype=output_dtype, device=remote_device)
        output_tensor_ids = [output_tensor.get_metadata_hash()]

        # Register tensor ID with device tracking system
        from ._storage import register_tensor_id
        device_index = remote_device.index
        register_tensor_id(output_tensor_ids[0], device_index)

    # Step 3: Execute remotely and request metadata return
    processed_args, processed_kwargs, input_metadata = (
        args_to_metadata_with_placeholders(args, kwargs)
    )

    orchestrator = _get_remote_orchestrator()
    result_metadata = orchestrator.execute_aten_operation(
        op_name,
        input_metadata,
        output_tensor_ids,
        processed_args,
        processed_kwargs,
        return_metadata=True,
    )

    # Step 4: Update output tensor metadata from remote execution results
    if not result_metadata or len(result_metadata) != 1:
        raise RuntimeError(
            f"Expected exactly 1 output metadata for {op_name}, got {len(result_metadata) if result_metadata else 0}"
        )

    metadata = result_metadata[0]

    # Validate dtype matches expectation
    actual_dtype = metadata["dtype"]
    if str(output_tensor.dtype) != actual_dtype:
        raise RuntimeError(
            f"Dtype mismatch for {op_name}: expected {output_tensor.dtype}, got {actual_dtype}"
        )

    # Resize storage to match remote result
    output_tensor.resize_([metadata["storage_nelements"]])

    # Update all tensor metadata at once using as_strided()
    output_tensor = torch.as_strided(
        output_tensor, metadata["shape"], metadata["stride"], metadata["storage_offset"]
    )

    # Step 5: Return single tensor result
    return output_tensor


def _has_static_output_shape(
    op_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> bool:
    """Determine if operation has predictable output shape for meta tensor inference."""

    # Always dynamic operations (output shape depends on data)
    ALWAYS_DYNAMIC = {
        "aten::masked_select",
        "aten::nonzero",
        "aten::unique",
        "aten::_unique2",
    }
    if op_name in ALWAYS_DYNAMIC:
        return False

    # Special case: aten::index can be static (tensor indexing) or dynamic (boolean indexing)
    if op_name == "aten::index":
        if len(args) >= 2 and isinstance(args[1], (tuple, list)):
            # Boolean indexing is dynamic, tensor indexing is static
            return not any(
                isinstance(idx, torch.Tensor) and idx.dtype == torch.bool
                for idx in args[1]
                if idx is not None
            )

    # TODO: Add more conditional operations here as needed:
    # if op_name == "aten::where":
    #     return len(args) != 1  # 1-arg form is dynamic, 3-arg form is static

    return True


def _execute_aten_operation(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    remote_device: torch.device,
) -> Any:
    """Execute operation on remote device - simplified from complex strategy pattern."""

    op_name = op.overloadpacket._qualified_op_name

    # Check for unsupported operations before meta execution
    if op_name == "aten::repeat_interleave":
        raise RuntimeError(
            "repeat_interleave is not supported on remote devices due to a PyTorch bug where tensor repeats "
            "are incorrectly dispatched to single-argument overload. "
            "Use tensor.repeat() or other alternatives instead."
        )

    # Step 1: Check if operation requires dynamic output handling
    has_static_output = _has_static_output_shape(op_name, args, kwargs)
    log.debug(f"🔍 Operation {op_name} has static output shape: {has_static_output}")

    if has_static_output:
        # Standard path: Use meta tensors for shape inference
        return _execute_with_static_outputs(op, args, kwargs, remote_device, op_name)
    else:
        # Dynamic path: Create placeholder outputs and get metadata from remote execution
        return _execute_with_dynamic_outputs(op, args, kwargs, remote_device, op_name)


def _remote_kernel_fallback(
    op: torch._ops.OpOverload, *args: Any, **kwargs: Any
) -> Any:
    """Execute PyTorch operations on remote devices using simple dispatch logic."""
    op_name = op.overloadpacket._qualified_op_name

    # Validate cross-device operations upfront and get the remote device
    remote_device = _validate_cross_device_operation(op_name, args, kwargs)

    # Check if operation is a view operation using schema alias information
    # View operations alias their input for reading (is_write=False)
    # In-place/out operations also have alias_info but with is_write=True
    schema = op._schema
    is_view_op = (
        schema.returns
        and len(schema.returns) > 0
        and hasattr(schema.returns[0], "alias_info")
        and schema.returns[0].alias_info is not None
        and not schema.returns[0].alias_info.is_write
    )

    if is_view_op:
        return _execute_view_operation(op, *args, **kwargs)
    else:
        return _execute_aten_operation(op, args, kwargs, remote_device)


def copy_from_device(from_: torch.Tensor) -> torch.Tensor:
    """Copy data from remote tensor to CPU tensor using tensor-based execution"""
    if from_.device.type != "mycelya":
        raise ValueError("copy_from_device requires a remote tensor")

    # Use remote execution to get the tensor data
    from ._device import get_device_registry

    # Get the device backend
    registry = get_device_registry()
    device = registry.get_device_by_index(from_.device.index)

    if device is None:
        raise RuntimeError(
            f"No RemoteMachine found for remote device index {from_.device.index}"
        )

    # Get tensor data using orchestrator for centralized client management
    tensor_id = from_.get_metadata_hash()
    log.info(f"Copying tensor ID {tensor_id} from remote to CPU")

    # Use orchestrator to get tensor data with automatic client routing
    from ._remote_orchestrator import remote_orchestrator

    result = remote_orchestrator.get_tensor_by_id(
        tensor_id,
        shape=list(from_.shape),
        stride=list(from_.stride()),
        storage_offset=from_.storage_offset(),
        dtype=str(from_.dtype),
    )

    log.info(
        f"Successfully copied contiguous tensor data for tensor ID {tensor_id} to CPU"
    )
    return result


def copy_from_host_to_device(from_: torch.Tensor, to_: torch.Tensor) -> torch.Tensor:
    """Copy data from CPU tensor to remote tensor using tensor-based execution"""
    if to_.device.type != "mycelya":
        raise ValueError("copy_from_host_to_device requires a remote target tensor")
    if from_.device.type != "cpu":
        raise ValueError("copy_from_host_to_device requires a CPU source tensor")

    # Use remote execution to send the tensor data
    from ._device import get_device_registry

    # Get the device backend
    registry = get_device_registry()
    device = registry.get_device_by_index(to_.device.index)

    if device is None:
        raise RuntimeError(
            f"No RemoteMachine found for remote device index {to_.device.index}"
        )

    # Use tensor-based approach with tensor IDs
    tensor_id = to_.get_metadata_hash()
    log.info(f"Copying CPU tensor to remote tensor ID {tensor_id}")

    # Pass CPU tensor directly to orchestrator without conversion
    from ._remote_orchestrator import remote_orchestrator

    # Use orchestrator to update tensor with automatic client routing
    # Pass tensor ID and raw data with tensor metadata for reconstruction
    remote_orchestrator.update_tensor(
        tensor_id,
        from_,  # Pass storage tensor directly
        source_shape=list(from_.shape),
        source_stride=list(from_.stride()),
        source_storage_offset=from_.storage_offset(),
        source_dtype=str(from_.dtype),
    )

    log.info(f"Successfully updated remote tensor with tensor ID {tensor_id}")
    return to_


def _copy_from(
    from_: torch.Tensor, to_: torch.Tensor, non_blocking: bool = False
) -> torch.Tensor:
    """Copy data from one tensor to another, handling remote device transfers.

    This function implements the core copy operation for remote tensors,
    supporting CPU↔remote transfers and same-device remote copies.
    Cross-device remote transfers and non-remote device copies are blocked.

    Args:
        from_: Source tensor to copy from
        to_: Target tensor to copy to
        non_blocking: Whether to perform the copy asynchronously (currently ignored)

    Returns:
        Target tensor with copied data

    Raises:
        RuntimeError: If attempting unsupported copy operations
    """
    # Only support CPU ↔ remote transfers

    if from_.device.type == "mycelya" and to_.device.type == "cpu":
        # Remote to CPU - supported
        host_mem = copy_from_device(from_)
        result = to_.copy_(host_mem)
    elif from_.device.type == "cpu" and to_.device.type == "mycelya":
        # CPU to remote - supported
        result = copy_from_host_to_device(from_, to_)
    elif from_.device.type == "mycelya" and to_.device.type == "mycelya":
        # Remote to remote transfers
        if from_.device.index == to_.device.index:
            # Same remote device - allowed (needed for gradients and internal operations)
            op = torch.ops.aten.copy_.default
            result = _remote_kernel_fallback(op, to_, from_)
        else:
            # Different remote devices - blocked (TODO: support in future)
            from mycelya_torch._device import get_device_registry

            device_registry = get_device_registry()
            from_device = device_registry.get_device_by_index(from_.device.index)
            to_device = device_registry.get_device_by_index(to_.device.index)

            raise RuntimeError(
                f"Cross-device remote transfers are not supported. "
                f'Source device: "{from_device.machine_id}" (index {from_.device.index}), '
                f'Target device: "{to_device.machine_id}" (index {to_.device.index}). '
                f"Only CPU↔remote and same-device transfers are allowed. Use CPU as intermediate."
            )
    else:
        # All other cases (non-remote device copies) - blocked
        raise RuntimeError(
            f"Copy operation from {from_.device.type} to {to_.device.type} is not supported. "
            f"Only CPU↔remote transfers are allowed."
        )

    return result


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
        ten2.shape,
        ten2.stride(),
    )


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

    # Get remote machine using device registry
    from ._device import get_device_registry

    registry = get_device_registry()
    machine = registry.get_device_by_index(self.device.index)

    if machine is None:
        raise RuntimeError(
            f"No RemoteMachine found for remote device index {self.device.index}"
        )

    # Get tensor ID and tensor data using tensor-based approach
    tensor_id = self.get_metadata_hash()

    # Get tensor data for this scalar using orchestrator
    from ._remote_orchestrator import remote_orchestrator

    cpu_tensor = remote_orchestrator.get_tensor_by_id(
        tensor_id,
        shape=list(self.shape),
        stride=list(self.stride()),
        storage_offset=self.storage_offset(),
        dtype=str(self.dtype),
    )

    # Call item() on the CPU tensor to get the Python scalar
    return cpu_tensor.item()


def _equal(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Custom implementation of torch.equal for remote tensors."""
    log.info("⚖️ torch.equal operation: comparing tensors on remote device")

    # Both tensors should be remote (validated by caller)
    # Check basic compatibility first
    if self.shape != other.shape:
        return False
    if self.dtype != other.dtype:
        return False

    # For torch.equal, we'll copy both tensors to CPU and compare locally
    # This is simpler than modifying the entire remote execution pipeline
    log.info("📥 Copying tensors to CPU for comparison")

    cpu_self = copy_from_device(self)
    cpu_other = copy_from_device(other)

    # Use PyTorch's native equal on CPU tensors
    result = torch.equal(cpu_self, cpu_other)

    log.info(f"✅ torch.equal completed: {result}")
    return result


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
_remote_lib_aten.impl("equal", _equal, dispatch_key="PrivateUse1")
