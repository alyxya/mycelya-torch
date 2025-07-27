# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Dict, List, Tuple

import torch
from torch.utils._pytree import tree_map

# Simple operation dispatch - no complex patterns needed
from ._logging import get_logger
from ._tensor_utils import RemoteTensorMetadata

log = get_logger(__name__)


def _get_remote_orchestrator():
    """Get the global remote orchestrator instance."""
    from ._remote_orchestrator import remote_orchestrator
    return remote_orchestrator


def _validate_cross_device_operation(op_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> torch.device:
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
                    f'({remote_device} and {obj.device}). '
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
) -> Tuple[Tuple[Any, ...], Dict[str, Any], List[RemoteTensorMetadata]]:
    """Convert args/kwargs, replacing remote tensors with placeholders and collecting metadata."""
    metadata_list: List[RemoteTensorMetadata] = []

    def replace_remote_tensor_with_placeholder(obj):
        """Replace remote tensors with placeholders and collect metadata."""
        if isinstance(obj, torch.Tensor):
            metadata = RemoteTensorMetadata.from_remote_tensor(obj)
            tensor_index = len(metadata_list)
            metadata_list.append(metadata)
            return f"__TENSOR_{tensor_index}"
        return obj

    # Use tree_map to handle nested structures automatically
    processed_args, processed_kwargs = tree_map(
        replace_remote_tensor_with_placeholder, (args, kwargs)
    )

    return processed_args, processed_kwargs, metadata_list


def _get_output_dtype_for_dynamic_operation(op_name: str, args: Tuple[Any, ...]) -> torch.dtype:
    """Determine the correct output dtype for dynamic output operations.

    Some operations have output dtypes that differ from their input dtypes.
    For example, nonzero always returns int64 indices regardless of input dtype.
    """
    if op_name == "aten::nonzero":
        # nonzero always returns int64 indices regardless of input dtype
        return torch.int64
    elif op_name == "aten::masked_select":
        # masked_select returns same dtype as the input tensor being selected from
        return args[0].dtype
    elif op_name == "aten::index":
        # index (boolean indexing) returns same dtype as the input tensor being indexed
        return args[0].dtype
    else:
        # Default: use input tensor dtype
        return args[0].dtype


def _execute_meta_operation(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    track_originals: bool = False
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
                    log.debug(f"Resizing empty 'out' tensor from {out_tensor.shape} to {meta_out.shape}")
                    out_tensor.resize_(meta_out.shape)

    return meta_result, original_tensors


def _create_output_tensors(
    meta_outputs: List, original_tensors: Dict, remote_device: torch.device
) -> tuple[List, List]:
    """Create output tensors based on meta execution results.

    Returns:
        tuple: (output_tensors, output_storage_ids)
            - output_tensors: List of created/reused tensors
            - output_storage_ids: List of storage IDs (int for new tensors, None for reused tensors)
    """
    output_tensors = []
    output_storage_ids = []

    for meta_output in meta_outputs:
        if meta_output in original_tensors:
            # Reuse original tensor (in-place operation)
            tensor = original_tensors[meta_output]
            output_tensors.append(tensor)
            output_storage_ids.append(None)  # No new storage created
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
            # Get storage ID from the newly created tensor
            output_storage_ids.append(new_tensor.untyped_storage().data_ptr())

    return output_tensors, output_storage_ids


def _execute_view_operation(
    op: torch._ops.OpOverload, *args: Any, **kwargs: Any
) -> torch.Tensor:
    """Handle view operations locally with shared storage IDs."""
    # Get the base tensor (first argument for most view operations)
    base_tensor = args[0]

    # Execute on meta tensors for shape inference
    meta_result, _ = _execute_meta_operation(op, args, kwargs, track_originals=False)

    # Create the result view using PyTorch's native as_strided
    return torch.as_strided(
        base_tensor,
        meta_result.shape,
        meta_result.stride(),
        meta_result.storage_offset()
    )


def _execute_with_static_outputs(
    op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any],
    remote_device: torch.device, op_name: str
) -> Any:
    """Execute operation using meta tensors for shape inference (original path)."""

    # Step 2: Execute the operation on meta tensors to determine outputs
    log.debug(f"🔧 Executing {op_name} on meta tensors for shape inference")

    try:
        meta_result, original_tensors = _execute_meta_operation(op, args, kwargs, track_originals=True)
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
        output_tensors, output_storage_ids = _create_output_tensors(
            meta_outputs, original_tensors, remote_device
        )
    else:
        output_tensors, output_storage_ids = [], []

    # Step 4: Execute remotely
    processed_args, processed_kwargs, input_metadata = (
        args_to_metadata_with_placeholders(args, kwargs)
    )

    orchestrator = _get_remote_orchestrator()
    orchestrator.execute_aten_operation(
        op_name, input_metadata, output_storage_ids, processed_args, processed_kwargs
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
    op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any],
    remote_device: torch.device, op_name: str
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
        # For out tensors, we need to get the storage ID for remote execution
        output_storage_ids = [output_tensor.untyped_storage().data_ptr()]
    else:
        # Step 1: Infer output dtype based on operation type
        output_dtype = _get_output_dtype_for_dynamic_operation(op_name, args)
        log.debug(f"Inferred output dtype: {output_dtype}")

        # Step 2: Create minimal placeholder tensor with 0 bytes
        output_tensor = torch.empty(0, dtype=output_dtype, device=remote_device)
        output_storage_ids = [output_tensor.untyped_storage().data_ptr()]

    # Step 3: Execute remotely and request metadata return
    processed_args, processed_kwargs, input_metadata = (
        args_to_metadata_with_placeholders(args, kwargs)
    )

    orchestrator = _get_remote_orchestrator()
    result_metadata = orchestrator.execute_aten_operation(
        op_name, input_metadata, output_storage_ids, processed_args, processed_kwargs,
        return_metadata=True
    )

    # Step 4: Update output tensor metadata from remote execution results
    if not result_metadata or len(result_metadata) != 1:
        raise RuntimeError(f"Expected exactly 1 output metadata for {op_name}, got {len(result_metadata) if result_metadata else 0}")

    metadata = result_metadata[0]

    # Validate dtype matches expectation
    actual_dtype = metadata["dtype"]
    if str(output_tensor.dtype) != actual_dtype:
        raise RuntimeError(f"Dtype mismatch for {op_name}: expected {output_tensor.dtype}, got {actual_dtype}")

    # Resize storage to match remote result
    output_tensor.resize_([metadata["storage_nelements"]])

    # Update all tensor metadata at once using as_strided()
    output_tensor = torch.as_strided(
        output_tensor,
        metadata["shape"],
        metadata["stride"],
        metadata["storage_offset"]
    )

    # Step 5: Return single tensor result
    return output_tensor


def _execute_aten_operation(
    op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any], remote_device: torch.device
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

    # Step 1: Check if operation requires dynamic output handling (hardcoded list)
    # These operations have data-dependent output shapes that meta tensors cannot handle
    DYNAMIC_OUTPUT_OPERATIONS = {
        "aten::masked_select",
        "aten::nonzero",
        "aten::index",  # Boolean indexing (tensor[mask]) uses nonzero internally
        # Add more operations here as needed:
        # "aten::unique",
    }

    has_meta_kernel = op_name not in DYNAMIC_OUTPUT_OPERATIONS
    log.debug(f"🔍 Operation {op_name} has meta kernel support: {has_meta_kernel}")

    if has_meta_kernel:
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
    is_view_op = (schema.returns and len(schema.returns) > 0 and
                  hasattr(schema.returns[0], 'alias_info') and
                  schema.returns[0].alias_info is not None and
                  not schema.returns[0].alias_info.is_write)

    if is_view_op:
        return _execute_view_operation(op, *args, **kwargs)
    else:
        return _execute_aten_operation(op, args, kwargs, remote_device)


def copy_from_device(from_: torch.Tensor) -> torch.Tensor:
    """Copy data from remote tensor to CPU tensor using remote execution"""
    if from_.device.type != "mycelya":
        raise ValueError("copy_from_device requires a remote tensor")

    # Use remote execution to get the tensor data
    from .device import get_device_registry

    # Get the device backend
    registry = get_device_registry()
    device = registry.get_device_by_index(from_.device.index)

    if device is None:
        raise RuntimeError(
            f"No RemoteMachine found for remote device index {from_.device.index}"
        )

    # Get storage data using orchestrator for centralized client management
    storage_id = from_.untyped_storage().data_ptr()
    log.info(f"Copying storage ID {storage_id} from remote to CPU")

    # Use orchestrator to get tensor data with automatic client routing
    from ._remote_orchestrator import remote_orchestrator

    tensor_data = remote_orchestrator.get_storage_data(
        storage_id,
        shape=list(from_.shape),
        stride=list(from_.stride()),
        storage_offset=from_.storage_offset(),
        dtype=str(from_.dtype),
    )

    # Deserialize to CPU tensor (always contiguous and packed)
    from ._tensor_utils import bytes_to_cpu_tensor
    result = bytes_to_cpu_tensor(tensor_data)

    log.info(
        f"Successfully copied contiguous tensor data for storage ID {storage_id} to CPU"
    )
    return result


def copy_from_host_to_device(from_: torch.Tensor, to_: torch.Tensor) -> torch.Tensor:
    """Copy data from CPU tensor to remote tensor using remote execution"""
    if to_.device.type != "mycelya":
        raise ValueError("copy_from_host_to_device requires a remote target tensor")
    if from_.device.type != "cpu":
        raise ValueError("copy_from_host_to_device requires a CPU source tensor")

    # Use remote execution to send the tensor data
    from .device import get_device_registry

    # Get the device backend
    registry = get_device_registry()
    device = registry.get_device_by_index(to_.device.index)

    if device is None:
        raise RuntimeError(
            f"No RemoteMachine found for remote device index {to_.device.index}"
        )

    # Send tensor data using orchestrator for centralized client management
    storage_id = to_.untyped_storage().data_ptr()
    log.info(f"Copying CPU tensor to remote storage ID {storage_id}")

    # Serialize the CPU tensor
    from ._remote_orchestrator import remote_orchestrator
    from ._tensor_utils import cpu_tensor_to_bytes

    tensor_data = cpu_tensor_to_bytes(from_)
    # Use orchestrator to update tensor with automatic client routing
    # Pass view parameters for proper handling of tensor views/slices
    remote_orchestrator.update_storage(
        storage_id,
        tensor_data,
        shape=list(to_.shape),
        stride=list(to_.stride()),
        storage_offset=to_.storage_offset(),
        dtype=str(to_.dtype)
    )
    log.info(f"Successfully created/updated remote tensor with ID {storage_id}")
    return to_


def _copy_from(from_: torch.Tensor, to_: torch.Tensor) -> torch.Tensor:
    """Copy data from one tensor to another, handling remote device transfers.

    This function implements the core copy operation for remote tensors,
    supporting CPU↔remote transfers and same-device remote copies.
    Cross-device remote transfers and non-remote device copies are blocked.

    Args:
        from_: Source tensor to copy from
        to_: Target tensor to copy to

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
            from mycelya_torch.device import get_device_registry

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

    # Get serialized tensor data for this scalar using orchestrator
    from ._remote_orchestrator import remote_orchestrator

    tensor_data = remote_orchestrator.get_storage_data(
        storage_id,
        shape=list(self.shape),
        stride=list(self.stride()),
        storage_offset=self.storage_offset(),
        dtype=str(self.dtype),
    )

    # Deserialize to CPU tensor
    from ._tensor_utils import bytes_to_cpu_tensor
    cpu_tensor = bytes_to_cpu_tensor(tensor_data)

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
