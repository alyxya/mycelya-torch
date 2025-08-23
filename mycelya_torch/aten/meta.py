# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Dict, List, Tuple

import torch

from .._logging import get_logger
from .._orchestrator import orchestrator
from .._utils import get_tensor_id
from .utils import args_to_tensors_with_ids_and_mask

log = get_logger(__name__)


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

    # Iterate over args
    meta_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            meta_arg = []
            for item in arg:
                meta_arg.append(to_meta_tensor(item))
            meta_args.append(type(arg)(meta_arg))
        else:
            meta_args.append(to_meta_tensor(arg))

    # Iterate over kwargs values
    meta_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple)):
            meta_value = []
            for item in value:
                meta_value.append(to_meta_tensor(item))
            meta_kwargs[key] = type(value)(meta_value)
        else:
            meta_kwargs[key] = to_meta_tensor(value)

    meta_args = tuple(meta_args)
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
            tensor_id = get_tensor_id(tensor)
            output_tensor_ids.append(tensor_id)

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
            tensor_id = str(get_tensor_id(new_tensor))
            output_tensor_ids.append(tensor_id)

    return output_tensors, output_tensor_ids


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
    elif isinstance(meta_result, (tuple, list)):
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
        output_tensors = []

    # Step 4: Execute remotely
    processed_args, processed_kwargs, input_tensors, tensor_mask = (
        args_to_tensors_with_ids_and_mask(args, kwargs)
    )

    orchestrator.execute_aten_operation(
        op_name,
        input_tensors,
        output_tensors,
        processed_args,
        processed_kwargs,
        tensor_mask,
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

    # Check for "out" kwarg - special handling needed
    out_tensor = kwargs.get("out", None)
    has_out_kwarg = out_tensor is not None

    if has_out_kwarg:
        log.debug(f"Operation {op_name} has 'out' kwarg, using existing tensor")
        output_tensor = out_tensor

    else:
        # Step 1: Infer output dtype based on operation type
        output_dtype = torch.int64 if op_name == "aten.nonzero" else args[0].dtype
        log.debug(f"Inferred output dtype: {output_dtype}")

        # Step 2: Create minimal placeholder tensor with 0 bytes
        output_tensor = torch.empty(0, dtype=output_dtype, device=remote_device)

    # Step 3: Execute remotely and request metadata return
    processed_args, processed_kwargs, input_tensors, tensor_mask = (
        args_to_tensors_with_ids_and_mask(args, kwargs)
    )

    result_metadata = orchestrator.execute_aten_operation(
        op_name,
        input_tensors,
        [output_tensor],
        processed_args,
        processed_kwargs,
        tensor_mask,
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
