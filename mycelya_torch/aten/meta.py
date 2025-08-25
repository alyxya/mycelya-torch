# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Dict, List, Tuple

import torch

from .._logging import get_logger
from .._orchestrator import orchestrator
from .._utils import dtype_to_str, get_tensor_id, map_args_kwargs

log = get_logger(__name__)


def _execute_meta_operation(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    track_originals: bool = False,
) -> tuple[Any, Dict]:
    """Execute operation on meta tensors for shape inference."""
    original_tensors = {}

    def to_meta_tensor(obj):
        if not isinstance(obj, torch.Tensor):
            return obj
        if track_originals:
            meta_tensor = obj.to("meta")
            original_tensors[meta_tensor] = obj
        else:
            meta_tensor = torch.empty(obj.shape, dtype=obj.dtype, device="meta")
            if obj.stride() != meta_tensor.stride():
                meta_tensor = torch.as_strided(meta_tensor, obj.shape, obj.stride(), obj.storage_offset())
        return meta_tensor

    meta_args, meta_kwargs = map_args_kwargs(to_meta_tensor, args, kwargs)
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


def _create_output_tensors(meta_outputs: List, original_tensors: Dict, remote_device: torch.device) -> tuple[List, List]:
    """Create output tensors based on meta execution results."""
    output_tensors, output_tensor_ids = [], []

    for meta_output in meta_outputs:
        if meta_output in original_tensors:
            # Reuse original tensor (in-place operation)
            tensor = original_tensors[meta_output]
            output_tensors.append(tensor)
            output_tensor_ids.append(get_tensor_id(tensor))
        else:
            # Create new tensor
            tensor = torch.empty(meta_output.shape, dtype=meta_output.dtype, device=remote_device)
            if meta_output.stride() != tensor.stride():
                tensor = torch.as_strided(tensor, meta_output.shape, meta_output.stride(), meta_output.storage_offset())
            output_tensors.append(tensor)
            output_tensor_ids.append(str(get_tensor_id(tensor)))

    return output_tensors, output_tensor_ids


def _execute_with_static_outputs(op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any], remote_device: torch.device, op_name: str) -> Any:
    """Execute operation using meta tensors for shape inference."""
    try:
        meta_result, original_tensors = _execute_meta_operation(op, args, kwargs, track_originals=True)
    except Exception as e:
        raise RuntimeError(f"Meta tensor execution failed for {op_name}: {e}")

    # Normalize meta_result to list
    meta_outputs = [meta_result] if isinstance(meta_result, torch.Tensor) else list(meta_result) if isinstance(meta_result, (tuple, list)) else []
    
    # Create output tensors and execute remotely
    output_tensors = _create_output_tensors(meta_outputs, original_tensors, remote_device)[0] if meta_outputs else []
    orchestrator.execute_aten_operation(op_name, args, kwargs, output_tensors)

    # Correct shapes and return results
    for tensor, meta in zip(output_tensors, meta_outputs):
        if tensor.shape != meta.shape:
            tensor.resize_(meta.shape)

    return tuple(output_tensors) if len(output_tensors) > 1 else output_tensors[0] if output_tensors else None


def _execute_with_dynamic_outputs(op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any], remote_device: torch.device, op_name: str) -> torch.Tensor:
    """Execute operation with dynamic output shapes."""
    # Use existing tensor or create placeholder
    output_tensor = kwargs.get("out") or torch.empty(
        0, 
        dtype=torch.int64 if op_name in ("aten.nonzero", "aten.nonzero.default") else args[0].dtype,
        device=remote_device
    )

    # Execute remotely and get metadata
    result = orchestrator.execute_aten_operation(op_name, args, kwargs, output_tensors=None)
    if not result or len(result) != 1:
        raise RuntimeError(f"Expected exactly 1 output metadata for {op_name}, got {len(result or [])}")

    metadata = result[0]
    
    # Validate dtype
    if dtype_to_str(output_tensor.dtype) != metadata["dtype"]:
        raise RuntimeError(f"Dtype mismatch for {op_name}: expected {dtype_to_str(output_tensor.dtype)}, got {metadata['dtype']}")

    # Update tensor shape and link to remote
    output_tensor.resize_([metadata["storage_nelements"]])
    output_tensor = torch.as_strided(output_tensor, metadata["shape"], metadata["stride"], metadata["storage_offset"])
    orchestrator.link_tensors([output_tensor], [metadata["temp_key"]])

    return output_tensor
