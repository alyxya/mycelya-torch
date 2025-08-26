# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Dict, List, Tuple

import torch

from .._logging import get_logger
from .._orchestrator import orchestrator
from .._utils import map_args_kwargs

log = get_logger(__name__)


def _execute_meta_operation(
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> tuple[Any, Dict]:
    """Execute operation on meta tensors for shape inference."""
    original_tensors = {}

    def to_meta_tensor(obj):
        if not isinstance(obj, torch.Tensor):
            return obj
        meta_tensor = obj.to("meta")
        original_tensors[meta_tensor] = obj
        return meta_tensor

    meta_args, meta_kwargs = map_args_kwargs(to_meta_tensor, args, kwargs)
    meta_result = op(*meta_args, **meta_kwargs)

    return meta_result, original_tensors


def _create_output_tensors(meta_outputs: List, original_tensors: Dict, remote_device: torch.device) -> List[torch.Tensor]:
    """Create output tensors based on meta execution results."""
    output_tensors = []

    for meta_output in meta_outputs:
        if meta_output in original_tensors:
            # Reuse original tensor (in-place operation) and correct shape if needed
            tensor = original_tensors[meta_output]
            if tensor.shape != meta_output.shape:
                tensor.resize_(meta_output.shape)
            output_tensors.append(tensor)
        else:
            # Create new tensor
            tensor = torch.empty(meta_output.shape, dtype=meta_output.dtype, device=remote_device)
            output_tensors.append(tensor)

    return output_tensors


def _execute_with_static_outputs(op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any], remote_device: torch.device, op_name: str, meta_result: Any, original_tensors: Dict) -> Any:
    """Execute operation using meta tensors for shape inference."""
    # Normalize meta_result to list
    meta_outputs = [meta_result] if isinstance(meta_result, torch.Tensor) else list(meta_result) if isinstance(meta_result, (tuple, list)) else []
    
    # Create output tensors 
    output_tensors = _create_output_tensors(meta_outputs, original_tensors, remote_device) if meta_outputs else []
    
    orchestrator.execute_aten_operation(op_name, args, kwargs, output_tensors)

    return tuple(output_tensors) if len(output_tensors) > 1 else output_tensors[0] if output_tensors else None


def _execute_with_dynamic_outputs(op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any], remote_device: torch.device, op_name: str) -> Any:
    """Execute operation with dynamic output shapes."""
    # Execute remotely and get metadata
    result = orchestrator.execute_aten_operation(op_name, args, kwargs, output_tensors=None)
    if not result:
        raise RuntimeError(f"No output metadata returned for {op_name}")

    # Create output tensors from metadata
    output_tensors = []
    temp_keys = []
    
    for metadata in result:
        # Parse dtype from string
        dtype_str = metadata["dtype"]
        if not hasattr(torch, dtype_str):
            raise RuntimeError(f"Unknown dtype {dtype_str} for {op_name}")
        dtype = getattr(torch, dtype_str)
        
        # Create tensor with exact shape from remote metadata
        storage_nelements = metadata["nbytes"] // dtype.itemsize
        output_tensor = torch.empty([storage_nelements], dtype=dtype, device=remote_device)
        output_tensor = torch.as_strided(output_tensor, metadata["shape"], metadata["stride"], metadata["storage_offset"])
        
        output_tensors.append(output_tensor)
        temp_keys.append(metadata["temp_key"])
    
    # Link all tensors to remote data
    orchestrator.link_tensors(output_tensors, temp_keys)
    
    # Return single tensor or tuple based on result count
    return tuple(output_tensors) if len(output_tensors) > 1 else output_tensors[0] if output_tensors else None


def _remote_kernel_fallback(
    op: torch._ops.OpOverload, *args: Any, **kwargs: Any
) -> Any:
    """Execute PyTorch operations on remote devices using simple dispatch logic."""
    
    # Get remote device from first tensor (orchestrator will handle validation)
    remote_device = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            remote_device = arg.device
            break
    
    if remote_device is None:
        # Check kwargs for tensors
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                remote_device = value.device
                break

    op_name = str(op)

    # Try meta tensor execution first, fall back to dynamic if not implemented
    try:
        meta_result, original_tensors = _execute_meta_operation(op, args, kwargs)
        # Meta tensor execution succeeded - use static output path
        return _execute_with_static_outputs(op, args, kwargs, remote_device, op_name, meta_result, original_tensors)
    except NotImplementedError:
        # Operation doesn't support meta tensor execution - use dynamic output path
        return _execute_with_dynamic_outputs(op, args, kwargs, remote_device, op_name)
