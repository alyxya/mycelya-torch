#!/usr/bin/env python3
"""
Modal remote execution app for torch_remote extension.

This module handles all Modal-specific functionality including:
- Device-specific app creation for different GPU types
- Remote execution of PyTorch operations
- Dynamic GPU selection and configuration

Part of: torch_remote PyTorch extension
"""
import modal
from typing import Any, Dict, List, Tuple, Optional
import os

# Create simplified image with just PyTorch and CUDA support
image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands([
        "pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121",
        "pip install numpy"
    ])
)

# Create app with dynamic GPU support
app = modal.App("torch-remote-extension")

# Common execution function implementation
def _execute_aten_operation_impl(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any],
    device_id: str,
    gpu_type: str
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Common implementation for executing an aten operation remotely."""
    import torch
    import io
    
    print(f"🚀 Modal {gpu_type} (device {device_id}) executing: {op_name}")
    print(f"Received {len(tensors_data)} tensors, {len(args)} args, {len(kwargs)} kwargs")
    
    try:
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Deserialize tensors and move to GPU
        tensors = []
        for i, (data, metadata) in enumerate(zip(tensors_data, tensor_metadata)):
            # Deserialize tensor
            buffer = io.BytesIO(data)
            tensor = torch.load(buffer, map_location='cpu')
            
            # Move to GPU
            tensor = tensor.to(device)
            tensors.append(tensor)
        
        # Replace tensor placeholders in args with actual tensors
        processed_args = []
        for arg in args:
            if isinstance(arg, str) and arg.startswith("__TENSOR_"):
                idx = int(arg.split("_")[-1])
                processed_args.append(tensors[idx])
            else:
                processed_args.append(arg)
        
        # Process kwargs similarly
        processed_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str) and value.startswith("__TENSOR_"):
                idx = int(value.split("_")[-1])
                processed_kwargs[key] = tensors[idx]
            else:
                processed_kwargs[key] = value
        
        # Get the operation
        # Convert aten::add.Tensor to aten.add.Tensor
        op_name_fixed = op_name.replace("::", ".")
        op_parts = op_name_fixed.split('.')
        op = torch.ops
        for part in op_parts:
            op = getattr(op, part)
        
        print(f"Executing operation with {len(processed_args)} args")
        
        # Execute the operation
        result = op(*processed_args, **processed_kwargs)
        
        print(f"✅ Completed: {op_name} -> {result.shape if hasattr(result, 'shape') else type(result).__name__}")
        
        # Handle different result types
        if isinstance(result, torch.Tensor):
            results = [result]
        elif isinstance(result, (list, tuple)):
            results = [r for r in result if isinstance(r, torch.Tensor)]
        else:
            # For scalar results, convert to tensor
            results = [torch.tensor(result, device=device)]
        
        # Serialize results
        serialized_results = []
        result_metadata = []
        
        for i, tensor in enumerate(results):
            # Move back to CPU for serialization
            cpu_tensor = tensor.cpu()
            
            # Serialize tensor
            buffer = io.BytesIO()
            torch.save(cpu_tensor, buffer)
            serialized_results.append(buffer.getvalue())
            
            # Store metadata
            metadata = {
                'shape': list(cpu_tensor.shape),
                'dtype': str(cpu_tensor.dtype),
                'size': cpu_tensor.numel(),
                'element_size': cpu_tensor.element_size()
            }
            result_metadata.append(metadata)
        
        return serialized_results, result_metadata
        
    except Exception as e:
        print(f"❌ Error executing {op_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Define separate Modal functions for each GPU type
@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    retries=2
)
def execute_aten_operation_t4(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any],
    device_id: str
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Execute an aten operation remotely on T4 GPU."""
    return _execute_aten_operation_impl(op_name, tensors_data, tensor_metadata, args, kwargs, device_id, "T4")

@app.function(
    image=image,
    gpu="L4",
    timeout=300,
    retries=2
)
def execute_aten_operation_l4(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any],
    device_id: str
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Execute an aten operation remotely on L4 GPU."""
    return _execute_aten_operation_impl(op_name, tensors_data, tensor_metadata, args, kwargs, device_id, "L4")

@app.function(
    image=image,
    gpu="A10G",
    timeout=450,
    retries=2
)
def execute_aten_operation_a10g(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any],
    device_id: str
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Execute an aten operation remotely on A10G GPU."""
    return _execute_aten_operation_impl(op_name, tensors_data, tensor_metadata, args, kwargs, device_id, "A10G")

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=450,
    retries=2
)
def execute_aten_operation_a100_40gb(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any],
    device_id: str
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Execute an aten operation remotely on A100-40GB GPU."""
    return _execute_aten_operation_impl(op_name, tensors_data, tensor_metadata, args, kwargs, device_id, "A100-40GB")

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=450,
    retries=2
)
def execute_aten_operation_a100_80gb(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any],
    device_id: str
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Execute an aten operation remotely on A100-80GB GPU."""
    return _execute_aten_operation_impl(op_name, tensors_data, tensor_metadata, args, kwargs, device_id, "A100-80GB")

@app.function(
    image=image,
    gpu="L40S",
    timeout=450,
    retries=2
)
def execute_aten_operation_l40s(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any],
    device_id: str
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Execute an aten operation remotely on L40S GPU."""
    return _execute_aten_operation_impl(op_name, tensors_data, tensor_metadata, args, kwargs, device_id, "L40S")

@app.function(
    image=image,
    gpu="H100",
    timeout=450,
    retries=2
)
def execute_aten_operation_h100(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any],
    device_id: str
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Execute an aten operation remotely on H100 GPU."""
    return _execute_aten_operation_impl(op_name, tensors_data, tensor_metadata, args, kwargs, device_id, "H100")

@app.function(
    image=image,
    gpu="H200",
    timeout=450,
    retries=2
)
def execute_aten_operation_h200(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any],
    device_id: str
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Execute an aten operation remotely on H200 GPU."""
    return _execute_aten_operation_impl(op_name, tensors_data, tensor_metadata, args, kwargs, device_id, "H200")

@app.function(
    image=image,
    gpu="B200",
    timeout=450,
    retries=2
)
def execute_aten_operation_b200(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any],
    device_id: str
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Execute an aten operation remotely on B200 GPU."""
    return _execute_aten_operation_impl(op_name, tensors_data, tensor_metadata, args, kwargs, device_id, "B200")

# Dictionary to map GPU types to their corresponding functions
GPU_FUNCTIONS = {
    "T4": execute_aten_operation_t4,
    "L4": execute_aten_operation_l4,
    "A10G": execute_aten_operation_a10g,
    "A100-40GB": execute_aten_operation_a100_40gb,
    "A100-80GB": execute_aten_operation_a100_80gb,
    "L40S": execute_aten_operation_l40s,
    "H100": execute_aten_operation_h100,
    "H200": execute_aten_operation_h200,
    "B200": execute_aten_operation_b200,
}


# Cache for GPU-specific apps
_gpu_apps: Dict[str, modal.App] = {}

def create_modal_app_for_gpu(gpu_type, app_name: Optional[str] = None) -> modal.App:
    """
    Create a Modal app configured for a specific GPU type.
    
    For now, just return the default app since Modal functions must be defined at module level.
    Device-specific apps would require separate modules per GPU type.
    """
    # Just return the default app for simplicity
    # The GPU type is controlled by environment variable in the default app
    return app


def get_modal_app_for_device(device) -> modal.App:
    """
    Get the Modal app for a specific device.
    
    Args:
        device: The BackendDevice to get the app for
        
    Returns:
        Modal app (currently uses default app for all devices)
    """
    if hasattr(device, 'provider') and device.provider.value != "modal":
        raise ValueError(f"Device provider {device.provider.value} is not Modal")
    
    # For now, return the default app
    # TODO: Implement per-GPU-type apps by setting environment variables
    return app


def set_default_gpu_type(gpu_type: str):
    """Set the GPU type for the default Modal app via environment variable."""
    import os
    os.environ["TORCH_REMOTE_GPU"] = gpu_type


def clear_app_cache():
    """Clear the app cache."""
    global _gpu_apps
    _gpu_apps.clear()


# Store the execution functions in the app for easy access
app._gpu_functions = GPU_FUNCTIONS

def get_gpu_function(gpu_type: str):
    """Get the appropriate Modal function for a given GPU type."""
    if gpu_type not in GPU_FUNCTIONS:
        raise ValueError(f"GPU type '{gpu_type}' is not supported. Available types: {list(GPU_FUNCTIONS.keys())}")
    return GPU_FUNCTIONS[gpu_type]