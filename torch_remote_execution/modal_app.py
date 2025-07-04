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
        "pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121"
    ])
)

# Create app with dynamic GPU support
app = modal.App("torch-remote-extension")

# Get GPU type from environment variable or default to T4 (cheaper option)
DEFAULT_GPU = os.environ.get("TORCH_REMOTE_GPU", "T4")

# Define the Modal function at global scope
@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=300 if DEFAULT_GPU in ["T4", "L4"] else 450,  # Longer timeout for better GPUs
    retries=2
)
def execute_aten_operation(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any],
    device_id: str = "default"
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Execute an aten operation remotely on specified GPU."""
    import torch
    import io
    
    gpu_type = os.environ.get("TORCH_REMOTE_GPU", "T4")
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


# Store the execution function in the app for easy access
app._execute_aten_operation = execute_aten_operation