"""
Modal device-specific app generator for torch_remote.

This module creates Modal apps configured for specific GPU types.
"""
import modal
from typing import Any, Dict, List, Tuple, Optional
from .device import BackendDevice, GPUType

# Cache for GPU-specific apps
_gpu_apps: Dict[str, modal.App] = {}

def create_modal_app_for_gpu(gpu_type: GPUType, app_name: Optional[str] = None) -> modal.App:
    """
    Create a Modal app configured for a specific GPU type.
    
    Args:
        gpu_type: The GPU type to configure for
        app_name: Optional custom app name
        
    Returns:
        Modal app configured for the specified GPU
    """
    gpu_spec = gpu_type.value
    
    # Check cache first
    if gpu_spec in _gpu_apps:
        return _gpu_apps[gpu_spec]
    
    # Create image with PyTorch and CUDA support
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install([
            "torch>=2.0.0", 
            "torchvision", 
            "torchaudio",
            "nvidia-ml-py3"
        ])
        .run_commands([
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        ])
    )
    
    # Create app with GPU-specific name
    if app_name is None:
        app_name = f"torch-remote-{gpu_spec.lower().replace('-', '_')}"
    
    app = modal.App(app_name)
    
    # Configure timeout based on GPU type (more powerful GPUs get more time)
    timeout = 300  # Default 5 minutes
    if gpu_type in [GPUType.H100, GPUType.H200, GPUType.B200]:
        timeout = 600  # 10 minutes for high-end GPUs
    elif gpu_type in [GPUType.A100_40GB, GPUType.A100_80GB]:
        timeout = 450  # 7.5 minutes for A100s
    
    @app.function(
        image=image,
        gpu=gpu_spec,
        timeout=timeout,
        retries=2
    )
    def execute_aten_operation(
        op_name: str, 
        tensors_data: List[bytes], 
        tensor_metadata: List[Dict[str, Any]], 
        args: List[Any], 
        kwargs: Dict[str, Any],
        device_id: str
    ) -> Tuple[List[bytes], List[Dict[str, Any]]]:
        """Execute an aten operation remotely on the specified GPU."""
        import torch
        import io
        
        print(f"🚀 Modal {gpu_spec} (device {device_id[:8]}) executing: {op_name}")
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
    
    # Store the execution function in the app for easy access
    app._execute_aten_operation = execute_aten_operation
    
    # Cache the app
    _gpu_apps[gpu_spec] = app
    
    return app


def get_modal_app_for_device(device: BackendDevice) -> modal.App:
    """
    Get the Modal app for a specific device.
    
    Args:
        device: The BackendDevice to get the app for
        
    Returns:
        Modal app configured for the device's GPU type
    """
    if device.provider.value != "modal":
        raise ValueError(f"Device provider {device.provider.value} is not Modal")
    
    return create_modal_app_for_gpu(device.gpu_type)


def clear_app_cache():
    """Clear the app cache."""
    global _gpu_apps
    _gpu_apps.clear()