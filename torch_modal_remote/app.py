#!/usr/bin/env python3
"""
Auto-generated Modal remote execution app for torch_modal extension.

This file is automatically created and managed by the torch_modal extension.
Users never need to interact with this file directly - it's used automatically
when remote execution is needed.

Part of: torch_modal PyTorch extension
"""
import modal
from typing import Any, Dict, List, Tuple

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

# Create app
app = modal.App("torch-modal-extension")

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=300,
    retries=2
)
def execute_aten_operation(
    op_name: str, 
    tensors_data: List[bytes], 
    tensor_metadata: List[Dict[str, Any]], 
    args: List[Any], 
    kwargs: Dict[str, Any]
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    """Execute an aten operation remotely on A100 GPU."""
    import torch
    import io
    
    print(f"🚀 Modal A100 executing: {op_name}")
    print(f"Received {len(tensors_data)} tensors, {len(args)} args, {len(kwargs)} kwargs")
    
    try:
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
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