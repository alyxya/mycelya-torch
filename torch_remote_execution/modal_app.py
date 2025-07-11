#!/usr/bin/env python3
"""
Modal remote execution app for torch_remote extension.

This module handles all Modal-specific functionality including:
- Dynamic device-specific app creation for different GPU types
- Remote execution of PyTorch operations
- Dynamic GPU selection and configuration

Part of: torch_remote PyTorch extension
"""
import modal
from typing import Any, Dict, List, Tuple, Optional
import os

# Create simplified image with just PyTorch and CUDA support
image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "torch")
)

# Cache for GPU-specific apps and their functions  
_gpu_apps: Dict[str, Tuple[modal.App, Any]] = {}
# Cache for RemoteGPUMachine instances
_gpu_machines: Dict[str, "RemoteGPUMachine"] = {}

# GPU configuration mapping
GPU_CONFIG = {
    "T4": {"timeout": 300, "retries": 2},
    "L4": {"timeout": 300, "retries": 2},
    "A10G": {"timeout": 450, "retries": 2},
    "A100-40GB": {"timeout": 450, "retries": 2},
    "A100-80GB": {"timeout": 450, "retries": 2},
    "L40S": {"timeout": 450, "retries": 2},
    "H100": {"timeout": 450, "retries": 2},
    "H200": {"timeout": 450, "retries": 2},
    "B200": {"timeout": 450, "retries": 2},
}

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
            tensor = torch.load(buffer, map_location='cpu', weights_only=True)
            
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


class RemoteGPUMachine:
    """
    Stateful wrapper representing a remote GPU machine running on Modal.
    
    This class encapsulates both the Modal app and executor, providing a clean
    interface for interacting with remote GPU execution while maintaining
    state and connection management.
    """
    
    def __init__(self, gpu_type: str, device_id: str):
        self.gpu_type = gpu_type
        self.device_id = device_id
        self._app = None
        self._executor_class = None
        self._executor_instance = None
        self._app_context = None
        
        # Initialize the Modal app and executor
        self._initialize()
    
    def _initialize(self):
        """Initialize the Modal app and executor class."""
        self._app, self._executor_class = _create_modal_app_for_gpu(
            self.gpu_type, self.device_id
        )
    
    def start(self):
        """Start the Modal app context for this machine."""
        if self._app_context is None:
            self._app_context = self._app.run()
            self._app_context.__enter__()
            # Create executor instance when app starts
            self._executor_instance = self._executor_class()
    
    def stop(self):
        """Stop the Modal app context for this machine."""
        if self._app_context is not None:
            self._app_context.__exit__(None, None, None)
            self._app_context = None
            self._executor_instance = None
    
    def is_running(self) -> bool:
        """Check if the machine is currently running."""
        return self._app_context is not None
    
    def execute_operation(
        self,
        op_name: str,
        tensors_data: List[bytes],
        tensor_metadata: List[Dict[str, Any]],
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Tuple[List[bytes], List[Dict[str, Any]]]:
        """
        Execute an operation on this remote GPU machine.
        
        Args:
            op_name: The operation name to execute
            tensors_data: Serialized tensor data
            tensor_metadata: Tensor metadata
            args: Operation arguments
            kwargs: Operation keyword arguments
            
        Returns:
            Tuple of (serialized_results, result_metadata)
            
        Raises:
            RuntimeError: If machine is not running
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.device_id} is not running. Call start() first.")
        
        return self._executor_instance.execute_aten_operation.remote(
            op_name, tensors_data, tensor_metadata, args, kwargs, self.device_id
        )
    
    def __enter__(self):
        """Context manager entry - starts the machine."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the machine."""
        self.stop()
    
    def __repr__(self):
        status = "running" if self.is_running() else "stopped"
        return f"RemoteGPUMachine(gpu_type='{self.gpu_type}', device_id='{self.device_id}', status='{status}')"


def create_modal_app_for_gpu(gpu_type: str, device_id: str) -> RemoteGPUMachine:
    """
    Create a RemoteGPUMachine for a specific GPU type and device.
    
    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB")
        device_id: The device ID (e.g., "modal-t4-f3a7d67e")
        
    Returns:
        RemoteGPUMachine instance representing the remote GPU
    """
    # Check cache first
    if device_id in _gpu_machines:
        return _gpu_machines[device_id]
    
    # Create new machine and cache it
    machine = RemoteGPUMachine(gpu_type, device_id)
    _gpu_machines[device_id] = machine
    return machine


def _create_modal_app_for_gpu(gpu_type: str, device_id: str) -> Tuple[modal.App, Any]:
    """
    Create a Modal app and class for a specific GPU type and device.
    
    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB")
        device_id: The device ID (e.g., "modal-t4-f3a7d67e")
        
    Returns:
        Tuple of (modal_app, executor_class) for the specified device
    """
    if device_id in _gpu_apps:
        return _gpu_apps[device_id]
    
    if gpu_type not in GPU_CONFIG:
        raise ValueError(f"GPU type '{gpu_type}' is not supported. Available types: {list(GPU_CONFIG.keys())}")
    
    config = GPU_CONFIG[gpu_type]
    app = modal.App(f"torch-remote-{device_id}")
    
    @app.cls(
        image=image,
        gpu=gpu_type,
        timeout=config["timeout"],
        retries=config["retries"],
        serialized=True
    )
    class PytorchOperationExecutor:
        
        @modal.method()
        def execute_aten_operation(
            self,
            op_name: str, 
            tensors_data: List[bytes], 
            tensor_metadata: List[Dict[str, Any]], 
            args: List[Any], 
            kwargs: Dict[str, Any],
            device_id: str
        ) -> Tuple[List[bytes], List[Dict[str, Any]]]:
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
                    tensor = torch.load(buffer, map_location='cpu', weights_only=True)
                    
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
    
    _gpu_apps[device_id] = (app, PytorchOperationExecutor)
    return app, PytorchOperationExecutor


def get_modal_app_for_device(device) -> RemoteGPUMachine:
    """
    Get the RemoteGPUMachine for a specific device.
    
    Args:
        device: The BackendDevice to get the machine for
        
    Returns:
        RemoteGPUMachine for the device's GPU type
    """
    if hasattr(device, 'provider') and device.provider.value != "modal":
        raise ValueError(f"Device provider {device.provider.value} is not Modal")
    
    return create_modal_app_for_gpu(device.gpu_type.value, device.device_id)


def clear_app_cache():
    """Clear the app cache."""
    global _gpu_apps
    _gpu_apps.clear()


def get_gpu_function(gpu_type: str):
    """Get the appropriate Modal function for a given GPU type."""
    # For backward compatibility, create a temporary device ID
    temp_device_id = f"temp-{gpu_type.lower().replace('-', '')}"
    app, function = create_modal_app_for_gpu(gpu_type, temp_device_id)
    return function