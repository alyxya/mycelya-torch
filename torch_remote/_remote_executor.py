"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""
import logging
from typing import Any, Dict, List, Tuple, Optional
import torch

log = logging.getLogger(__name__)

# Try to import the remote app (Modal provider implementation)
try:
    from torch_remote_execution.modal_app import get_gpu_function
    log.info("Loaded torch_remote_execution app")
except Exception as e:
    log.warning(f"Remote execution not available: {e}")
    get_gpu_function = None


class RemoteExecutor:
    """Handles remote execution of aten operations on remote GPUs.
    
    This is the Modal provider implementation. Future providers can be added
    by extending this class or creating provider-specific executors.
    """
    
    def __init__(self):
        self._device_apps: Dict[str, Any] = {}  # Cache for device-specific apps
    
    def _get_device_app_and_function(self, device_id: str):
        """Get the Modal app and function for a specific device."""
        if device_id in self._device_apps:
            return self._device_apps[device_id]
        
        # Import here to avoid circular imports
        from .device import get_device_registry
        from torch_remote_execution.modal_app import get_modal_app_for_device
        
        # Get the device from registry
        registry = get_device_registry()
        device = registry.get_device_by_id(device_id)
        
        if device is None:
            raise RuntimeError(f"Device {device_id} not found in registry")
        
        # Get device-specific app and function
        app, function = get_modal_app_for_device(device)
        self._device_apps[device_id] = (app, function)
        
        return app, function
        
    def execute_remote_operation(
        self, 
        op_name: str, 
        args: Tuple[Any, ...], 
        kwargs: Dict[str, Any],
        device_id: Optional[str] = None
    ) -> Any:
        """
        Execute an aten operation remotely on GPU.
        
        Args:
            op_name: The aten operation name
            args: Operation arguments (may contain tensors)
            kwargs: Operation keyword arguments (may contain tensors)
            device_id: Optional device ID for device-specific execution
            
        Returns:
            Result of the operation (tensors moved back to remote device)
        """
        try:
            # Detect device from tensors if not specified
            if device_id is None:
                device_id = self._detect_device_from_tensors(args, kwargs)
            
            # Get the appropriate app and function (device-specific)
            if device_id is None:
                raise ValueError("device_id is required for remote execution")
            
            remote_app, execute_function = self._get_device_app_and_function(device_id)
            
            # Separate tensors from other arguments
            tensors_data = []
            tensor_metadata = []
            processed_args = []
            processed_kwargs = {}
            
            # Process args
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
                    # Convert remote tensor data to regular CPU tensor
                    if hasattr(arg, '__class__') and 'ModalTensorData' in str(arg.__class__):
                        # This is a ModalTensorData (provider-specific), convert to CPU tensor
                        cpu_tensor = self._remote_tensor_to_cpu(arg)
                    else:
                        # Regular remote tensor, copy to CPU
                        cpu_tensor = arg.cpu()
                    
                    tensor_data = self._serialize_tensor(cpu_tensor)
                    metadata = self._get_tensor_metadata(cpu_tensor)
                    
                    tensors_data.append(tensor_data)
                    tensor_metadata.append(metadata)
                    processed_args.append(f"__TENSOR_{len(tensors_data)-1}")
                else:
                    processed_args.append(arg)
            
            # Process kwargs
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.device.type == "remote":
                    # Convert remote tensor data to regular CPU tensor
                    if hasattr(value, '__class__') and 'ModalTensorData' in str(value.__class__):
                        # This is a ModalTensorData (provider-specific), convert to CPU tensor
                        cpu_tensor = self._remote_tensor_to_cpu(value)
                    else:
                        # Regular remote tensor, copy to CPU
                        cpu_tensor = value.cpu()
                    
                    tensor_data = self._serialize_tensor(cpu_tensor)
                    metadata = self._get_tensor_metadata(cpu_tensor)
                    
                    tensors_data.append(tensor_data)
                    tensor_metadata.append(metadata)
                    processed_kwargs[key] = f"__TENSOR_{len(tensors_data)-1}"
                else:
                    processed_kwargs[key] = value
            
            log.info(f"Executing {op_name} remotely with {len(tensors_data)} tensors")
            
            # Execute remotely with app context (Modal provider implementation)
            log.info(f"🚀 Starting remote app context for {op_name}")
            with remote_app.run():
                log.info(f"📡 App context started, calling remote function for {op_name}")
                log.info(f"🔧 Executor class: {execute_function}")
                log.info(f"📊 Args: op_name={op_name}, tensors={len(tensors_data)}, device_id={device_id}")
                
                try:
                    # Instantiate the executor class and call the method
                    executor_instance = execute_function()
                    serialized_results, result_metadata = executor_instance.execute_aten_operation.remote(
                        op_name, tensors_data, tensor_metadata, processed_args, processed_kwargs, device_id
                    )
                    log.info(f"✅ Remote function call completed for {op_name}")
                    log.info(f"📥 Received {len(serialized_results)} results")
                except Exception as remote_ex:
                    log.error(f"❌ Remote function call failed for {op_name}: {remote_ex}")
                    log.error(f"Exception type: {type(remote_ex).__name__}")
                    import traceback
                    log.error(f"Traceback: {traceback.format_exc()}")
                    raise
            
            # Deserialize results and create remote tensors
            results = []
            for data, metadata in zip(serialized_results, result_metadata):
                cpu_tensor = self._deserialize_tensor(data)
                remote_tensor = self._cpu_tensor_to_remote(cpu_tensor, device_id)
                results.append(remote_tensor)
            
            # Return single tensor or tuple based on original operation
            if len(results) == 1:
                return results[0]
            else:
                return tuple(results)
                
        except Exception as e:
            log.error(f"Remote execution failed for {op_name}: {str(e)}")
            raise RuntimeError(f"Remote execution failed: {str(e)}")
    
    
    def cleanup(self):
        """Clean up the remote app context."""
        self._device_apps.clear()
    
    def _remote_tensor_to_cpu(self, remote_tensor: torch.Tensor) -> torch.Tensor:
        """Convert remote tensor to CPU tensor without triggering remote execution."""
        try:
            # Use the device daemon's copy_from_device to avoid recursion
            from ._device_daemon import driver
            return driver.exec("copy_from_device", remote_tensor)
        except Exception:
            # If that fails, use direct tensor data access (avoid .cpu() recursion)
            # This creates a new CPU tensor with the same data
            return torch.tensor(remote_tensor.detach().numpy(), device='cpu')
    
    def _cpu_tensor_to_remote(self, cpu_tensor: torch.Tensor, device_id: Optional[str] = None) -> torch.Tensor:
        """Convert CPU tensor to remote tensor (Modal provider implementation)."""
        # Create a new remote tensor from the CPU tensor
        # This is simpler than using copy_from_host_to_device which has issues with provider-specific tensor data
        if device_id is None:
            raise RuntimeError("Device ID must be explicitly specified for remote tensor creation")
        
        # Get the BackendDevice to preserve device_id
        from .device import get_device_registry
        registry = get_device_registry()
        device = registry.get_device_by_id(device_id)
        if device is None:
            raise RuntimeError(f"Device {device_id} not found in registry")
        
        # Use the BackendDevice to ensure _device_id is preserved
        return cpu_tensor.to(device)
    
    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """Serialize tensor to bytes."""
        import io
        buffer = io.BytesIO()
        # Convert to pure CPU tensor to avoid torch_remote dependencies in serialization
        cpu_tensor = tensor.cpu().detach()
        torch.save(cpu_tensor, buffer)
        return buffer.getvalue()
    
    def _deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """Deserialize tensor from bytes."""
        import io
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location='cpu')
    
    def _get_tensor_metadata(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Get tensor metadata."""
        return {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'size': tensor.numel(),
            'element_size': tensor.element_size()
        }
    
    def _detect_device_from_tensors(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[str]:
        """Detect device ID from tensors in arguments."""
        detected_device_id = None
        
        # Check args for remote tensors
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
                # Try to get device ID from tensor metadata
                if hasattr(arg, '_device_id'):
                    if detected_device_id is None:
                        detected_device_id = arg._device_id
                    elif detected_device_id != arg._device_id:
                        # Different devices found - this is not allowed
                        raise RuntimeError(
                            f"Cannot perform operations between tensors on different remote devices: "
                            f"'{detected_device_id}' and '{arg._device_id}'"
                        )
        
        # Check kwargs for remote tensors
        for value in kwargs.values():
            if isinstance(value, torch.Tensor) and value.device.type == "remote":
                # Try to get device ID from tensor metadata
                if hasattr(value, '_device_id'):
                    if detected_device_id is None:
                        detected_device_id = value._device_id
                    elif detected_device_id != value._device_id:
                        # Different devices found - this is not allowed
                        raise RuntimeError(
                            f"Cannot perform operations between tensors on different remote devices: "
                            f"'{detected_device_id}' and '{value._device_id}'"
                        )
        
        return detected_device_id


# Global executor instance (Modal provider implementation)
remote_executor = RemoteExecutor()