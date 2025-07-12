"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""
import logging
from typing import Any, Dict, List, Tuple, Optional
import torch

from .device import BackendDevice, get_device_registry

log = logging.getLogger(__name__)

# Try to load the remote execution module (Modal provider implementation)
try:
    import torch_remote_execution.modal_app
    log.info("Loaded torch_remote_execution app")
except Exception as e:
    log.warning(f"Remote execution not available: {e}")


class RemoteExecutor:
    """Handles remote execution of aten operations on remote GPUs.
    
    This is the Modal provider implementation. Future providers can be added
    by extending this class or creating provider-specific executors.
    """
    
    def __init__(self):
        self._device_apps: Dict[str, Any] = {}  # Cache for device-specific GPU machines
    
    def _get_device_gpu_machine(self, device: 'BackendDevice'):
        """Get the active RemoteGPUMachine for a specific device."""
        # Get the pre-started GPU machine from the device
        gpu_machine = device.get_gpu_machine()
        
        if gpu_machine is None:
            raise RuntimeError(f"No GPU machine available for device {device.device_id}")
        
        if not gpu_machine.is_running():
            raise RuntimeError(f"GPU machine for device {device.device_id} is not running")
        
        return gpu_machine
        
    def execute_remote_operation(
        self, 
        op_name: str, 
        args: Tuple[Any, ...], 
        kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute an aten operation remotely on GPU.
        
        Args:
            op_name: The aten operation name
            args: Operation arguments (may contain tensors)
            kwargs: Operation keyword arguments (may contain tensors)
            
        Returns:
            Result of the operation (tensors moved back to remote device)
        """
        try:
            # Detect device from tensors
            device = self._detect_device_from_tensors(args, kwargs)
            
            # Get the pre-started GPU machine (device-specific)
            if device is None:
                raise ValueError("BackendDevice is required for remote execution")
            
            gpu_machine = self._get_device_gpu_machine(device)
            
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
            
            # Execute remotely using pre-started RemoteGPUMachine
            log.info(f"🚀 Using active GPU machine for {op_name}: {gpu_machine}")
            log.info(f"📊 Args: op_name={op_name}, tensors={len(tensors_data)}, device_id={device.device_id}")
            
            try:
                serialized_results, result_metadata = gpu_machine.execute_operation(
                    op_name, tensors_data, tensor_metadata, processed_args, processed_kwargs
                )
                log.info(f"✅ Remote operation completed for {op_name}")
                log.info(f"📥 Received {len(serialized_results)} results")
            except Exception as remote_ex:
                log.error(f"❌ Remote operation failed for {op_name}: {remote_ex}")
                log.error(f"Exception type: {type(remote_ex).__name__}")
                import traceback
                log.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Deserialize results and create remote tensors
            results = []
            for data, metadata in zip(serialized_results, result_metadata):
                cpu_tensor = self._deserialize_tensor(data)
                remote_tensor = self._cpu_tensor_to_remote(cpu_tensor, device)
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
        """Clean up the remote executor."""
        # No longer needed since machines are managed by devices
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
    
    def _cpu_tensor_to_remote(self, cpu_tensor: torch.Tensor, device: 'BackendDevice') -> torch.Tensor:
        """Convert CPU tensor to remote tensor."""
        # Create a new remote tensor from the CPU tensor using the BackendDevice
        if device is None:
            raise RuntimeError("BackendDevice must be specified for remote tensor creation")
        
        # Convert to remote device - no need to manually attach _device_id anymore
        result = cpu_tensor.to(device.device())
        return result
    
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
    
    def _detect_device_from_tensors(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[BackendDevice]:
        """Detect BackendDevice from tensors in arguments."""
        detected_device = None
        
        def check_tensor(tensor):
            nonlocal detected_device
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "remote":
                # Get device from registry using device index
                registry = get_device_registry()
                device = registry.get_device_by_index(tensor.device.index)
                
                if device is None:
                    raise RuntimeError(f"No BackendDevice found for remote device index {tensor.device.index}")
                
                if detected_device is None:
                    detected_device = device
                elif detected_device is not device:
                    # Different devices found - this is not allowed
                    raise RuntimeError(
                        f"Cannot perform operations between tensors on different remote devices: "
                        f"'{detected_device.device_id}' (index {detected_device.remote_index}) and "
                        f"'{device.device_id}' (index {device.remote_index})"
                    )
        
        # Check args for remote tensors
        for arg in args:
            check_tensor(arg)
        
        # Check kwargs for remote tensors  
        for value in kwargs.values():
            check_tensor(value)
        
        return detected_device


# Global executor instance (Modal provider implementation)
remote_executor = RemoteExecutor()