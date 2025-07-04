"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""
import logging
from typing import Any, Dict, List, Tuple
import torch

log = logging.getLogger(__name__)

# Global remote app - will be auto-imported (Modal provider implementation)
_remote_app = None
_execute_aten_operation = None

# Try to import the remote app (Modal provider implementation)
try:
    from torch_remote_execution.modal_app import app as _remote_app, execute_aten_operation as _execute_aten_operation
    log.info("Loaded torch_remote_execution app")
except Exception as e:
    log.warning(f"Remote execution not available: {e}")
    _remote_app = None
    _execute_aten_operation = None

def _get_remote_app():
    """Get the remote app for remote execution (Modal provider implementation)."""
    global _remote_app, _execute_aten_operation
    
    if _remote_app is None:
        raise RuntimeError("Remote execution not available. Install provider dependencies (e.g., pip install modal)")
    
    # Store the function in the app for easy access
    _remote_app._execute_aten_operation = _execute_aten_operation
    return _remote_app


class RemoteExecutor:
    """Handles remote execution of aten operations on remote GPUs.
    
    This is the Modal provider implementation. Future providers can be added
    by extending this class or creating provider-specific executors.
    """
    
    def __init__(self):
        self._remote_app = None
        self._app_context = None
        
    def _get_app_context(self):
        """Get or create the remote app context for ephemeral runs (Modal provider implementation)."""
        if self._remote_app is None:
            self._remote_app = _get_remote_app()
        
        # For ephemeral execution, we don't need to manage the context manually
        # The provider will handle the app lifecycle automatically
        return self._remote_app
        
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
            # Get the remote app (Modal provider implementation)
            remote_app = self._get_app_context()
            
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
            with remote_app.run():
                # Get the function from the app's registered functions
                execute_function = remote_app.registered_functions.get('execute_aten_operation')
                if execute_function is None:
                    # Fallback to the stored function
                    execute_function = remote_app._execute_aten_operation
                
                serialized_results, result_metadata = execute_function.remote(
                    op_name, tensors_data, tensor_metadata, processed_args, processed_kwargs
                )
            
            # Deserialize results and create remote tensors
            results = []
            for data, metadata in zip(serialized_results, result_metadata):
                cpu_tensor = self._deserialize_tensor(data)
                remote_tensor = self._cpu_tensor_to_remote(cpu_tensor)
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
        if self._app_context is not None:
            try:
                self._app_context.__exit__(None, None, None)
            except Exception:
                pass
            self._app_context = None
    
    def _remote_tensor_to_cpu(self, remote_tensor: torch.Tensor) -> torch.Tensor:
        """Convert remote tensor to CPU tensor (Modal provider implementation)."""
        try:
            # Try the provider-specific copy first
            from ._aten_impl import copy_from_device
            return copy_from_device(remote_tensor)
        except Exception:
            # Fallback to direct CPU conversion
            return remote_tensor.cpu()
    
    def _cpu_tensor_to_remote(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """Convert CPU tensor to remote tensor (Modal provider implementation)."""
        # Create a new remote tensor from the CPU tensor
        # This is simpler than using copy_from_host_to_device which has issues with provider-specific tensor data
        return cpu_tensor.to("remote")
    
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


# Global executor instance (Modal provider implementation)
remote_executor = RemoteExecutor()