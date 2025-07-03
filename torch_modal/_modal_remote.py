"""
Modal remote execution system for aten operations on A100 GPUs.
Uses ephemeral Modal runs (no deployment needed).
"""
import logging
from typing import Any, Dict, List, Tuple
import torch

log = logging.getLogger(__name__)

# Global modal app - will be auto-imported
_modal_app = None
_execute_aten_operation = None

# Try to import the Modal app
try:
    from torch_modal_remote.app import app as _modal_app, execute_aten_operation as _execute_aten_operation
    log.info("Loaded torch_modal remote execution app")
except Exception as e:
    log.warning(f"Modal remote execution not available: {e}")
    _modal_app = None
    _execute_aten_operation = None

def _get_modal_app():
    """Get the Modal app for remote execution."""
    global _modal_app, _execute_aten_operation
    
    if _modal_app is None:
        raise RuntimeError("Modal remote execution not available. Install with: pip install modal")
    
    # Store the function in the app for easy access
    _modal_app._execute_aten_operation = _execute_aten_operation
    return _modal_app


class ModalRemoteExecutor:
    """Handles remote execution of aten operations on Modal A100 GPUs using ephemeral runs."""
    
    def __init__(self):
        self._modal_app = None
        self._app_context = None
        
    def _get_app_context(self):
        """Get or create the Modal app context for ephemeral runs."""
        if self._modal_app is None:
            self._modal_app = _get_modal_app()
        
        # For ephemeral execution, we don't need to manage the context manually
        # Modal will handle the app lifecycle automatically
        return self._modal_app
        
    def execute_remote_operation(
        self, 
        op_name: str, 
        args: Tuple[Any, ...], 
        kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute an aten operation remotely on A100 GPU.
        
        Args:
            op_name: The aten operation name
            args: Operation arguments (may contain tensors)
            kwargs: Operation keyword arguments (may contain tensors)
            
        Returns:
            Result of the operation (tensors moved back to modal device)
        """
        try:
            # Get the Modal app
            modal_app = self._get_app_context()
            
            # Separate tensors from other arguments
            tensors_data = []
            tensor_metadata = []
            processed_args = []
            processed_kwargs = {}
            
            # Process args
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.device.type == "modal":
                    # Convert ModalTensorData to regular CPU tensor
                    if hasattr(arg, '__class__') and 'ModalTensorData' in str(arg.__class__):
                        # This is a ModalTensorData, convert to CPU tensor
                        cpu_tensor = self._modal_tensor_to_cpu(arg)
                    else:
                        # Regular modal tensor, copy to CPU
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
                if isinstance(value, torch.Tensor) and value.device.type == "modal":
                    # Convert ModalTensorData to regular CPU tensor
                    if hasattr(value, '__class__') and 'ModalTensorData' in str(value.__class__):
                        # This is a ModalTensorData, convert to CPU tensor
                        cpu_tensor = self._modal_tensor_to_cpu(value)
                    else:
                        # Regular modal tensor, copy to CPU
                        cpu_tensor = value.cpu()
                    
                    tensor_data = self._serialize_tensor(cpu_tensor)
                    metadata = self._get_tensor_metadata(cpu_tensor)
                    
                    tensors_data.append(tensor_data)
                    tensor_metadata.append(metadata)
                    processed_kwargs[key] = f"__TENSOR_{len(tensors_data)-1}"
                else:
                    processed_kwargs[key] = value
            
            log.info(f"Executing {op_name} remotely with {len(tensors_data)} tensors")
            
            # Execute remotely with app context
            with modal_app.run():
                # Get the function from the app's registered functions
                execute_function = modal_app.registered_functions.get('execute_aten_operation')
                if execute_function is None:
                    # Fallback to the stored function
                    execute_function = modal_app._execute_aten_operation
                
                serialized_results, result_metadata = execute_function.remote(
                    op_name, tensors_data, tensor_metadata, processed_args, processed_kwargs
                )
            
            # Deserialize results and create modal tensors
            results = []
            for data, metadata in zip(serialized_results, result_metadata):
                cpu_tensor = self._deserialize_tensor(data)
                modal_tensor = self._cpu_tensor_to_modal(cpu_tensor)
                results.append(modal_tensor)
            
            # Return single tensor or tuple based on original operation
            if len(results) == 1:
                return results[0]
            else:
                return tuple(results)
                
        except Exception as e:
            log.error(f"Remote execution failed for {op_name}: {str(e)}")
            raise RuntimeError(f"Remote execution failed: {str(e)}")
    
    def cleanup(self):
        """Clean up the Modal app context."""
        if self._app_context is not None:
            try:
                self._app_context.__exit__(None, None, None)
            except Exception:
                pass
            self._app_context = None
    
    def _modal_tensor_to_cpu(self, modal_tensor: torch.Tensor) -> torch.Tensor:
        """Convert modal tensor to CPU tensor."""
        try:
            # Try the modal-specific copy first
            from ._aten_impl import copy_from_device
            return copy_from_device(modal_tensor)
        except Exception:
            # Fallback to direct CPU conversion
            return modal_tensor.cpu()
    
    def _cpu_tensor_to_modal(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """Convert CPU tensor to modal tensor."""
        # Create a new modal tensor from the CPU tensor
        # This is simpler than using copy_from_host_to_device which has issues with ModalTensorData
        return cpu_tensor.to("modal")
    
    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """Serialize tensor to bytes."""
        import io
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
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


# Global executor instance
remote_executor = ModalRemoteExecutor()