# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution system for aten operations on remote GPUs.
Supports multiple remote execution providers.

This module provides a generic interface for remote execution of PyTorch operations.
Currently supports Modal as the first provider implementation.
"""
import logging
from typing import Any, Dict, List, Tuple, Optional
import torch
import time

from .device import RemoteBackend, get_device_registry

log = logging.getLogger(__name__)


class RemoteTensorError(Exception):
    """Base exception for remote tensor operations."""
    pass


class StaleReferenceError(RemoteTensorError):
    """Raised when trying to access a tensor that no longer exists on remote machine."""
    pass


class ConnectionError(RemoteTensorError):
    """Raised when remote machine connection is lost."""
    pass


class RemoteExecutionError(RemoteTensorError):
    """Raised when remote execution fails."""
    pass


def with_error_handling(func):
    """Decorator to add error handling to remote operations."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check for specific error patterns
            error_msg = str(e).lower()
            
            if "tensor id" in error_msg and "not found" in error_msg:
                raise StaleReferenceError(f"Tensor reference is stale: {e}")
            elif "machine" in error_msg and "not running" in error_msg:
                raise ConnectionError(f"Remote machine connection lost: {e}")
            elif "remote execution failed" in error_msg:
                raise RemoteExecutionError(f"Remote execution failed: {e}")
            else:
                # Re-raise as generic remote tensor error
                raise RemoteTensorError(f"Remote operation failed: {e}")
    
    return wrapper


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
        self._last_heartbeat: Dict[str, float] = {}  # Track last successful communication per device
    
    def _get_device_gpu_machine(self, device: "RemoteBackend"):
        """Get the active RemoteGPUMachine for a specific device."""
        # Get the pre-started GPU machine from the device
        gpu_machine = device.get_gpu_machine()
        
        if gpu_machine is None:
            raise RuntimeError(f"No GPU machine available for device {device.machine_id}")
        
        if not gpu_machine.is_running():
            raise RuntimeError(f"GPU machine for device {device.machine_id} is not running")
        
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
                raise ValueError("RemoteBackend is required for remote execution")
            
            gpu_machine = self._get_device_gpu_machine(device)
            
            # Separate input tensors from output tensors
            input_tensors_data = []
            input_tensor_metadata = []
            processed_args = []
            processed_kwargs = {}
            output_tensors = []  # Track pre-allocated output tensors
            
            # Process args - these are always input tensors
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
                    # Convert remote tensor data to regular CPU tensor
                    # Use proper remote-to-CPU conversion for INPUT tensors only
                    cpu_tensor = self._remote_tensor_to_cpu(arg)
                    
                    tensor_data = self._serialize_tensor(cpu_tensor)
                    metadata = self._get_tensor_metadata(cpu_tensor)
                    
                    input_tensors_data.append(tensor_data)
                    input_tensor_metadata.append(metadata)
                    processed_args.append(f"__TENSOR_{len(input_tensors_data)-1}")
                    
                    # Log input tensor details
                    tensor_id = arg.untyped_storage().data_ptr()
                    log.info(f"📥 INPUT tensor arg[{i}]: ID={tensor_id}, shape={arg.shape}, dtype={arg.dtype}, device={arg.device}")
                else:
                    processed_args.append(arg)
            
            # Process kwargs - distinguish between input and output tensors
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.device.type == "remote":
                    if key == "out":
                        # This is an OUTPUT tensor - don't read its data, just track it
                        output_tensors.append(value)
                        
                        # Create empty tensor placeholder for the remote operation
                        empty_tensor = torch.empty(
                            value.shape,
                            dtype=value.dtype,
                            device="cpu"
                        )
                        tensor_data = self._serialize_tensor(empty_tensor)
                        metadata = self._get_tensor_metadata(empty_tensor)
                        
                        input_tensors_data.append(tensor_data)
                        input_tensor_metadata.append(metadata)
                        processed_kwargs[key] = f"__TENSOR_{len(input_tensors_data)-1}"
                        
                        # Log output tensor details
                        tensor_id = value.untyped_storage().data_ptr()
                        log.info(f"📤 OUTPUT tensor kwarg[{key}]: ID={tensor_id}, shape={value.shape}, dtype={value.dtype}, device={value.device}")
                    else:
                        # This is an INPUT tensor - read its data
                        cpu_tensor = self._remote_tensor_to_cpu(value)
                        
                        tensor_data = self._serialize_tensor(cpu_tensor)
                        metadata = self._get_tensor_metadata(cpu_tensor)
                        
                        input_tensors_data.append(tensor_data)
                        input_tensor_metadata.append(metadata)
                        processed_kwargs[key] = f"__TENSOR_{len(input_tensors_data)-1}"
                        
                        # Log input tensor details
                        tensor_id = value.untyped_storage().data_ptr()
                        log.info(f"📥 INPUT tensor kwarg[{key}]: ID={tensor_id}, shape={value.shape}, dtype={value.dtype}, device={value.device}")
                else:
                    processed_kwargs[key] = value
            
            log.info(f"Executing {op_name} remotely with {len(input_tensors_data)} input tensors, {len(output_tensors)} output tensors")
            
            # Execute remotely using pre-started RemoteGPUMachine
            log.info(f"🚀 Using active GPU machine for {op_name}: {gpu_machine}")
            log.info(f"📊 Args: op_name={op_name}, input_tensors={len(input_tensors_data)}, output_tensors={len(output_tensors)}, machine_id={device.machine_id}")
            
            try:
                serialized_results, result_metadata = gpu_machine.execute_operation(
                    op_name, input_tensors_data, input_tensor_metadata, processed_args, processed_kwargs
                )
                log.info(f"✅ Remote operation completed for {op_name}")
                log.info(f"📥 Received {len(serialized_results)} results")
            except Exception as remote_ex:
                log.error(f"❌ Remote operation failed for {op_name}: {remote_ex}")
                log.error(f"Exception type: {type(remote_ex).__name__}")
                import traceback
                log.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Handle result tensor allocation - use pre-allocated output tensors when available
            results = []
            gpu_machine = self._get_device_gpu_machine(device)
            
            for i, (data, metadata) in enumerate(zip(serialized_results, result_metadata)):
                cpu_tensor = self._deserialize_tensor(data)
                
                # Check if we have a pre-allocated output tensor for this result
                if i < len(output_tensors):
                    # Use the pre-allocated output tensor
                    output_tensor = output_tensors[i]
                    tensor_id_int = output_tensor.untyped_storage().data_ptr()
                    tensor_id_str = str(tensor_id_int)
                    
                    # Store the result data in the pre-allocated tensor's ID
                    gpu_machine.create_tensor(data, tensor_id_str)
                    
                    # Log output tensor result details
                    log.info(f"📤 OUTPUT RESULT tensor[{i}]: ID={tensor_id_int}, shape={metadata.get('shape', 'unknown')}, dtype={metadata.get('dtype', 'unknown')}, size={cpu_tensor.numel()}")
                    
                    results.append(output_tensor)
                else:
                    # Create a new remote tensor for this result
                    remote_tensor = self._cpu_tensor_to_remote(cpu_tensor, device)
                    
                    # Register the tensor data with the GPU machine using the tensor's ID
                    tensor_id_int = remote_tensor.untyped_storage().data_ptr()
                    tensor_id_str = str(tensor_id_int)
                    
                    # Store the tensor data on the GPU machine so future operations can access it
                    gpu_machine.create_tensor(data, tensor_id_str)
                    
                    # Log new result tensor details
                    log.info(f"📤 NEW RESULT tensor[{i}]: ID={tensor_id_int}, shape={metadata.get('shape', 'unknown')}, dtype={metadata.get('dtype', 'unknown')}, size={cpu_tensor.numel()}")
                    
                    results.append(remote_tensor)
            
            # Return single tensor or tuple based on original operation
            if len(results) == 1:
                return results[0]
            else:
                return tuple(results)
                
        except Exception as e:
            log.error(f"Remote execution failed for {op_name}: {str(e)}")
            raise RuntimeError(f"Remote execution failed: {str(e)}")
    
    
    def create_tensor_on_remote(
        self,
        tensor_data: bytes, 
        device: "RemoteBackend",
        tensor_id: Optional[str] = None
    ) -> str:
        """
        Create a tensor on the remote machine and return its ID.
        
        Args:
            tensor_data: Serialized tensor data
            device: Target device
            tensor_id: Optional specific ID to use
            
        Returns:
            The tensor ID
        """
        gpu_machine = self._get_device_gpu_machine(device)
        return gpu_machine.create_tensor(tensor_data, tensor_id)
    
    def get_tensor_data_from_remote(self, tensor_id: str, device_index: int) -> torch.Tensor:
        """
        Get tensor data from remote machine by ID.
        
        Args:
            tensor_id: The tensor ID
            device_index: The remote device index
            
        Returns:
            CPU tensor with the data
        """
        registry = get_device_registry()
        device = registry.get_device_by_index(device_index)
        if device is None:
            raise RuntimeError(f"No device found for index {device_index}")
        
        gpu_machine = self._get_device_gpu_machine(device)
        tensor_data = gpu_machine.get_tensor_data(tensor_id)
        return self._deserialize_tensor(tensor_data)
    
    def execute_remote_operation_with_ids(
        self,
        op_name: str,
        tensor_ids: List[str],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        device: "RemoteBackend"
    ) -> List[str]:
        """
        Execute an operation using tensor IDs.
        
        Args:
            op_name: The operation name
            tensor_ids: Input tensor IDs
            args: Operation arguments
            kwargs: Operation keyword arguments
            device: Target device
            
        Returns:
            Result tensor IDs
        """
        gpu_machine = self._get_device_gpu_machine(device)
        return gpu_machine.execute_operation_with_ids(op_name, tensor_ids, list(args), kwargs)
    
    def create_factory_tensor_on_remote(
        self,
        factory_op: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        device: "RemoteBackend"
    ) -> str:
        """
        Create a tensor using a factory operation on remote machine.
        
        Args:
            factory_op: Factory operation name (e.g., "randn")
            args: Factory arguments
            kwargs: Factory keyword arguments  
            device: Target device
            
        Returns:
            Created tensor ID
        """
        gpu_machine = self._get_device_gpu_machine(device)
        return gpu_machine.factory_tensor(factory_op, args, kwargs)
    
    def remove_tensor_from_remote(self, tensor_id: str, device: "RemoteBackend") -> bool:
        """
        Remove a tensor from remote machine.
        
        Args:
            tensor_id: The tensor ID
            device: The device
            
        Returns:
            True if removed, False if not found
        """
        gpu_machine = self._get_device_gpu_machine(device)
        return gpu_machine.remove_tensor(tensor_id)
    
    def check_tensor_exists(self, tensor_id: str, device: "RemoteBackend") -> bool:
        """
        Check if a tensor exists on the remote machine.
        
        Args:
            tensor_id: The tensor ID
            device: The device
            
        Returns:
            True if tensor exists, False otherwise
        """
        try:
            gpu_machine = self._get_device_gpu_machine(device)
            metadata = gpu_machine.get_tensor_metadata(tensor_id)
            self._update_heartbeat(device.machine_id)
            return metadata is not None
        except Exception:
            return False
    
    def validate_tensor_reference(self, tensor_id: str, device: "RemoteBackend") -> bool:
        """
        Validate that a tensor reference is still valid.
        
        Args:
            tensor_id: The tensor ID to validate
            device: The device
            
        Returns:
            True if reference is valid, False otherwise
        """
        if not self.is_device_connected(device):
            return False
        
        return self.check_tensor_exists(tensor_id, device)
    
    def is_device_connected(self, device: "RemoteBackend") -> bool:
        """
        Check if a device is connected and responsive.
        
        Args:
            device: The device to check
            
        Returns:
            True if connected, False otherwise
        """
        try:
            gpu_machine = self._get_device_gpu_machine(device)
            if not gpu_machine.is_running():
                return False
            
            # Try to get registry stats as a ping
            stats = gpu_machine.get_registry_stats()
            self._update_heartbeat(device.machine_id)
            return True
        except Exception:
            return False
    
    def _update_heartbeat(self, machine_id: str) -> None:
        """Update the last successful communication timestamp for a device."""
        self._last_heartbeat[machine_id] = time.time()
    
    def get_last_heartbeat(self, machine_id: str) -> Optional[float]:
        """Get the timestamp of last successful communication with a device."""
        return self._last_heartbeat.get(machine_id)
    
    def reconnect_device(self, device: "RemoteBackend") -> bool:
        """
        Attempt to reconnect to a device.
        
        Args:
            device: The device to reconnect
            
        Returns:
            True if reconnection successful, False otherwise
        """
        try:
            gpu_machine = device.get_gpu_machine()
            if gpu_machine:
                # Stop and restart the machine
                gpu_machine.stop()
                gpu_machine.start()
                
                # Test connection
                if self.is_device_connected(device):
                    log.info(f"Successfully reconnected to device {device.machine_id}")
                    return True
                else:
                    log.warning(f"Failed to reconnected to device {device.machine_id}")
                    return False
        except Exception as e:
            log.error(f"Error during reconnection to device {device.machine_id}: {e}")
            return False
    
    @with_error_handling
    def safe_execute_remote_operation_with_ids(
        self,
        op_name: str,
        tensor_ids: List[str],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        device: "RemoteBackend"
    ) -> List[str]:
        """
        Safely execute an operation using tensor IDs with error handling.
        
        Args:
            op_name: The operation name
            tensor_ids: Input tensor IDs
            args: Operation arguments
            kwargs: Operation keyword arguments
            device: Target device
            
        Returns:
            Result tensor IDs
            
        Raises:
            StaleReferenceError: If tensor references are invalid
            ConnectionError: If device is not connected
            RemoteExecutionError: If execution fails
        """
        # Validate device connection
        if not self.is_device_connected(device):
            raise ConnectionError(f"Device {device.machine_id} is not connected")
        
        # Validate tensor references
        for tensor_id in tensor_ids:
            if not self.check_tensor_exists(tensor_id, device):
                raise StaleReferenceError(f"Tensor {tensor_id} not found on device {device.machine_id}")
        
        # Execute operation
        return self.execute_remote_operation_with_ids(op_name, tensor_ids, args, kwargs, device)
    
    def cleanup(self):
        """Clean up the remote executor."""
        # No longer needed since machines are managed by devices
        self._device_apps.clear()
        self._last_heartbeat.clear()
    
    def _remote_tensor_to_cpu(self, remote_tensor: torch.Tensor) -> torch.Tensor:
        """Convert remote tensor to CPU tensor without triggering remote execution."""
        from .device import get_device_registry
        
        # Get the device backend
        registry = get_device_registry()
        device = registry.get_device_by_index(remote_tensor.device.index)
        
        if device is None:
            raise RuntimeError(f"No RemoteBackend found for remote device index {remote_tensor.device.index}")
        
        # Get the GPU machine for this device
        gpu_machine = device.get_gpu_machine()
        if gpu_machine is None or not gpu_machine.is_running():
            raise RuntimeError(f"GPU machine not available for device {device.machine_id}")
        
        # Get tensor data using tensor ID (convert int to string for GPU machine)
        tensor_id_int = remote_tensor.untyped_storage().data_ptr()
        tensor_id_str = str(tensor_id_int)
        
        # Use GPU machine to get tensor data by ID
        tensor_data = gpu_machine.get_tensor_data(tensor_id_str)
        
        # Deserialize the tensor
        return self._deserialize_tensor(tensor_data)
    
    def _cpu_tensor_to_remote(self, cpu_tensor: torch.Tensor, device: "RemoteBackend") -> torch.Tensor:
        """Convert CPU tensor to remote tensor."""
        # Create a new remote tensor from the CPU tensor using the RemoteBackend
        if device is None:
            raise RuntimeError("RemoteBackend must be specified for remote tensor creation")
        
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
        return torch.load(buffer, map_location="cpu")
    
    def _get_tensor_metadata(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Get tensor metadata."""
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "size": tensor.numel(),
            "element_size": tensor.element_size()
        }
    
    def _detect_device_from_tensors(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[RemoteBackend]:
        """Detect RemoteBackend from tensors in arguments."""
        detected_device = None
        
        def check_tensor(tensor):
            nonlocal detected_device
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "remote":
                # Get device from registry using device index
                registry = get_device_registry()
                device = registry.get_device_by_index(tensor.device.index)
                
                if device is None:
                    raise RuntimeError(f"No RemoteBackend found for remote device index {tensor.device.index}")
                
                if detected_device is None:
                    detected_device = device
                elif detected_device is not device:
                    # Different devices found - this is not allowed
                    raise RuntimeError(
                        f"Cannot perform operations between tensors on different remote devices: "
                        f"\"{detected_device.machine_id}\" (index {detected_device.remote_index}) and "
                        f"\"{device.machine_id}\" (index {device.remote_index})\""
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

def _get_remote_executor() -> Optional[RemoteExecutor]:
    """Get the global remote executor instance."""
    return remote_executor