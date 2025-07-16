# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

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
import uuid
import weakref
import threading
from collections import defaultdict

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
    "A100": {"timeout": 450, "retries": 2},
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
            tensor = torch.load(buffer, map_location="cpu", weights_only=True)
            
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
        op_parts = op_name_fixed.split(".")
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
                "shape": list(cpu_tensor.shape),
                "dtype": str(cpu_tensor.dtype),
                "size": cpu_tensor.numel(),
                "element_size": cpu_tensor.element_size()
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
            try:
                self._app_context.__exit__(None, None, None)
            except Exception:
                # Silently ignore cleanup errors during atexit
                pass
            finally:
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
        Execute an operation on this remote GPU machine (legacy method).
        
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
    
    def create_tensor(self, tensor_data: bytes, tensor_id: Optional[str] = None) -> str:
        """
        Create a tensor on the remote machine.
        
        Args:
            tensor_data: Serialized tensor data
            tensor_id: Optional specific ID to use
            
        Returns:
            The tensor ID
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.device_id} is not running. Call start() first.")
        
        return self._executor_instance.create_tensor.remote(tensor_data, tensor_id)
    
    def get_tensor_data(self, tensor_id: str) -> bytes:
        """
        Get tensor data by ID for device transfer.
        
        Args:
            tensor_id: The tensor ID
            
        Returns:
            Serialized tensor data
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.device_id} is not running. Call start() first.")
        
        return self._executor_instance.get_tensor_data.remote(tensor_id)
    
    def get_tensor_metadata(self, tensor_id: str) -> Dict[str, Any]:
        """
        Get tensor metadata by ID.
        
        Args:
            tensor_id: The tensor ID
            
        Returns:
            Tensor metadata
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.device_id} is not running. Call start() first.")
        
        return self._executor_instance.get_tensor_metadata.remote(tensor_id)
    
    def execute_operation_with_ids(
        self,
        op_name: str,
        tensor_ids: List[str],
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[str]:
        """
        Execute an operation using tensor IDs.
        
        Args:
            op_name: The operation name
            tensor_ids: Input tensor IDs
            args: Operation arguments
            kwargs: Operation keyword arguments
            
        Returns:
            Result tensor IDs
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.device_id} is not running. Call start() first.")
        
        return self._executor_instance.execute_aten_operation_with_ids.remote(
            op_name, tensor_ids, args, kwargs, self.device_id
        )
    
    def factory_tensor(
        self,
        factory_op: str,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> str:
        """
        Create a tensor using a factory operation.
        
        Args:
            factory_op: Factory operation name (e.g., "randn")
            args: Factory arguments
            kwargs: Factory keyword arguments
            
        Returns:
            Created tensor ID
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.device_id} is not running. Call start() first.")
        
        return self._executor_instance.factory_tensor.remote(
            factory_op, args, kwargs, self.device_id
        )
    
    def remove_tensor(self, tensor_id: str) -> bool:
        """
        Remove a tensor from the remote machine.
        
        Args:
            tensor_id: The tensor ID
            
        Returns:
            True if removed, False if not found
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.device_id} is not running. Call start() first.")
        
        return self._executor_instance.remove_tensor.remote(tensor_id)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get tensor registry statistics."""
        if not self.is_running():
            raise RuntimeError(f"Machine {self.device_id} is not running. Call start() first.")
        
        return self._executor_instance.get_registry_stats.remote()
    
    def garbage_collect(self, active_tensor_ids: Optional[List[str]] = None) -> int:
        """
        Perform garbage collection on the remote machine.
        
        Args:
            active_tensor_ids: Optional list of tensor IDs to keep alive
            
        Returns:
            Number of tensors removed
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.device_id} is not running. Call start() first.")
        
        return self._executor_instance.garbage_collect.remote(active_tensor_ids)
    
    def get_memory_pressure(self) -> Dict[str, Any]:
        """Get memory pressure information from remote machine."""
        if not self.is_running():
            raise RuntimeError(f"Machine {self.device_id} is not running. Call start() first.")
        
        return self._executor_instance.get_memory_pressure.remote()
    
    def auto_garbage_collect(self) -> int:
        """Perform automatic garbage collection based on memory pressure."""
        if not self.is_running():
            raise RuntimeError(f"Machine {self.device_id} is not running. Call start() first.")
        
        return self._executor_instance.auto_garbage_collect.remote()
    
    def __enter__(self):
        """Context manager entry - starts the machine."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the machine."""
        self.stop()
    
    
    def __repr__(self):
        status = "running" if self.is_running() else "stopped"
        return f"RemoteGPUMachine(gpu_type=\"{self.gpu_type}\", device_id=\"{self.device_id}\", status=\"{status}\")"


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
        raise ValueError(f"GPU type \"{gpu_type}\" is not supported. Available types: {list(GPU_CONFIG.keys())}")
    
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
        
        def _get_tensor_registry(self):
            """Get or create tensor registry for this executor instance."""
            if not hasattr(self, "_tensor_registry"):
                # Define TensorRegistry class locally to avoid import issues
                import threading
                from collections import defaultdict
                import uuid
                from typing import Dict, List, Optional, Any
                
                class TensorRegistry:
                    """Server-side registry for managing persistent tensor storage with unique IDs."""
                    
                    def __init__(self):
                        self._tensors: Dict[str, Any] = {}
                        self._metadata: Dict[str, Dict[str, Any]] = {}
                        self._ref_counts: Dict[str, int] = defaultdict(int)
                        self._lock = threading.RLock()
                        
                    def register_tensor(self, tensor: Any, tensor_id: Optional[str] = None) -> str:
                        if tensor_id is None:
                            tensor_id = str(uuid.uuid4())
                            
                        with self._lock:
                            self._tensors[tensor_id] = tensor
                            self._metadata[tensor_id] = {
                                "shape": list(tensor.shape),
                                "dtype": str(tensor.dtype),
                                "device": str(tensor.device),
                                "numel": tensor.numel(),
                                "element_size": tensor.element_size(),
                                "requires_grad": tensor.requires_grad,
                            }
                            self._ref_counts[tensor_id] += 1
                            
                        return tensor_id
                    
                    def get_tensor(self, tensor_id: str) -> Any:
                        with self._lock:
                            if tensor_id not in self._tensors:
                                raise KeyError(f"Tensor ID {tensor_id} not found in registry")
                            return self._tensors[tensor_id]
                    
                    def get_metadata(self, tensor_id: str) -> Dict[str, Any]:
                        with self._lock:
                            if tensor_id not in self._metadata:
                                raise KeyError(f"Tensor ID {tensor_id} not found in registry")
                            return self._metadata[tensor_id].copy()
                    
                    def remove_tensor(self, tensor_id: str) -> bool:
                        with self._lock:
                            if tensor_id not in self._tensors:
                                return False
                            
                            del self._tensors[tensor_id]
                            del self._metadata[tensor_id]
                            if tensor_id in self._ref_counts:
                                del self._ref_counts[tensor_id]
                            return True
                    
                    def get_stats(self) -> Dict[str, Any]:
                        with self._lock:
                            total_tensors = len(self._tensors)
                            total_memory = 0
                            for tensor in self._tensors.values():
                                if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                                    total_memory += tensor.numel() * tensor.element_size()
                            
                            return {
                                "total_tensors": total_tensors,
                                "total_memory_bytes": total_memory,
                                "tensor_ids": list(self._tensors.keys())
                            }
                    
                    def clear(self):
                        with self._lock:
                            self._tensors.clear()
                            self._metadata.clear()
                            self._ref_counts.clear()
                    
                    def garbage_collect(self) -> int:
                        """Remove tensors with zero reference counts."""
                        with self._lock:
                            to_remove = []
                            for tensor_id, ref_count in self._ref_counts.items():
                                if ref_count <= 0:
                                    to_remove.append(tensor_id)
                            
                            for tensor_id in to_remove:
                                if tensor_id in self._tensors:
                                    del self._tensors[tensor_id]
                                if tensor_id in self._metadata:
                                    del self._metadata[tensor_id]
                                if tensor_id in self._ref_counts:
                                    del self._ref_counts[tensor_id]
                            
                            return len(to_remove)
                
                self._tensor_registry = TensorRegistry()
            
            return self._tensor_registry
        
        @property
        def tensor_registry(self):
            """Get the tensor registry (lazy initialization)."""
            return self._get_tensor_registry()
        
        @modal.method()
        def create_tensor(
            self,
            tensor_data: bytes,
            tensor_id: Optional[str] = None
        ) -> str:
            """
            Create a new tensor on the remote machine and return its ID.
            
            Args:
                tensor_data: Serialized tensor data
                tensor_id: Optional specific ID to use
                
            Returns:
                The tensor ID
            """
            import torch
            import io
            
            # Deserialize tensor
            buffer = io.BytesIO(tensor_data)
            tensor = torch.load(buffer, map_location="cpu", weights_only=True)
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = tensor.to(device)
            
            # Register in tensor registry
            tensor_id = self.tensor_registry.register_tensor(tensor, tensor_id)
            print(f"📦 Created tensor {tensor_id} with shape {tensor.shape} on {device}")
            
            return tensor_id
        
        @modal.method()
        def get_tensor_data(self, tensor_id: str) -> bytes:
            """
            Retrieve tensor data by ID for transfer to client.
            
            Args:
                tensor_id: The tensor ID
                
            Returns:
                Serialized tensor data
            """
            import torch
            import io
            
            tensor = self.tensor_registry.get_tensor(tensor_id)
            
            # Move to CPU for serialization
            cpu_tensor = tensor.cpu()
            
            # Serialize tensor
            buffer = io.BytesIO()
            torch.save(cpu_tensor, buffer)
            
            return buffer.getvalue()
        
        @modal.method()
        def get_tensor_metadata(self, tensor_id: str) -> Dict[str, Any]:
            """
            Get metadata for a tensor by ID.
            
            Args:
                tensor_id: The tensor ID
                
            Returns:
                Tensor metadata
            """
            return self.tensor_registry.get_metadata(tensor_id)
        
        @modal.method()
        def remove_tensor(self, tensor_id: str) -> bool:
            """
            Remove a tensor from the registry.
            
            Args:
                tensor_id: The tensor ID
                
            Returns:
                True if removed, False if not found
            """
            removed = self.tensor_registry.remove_tensor(tensor_id)
            if removed:
                print(f"🗑️ Removed tensor {tensor_id}")
            return removed
        
        @modal.method()
        def execute_aten_operation_with_ids(
            self,
            op_name: str,
            tensor_ids: List[str],
            args: List[Any],
            kwargs: Dict[str, Any],
            device_id: str
        ) -> List[str]:
            """
            Execute an operation using tensor IDs and return result tensor IDs.
            
            Args:
                op_name: The operation name to execute
                tensor_ids: List of input tensor IDs
                args: Operation arguments (with tensor placeholders)
                kwargs: Operation keyword arguments (with tensor placeholders)
                device_id: Device ID for logging
                
            Returns:
                List of result tensor IDs
            """
            import torch
            
            print(f"🚀 Modal {gpu_type} (device {device_id}) executing: {op_name}")
            print(f"Using tensor IDs: {tensor_ids}")
            
            try:
                # Get tensors from registry
                tensors = []
                for i, tensor_id in enumerate(tensor_ids):
                    tensor = self.tensor_registry.get_tensor(tensor_id)
                    tensors.append(tensor)
                    # Log input tensor details on modal side
                    print(f"📥 MODAL INPUT tensor[{i}]: ID={tensor_id}, shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, size={tensor.numel()}")
                    
                    # Log tensor data summary for debugging
                    if tensor.numel() > 0:
                        print(f"   Data range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}], mean={tensor.mean().item():.6f}")
                
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
                op_name_fixed = op_name.replace("::", ".")
                op_parts = op_name_fixed.split(".")
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
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    results = [torch.tensor(result, device=device)]
                
                # Register result tensors and return their IDs
                result_ids = []
                for i, tensor in enumerate(results):
                    tensor_id = self.tensor_registry.register_tensor(tensor)
                    result_ids.append(tensor_id)
                    
                    # Log output tensor details on modal side
                    print(f"📤 MODAL OUTPUT tensor[{i}]: ID={tensor_id}, shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, size={tensor.numel()}")
                    
                    # Log tensor data summary for debugging
                    if tensor.numel() > 0:
                        print(f"   Data range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}], mean={tensor.mean().item():.6f}")
                
                return result_ids
                
            except Exception as e:
                print(f"❌ Error executing {op_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        
        @modal.method()
        def factory_tensor(
            self,
            factory_op: str,
            args: List[Any],
            kwargs: Dict[str, Any],
            device_id: str
        ) -> str:
            """
            Create a tensor using a factory operation (e.g., torch.randn).
            
            Args:
                factory_op: The factory operation name (e.g., "randn", "zeros")
                args: Factory operation arguments
                kwargs: Factory operation keyword arguments
                device_id: Device ID for logging
                
            Returns:
                The created tensor ID
            """
            import torch
            
            print(f"🏭 Modal {gpu_type} (device {device_id}) creating tensor: {factory_op}")
            
            try:
                # Get GPU device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Update kwargs to use local device
                if "device" in kwargs:
                    kwargs["device"] = device
                
                # Get the factory function
                factory_func = getattr(torch, factory_op)
                
                # Create tensor
                tensor = factory_func(*args, **kwargs)
                
                # Register tensor and return ID
                tensor_id = self.tensor_registry.register_tensor(tensor)
                
                print(f"✅ Created {factory_op} tensor {tensor_id} with shape {tensor.shape}")
                return tensor_id
                
            except Exception as e:
                print(f"❌ Error creating {factory_op} tensor: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        
        @modal.method()
        def get_registry_stats(self) -> Dict[str, Any]:
            """Get tensor registry statistics."""
            return self.tensor_registry.get_stats()
        
        @modal.method()
        def garbage_collect(self, active_tensor_ids: Optional[List[str]] = None) -> int:
            """
            Perform garbage collection on the tensor registry.
            
            Args:
                active_tensor_ids: Optional list of tensor IDs to keep alive
                
            Returns:
                Number of tensors removed
            """
            removed_count = self.tensor_registry.garbage_collect(active_tensor_ids)
            if removed_count > 0:
                print(f"🗑️ Garbage collected {removed_count} tensors")
            return removed_count
        
        @modal.method()
        def get_memory_pressure(self) -> Dict[str, Any]:
            """Get memory pressure information."""
            return self.tensor_registry.get_memory_pressure()
        
        @modal.method()
        def auto_garbage_collect(self) -> int:
            """
            Perform automatic garbage collection.
            
            Returns:
                Number of tensors removed
            """
            # Get registry stats to check for cleanup opportunities
            stats = self.tensor_registry.get_stats()
            
            removed_count = 0
            
            # Perform garbage collection to clean up unreferenced tensors
            removed_count = self.tensor_registry.garbage_collect()
            
            if removed_count > 0:
                print(f"🧹 Auto-GC removed {removed_count} tensors")
            
            return removed_count
        
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
                    tensor = torch.load(buffer, map_location="cpu", weights_only=True)
                    
                    # Move to GPU
                    tensor = tensor.to(device)
                    tensors.append(tensor)
                    
                    # Log input tensor details on modal side (legacy method)
                    print(f"📥 MODAL INPUT tensor[{i}]: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, size={tensor.numel()}")
                    
                    # Log tensor data summary for debugging
                    if tensor.numel() > 0:
                        print(f"   Data range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}], mean={tensor.mean().item():.6f}")
                
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
                op_parts = op_name_fixed.split(".")
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
                    
                    # Log output tensor details on modal side (legacy method)
                    print(f"📤 MODAL OUTPUT tensor[{i}]: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, size={tensor.numel()}")
                    
                    # Log tensor data summary for debugging
                    if tensor.numel() > 0:
                        print(f"   Data range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}], mean={tensor.mean().item():.6f}")
                    
                    # Serialize tensor
                    buffer = io.BytesIO()
                    torch.save(cpu_tensor, buffer)
                    serialized_results.append(buffer.getvalue())
                    
                    # Store metadata
                    metadata = {
                        "shape": list(cpu_tensor.shape),
                        "dtype": str(cpu_tensor.dtype),
                        "size": cpu_tensor.numel(),
                        "element_size": cpu_tensor.element_size()
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
        device: The RemoteBackend to get the machine for
        
    Returns:
        RemoteGPUMachine for the device's GPU type
    """
    if hasattr(device, "provider") and device.provider.value != "modal":
        raise ValueError(f"Device provider {device.provider.value} is not Modal")
    
    return create_modal_app_for_gpu(device.gpu_type.value, device.device_id)


def clear_app_cache():
    """Clear the app cache."""
    global _gpu_apps
    _gpu_apps.clear()


