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
import random
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




class RemoteGPUMachine:
    """
    Stateful wrapper representing a remote GPU machine running on Modal.
    
    This class encapsulates both the Modal app and executor, providing a clean
    interface for interacting with remote GPU execution while maintaining
    state and connection management.
    """
    
    def __init__(self, gpu_type: str, machine_id: str):
        self.gpu_type = gpu_type
        self.machine_id = machine_id
        self._app = None
        self._executor_class = None
        self._executor_instance = None
        self._app_context = None
        
        # Initialize the Modal app and executor
        self._initialize()
    
    def _initialize(self):
        """Initialize the Modal app and executor class."""
        self._app, self._executor_class = _create_modal_app_for_gpu(
            self.gpu_type, self.machine_id
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
    
    def create_tensor(self, tensor_data: bytes, storage_id: Optional[str] = None) -> str:
        """
        Create a tensor on the remote machine.
        
        Args:
            tensor_data: Serialized tensor data
            storage_id: Optional specific ID to use
            
        Returns:
            The tensor ID
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._executor_instance.create_tensor.remote(tensor_data, storage_id)
    
    def get_tensor_data(self, storage_id: str) -> bytes:
        """
        Get tensor data by ID for device transfer.
        
        Args:
            storage_id: The tensor ID
            
        Returns:
            Serialized tensor data
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._executor_instance.get_tensor_data.remote(storage_id)
    
    def get_tensor_metadata(self, storage_id: str) -> Dict[str, Any]:
        """
        Get tensor metadata by ID.
        
        Args:
            storage_id: The tensor ID
            
        Returns:
            Tensor metadata
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._executor_instance.get_tensor_metadata.remote(storage_id)
    
    def execute_operation_with_ids(
        self,
        op_name: str,
        storage_ids: List[str],
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[str]:
        """
        Execute an operation using tensor IDs.
        
        Args:
            op_name: The operation name
            storage_ids: Input tensor IDs
            args: Operation arguments
            kwargs: Operation keyword arguments
            
        Returns:
            Result tensor IDs
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._executor_instance.execute_aten_operation_with_ids.remote(
            op_name, storage_ids, args, kwargs, self.machine_id
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
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._executor_instance.factory_tensor.remote(
            factory_op, args, kwargs, self.machine_id
        )
    
    def remove_tensor(self, storage_id: str) -> bool:
        """
        Remove a tensor from the remote machine.
        
        Args:
            storage_id: The tensor ID
            
        Returns:
            True if removed, False if not found
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._executor_instance.remove_tensor.remote(storage_id)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get tensor registry statistics."""
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._executor_instance.get_registry_stats.remote()
    
    def garbage_collect(self, active_storage_ids: Optional[List[str]] = None) -> int:
        """
        Perform garbage collection on the remote machine.
        
        Args:
            active_storage_ids: Optional list of tensor IDs to keep alive
            
        Returns:
            Number of tensors removed
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._executor_instance.garbage_collect.remote(active_storage_ids)
    
    def get_memory_pressure(self) -> Dict[str, Any]:
        """Get memory pressure information from remote machine."""
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._executor_instance.get_memory_pressure.remote()
    
    def auto_garbage_collect(self) -> int:
        """Perform automatic garbage collection based on memory pressure."""
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
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
        return f"RemoteGPUMachine(gpu_type=\"{self.gpu_type}\", machine_id=\"{self.machine_id}\", status=\"{status}\")"


def create_modal_app_for_gpu(gpu_type: str, machine_id: str) -> RemoteGPUMachine:
    """
    Create a RemoteGPUMachine for a specific GPU type and device.
    
    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB")
        machine_id: The machine ID (e.g., "modal-t4-f3a7d67e")
        
    Returns:
        RemoteGPUMachine instance representing the remote GPU
    """
    # Check cache first
    if machine_id in _gpu_machines:
        return _gpu_machines[machine_id]
    
    # Create new machine and cache it
    machine = RemoteGPUMachine(gpu_type, machine_id)
    _gpu_machines[machine_id] = machine
    return machine


def _create_modal_app_for_gpu(gpu_type: str, machine_id: str) -> Tuple[modal.App, Any]:
    """
    Create a Modal app and class for a specific GPU type and device.
    
    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB")
        machine_id: The machine ID (e.g., "modal-t4-f3a7d67e")
        
    Returns:
        Tuple of (modal_app, executor_class) for the specified device
    """
    if machine_id in _gpu_apps:
        return _gpu_apps[machine_id]
    
    if gpu_type not in GPU_CONFIG:
        raise ValueError(f"GPU type \"{gpu_type}\" is not supported. Available types: {list(GPU_CONFIG.keys())}")
    
    config = GPU_CONFIG[gpu_type]
    app = modal.App(f"torch-remote-{machine_id}")
    
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
                        
                    def register_tensor(self, tensor: Any, storage_id: Optional[str] = None) -> str:
                        if storage_id is None:
                            storage_id = str(uuid.uuid4())
                            
                        with self._lock:
                            self._tensors[storage_id] = tensor
                            self._metadata[storage_id] = {
                                "shape": list(tensor.shape),
                                "dtype": str(tensor.dtype),
                                "device": str(tensor.device),
                                "numel": tensor.numel(),
                                "element_size": tensor.element_size(),
                                "requires_grad": tensor.requires_grad,
                            }
                            self._ref_counts[storage_id] += 1
                            
                        return storage_id
                    
                    def get_tensor(self, storage_id: str) -> Any:
                        with self._lock:
                            if storage_id not in self._tensors:
                                raise KeyError(f"Tensor ID {storage_id} not found in registry")
                            return self._tensors[storage_id]
                    
                    def get_metadata(self, storage_id: str) -> Dict[str, Any]:
                        with self._lock:
                            if storage_id not in self._metadata:
                                raise KeyError(f"Tensor ID {storage_id} not found in registry")
                            return self._metadata[storage_id].copy()
                    
                    def remove_tensor(self, storage_id: str) -> bool:
                        with self._lock:
                            if storage_id not in self._tensors:
                                return False
                            
                            del self._tensors[storage_id]
                            del self._metadata[storage_id]
                            if storage_id in self._ref_counts:
                                del self._ref_counts[storage_id]
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
                                "storage_ids": list(self._tensors.keys())
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
                            for storage_id, ref_count in self._ref_counts.items():
                                if ref_count <= 0:
                                    to_remove.append(storage_id)
                            
                            for storage_id in to_remove:
                                if storage_id in self._tensors:
                                    del self._tensors[storage_id]
                                if storage_id in self._metadata:
                                    del self._metadata[storage_id]
                                if storage_id in self._ref_counts:
                                    del self._ref_counts[storage_id]
                            
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
            storage_id: Optional[str] = None
        ) -> str:
            """
            Create a new tensor on the remote machine and return its ID.
            
            Args:
                tensor_data: Serialized tensor data
                storage_id: Optional specific ID to use
                
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
            storage_id = self.tensor_registry.register_tensor(tensor, storage_id)
            print(f"📦 Created tensor {storage_id} with shape {tensor.shape} on {device}")
            
            return storage_id
        
        @modal.method()
        def get_tensor_data(self, storage_id: str) -> bytes:
            """
            Retrieve tensor data by ID for transfer to client.
            
            Args:
                storage_id: The tensor ID
                
            Returns:
                Serialized tensor data
            """
            import torch
            import io
            
            tensor = self.tensor_registry.get_tensor(storage_id)
            
            # Move to CPU for serialization
            cpu_tensor = tensor.cpu()
            
            # Serialize tensor
            buffer = io.BytesIO()
            torch.save(cpu_tensor, buffer)
            
            return buffer.getvalue()
        
        @modal.method()
        def get_tensor_metadata(self, storage_id: str) -> Dict[str, Any]:
            """
            Get metadata for a tensor by ID.
            
            Args:
                storage_id: The tensor ID
                
            Returns:
                Tensor metadata
            """
            return self.tensor_registry.get_metadata(storage_id)
        
        @modal.method()
        def remove_tensor(self, storage_id: str) -> bool:
            """
            Remove a tensor from the registry.
            
            Args:
                storage_id: The tensor ID
                
            Returns:
                True if removed, False if not found
            """
            removed = self.tensor_registry.remove_tensor(storage_id)
            if removed:
                print(f"🗑️ Removed tensor {storage_id}")
            return removed
        
        @modal.method()
        def execute_aten_operation_with_ids(
            self,
            op_name: str,
            storage_ids: List[str],
            args: List[Any],
            kwargs: Dict[str, Any],
            machine_id: str
        ) -> List[str]:
            """
            Execute an operation using tensor IDs and return result tensor IDs.
            
            Args:
                op_name: The operation name to execute
                storage_ids: List of input tensor IDs
                args: Operation arguments (with tensor placeholders)
                kwargs: Operation keyword arguments (with tensor placeholders)
                machine_id: Machine ID for logging
                
            Returns:
                List of result tensor IDs
            """
            import torch
            
            print(f"🚀 Modal {gpu_type} (machine {machine_id}) executing: {op_name}")
            print(f"Using tensor IDs: {storage_ids}")
            
            try:
                # Get tensors from registry
                tensors = []
                for i, storage_id in enumerate(storage_ids):
                    tensor = self.tensor_registry.get_tensor(storage_id)
                    tensors.append(tensor)
                    # Log input tensor details on modal side
                    print(f"📥 MODAL INPUT tensor[{i}]: ID={storage_id}, shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, size={tensor.numel()}")
                    
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
                # Generate integer storage IDs like the C++ allocator does
                result_ids = []
                for i, tensor in enumerate(results):
                    # Generate integer storage ID (compatible with C++ allocator)
                    storage_id = str(random.randint(1, 2**64 - 1))
                    storage_id = self.tensor_registry.register_tensor(tensor, storage_id)
                    result_ids.append(storage_id)
                    
                    # Log output tensor details on modal side
                    print(f"📤 MODAL OUTPUT tensor[{i}]: ID={storage_id}, shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, size={tensor.numel()}")
                    
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
            machine_id: str
        ) -> str:
            """
            Create a tensor using a factory operation (e.g., torch.randn).
            
            Args:
                factory_op: The factory operation name (e.g., "randn", "zeros")
                args: Factory operation arguments
                kwargs: Factory operation keyword arguments
                machine_id: Machine ID for logging
                
            Returns:
                The created tensor ID
            """
            import torch
            
            print(f"🏭 Modal {gpu_type} (machine {machine_id}) creating tensor: {factory_op}")
            
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
                # Generate integer storage ID (compatible with C++ allocator)
                storage_id = str(random.randint(1, 2**64 - 1))
                storage_id = self.tensor_registry.register_tensor(tensor, storage_id)
                
                print(f"✅ Created {factory_op} tensor {storage_id} with shape {tensor.shape}")
                return storage_id
                
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
        def garbage_collect(self, active_storage_ids: Optional[List[str]] = None) -> int:
            """
            Perform garbage collection on the tensor registry.
            
            Args:
                active_storage_ids: Optional list of tensor IDs to keep alive
                
            Returns:
                Number of tensors removed
            """
            removed_count = self.tensor_registry.garbage_collect(active_storage_ids)
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
    
    _gpu_apps[machine_id] = (app, PytorchOperationExecutor)
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
    
    return create_modal_app_for_gpu(device.gpu_type.value, device.machine_id)


def clear_app_cache():
    """Clear the app cache."""
    global _gpu_apps
    _gpu_apps.clear()


