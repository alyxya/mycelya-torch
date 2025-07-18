# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal client implementation for torch_remote.

This module provides the ModalClient class for interfacing with Modal cloud GPUs,
along with related functionality for creating and managing Modal applications.
"""

from typing import Any, Dict, List, Optional
import logging

from ..client_interface import ClientInterface

log = logging.getLogger(__name__)

# Cache for ModalClient instances
_gpu_machines: Dict[str, "ModalClient"] = {}


class ModalClient(ClientInterface):
    """
    Client interface for Modal cloud GPU execution.
    
    This class provides a client-side interface to Modal's cloud GPU infrastructure,
    encapsulating Modal app management, server instances, and communication
    protocols while maintaining state and connection management.
    """
    
    def __init__(self, gpu_type: str, machine_id: str):
        super().__init__(gpu_type, machine_id)
        self._app = None
        self._server_class = None
        self._server_instance = None
        self._app_context = None
        
        # Initialize the Modal app and server
        self._initialize()
    
    def _initialize(self):
        """Initialize the Modal app and server class."""
        # Import here to avoid circular imports
        from torch_remote_modal.modal_app import create_modal_app_for_gpu
        self._app, self._server_class = create_modal_app_for_gpu(
            self.gpu_type, self.machine_id
        )
    
    def start(self):
        """Start the Modal app context for this machine."""
        if self._app_context is None:
            self._app_context = self._app.run()
            self._app_context.__enter__()
            # Create server instance when app starts
            self._server_instance = self._server_class()
    
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
                self._server_instance = None
    
    def is_running(self) -> bool:
        """Check if the machine is currently running."""
        return self._app_context is not None
    
    def create_storage(self, tensor_data: bytes, storage_id: Optional[str] = None) -> str:
        """
        Create a storage on the remote machine.
        
        Args:
            tensor_data: Serialized tensor data
            storage_id: Optional specific ID to use
            
        Returns:
            The storage ID
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._server_instance.create_storage.remote(tensor_data, storage_id)
    
    def get_storage_data(self, storage_id: str, shape: List[int] = None, stride: List[int] = None, 
                        storage_offset: int = 0, dtype: str = None) -> bytes:
        """
        Get storage data by ID for device transfer, optionally as a specific view.
        
        Args:
            storage_id: The storage ID
            shape: Tensor shape for view (if None, returns full storage)
            stride: Tensor stride for view  
            storage_offset: Storage offset for view
            dtype: Tensor data type
            
        Returns:
            Serialized tensor data (contiguous representation of the view)
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._server_instance.get_storage_data.remote(storage_id, shape, stride, storage_offset, dtype)
    
    
    def execute_aten_operation(
        self,
        op_name: str,
        storage_ids: List[str],
        tensor_metadata: List[Dict[str, Any]],
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[str]:
        """
        Execute an aten operation using tensor IDs and metadata.
        
        Args:
            op_name: The aten operation name
            storage_ids: Input tensor storage IDs
            tensor_metadata: Metadata for reconstructing tensors (shape, stride, offset, storage_id)
            args: Operation arguments
            kwargs: Operation keyword arguments
            
        Returns:
            Result tensor IDs
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._server_instance.execute_aten_operation.remote(
            op_name, storage_ids, tensor_metadata, args, kwargs, self.machine_id
        )
    
    def remove_storage(self, storage_id: str) -> bool:
        """
        Remove a storage from the remote machine.
        
        Args:
            storage_id: The storage ID
            
        Returns:
            True if removed, False if not found
        """
        if not self.is_running():
            raise RuntimeError(f"Machine {self.machine_id} is not running. Call start() first.")
        
        return self._server_instance.remove_storage.remote(storage_id)
    
    
    
    def __enter__(self):
        """Context manager entry - starts the machine."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the machine."""
        self.stop()
    
    
    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return f"ModalClient(gpu_type=\"{self.gpu_type}\", machine_id=\"{self.machine_id}\", status=\"{status}\")"


def create_modal_app_for_gpu(gpu_type: str, machine_id: str) -> ModalClient:
    """
    Create a ModalClient for a specific GPU type and machine.
    
    Args:
        gpu_type: The GPU type (e.g., "T4", "A100-40GB")
        machine_id: The machine ID (e.g., "modal-t4-f3a7d67e")
        
    Returns:
        ModalClient instance for communicating with Modal GPU infrastructure
    """
    # Check cache first
    if machine_id in _gpu_machines:
        return _gpu_machines[machine_id]
    
    # Create new client and cache it
    client = ModalClient(gpu_type, machine_id)
    _gpu_machines[machine_id] = client
    return client


# PytorchServer and app creation logic is in torch_remote_modal.modal_app


def get_modal_app_for_device(device) -> ModalClient:
    """
    Get the ModalClient for a specific machine.
    
    Args:
        device: The RemoteMachine to get the client for
        
    Returns:
        ModalClient for the machine's GPU type
    """
    if hasattr(device, "provider") and device.provider.value != "modal":
        raise ValueError(f"Device provider {device.provider.value} is not Modal")
    
    return create_modal_app_for_gpu(device.gpu_type.value, device.machine_id)


def clear_app_cache():
    """Clear the app cache."""
    global _gpu_machines
    _gpu_machines.clear()