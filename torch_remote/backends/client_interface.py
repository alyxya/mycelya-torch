# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Abstract client interface for torch_remote cloud providers.

This module defines the base interface that all cloud provider clients must implement,
ensuring consistent API across different backends (Modal, AWS, GCP, Azure, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ClientInterface(ABC):
    """
    Abstract base class defining the interface for cloud provider clients.
    
    All cloud provider clients (ModalClient, AWSClient, etc.) must inherit from this
    class and implement all abstract methods to ensure consistent API across providers.
    """
    
    def __init__(self, gpu_type: str, machine_id: str):
        """
        Initialize the client with GPU type and machine ID.
        
        Args:
            gpu_type: The GPU type (e.g., "T4", "A100-40GB")
            machine_id: Unique machine identifier
        """
        self.gpu_type = gpu_type
        self.machine_id = machine_id
    
    @abstractmethod
    def start(self) -> None:
        """
        Start the cloud provider's compute resources.
        
        This method should initialize and start the remote GPU machine,
        making it ready to accept operations.
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """
        Stop the cloud provider's compute resources.
        
        This method should cleanly shutdown the remote GPU machine
        and release any associated resources.
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the machine is currently running and ready.
        
        Returns:
            True if the machine is running and can accept operations, False otherwise
        """
        pass
    
    @abstractmethod
    def create_storage(self, tensor_data: bytes, storage_id: Optional[str] = None) -> str:
        """
        Create a storage on the remote machine.
        
        Args:
            tensor_data: Serialized tensor data to store
            storage_id: Optional specific ID to use for the storage
            
        Returns:
            The storage ID assigned to the created storage
        """
        pass
    
    @abstractmethod
    def get_storage_data(self, storage_id: str, shape: List[int] = None, stride: List[int] = None, 
                        storage_offset: int = 0, dtype: str = None) -> bytes:
        """
        Retrieve storage data by ID, optionally as a specific view.
        
        Args:
            storage_id: The storage ID to retrieve
            shape: Tensor shape for view (if None, returns full storage)
            stride: Tensor stride for view
            storage_offset: Storage offset for view
            dtype: Tensor data type
            
        Returns:
            Serialized tensor data (contiguous representation of the view)
        """
        pass
    
    @abstractmethod
    def execute_aten_operation(
        self,
        op_name: str,
        storage_ids: List[str],
        tensor_metadata: List[Dict[str, Any]],
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[str]:
        """
        Execute an aten operation on the remote machine.
        
        Args:
            op_name: The aten operation name to execute
            storage_ids: List of input tensor storage IDs
            tensor_metadata: Metadata for reconstructing tensors (shape, stride, offset, etc.)
            args: Operation arguments
            kwargs: Operation keyword arguments
            
        Returns:
            List of result tensor storage IDs
        """
        pass
    
    @abstractmethod
    def remove_storage(self, storage_id: str) -> bool:
        """
        Remove a storage from the remote machine.
        
        Args:
            storage_id: The storage ID to remove
            
        Returns:
            True if storage was removed, False if not found
        """
        pass
    
    # Context manager methods (optional to override, but provide default behavior)
    def __enter__(self):
        """Context manager entry - starts the machine."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the machine."""
        self.stop()
    
    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation of the client.
        
        Returns:
            Human-readable string describing the client state
        """
        pass