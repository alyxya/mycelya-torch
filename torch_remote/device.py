# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Backend device management for torch_remote.

This module provides device abstraction for different GPU cloud providers and GPU types.
"""
import uuid
import atexit
from typing import Dict, Any, Optional, Union
from enum import Enum
import torch


class GPUType(Enum):
    """Supported GPU types across cloud providers."""
    T4 = "T4"
    L4 = "L4"
    A10G = "A10G"
    A100 = "A100"
    A100_40GB = "A100-40GB"
    A100_80GB = "A100-80GB"
    L40S = "L40S"
    H100 = "H100"
    H200 = "H200"
    B200 = "B200"


class BackendProvider(Enum):
    """Supported cloud providers."""
    MODAL = "modal"
    # Future providers can be added here
    # RUNPOD = "runpod"
    # LAMBDA = "lambda"


class RemoteBackend:
    """
    Represents a remote GPU device with specific provider and GPU type.

    Each RemoteBackend instance represents a unique remote GPU instance.
    Operations between different RemoteBackend instances are blocked with explicit error messages.
    """

    def __init__(self, provider: BackendProvider, gpu_type: GPUType, **kwargs):
        """
        Initialize a backend device.

        Args:
            provider: The cloud provider (e.g., Modal)
            gpu_type: The GPU type (e.g., A100-40GB)
            **kwargs: Additional provider-specific configuration
        """
        self.provider = provider
        self.gpu_type = gpu_type
        self.machine_id = self._generate_machine_id()
        self.config = kwargs
        self._initialized = False
        self._gpu_machine = None

        # Validate GPU type is supported by provider
        self._validate_gpu_support()
        
        # Create and start the GPU machine
        self._create_and_start_gpu_machine()

    def _generate_machine_id(self) -> str:
        """Generate a human-readable machine ID with provider and GPU info."""
        # Get short UUID for uniqueness
        short_uuid = str(uuid.uuid4())[:8]

        # Clean up GPU type for ID (remove special chars)
        gpu_clean = self.gpu_type.value.replace("-", "").replace("_", "").lower()

        # Format: provider-gpu-uuid
        return f"{self.provider.value}-{gpu_clean}-{short_uuid}"

    def _validate_gpu_support(self):
        """Validate that the GPU type is supported by the provider."""
        if self.provider == BackendProvider.MODAL:
            # Modal supports all current GPU types
            supported_gpus = set(GPUType)
            if self.gpu_type not in supported_gpus:
                raise ValueError(f"GPU type {self.gpu_type.value} not supported by {self.provider.value}")
        else:
            raise ValueError(f"Provider {self.provider.value} not implemented yet")
    
    def _create_and_start_gpu_machine(self):
        """Create and start the GPU machine for this device."""
        try:
            if self.provider == BackendProvider.MODAL:
                # Import here to avoid circular imports
                from torch_remote_execution.modal_app import create_modal_app_for_gpu
                self._gpu_machine = create_modal_app_for_gpu(self.gpu_type.value, self.machine_id)
                self._gpu_machine.start()
                print(f"Started GPU machine: {self._gpu_machine}")
            else:
                raise ValueError(f"Provider {self.provider.value} not implemented yet")
        except ImportError as e:
            print(f"Remote execution not available: {e}")
            # Continue without remote execution capability
        except Exception as e:
            print(f"Failed to start GPU machine: {e}")
            # Continue without remote execution capability
    
    def get_gpu_machine(self):
        """Get the active GPU machine for this device."""
        return self._gpu_machine
    
    def stop_gpu_machine(self):
        """Stop the GPU machine for this device."""
        if self._gpu_machine and self._gpu_machine.is_running():
            try:
                self._gpu_machine.stop()
                print(f"Stopped GPU machine: {self.machine_id}")
            except Exception as e:
                # Don't print full stack traces during shutdown
                print(f"Error stopping GPU machine {self.machine_id}: {type(e).__name__}")
        self._gpu_machine = None

    def __str__(self):
        return f"RemoteBackend(provider={self.provider.value}, gpu={self.gpu_type.value}, id={self.machine_id})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        """Two devices are equal only if they have the same machine_id."""
        if not isinstance(other, RemoteBackend):
            return False
        return self.machine_id == other.machine_id

    def __hash__(self):
        return hash(self.machine_id)
    

    @property
    def device_name(self):
        """Get a human-readable device name."""
        return f"{self.provider.value.title()} {self.gpu_type.value}"

    @property
    def modal_gpu_spec(self):
        """Get the Modal GPU specification string."""
        if self.provider != BackendProvider.MODAL:
            raise ValueError("modal_gpu_spec only available for Modal provider")
        return self.gpu_type.value

    @property
    def remote_index(self):
        """Get the device's index in the device registry."""
        registry = get_device_registry()
        return registry.get_device_index(self)
    
    def device(self):
        """
        Get a PyTorch device object for this RemoteBackend.
        
        Returns:
            torch.device: A PyTorch device object with type "remote" and the device's index
            
        Example:
            >>> backend_device = create_modal_device("A100-40GB")
            >>> torch_device = backend_device.device()
            >>> tensor = torch.randn(3, 3, device=torch_device)
        """
        return torch.device("remote", self.remote_index)



class DeviceRegistry:
    """
    Registry to manage active RemoteBackend instances.

    Maps device indices directly to RemoteBackend instances for simple lookups.
    """

    def __init__(self):
        self._devices: Dict[int, RemoteBackend] = {}  # index -> RemoteBackend
        self._next_index = 0

    def register_device(self, device: RemoteBackend) -> int:
        """
        Register a device and return its index.

        Args:
            device: The RemoteBackend to register

        Returns:
            The assigned device index
        """
        # Check if device is already registered
        for index, existing_device in self._devices.items():
            if existing_device is device:
                return index

        # Assign new index
        index = self._next_index
        self._next_index += 1

        # Store direct mapping
        self._devices[index] = device

        return index

    def get_device_by_index(self, index: int) -> Optional[RemoteBackend]:
        """Get device by its index."""
        return self._devices.get(index)

    def get_device_by_id(self, machine_id: str) -> Optional[RemoteBackend]:
        """Get device by its ID (for backwards compatibility)."""
        for device in self._devices.values():
            if device.machine_id == machine_id:
                return device
        return None

    def get_device_index(self, device: RemoteBackend) -> Optional[int]:
        """Get the index of a device."""
        for index, existing_device in self._devices.items():
            if existing_device is device:
                return index
        return None

    def devices_compatible(self, device1: RemoteBackend, device2: RemoteBackend) -> bool:
        """Check if two devices are compatible for operations."""
        # Devices are compatible if they are the same instance
        return device1 is device2

    def clear(self):
        """Clear all registered devices."""
        self._devices.clear()
        self._next_index = 0
    
    def shutdown_all_machines(self):
        """Stop all GPU machines without clearing the registry."""
        for device in self._devices.values():
            if device._gpu_machine and device._gpu_machine.is_running():
                try:
                    device._gpu_machine.stop()
                except Exception:
                    # Silently ignore errors during shutdown
                    pass


# Global device registry
_device_registry = DeviceRegistry()

# Device cleanup is handled via atexit registration (see line 288)
# Modal handles its own async context cleanup, but we still register explicit cleanup
# for proper resource management in standalone usage scenarios.


def create_modal_device(gpu: Union[str, GPUType], **kwargs) -> RemoteBackend:
    """
    Create a Modal backend device with the specified GPU type.

    Args:
        gpu: GPU type (e.g., "A100-40GB" or GPUType.A100_40GB)
        **kwargs: Additional Modal-specific configuration

    Returns:
        RemoteBackend instance for the specified GPU

    Example:
        >>> device = create_modal_device("A100-40GB")
        >>> tensor = torch.randn(3, 3, device=device)
    """
    if isinstance(gpu, str):
        try:
            gpu_type = GPUType(gpu)
        except ValueError:
            valid_gpus = [g.value for g in GPUType]
            raise ValueError(f"Invalid GPU type \"{gpu}\". Valid types: {valid_gpus}")
    else:
        gpu_type = gpu

    device = RemoteBackend(
        provider=BackendProvider.MODAL,
        gpu_type=gpu_type,
        **kwargs
    )

    # Register the device
    _device_registry.register_device(device)
    
    # Register atexit cleanup for this specific device
    atexit.register(device.stop_gpu_machine)

    return device


def get_device_registry() -> DeviceRegistry:
    """Get the global device registry."""
    return _device_registry
