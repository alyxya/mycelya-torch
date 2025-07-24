# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Backend device management for torch_remote.

This module provides device abstraction for different GPU cloud providers and GPU types.
"""

import atexit
import uuid
from enum import Enum
from typing import Any, Dict, Optional, Union

import torch

from ._logging import get_logger

log = get_logger(__name__)


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


class CloudProvider(Enum):
    """Supported cloud providers."""

    MODAL = "modal"
    # Future providers can be added here
    # RUNPOD = "runpod"
    # LAMBDA = "lambda"


class RemoteMachine:
    """
    Represents a remote machine with specific provider and GPU type(s).

    Each RemoteMachine instance represents a unique remote machine instance
    that can host one or more GPUs. Operations between different RemoteMachine
    instances are blocked with explicit error messages.

    Can be used as a context manager for automatic resource cleanup:

        >>> with RemoteMachine(CloudProvider.MODAL, GPUType.T4) as machine:
        ...     x = torch.randn(100, 100, device=machine.device())
        ...     result = x @ x.T
        >>> # Machine automatically stopped when exiting context
    """

    def __init__(
        self, provider: CloudProvider, gpu_type: GPUType, start: bool = True
    ) -> None:
        """
        Initialize a backend device.

        Args:
            provider: The cloud provider (e.g., Modal)
            gpu_type: The GPU type (e.g., A100-40GB)
            start: Whether to start the client immediately (default: True)
        """
        self.provider = provider
        self.gpu_type = gpu_type
        self.machine_id = self._generate_machine_id()
        self._client = None

        # Validate GPU type is supported by provider
        self._validate_gpu_support()

        # Create the client
        self._create_client()

        # Start the client if requested
        if start:
            self.start()

        # Register cleanup on exit
        atexit.register(self.stop)

    def _generate_machine_id(self) -> str:
        """Generate a human-readable machine ID with provider and GPU info."""
        # Get short UUID for uniqueness
        short_uuid = str(uuid.uuid4())[:8]

        # Clean up GPU type for ID (remove special chars)
        gpu_clean = self.gpu_type.value.replace("-", "").replace("_", "").lower()

        # Format: provider-gpu-uuid
        return f"{self.provider.value}-{gpu_clean}-{short_uuid}"

    def _validate_gpu_support(self) -> None:
        """Validate that the GPU type is supported by the provider."""
        if self.provider == CloudProvider.MODAL:
            # Modal supports all current GPU types
            supported_gpus = set(GPUType)
            if self.gpu_type not in supported_gpus:
                raise ValueError(
                    f"GPU type {self.gpu_type.value} not supported by {self.provider.value}"
                )
        else:
            raise ValueError(f"Provider {self.provider.value} not implemented yet")

    def _create_client(self) -> None:
        """Create the client for this device."""
        try:
            if self.provider == CloudProvider.MODAL:
                # Import here to avoid circular imports
                from .backends.modal.client import create_modal_app_for_gpu

                self._client = create_modal_app_for_gpu(
                    self.gpu_type.value, self.machine_id
                )
            else:
                raise ValueError(f"Provider {self.provider.value} not implemented yet")
        except ImportError as e:
            log.warning(f"Remote execution not available: {e}")
            # Continue without remote execution capability
        except Exception as e:
            log.error(f"Failed to create client: {e}")
            # Continue without remote execution capability

    def start(self) -> None:
        """Start the client for this device."""
        if self._client is None:
            log.warning("Cannot start: client not created")
            return

        try:
            self._client.start()
            log.info(f"Started client: {self._client}")
        except Exception as e:
            log.error(f"Failed to start client: {e}")
            # Continue without remote execution capability

    def stop(self) -> None:
        """Stop the client for this device."""
        if self._client and self._client.is_running():
            try:
                self._client.stop()
                log.info(f"Stopped client: {self.machine_id}")
            except Exception as e:
                # Don't log full stack traces during shutdown
                log.warning(
                    f"Error stopping client {self.machine_id}: {type(e).__name__}"
                )
        self._client = None

    def __enter__(self) -> "RemoteMachine":
        """Enter the context manager and ensure client is started."""
        if self._client is None or not self._client.is_running():
            self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager and clean up resources."""
        self.stop()

    def __str__(self) -> str:
        return f"RemoteMachine(provider={self.provider.value}, gpu={self.gpu_type.value}, id={self.machine_id})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        """Two devices are equal only if they have the same machine_id."""
        if not isinstance(other, RemoteMachine):
            return False
        return self.machine_id == other.machine_id

    def __hash__(self) -> int:
        return hash(self.machine_id)

    @property
    def device_name(self) -> str:
        """Get a human-readable device name."""
        return f"{self.provider.value.title()} {self.gpu_type.value}"

    @property
    def modal_gpu_spec(self) -> str:
        """Get the Modal GPU specification string."""
        if self.provider != CloudProvider.MODAL:
            raise ValueError("modal_gpu_spec only available for Modal provider")
        return self.gpu_type.value

    @property
    def remote_index(self) -> Optional[int]:
        """Get the device's index in the device registry."""
        registry = get_device_registry()
        return registry.get_device_index(self)

    def device(self) -> torch.device:
        """
        Get a PyTorch device object for this RemoteMachine.

        Returns:
            torch.device: A PyTorch device object with type "remote" and the device's index

        Example:
            >>> backend_device = create_modal_machine("A100-40GB")
            >>> torch_device = backend_device.device()
            >>> tensor = torch.randn(3, 3, device=torch_device)
        """
        remote_index = self.remote_index
        if remote_index is None:
            raise RuntimeError("Device not registered in device registry")
        return torch.device("remote", remote_index)


class DeviceRegistry:
    """
    Registry to manage active RemoteMachine instances.

    Maps device indices directly to RemoteMachine instances for simple lookups.
    """

    def __init__(self) -> None:
        self._devices: Dict[int, RemoteMachine] = {}  # index -> RemoteMachine
        self._next_index = 0

    def register_device(self, machine: RemoteMachine) -> int:
        """
        Register a device and return its index.

        Args:
            machine: The RemoteMachine to register

        Returns:
            The assigned device index
        """
        # Check if device is already registered
        for index, existing_device in self._devices.items():
            if existing_device is machine:
                return index

        # Assign new index
        index = self._next_index
        self._next_index += 1

        # Store direct mapping
        self._devices[index] = machine

        return index

    def get_device_by_index(self, index: int) -> Optional[RemoteMachine]:
        """Get device by its index."""
        return self._devices.get(index)

    def get_device_index(self, machine: RemoteMachine) -> Optional[int]:
        """Get the index of a machine."""
        for index, existing_machine in self._devices.items():
            if existing_machine is machine:
                return index
        return None


# Global device registry
_device_registry = DeviceRegistry()

# Device cleanup is handled via atexit registration (see line 288)
# Modal handles its own async context cleanup, but we still register explicit cleanup
# for proper resource management in standalone usage scenarios.


def create_modal_machine(gpu: Union[str, GPUType], start: bool = True) -> RemoteMachine:
    """
    Create a Modal remote machine with the specified GPU type.

    Args:
        gpu: GPU type (e.g., "A100-40GB" or GPUType.A100_40GB)
        start: Whether to start the client immediately (default: True)

    Returns:
        RemoteMachine instance for the specified GPU

    Example:
        >>> machine = create_modal_machine("A100-40GB")
        >>> tensor = torch.randn(3, 3, device=machine.device())
        >>>
        >>> # Create without starting
        >>> machine = create_modal_machine("A100-40GB", start=False)
        >>> machine.start()  # Start manually later
    """
    if isinstance(gpu, str):
        try:
            gpu_type = GPUType(gpu)
        except ValueError:
            valid_gpus = [g.value for g in GPUType]
            raise ValueError(f'Invalid GPU type "{gpu}". Valid types: {valid_gpus}')
    else:
        gpu_type = gpu

    machine = RemoteMachine(
        provider=CloudProvider.MODAL, gpu_type=gpu_type, start=start
    )

    # Register the machine
    _device_registry.register_device(machine)

    return machine


def get_device_registry() -> DeviceRegistry:
    """Get the global device registry."""
    return _device_registry
