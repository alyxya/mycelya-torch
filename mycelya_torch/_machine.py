# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote machine management for mycelya_torch.

This module provides RemoteMachine abstraction and factory functions for creating
machines with different cloud providers and GPU types.
"""

import atexit
import uuid
from enum import Enum
from typing import Any, Optional, Union

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
    MOCK = "mock"
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
        self,
        provider: Union[str, CloudProvider],
        gpu_type: Union[str, GPUType, None] = None,
        timeout: int = 300,
        retries: int = None,
        start: bool = True,
    ) -> None:
        """
        Initialize a remote machine.

        Args:
            provider: The cloud provider (e.g., "modal" or CloudProvider.MODAL)
            gpu_type: The GPU type (e.g., "A100-40GB" or GPUType.A100_40GB).
                     Required for modal provider, ignored for mock provider.
            timeout: Function timeout in seconds (default: 300)
            retries: Number of retries on failure (default: 1 for modal, 0 for mock)
            start: Whether to start the client immediately (default: True)
        """
        # Handle string providers
        if isinstance(provider, str):
            try:
                self.provider = CloudProvider(provider)
            except ValueError:
                valid_providers = [p.value for p in CloudProvider]
                raise ValueError(
                    f'Invalid provider "{provider}". Valid providers: {valid_providers}'
                )
        else:
            self.provider = provider

        # Handle GPU type validation based on provider
        if self.provider == CloudProvider.MODAL:
            # Modal provider requires GPU type
            if gpu_type is None:
                raise ValueError("Modal provider requires gpu_type to be specified")

            if isinstance(gpu_type, str):
                try:
                    self.gpu_type = GPUType(gpu_type)
                except ValueError:
                    valid_gpus = [g.value for g in GPUType]
                    raise ValueError(
                        f'Invalid GPU type "{gpu_type}". Valid types: {valid_gpus}'
                    )
            else:
                self.gpu_type = gpu_type
        else:
            # Mock provider doesn't need GPU type, use a default
            self.gpu_type = (
                GPUType.T4
                if gpu_type is None
                else (GPUType(gpu_type) if isinstance(gpu_type, str) else gpu_type)
            )
        self.timeout = timeout

        # Set default retries based on provider
        if retries is None:
            self.retries = 0 if self.provider == CloudProvider.MOCK else 1
        else:
            self.retries = retries
        self.machine_id = self._generate_machine_id()
        self._client = None

        # Validate GPU type is supported by provider
        self._validate_gpu_support()

        # Create the client
        self._create_client()

        # Start the client if requested
        if start:
            self.start()

        # Register with device registry
        from ._device import get_device_registry

        registry = get_device_registry()
        registry.register_device(self)

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
        elif self.provider == CloudProvider.MOCK:
            # Mock provider supports all GPU types (simulated locally)
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
                from .backends.modal.client import ModalClient

                self._client = ModalClient(
                    self.gpu_type.value,
                    self.machine_id,
                    self.timeout,
                    self.retries,
                )
            elif self.provider == CloudProvider.MOCK:
                # Import here to avoid circular imports
                from .backends.mock.client import MockClient

                self._client = MockClient(
                    self.gpu_type.value,
                    self.machine_id,
                    self.timeout,
                    self.retries,
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

    def get_client(self):
        """Get the client for this device.

        Returns:
            The client interface for this device

        Raises:
            RuntimeError: If client is not available or not running
        """
        if self._client is None:
            raise RuntimeError(f"No client available for machine {self.machine_id}")
        if not self._client.is_running():
            raise RuntimeError(f"Client for machine {self.machine_id} is not running")
        return self._client

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
        from ._device import get_device_registry

        registry = get_device_registry()
        return registry.get_device_index(self)

    def device(self) -> torch.device:
        """
        Get a PyTorch device object for this RemoteMachine.

        Returns:
            torch.device: A PyTorch device object with type "mycelya" and the device's index

        Example:
            >>> # Modal machine (GPU required)
            >>> machine = RemoteMachine("modal", "A100-40GB")
            >>> torch_device = machine.device()
            >>> tensor = torch.randn(3, 3, device=torch_device)
            >>>
            >>> # Mock machine (GPU optional)
            >>> machine = RemoteMachine("mock")
            >>> torch_device = machine.device()
        """
        remote_index = self.remote_index
        if remote_index is None:
            raise RuntimeError("Device not registered in device registry")
        return torch.device("mycelya", remote_index)
