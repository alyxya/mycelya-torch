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

from ._device import get_device_manager
from ._logging import get_logger
from ._orchestrator import orchestrator

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

        >>> with RemoteMachine("modal", "T4") as machine:
        ...     x = torch.randn(100, 100, device=machine.device())
        ...     result = x @ x.T
        >>> # Machine automatically stopped when exiting context

    Or created directly (starts automatically by default):

        >>> machine = RemoteMachine("modal", "T4")
        >>> x = torch.randn(100, 100, device=machine.device())
        >>> result = x @ x.T
    """

    # Class-level tracking of all machine instances
    _all_machines: list["RemoteMachine"] = []

    def __init__(
        self,
        provider: Union[str, CloudProvider],
        gpu_type: Union[str, GPUType, None] = None,
        start: bool = True,
        _batching: bool = True,
    ) -> None:
        """
        Initialize a remote machine.

        Args:
            provider: The cloud provider (e.g., "modal")
            gpu_type: The GPU type (e.g., "A100-40GB").
                     Required for modal provider, ignored for mock provider.
            start: Whether to start the client immediately (default: True)
            _batching: Whether to enable operation batching (default: True)
        """
        # Parse and validate provider
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

        # Parse GPU type
        if self.provider != CloudProvider.MOCK and gpu_type is None:
            raise ValueError(
                f"{self.provider.value} provider requires gpu_type to be specified"
            )

        if gpu_type is None:
            self.gpu_type = GPUType.T4  # Default for mock provider
        elif isinstance(gpu_type, str):
            try:
                self.gpu_type = GPUType(gpu_type)
            except ValueError:
                valid_gpus = [g.value for g in GPUType]
                raise ValueError(
                    f'Invalid GPU type "{gpu_type}". Valid types: {valid_gpus}'
                )
        else:
            self.gpu_type = gpu_type

        # Generate unique machine ID
        short_uuid = str(uuid.uuid4())[:8]
        gpu_clean = self.gpu_type.value.replace("-", "").replace("_", "").lower()
        self.machine_id = f"{self.provider.value}-{gpu_clean}-{short_uuid}"

        self._batching = _batching

        # Create and register client with orchestrator
        device_index = self.device().index
        orchestrator.create_client(
            self.machine_id,
            self.provider.value,
            self.gpu_type.value,
            device_index,
            self._batching,
        )

        # Start client if requested and register cleanup
        if start:
            self.start()
        atexit.register(self.stop)

        # Track all machine instances
        RemoteMachine._all_machines.append(self)

    def start(self) -> None:
        """Start the client for this device."""
        orchestrator.start_client(self.machine_id)
        log.info(f"Started machine: {self}")

    def stop(self) -> None:
        """Stop the client for this device."""
        try:
            orchestrator.stop_client(self.machine_id)
            log.info(f"Stopped machine: {self}")
        except Exception as e:
            # Don't log full stack traces during shutdown
            log.warning(f"Error stopping machine {self}: {type(e).__name__}")

    def get_client(self):
        """Get the client for this device.

        Returns:
            The client interface for this device

        Raises:
            RuntimeError: If client is not available or not running
        """
        return orchestrator.get_client(self.machine_id)

    def __enter__(self) -> "RemoteMachine":
        """Enter the context manager and ensure client is started."""
        if not orchestrator.is_client_running(self.machine_id):
            self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager and clean up resources."""
        self.stop()

    def __str__(self) -> str:
        return f"RemoteMachine(provider={self.provider.value}, gpu={self.gpu_type.value}, id={self.machine_id})"

    def __repr__(self) -> str:
        return self.__str__()

    def device(
        self, type: Optional[str] = None, index: Optional[int] = None
    ) -> torch.device:
        """Get a PyTorch device object for this RemoteMachine.

        Args:
            type: Device type ("cuda", "cpu", "mps", or "cuda:1" format).
                 Defaults: modal="cuda", mock="cpu".
            index: Device index (default: 0). Cannot be used with "type:index" format.

        Returns:
            torch.device with type "mycelya" and mapped index.
        """
        # Parse "type:index" format
        if type and ":" in type:
            if index is not None:
                raise ValueError(
                    f"Cannot specify both index ({index}) and type:index format ('{type}')"
                )
            type, index = type.split(":", 1)
            index = int(index)

        # Apply defaults
        type = type or ("cpu" if self.provider == CloudProvider.MOCK else "cuda")
        index = index or 0

        # Validate device type for provider
        valid_types = (
            ["cpu", "mps"] if self.provider == CloudProvider.MOCK else ["cuda", "cpu"]
        )
        if type not in valid_types:
            raise ValueError(
                f"{self.provider.value} provider only supports {valid_types}, got '{type}'"
            )

        return get_device_manager().get_device(self.machine_id, type=type, index=index)


def get_all_machines() -> list[RemoteMachine]:
    """
    Get a list of all created machines.

    Returns:
        List of all RemoteMachine instances that have been created.
        This maintains strong references to keep machines alive.

    Example:
        >>> machine1 = RemoteMachine("modal", "T4")
        >>> machine2 = RemoteMachine("mock")
        >>> machines = get_all_machines()
        >>> print(f"Created {len(machines)} machines")
        Created 2 machines
    """
    return list(RemoteMachine._all_machines)
