# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote machine management for mycelya_torch.

This module provides RemoteMachine abstraction and factory functions for creating
machines with different cloud providers and GPU types.
"""

import atexit
import uuid
from typing import Any

import torch

from ._device import device_manager
from ._logging import get_logger
from ._orchestrator import orchestrator

log = get_logger(__name__)


class RemoteMachine:
    """
    Represents a remote machine with specific provider and GPU type(s).

    Each RemoteMachine instance represents a unique remote machine instance
    that can host one or more GPUs. Operations between different RemoteMachine
    instances are blocked with explicit error messages.

    Can be used as a context manager for automatic resource cleanup:

        >>> with RemoteMachine("modal", "T4") as machine:
        ...     x = torch.randn(100, 100, device=machine.device("cuda"))
        ...     result = x @ x.T
        >>> # Machine automatically stopped when exiting context

    Or created directly (starts automatically by default):

        >>> machine = RemoteMachine("modal", "T4")
        >>> x = torch.randn(100, 100, device=machine.device("cuda"))
        >>> result = x @ x.T
    """

    # Class-level tracking of all machine instances
    _all_machines: list["RemoteMachine"] = []

    def __init__(
        self,
        provider: str,
        gpu_type: str = "",
        *,
        gpu_count: int = 1,
        pip_packages: list[str] | None = None,
        idle_timeout: int | None = None,
        modal_timeout: int | None = 3600,
        _start: bool = True,
        _batching: bool = True,
    ) -> None:
        """
        Initialize a remote machine.

        Args:
            provider: The cloud provider (e.g., "modal", "mock")
            gpu_type: The GPU type (e.g., "A100", "T4").
                     Required for modal provider, ignored for mock provider.
            gpu_count: Number of GPUs (1-8, default: 1). Ignored for mock provider.
            pip_packages: Additional pip packages to install in the modal app.
                         These will be added to the default packages (default: None)
            idle_timeout: Number of seconds of inactivity before machine pauses (default: None, no timeout)
            modal_timeout: Timeout in seconds for modal provider (default: 3600)
            _start: Whether to start the client immediately (default: True)
            _batching: Whether to enable operation batching (default: True)
        """
        self._provider = provider

        # Validate gpu_count
        if not isinstance(gpu_count, int) or gpu_count < 1 or gpu_count > 8:
            raise ValueError(
                f"gpu_count must be an integer between 1 and 8, got {gpu_count}"
            )

        # Handle GPU type based on provider
        if provider == "modal":
            # Validate GPU type for modal
            valid_gpu_types = [
                "T4",
                "L4",
                "A10G",
                "A100",
                "A100-40GB",
                "A100-80GB",
                "L40S",
                "H100",
                "H200",
                "B200",
            ]
            if not gpu_type:
                raise ValueError(
                    f"Missing GPU type for modal provider. "
                    f"Valid types: {valid_gpu_types}"
                )
            elif gpu_type not in valid_gpu_types:
                raise ValueError(
                    f"Invalid GPU type '{gpu_type}' for modal provider. "
                    f"Valid types: {valid_gpu_types}"
                )
        elif provider == "mock":
            if gpu_type:
                log.warning(
                    f"GPU type '{gpu_type}' provided for mock provider but will be ignored"
                )
        else:
            raise ValueError(
                f"Unsupported provider '{provider}'. Supported providers: modal"
            )

        # Generate unique machine ID
        machine_id = str(uuid.uuid4())[:8]

        # Create and register client with orchestrator
        self._client_manager = orchestrator.create_client(
            machine_id,
            provider,
            gpu_type,
            gpu_count,
            _batching,
            idle_timeout,
            modal_timeout,
        )

        # Install additional pip packages if specified
        if pip_packages:
            self._client_manager.pip_install(pip_packages)

        # Start client if requested and register cleanup
        if _start:
            self.start()
        atexit.register(self.force_stop)

        # Track all machine instances
        RemoteMachine._all_machines.append(self)

    @property
    def provider(self) -> str:
        """Get the provider string."""
        return self._provider

    @property
    def gpu_type(self) -> str:
        """Get the GPU type from the client manager."""
        return self._client_manager.gpu_type

    @property
    def gpu_count(self) -> int:
        """Get the GPU count from the client manager."""
        return self._client_manager.gpu_count

    @property
    def machine_id(self) -> str:
        """Get the machine ID from the client manager."""
        return self._client_manager.machine_id

    @property
    def packages(self) -> list[str]:
        """Get the packages list from the client manager."""
        return self._client_manager.packages

    @property
    def batching(self) -> bool:
        """Get the batching setting from the client manager."""
        return self._client_manager.batching

    def start(self) -> None:
        """Start the cloud provider's compute resources."""
        self._client_manager.start()

    def stop(self) -> None:
        """Stop the cloud provider's compute resources."""
        self._client_manager.stop()

    def force_stop(self) -> None:
        """Force immediate stop of the machine, clearing all pending operations.

        This method bypasses normal graceful shutdown and immediately stops
        the machine, clearing all pending operations. It is useful for
        emergency shutdown scenarios like atexit handlers.
        """
        self._client_manager.force_stop()

    def pause(self) -> None:
        """Pause the machine (offload state and stop compute resources)."""
        self._client_manager.pause()

    def resume(self) -> None:
        """Resume the machine from paused state (start and reload state)."""
        self._client_manager.resume()

    def __enter__(self) -> "RemoteMachine":
        """Enter the context manager and ensure client is started."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager and clean up resources."""
        self.stop()

    def __str__(self) -> str:
        return f"RemoteMachine(provider={self.provider}, gpu={self.gpu_type}, count={self.gpu_count}, id={self.machine_id})"

    def __repr__(self) -> str:
        return self.__str__()

    def pip_install(self, packages: str | list[str]) -> "RemoteMachine":
        """Install packages using pip on the remote machine.

        Args:
            packages: Package name(s) to install. Can be a single string or list of strings.
                     Examples: "numpy", ["numpy", "scipy"], "torch==2.0.0"

        Returns:
            Self for method chaining.
        """
        # Convert single string to list
        if isinstance(packages, str):
            packages = [packages]

        # Delegate to client manager
        self._client_manager.pip_install(packages)
        return self

    def device(self, type: str, index: int | None = None) -> torch.device:
        """Get a PyTorch device object for this RemoteMachine.

        Args:
            type: Device type ("cuda", "cpu", "mps", or "cuda:1" format). Required.
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

        # Default index if not specified
        index = index or 0

        # Validate device type for provider
        valid_types = ["cpu", "mps"] if self.provider == "mock" else ["cuda", "cpu"]
        if type not in valid_types:
            raise ValueError(
                f"{self.provider} provider only supports {valid_types}, got '{type}'"
            )

        return device_manager.get_mycelya_device(
            self.machine_id, type=type, index=index
        )


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
