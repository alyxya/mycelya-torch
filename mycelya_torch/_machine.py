# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote machine management for mycelya_torch.

This module provides RemoteMachine abstraction and factory functions for creating
machines with different cloud providers and GPU types.
"""

import atexit
import uuid
from typing import Any, List, Optional

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

    # Default packages required for modal apps
    _default_packages = [
        "numpy",
        "torch",
        "huggingface_hub",
        "safetensors",
        "cloudpickle",
    ]

    @classmethod
    def _combine_packages(cls, additional_packages: List[str]) -> List[str]:
        """
        Combine default packages with additional packages, removing duplicates.

        Args:
            additional_packages: Additional packages to include

        Returns:
            Combined list of packages with duplicates removed
        """
        all_packages = cls._default_packages.copy()

        if additional_packages:
            # Extract package names (without version specifiers) for deduplication
            base_names = {
                pkg.split("==")[0]
                .split(">=")[0]
                .split("<=")[0]
                .split("~=")[0]
                .split("!=")[0]
                for pkg in cls._default_packages
            }
            for pkg in additional_packages:
                pkg_name = (
                    pkg.split("==")[0]
                    .split(">=")[0]
                    .split("<=")[0]
                    .split("~=")[0]
                    .split("!=")[0]
                )
                if pkg_name not in base_names:
                    all_packages.append(pkg)
                    base_names.add(pkg_name)

        return all_packages

    def __init__(
        self,
        provider: str,
        gpu_type: str = "",
        *,
        pip_packages: List[str] = [],  # noqa: B006
        start: bool = True,
        _batching: bool = True,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Initialize a remote machine.

        Args:
            provider: The cloud provider (e.g., "modal", "mock")
            gpu_type: The GPU type (e.g., "A100", "T4").
                     Required for modal provider, ignored for mock provider.
            pip_packages: Additional pip packages to install in the modal app.
                         These will be added to the default packages (default: [])
            start: Whether to start the client immediately (default: True)
            _batching: Whether to enable operation batching (default: True)
            timeout: Timeout in seconds for modal provider (default: None)
        """
        self.provider = provider
        self.gpu_type = gpu_type
        self.pip_packages = pip_packages

        # Combine default packages with additional packages, removing duplicates
        self.final_packages = self._combine_packages(pip_packages)

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
        short_uuid = str(uuid.uuid4())[:8]
        gpu_clean = self.gpu_type.replace("-", "").replace("_", "").lower()
        self.machine_id = f"{self.provider}-{gpu_clean}-{short_uuid}"

        self._batching = _batching
        self.timeout = timeout

        # Create and register client with orchestrator
        orchestrator.create_client(
            self.machine_id,
            self.provider,
            self.gpu_type,
            self.final_packages,
            self._batching,
            self.timeout,
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

    def stop(self) -> None:
        """Stop the client for this device."""
        try:
            orchestrator.stop_client(self.machine_id)
        except Exception as e:
            # Don't log full stack traces during shutdown
            log.warning(f"Error stopping machine {self}: {type(e).__name__}")

    def __enter__(self) -> "RemoteMachine":
        """Enter the context manager and ensure client is started."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager and clean up resources."""
        self.stop()

    def __str__(self) -> str:
        return f"RemoteMachine(provider={self.provider}, gpu={self.gpu_type}, id={self.machine_id})"

    def __repr__(self) -> str:
        return self.__str__()

    def device(self, type: str, index: Optional[int] = None) -> torch.device:
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
