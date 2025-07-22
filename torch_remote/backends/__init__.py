# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution backends for torch_remote.

This module contains backend implementations for different cloud GPU providers.
Currently supports:
- Modal (torch_remote.backends.modal)

Future backends may include:
- RunPod (torch_remote.backends.runpod)
- Lambda Labs (torch_remote.backends.lambda_labs)
"""

# Import standardized interface components
from .client_interface import (
    ClientConfig,
    ClientError,
    ClientInterface,
    ConnectionError,
    RemoteExecutionError,
    ResourceNotFoundError,
    StorageError,
    extract_storage_ids,
)

# Import available backends
try:
    from . import modal
    __all__ = [
        "ClientInterface",
        "ClientConfig",
        "ClientError",
        "ConnectionError",
        "RemoteExecutionError",
        "StorageError",
        "ResourceNotFoundError",
        "extract_storage_ids",
        "modal"
    ]
except ImportError:
    __all__ = [
        "ClientInterface",
        "ClientConfig",
        "ClientError",
        "ConnectionError",
        "RemoteExecutionError",
        "StorageError",
        "ResourceNotFoundError",
        "extract_storage_ids",
    ]
