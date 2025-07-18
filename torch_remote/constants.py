# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Constants used throughout the torch_remote package.

This module centralizes string literals and other constants to improve
maintainability and reduce magic string usage across the codebase.
"""

# Device type constants
REMOTE_DEVICE_TYPE = "remote"
CPU_DEVICE_TYPE = "cpu"
CUDA_DEVICE_TYPE = "cuda"
META_DEVICE_TYPE = "meta"

# Tensor placeholder constants for remote execution
TENSOR_PLACEHOLDER_PREFIX = "__TENSOR_"

# PyTorch dispatch constants
PRIVATEUSE1_DISPATCH_KEY = "PrivateUse1"

# Backend provider constants
MODAL_PROVIDER = "modal"

# Storage ID constants (imported from _device_daemon.py for centralization)
from ._device_daemon import MIN_STORAGE_ID, MAX_STORAGE_ID, MAX_ID_GENERATION_ATTEMPTS

__all__ = [
    "REMOTE_DEVICE_TYPE",
    "CPU_DEVICE_TYPE",
    "CUDA_DEVICE_TYPE",
    "META_DEVICE_TYPE",
    "TENSOR_PLACEHOLDER_PREFIX",
    "PRIVATEUSE1_DISPATCH_KEY",
    "MODAL_PROVIDER",
    "MIN_STORAGE_ID",
    "MAX_STORAGE_ID",
    "MAX_ID_GENERATION_ATTEMPTS",
]
