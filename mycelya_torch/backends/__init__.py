# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution backends for mycelya_torch.

This module contains backend implementations for different cloud GPU providers.
Currently supports:
- Modal (mycelya_torch.backends.modal)

Future backends may include:
- RunPod (mycelya_torch.backends.runpod)
- Lambda Labs (mycelya_torch.backends.lambda_labs)
"""

# Import standardized interface components
from .client import Client

# Import available backends
try:
    from . import modal

    __all__ = ["Client", "modal"]
except ImportError:
    __all__ = ["Client"]
