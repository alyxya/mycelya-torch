# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution clients for mycelya_torch.

This module contains client implementations for different cloud GPU providers.
Currently supports:
- Modal (mycelya_torch.clients.modal)

Future clients may include:
- RunPod (mycelya_torch.clients.runpod)
- Lambda Labs (mycelya_torch.clients.lambda_labs)
"""

# Import standardized interface components
from .base_client import Client

# Import available clients
try:
    from . import modal

    __all__ = ["Client", "modal"]
except ImportError:
    __all__ = ["Client"]
