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

# Import available backends
try:
    from . import modal
    __all__ = ['modal']
except ImportError:
    __all__ = []