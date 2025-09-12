# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal server implementation for mycelya_torch.

This module provides the Modal cloud GPU server implementation.
"""

from .server import create_modal_app_for_gpu

__all__ = ["create_modal_app_for_gpu"]
