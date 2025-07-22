# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Core infrastructure for torch_remote.

This module provides core infrastructure components including dependency injection,
service management, and other foundational elements for the torch_remote system.
"""

from .container import (
    ServiceContainer,
    clear_container,
    get_container,
    get_service,
    register_default_services,
    register_instance,
    register_service,
)

__all__ = [
    "ServiceContainer",
    "get_container",
    "register_service",
    "register_instance",
    "get_service",
    "clear_container",
    "register_default_services",
]
