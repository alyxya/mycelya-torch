# Copyright (C) 2025 alyxya, SPDX-License-Identifier: AGPL-3.0-or-later

"""Mock server module for local testing."""

from typing import Any, Tuple

from ..modal.server import create_modal_app_for_gpu


def create_mock_modal_app() -> Tuple[Any, Any]:
    """
    Create a mock Modal app for local testing.

    This function wraps create_modal_app_for_gpu() with gpu_type="local"
    to enable local execution without cloud infrastructure.

    Returns:
        Tuple of (modal_app, server_class) for local execution
    """
    return create_modal_app_for_gpu(gpu_type="local", packages=[], python_version="3.13")
