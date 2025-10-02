# Copyright (C) 2025 alyxya, SPDX-License-Identifier: AGPL-3.0-or-later

"""Mock server module for local testing."""

from typing import Any

from ..modal.server import create_modal_app_for_gpu


def create_mock_modal_app(machine_id: str) -> tuple[Any, Any]:
    """
    Create a mock Modal app for local testing.

    This function wraps create_modal_app_for_gpu() with gpu_type="local"
    to enable local execution without cloud infrastructure.

    Args:
        machine_id: Unique machine ID used as the app name

    Returns:
        Tuple of (modal_app, server_class) for local execution
    """
    return create_modal_app_for_gpu(
        machine_id=machine_id,
        gpu_type="local",
        packages=[],
        python_version="3.13",
    )
