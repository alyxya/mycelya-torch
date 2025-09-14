# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution decorator for mycelya tensors.

This module provides the remote decorator that enables transparent remote execution
of functions on mycelya tensors by automatically handling serialization and
orchestrator coordination.
"""

import functools
from typing import Any, Callable, Optional

from ._orchestrator import orchestrator


def remote(_func: Optional[Callable[..., Any]] = None, *, run_async: bool = False):
    """
    Dual-mode decorator that converts a function to execute remotely on mycelya tensors.

    Can be used either as @remote or @remote() with identical behavior.

    This decorator:
    1. Analyzes function arguments to determine target remote machine
    2. Serializes function and arguments using cloudpickle.Pickler-based MycelyaPickler
    3. Executes function remotely via orchestrator coordination
    4. Deserializes results back to local mycelya tensors with proper linking

    Args:
        _func: Function to decorate (when used as @remote) or None (when used as @remote())
        run_async: Whether to run the function asynchronously (unused for now, defaults to False)

    Returns:
        Decorated function (when used as @remote) or decorator function (when used as @remote())

    Examples:
        # Both of these work identically:

        @remote
        def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b

        @remote()
        def matrix_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        # Future async support:
        @remote(run_async=True)
        def async_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        machine = RemoteMachine("modal", "A100")
        x = torch.randn(100, 100, device=machine.device())
        y = torch.randn(100, 100, device=machine.device())
        result1 = matrix_multiply(x, y)  # Executes remotely
        result2 = matrix_add(x, y)       # Executes remotely
    """

    def create_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(func):
            raise TypeError(
                f"@remote decorator expected a callable function, got {type(func).__name__}"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the function remotely via orchestrator
            # Machine inference happens during pickling via Pickler.machine_id
            return orchestrator.execute_pickled_function(func, args, kwargs)

        return wrapper

    # Dual-mode logic: detect if used as @remote or @remote()
    if _func is None:
        # Called as @remote() - return decorator function
        return create_wrapper
    else:
        # Called as @remote - directly decorate the function
        return create_wrapper(_func)
