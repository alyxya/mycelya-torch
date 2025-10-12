# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote execution decorator for mycelya tensors.

This module provides the remote decorator that enables transparent remote execution
of functions on mycelya tensors by automatically handling serialization and
orchestrator coordination.
"""

import functools
from typing import Any, Callable

from ._orchestrator import orchestrator


def remote(_func: Callable[..., Any] | None = None, *, run_async: bool = False):
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
        run_async: Whether to run the function asynchronously (defaults to False)
                   - If False: Blocks until result is ready and returns the result directly
                   - If True: Returns a Future immediately without blocking

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

        # Asynchronous execution:
        @remote(run_async=True)
        def async_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        machine = RemoteMachine("modal", "A100")
        x = torch.randn(100, 100, device=machine.device("cuda"))
        y = torch.randn(100, 100, device=machine.device("cuda"))

        # Synchronous execution (blocks until result is ready)
        result1 = matrix_multiply(x, y)
        result2 = matrix_add(x, y)

        # Asynchronous execution (returns Future immediately)
        future = async_function(x, y)  # Returns immediately
        # ... do other work ...
        result3 = future.result()  # Block only when needed
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
            if run_async:
                # Asynchronous execution - return Future immediately
                return orchestrator.execute_function_future(func, args, kwargs)
            else:
                # Synchronous execution - block until result is ready
                return orchestrator.execute_function(func, args, kwargs)

        return wrapper

    # Dual-mode logic: detect if used as @remote or @remote()
    if _func is None:
        # Called as @remote() - return decorator function
        return create_wrapper
    else:
        # Called as @remote - directly decorate the function
        return create_wrapper(_func)
