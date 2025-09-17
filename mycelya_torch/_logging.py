# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Centralized logging configuration for mycelya_torch.

This module provides utilities to configure logging behavior across all mycelya_torch modules.
"""

import logging

# Root logger name for all mycelya_torch modules
MYCELYA_TORCH_LOGGER = "mycelya_torch"

# Default logging level
DEFAULT_LEVEL = logging.WARNING


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a mycelya_torch module.

    This ensures all mycelya_torch loggers are children of the main mycelya_torch logger,
    allowing for centralized configuration.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance configured for mycelya_torch
    """
    # Ensure the name starts with mycelya_torch
    if not name.startswith("mycelya_torch"):
        if name == "__main__":
            name = "mycelya_torch"
        else:
            name = f"mycelya_torch.{name}"

    logger = logging.getLogger(name)

    # Set up the root mycelya_torch logger if this is the first time
    root_logger = logging.getLogger(MYCELYA_TORCH_LOGGER)
    if not root_logger.handlers:
        _setup_default_logging()

    return logger


def _setup_default_logging():
    """Set up default logging configuration for mycelya_torch."""
    root_logger = logging.getLogger(MYCELYA_TORCH_LOGGER)
    root_logger.setLevel(DEFAULT_LEVEL)

    # Only add handler if none exists to avoid duplicates
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

        # Prevent propagation to avoid duplicate messages
        root_logger.propagate = False


def set_logging_level(level: int | str) -> None:
    """
    Set the logging level for all mycelya_torch modules.

    This function provides a simple way to control the verbosity of mycelya_torch
    logging output, making it easier to debug issues without modifying code.

    Args:
        level: Logging level. Can be:
            - String: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
            - Integer: logging.DEBUG (10), logging.INFO (20), etc.

    Examples:
        >>> import mycelya_torch
        >>> mycelya_torch.set_logging_level('DEBUG')  # Show all debug messages
        >>> mycelya_torch.set_logging_level('INFO')   # Show info and above
        >>> mycelya_torch.set_logging_level('WARNING') # Show warnings and above (default)
        >>> mycelya_torch.set_logging_level(logging.DEBUG) # Using logging constants
    """
    if isinstance(level, str):
        level = level.upper()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if level not in level_map:
            raise ValueError(
                f"Invalid logging level: {level}. Must be one of {list(level_map.keys())}"
            )
        level = level_map[level]

    # Set the level on the root mycelya_torch logger
    root_logger = logging.getLogger(MYCELYA_TORCH_LOGGER)
    root_logger.setLevel(level)

    # Ensure logging is set up
    if not root_logger.handlers:
        _setup_default_logging()
        root_logger.setLevel(level)  # Apply level after setup


def get_logging_level() -> int:
    """
    Get the current logging level for mycelya_torch.

    Returns:
        Current logging level as an integer
    """
    root_logger = logging.getLogger(MYCELYA_TORCH_LOGGER)
    return root_logger.level


def disable_logging() -> None:
    """Disable all mycelya_torch logging output."""
    set_logging_level(logging.CRITICAL + 1)


def enable_debug_logging() -> None:
    """Enable debug logging for mycelya_torch (shows all messages)."""
    set_logging_level(logging.DEBUG)


def enable_info_logging() -> None:
    """Enable info logging for mycelya_torch (shows info, warning, error)."""
    set_logging_level(logging.INFO)


def reset_logging() -> None:
    """Reset logging to default level (WARNING)."""
    set_logging_level(DEFAULT_LEVEL)


# Public API for this module
__all__ = [
    "enable_debug_logging",
    "enable_info_logging",
    "disable_logging",
    "get_logging_level",
    "set_logging_level",
    "reset_logging",
    "get_logger",  # Also useful for external modules
]
