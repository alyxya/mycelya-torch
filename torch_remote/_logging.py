# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Centralized logging configuration for torch_remote.

This module provides utilities to configure logging behavior across all torch_remote modules.
"""

import logging
from typing import Union

# Root logger name for all torch_remote modules
TORCH_REMOTE_LOGGER = "torch_remote"

# Default logging level
DEFAULT_LEVEL = logging.WARNING


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a torch_remote module.

    This ensures all torch_remote loggers are children of the main torch_remote logger,
    allowing for centralized configuration.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance configured for torch_remote
    """
    # Ensure the name starts with torch_remote
    if not name.startswith("torch_remote"):
        if name == "__main__":
            name = "torch_remote"
        else:
            name = f"torch_remote.{name}"

    logger = logging.getLogger(name)

    # Set up the root torch_remote logger if this is the first time
    root_logger = logging.getLogger(TORCH_REMOTE_LOGGER)
    if not root_logger.handlers:
        _setup_default_logging()

    return logger


def _setup_default_logging():
    """Set up default logging configuration for torch_remote."""
    root_logger = logging.getLogger(TORCH_REMOTE_LOGGER)
    root_logger.setLevel(DEFAULT_LEVEL)

    # Only add handler if none exists to avoid duplicates
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

        # Prevent propagation to avoid duplicate messages
        root_logger.propagate = False


def set_logging_level(level: Union[int, str]) -> None:
    """
    Set the logging level for all torch_remote modules.

    This function provides a simple way to control the verbosity of torch_remote
    logging output, making it easier to debug issues without modifying code.

    Args:
        level: Logging level. Can be:
            - String: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
            - Integer: logging.DEBUG (10), logging.INFO (20), etc.

    Examples:
        >>> import torch_remote
        >>> torch_remote.set_logging_level('DEBUG')  # Show all debug messages
        >>> torch_remote.set_logging_level('INFO')   # Show info and above
        >>> torch_remote.set_logging_level('WARNING') # Show warnings and above (default)
        >>> torch_remote.set_logging_level(logging.DEBUG) # Using logging constants
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

    # Set the level on the root torch_remote logger
    root_logger = logging.getLogger(TORCH_REMOTE_LOGGER)
    root_logger.setLevel(level)

    # Ensure logging is set up
    if not root_logger.handlers:
        _setup_default_logging()
        root_logger.setLevel(level)  # Apply level after setup


def get_logging_level() -> int:
    """
    Get the current logging level for torch_remote.

    Returns:
        Current logging level as an integer
    """
    root_logger = logging.getLogger(TORCH_REMOTE_LOGGER)
    return root_logger.level


def disable_logging() -> None:
    """Disable all torch_remote logging output."""
    set_logging_level(logging.CRITICAL + 1)


def enable_debug_logging() -> None:
    """Enable debug logging for torch_remote (shows all messages)."""
    set_logging_level(logging.DEBUG)


def enable_info_logging() -> None:
    """Enable info logging for torch_remote (shows info, warning, error)."""
    set_logging_level(logging.INFO)


def reset_logging() -> None:
    """Reset logging to default level (WARNING)."""
    set_logging_level(DEFAULT_LEVEL)
