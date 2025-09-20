# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Version synchronization utilities for matching local and remote package versions.

This module provides functionality to detect local package versions and Python version
to ensure Modal containers use the same versions as the local development environment.
"""

import importlib.metadata
import sys


def get_python_version() -> str:
    """
    Get the current Python version in X.Y format (e.g., "3.11").

    Returns:
        Python version string compatible with Modal Image.from_registry()
    """
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_local_package_version(package_name: str) -> str | None:
    """
    Get the version of an installed package.

    Args:
        package_name: Name of the package to check

    Returns:
        Package version string if installed, None if not installed
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_versioned_packages(package_names: list[str]) -> list[str]:
    """
    Convert package names to versioned package specifications.

    For packages installed locally, returns "package==version".
    For packages not installed locally, returns "package" (no version).

    Args:
        package_names: List of package names to check

    Returns:
        List of package specifications with versions where available
    """
    versioned_packages = []

    for package_name in package_names:
        local_version = get_local_package_version(package_name)
        if local_version is not None:
            versioned_packages.append(f"{package_name}=={local_version}")
        else:
            versioned_packages.append(package_name)

    return versioned_packages


def module_name_to_package_name(module_name: str) -> str | None:
    """
    Convert module name to package name using package distribution mapping.

    This handles cases where the module name differs from the package name
    (e.g., "PIL" module name from "pillow" package, "bs4" module name from "beautifulsoup4" package).

    Args:
        module_name: Name used in import statements (e.g., "PIL")

    Returns:
        Name used for package installation (e.g., "pillow") or None if not found
    """
    pkg_to_dist = importlib.metadata.packages_distributions()
    distributions = pkg_to_dist.get(module_name)
    return distributions[0] if distributions else None
