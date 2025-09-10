# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Version synchronization utilities for matching local and remote package versions.

This module provides functionality to detect local package versions and Python version
to ensure Modal containers use the same versions as the local development environment.
"""

import sys
from typing import List, Optional
import importlib.metadata


def get_python_version() -> str:
    """
    Get the current Python version in X.Y format (e.g., "3.11").
    
    Returns:
        Python version string compatible with Modal Image.from_registry()
    """
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_local_package_version(package_name: str) -> Optional[str]:
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


def get_versioned_packages(package_names: List[str]) -> List[str]:
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
