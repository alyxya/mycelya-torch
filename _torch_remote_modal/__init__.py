# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Private package for torch_remote execution.

This package contains remote execution apps for multiple cloud providers 
and should not be imported directly. It's used internally by torch_remote.
"""

__version__ = "0.1.0"

# Modal execution components are imported lazily when needed
# to avoid Modal initialization errors during package import