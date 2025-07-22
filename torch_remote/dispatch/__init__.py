# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Dispatch module for torch-remote.

This module contains the operation dispatch system:
- OperationClassifier: Categorizes PyTorch operations
- ExecutionStrategies: Handles different execution patterns
- Operation dispatch logic and routing

These components provide better separation of concerns and cleaner
operation handling compared to having all dispatch logic in _aten_impl.py.
"""

from .execution_strategies import ExecutionStrategy, ExecutionStrategyFactory
from .operation_classifier import OperationClassifier, OperationType

__all__ = ["OperationClassifier", "OperationType", "ExecutionStrategy", "ExecutionStrategyFactory"]
