# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Operation classifier for categorizing PyTorch operations.

This module provides a centralized system for classifying PyTorch ATen operations
into different categories based on their characteristics and execution requirements.
This replaces the inline operation detection logic in _aten_impl.py.
"""

import logging
from enum import Enum
from typing import Dict, Set

import torch

log = logging.getLogger(__name__)


class OperationType(Enum):
    """Categories of PyTorch operations based on execution requirements."""

    VIEW_OPERATION = "view"            # Operations that create views (executed locally)
    COMPUTE_OPERATION = "compute"      # Operations that perform computation (executed remotely)
    MEMORY_OPERATION = "memory"        # Memory management operations
    CREATION_OPERATION = "creation"    # Tensor creation operations
    SCALAR_OPERATION = "scalar"        # Operations that extract scalar values


class OperationClassifier:
    """Classifies PyTorch operations into execution categories.
    
    This class provides a centralized system for determining how different
    PyTorch operations should be handled in the remote execution system.
    """

    # Known view operations that should always be treated as views
    _KNOWN_VIEW_OPERATIONS: Set[str] = {
        "aten::view",
        "aten::reshape",
        "aten::transpose",
        "aten::permute",
        "aten::squeeze",
        "aten::unsqueeze",
        "aten::flatten",
        "aten::unflatten",
        "aten::select",
        "aten::slice",
        "aten::narrow",
        "aten::expand",
        "aten::repeat",
        "aten::as_strided",
        "aten::t",
        "aten::T",
        "aten::mT",
        "aten::mH",
        "aten::real",
        "aten::imag",
        "aten::conj",
        "aten::detach",
        "aten::alias",
    }

    # Operations that create new tensors from existing ones (usually views)
    _TENSOR_FACTORY_OPERATIONS: Set[str] = {
        "aten::contiguous",
        "aten::clone",
        "aten::to",
        "aten::type_as",
    }

    # Scalar extraction operations
    _SCALAR_OPERATIONS: Set[str] = {
        "aten::item",
        "aten::numel",
        "aten::size",
        "aten::stride",
        "aten::storage_offset",
        "aten::dim",
        "aten::element_size",
        "aten::is_contiguous",
        "aten::is_complex",
        "aten::is_floating_point",
        "aten::is_signed",
    }

    # Memory management operations
    _MEMORY_OPERATIONS: Set[str] = {
        "aten::resize_",
        "aten::set_",
        "aten::storage",
        "aten::data_ptr",
    }

    # Operations that should never be treated as views even if they have alias_info
    _NEVER_VIEW_OPERATIONS: Set[str] = {
        "aten::add_",      # In-place operations
        "aten::mul_",
        "aten::sub_",
        "aten::div_",
        "aten::copy_",
        "aten::fill_",
        "aten::zero_",
        "aten::uniform_",
        "aten::normal_",
        "aten::bernoulli_",
        "aten::exponential_",
    }

    @classmethod
    def classify_operation(cls, op: torch._ops.OpOverload) -> OperationType:
        """Classify a PyTorch operation into its execution category.
        
        Args:
            op: PyTorch operation overload to classify
            
        Returns:
            OperationType indicating how the operation should be handled
        """
        op_name = op.overloadpacket._qualified_op_name

        # Check explicit operation lists first
        if op_name in cls._SCALAR_OPERATIONS:
            return OperationType.SCALAR_OPERATION

        if op_name in cls._MEMORY_OPERATIONS:
            return OperationType.MEMORY_OPERATION

        if op_name in cls._KNOWN_VIEW_OPERATIONS:
            return OperationType.VIEW_OPERATION

        if op_name in cls._NEVER_VIEW_OPERATIONS:
            return OperationType.COMPUTE_OPERATION

        # Use PyTorch's schema information for remaining operations
        has_alias_info = any(r.alias_info is not None for r in op._schema.returns)
        is_mutable = op._schema.is_mutable
        is_view_operation = has_alias_info and not is_mutable

        if is_view_operation:
            log.debug(f"🔍 View operation detected via schema: {op_name}")
            return OperationType.VIEW_OPERATION

        # Log mutable operations with alias info for debugging
        if has_alias_info and is_mutable:
            log.debug(f"🚨 Mutable operation with alias_info: {op_name}")

        # Default to compute operation
        return OperationType.COMPUTE_OPERATION

    @classmethod
    def is_view_operation(cls, op: torch._ops.OpOverload) -> bool:
        """Check if an operation is a view operation.
        
        Args:
            op: PyTorch operation overload to check
            
        Returns:
            True if the operation creates a view, False otherwise
        """
        return cls.classify_operation(op) == OperationType.VIEW_OPERATION

    @classmethod
    def is_remote_execution_required(cls, op: torch._ops.OpOverload) -> bool:
        """Check if an operation requires remote execution.
        
        Args:
            op: PyTorch operation overload to check
            
        Returns:
            True if operation should be executed remotely, False for local execution
        """
        op_type = cls.classify_operation(op)
        return op_type in {
            OperationType.COMPUTE_OPERATION,
            OperationType.MEMORY_OPERATION,
        }

    @classmethod
    def is_local_execution_allowed(cls, op: torch._ops.OpOverload) -> bool:
        """Check if an operation can be executed locally.
        
        Args:
            op: PyTorch operation overload to check
            
        Returns:
            True if operation can be executed locally, False otherwise
        """
        op_type = cls.classify_operation(op)
        return op_type in {
            OperationType.VIEW_OPERATION,
            OperationType.SCALAR_OPERATION,
        }

    @classmethod
    def get_operation_stats(cls) -> Dict[str, int]:
        """Get statistics about known operation categories.
        
        Returns:
            Dictionary with counts of operations in each category
        """
        return {
            "known_view_operations": len(cls._KNOWN_VIEW_OPERATIONS),
            "tensor_factory_operations": len(cls._TENSOR_FACTORY_OPERATIONS),
            "scalar_operations": len(cls._SCALAR_OPERATIONS),
            "memory_operations": len(cls._MEMORY_OPERATIONS),
            "never_view_operations": len(cls._NEVER_VIEW_OPERATIONS),
        }

    @classmethod
    def add_known_view_operation(cls, op_name: str) -> None:
        """Add an operation to the known view operations set.
        
        This allows for runtime registration of new view operations.
        
        Args:
            op_name: Qualified operation name (e.g., "aten::my_view_op")
        """
        cls._KNOWN_VIEW_OPERATIONS.add(op_name)
        log.info(f"Added {op_name} to known view operations")

    @classmethod
    def add_never_view_operation(cls, op_name: str) -> None:
        """Add an operation to the never-view operations set.
        
        This allows for runtime registration of operations that should
        never be treated as views even if they have alias_info.
        
        Args:
            op_name: Qualified operation name (e.g., "aten::my_compute_op")  
        """
        cls._NEVER_VIEW_OPERATIONS.add(op_name)
        log.info(f"Added {op_name} to never-view operations")
