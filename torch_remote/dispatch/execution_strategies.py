# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Execution strategies for different types of PyTorch operations.

This module implements the Strategy pattern for handling different categories
of operations identified by the OperationClassifier. This replaces the
large conditional logic in _aten_impl.py with focused strategy classes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from torch.utils._pytree import tree_map

from .._meta_parser import TensorMetadataConverter
from .operation_classifier import OperationType

log = logging.getLogger(__name__)


class ExecutionStrategy(ABC):
    """Abstract base class for operation execution strategies."""

    @abstractmethod
    def execute(self, op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute the operation using this strategy.
        
        Args:
            op: PyTorch operation to execute
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Result of the operation
        """
        pass


class ViewExecutionStrategy(ExecutionStrategy):
    """Strategy for executing view operations locally."""

    def execute(self, op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute view operation locally with proper device validation."""
        from .._aten_impl import _handle_view_operation
        return _handle_view_operation(op, *args, **kwargs)


class RemoteExecutionStrategy(ExecutionStrategy):
    """Strategy for executing compute operations on remote devices."""

    def __init__(self):
        # Use lazy import to avoid circular dependency during initialization
        self._orchestrator = None

    def _get_orchestrator(self):
        """Lazy initialization of orchestrator to avoid circular imports."""
        if self._orchestrator is None:
            from .._remote_orchestrator import remote_orchestrator
            self._orchestrator = remote_orchestrator
        return self._orchestrator

    def execute(self, op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute operation on remote device through orchestrator."""
        op_name = op.overloadpacket._qualified_op_name
        log.info(f"🚀 Remote execution strategy: {op_name}")

        # Use the restored compute logic with proper meta tensor execution
        return self._execute_with_meta_tensors(op, args, kwargs)

    def _execute_with_meta_tensors(self, op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute operation using meta tensor approach for proper output handling."""
        op_name = op.overloadpacket._qualified_op_name

        # Step 1: Validate all remote tensors are on the same device and convert to meta tensors
        remote_device = None
        original_tensors = {}  # Maps meta tensor id to original tensor

        def validate_and_convert_to_meta_tensor(tensor: torch.Tensor) -> torch.Tensor:
            """Validate device consistency and convert a remote tensor to a meta tensor."""
            nonlocal remote_device

            # Check device consistency for remote tensors
            if tensor.device.type == "remote":
                if remote_device is None:
                    remote_device = tensor.device
                    log.debug(f"Using remote device: {remote_device}")
                elif tensor.device != remote_device:
                    # Cross-device validation using storage resolver
                    from ..services.storage_resolver import StorageMachineResolver
                    resolver = StorageMachineResolver()

                    # Extract storage IDs for cross-device validation
                    storage_ids = [
                        tensor.untyped_storage().data_ptr(),
                        remote_device.index  # Use a placeholder for now
                    ]
                    try:
                        resolver.validate_cross_device_operation(storage_ids)
                    except RuntimeError as e:
                        raise RuntimeError(f"Operation {op_name}: {e}")

                # Convert to meta tensor while preserving properties
                meta_tensor = torch.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device="meta",
                    requires_grad=tensor.requires_grad,
                )
                if tensor.stride() != meta_tensor.stride():
                    meta_tensor = torch.as_strided(
                        meta_tensor, tensor.shape, tensor.stride(), tensor.storage_offset()
                    )

                original_tensors[id(meta_tensor)] = tensor
                return meta_tensor
            return tensor

        # Convert args and kwargs with device validation using tree_map
        def convert_to_meta_tensor(obj):
            if isinstance(obj, torch.Tensor) and obj.device.type == "remote":
                return validate_and_convert_to_meta_tensor(obj)
            return obj

        meta_args, meta_kwargs = tree_map(convert_to_meta_tensor, (args, kwargs))

        # If no remote tensors found, this indicates a dispatch logic error
        if remote_device is None:
            raise RuntimeError(
                f"Remote kernel fallback called for operation {op_name} with no remote tensors. "
                "This indicates a dispatch logic error."
            )

        # Step 2: Execute the operation on meta tensors to determine outputs
        log.debug(f"🔧 Executing {op_name} on meta tensors for shape inference")

        try:
            meta_result = op(*meta_args, **meta_kwargs)
            log.debug(f"✅ Meta execution completed successfully for {op_name}")
        except Exception as e:
            log.error(f"Meta execution failed for {op_name}: {e}")
            raise RuntimeError(
                f"Meta tensor execution failed for {op_name}: {e}. "
                "This operation cannot be executed remotely without meta tensor support."
            )

        # Handle both single tensor and tuple results
        if isinstance(meta_result, torch.Tensor):
            meta_outputs = [meta_result]
            return_single = True
        elif isinstance(meta_result, tuple):
            meta_outputs = list(meta_result)
            return_single = False
        else:
            # Non-tensor result, execute remotely and return as-is
            return self._execute_non_tensor_result(op, args, kwargs, meta_result)

        # Step 3: Create output tensors
        output_tensors = self._create_output_tensors(
            op, args, kwargs, meta_outputs, original_tensors, remote_device
        )

        # Step 4: Execute remotely with orchestrator
        self._execute_remote_operation(op, args, kwargs, output_tensors)

        # Step 5: Return results
        if return_single:
            return output_tensors[0]
        else:
            return tuple(output_tensors)

    def _execute_non_tensor_result(self, op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any], meta_result: Any) -> Any:
        """Handle operations that return non-tensor results."""
        op_name = op.overloadpacket._qualified_op_name
        log.debug(f"Non-tensor result from {op_name}, executing remotely")

        # Use the clean abstraction - convert tensors to metadata first
        processed_args, processed_kwargs, input_metadata = (
            TensorMetadataConverter.args_to_metadata_with_placeholders(
                args, kwargs, operation_context=op_name
            )
        )

        # No output tensors for non-tensor results
        output_metadata = []

        # Execute with clean interface - only metadata crosses boundary
        orchestrator = self._get_orchestrator()
        orchestrator.execute_remote_aten_operation(
            op_name,
            input_metadata,
            output_metadata,
            processed_args,
            processed_kwargs,
        )
        return meta_result

    def _create_output_tensors(self, op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                             meta_outputs: List, original_tensors: Dict, remote_device: torch.device) -> List:
        """Create output tensors based on meta execution results."""
        op_name = op.overloadpacket._qualified_op_name

        # Check if any output meta tensors are the same object as input meta tensors
        input_meta_tensor_ids = {
            id(tensor) for tensor in args if isinstance(tensor, torch.Tensor)
        }
        input_meta_tensor_ids.update({
            id(tensor) for tensor in kwargs.values() if isinstance(tensor, torch.Tensor)
        })

        output_tensors = []

        # Special handling for "out" parameter
        out_tensor = None
        if ("out" in kwargs and isinstance(kwargs["out"], torch.Tensor) and
            kwargs["out"].device.type == "remote"):
            out_tensor = kwargs["out"]
            log.debug(f"Found 'out' parameter tensor with shape {out_tensor.shape}")

        # Process each output tensor
        for i, meta_output in enumerate(meta_outputs):
            log.debug(f"🔧 Processing output tensor {i}: meta_output.shape={meta_output.shape}")

            if id(meta_output) in input_meta_tensor_ids:
                # Output shares storage with input - reuse the original input tensor
                original_tensor = original_tensors[id(meta_output)]
                output_tensors.append(original_tensor)
                log.debug(f"✅ Output tensor {i} reuses input tensor storage (in-place operation)")
            else:
                # Check if this meta output corresponds to the "out" parameter
                if out_tensor is not None and len(output_tensors) == i:
                    # Use the existing "out" tensor as the output tensor
                    output_tensors.append(out_tensor)
                    log.debug(f"✅ Using existing 'out' tensor as output {i}")
                else:
                    log.debug(f"🔧 Creating new output tensor {i} with shape {meta_output.shape}")

                    # Create new output tensor with lazy allocation
                    new_tensor = self._create_new_output_tensor(meta_output, remote_device)
                    output_tensors.append(new_tensor)
                    log.debug(f"✅ Created new output tensor {i} with shape {meta_output.shape}")

        return output_tensors

    def _create_new_output_tensor(self, meta_output: torch.Tensor, remote_device: torch.device) -> torch.Tensor:
        """Create a new output tensor on the remote device."""
        # Import here to avoid circular imports
        try:
            from .._C import empty_remote

            # Create empty remote tensor
            new_tensor = empty_remote(
                meta_output.shape,
                dtype=meta_output.dtype,
                device=remote_device,
                requires_grad=meta_output.requires_grad,
            )

            # Apply stride if different from default
            if meta_output.stride() != new_tensor.stride():
                new_tensor = torch.as_strided(
                    new_tensor,
                    meta_output.shape,
                    meta_output.stride(),
                    meta_output.storage_offset(),
                )

            return new_tensor
        except ImportError:
            # Fallback to regular tensor creation (will be handled by PyTorch backend)
            return torch.empty(
                meta_output.shape,
                dtype=meta_output.dtype,
                device=remote_device,
                requires_grad=meta_output.requires_grad,
            )

    def _execute_remote_operation(self, op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any], output_tensors: List) -> None:
        """Execute the operation remotely using the orchestrator."""
        op_name = op.overloadpacket._qualified_op_name

        log.debug(f"🔧 Executing {op_name} remotely with meta-based output handling")

        # Convert tensors to metadata at PyTorch boundary (early conversion)
        processed_args, processed_kwargs, input_metadata = (
            TensorMetadataConverter.args_to_metadata_with_placeholders(
                args, kwargs, operation_context=op_name
            )
        )

        # Convert output tensors to metadata as well
        output_metadata = [
            TensorMetadataConverter.tensor_to_metadata(tensor, f"{op_name}_output")
            for tensor in output_tensors
        ]

        log.debug(
            f"🔧 About to call orchestrator with {len(input_metadata)} input tensors, {len(output_metadata)} output tensors"
        )

        # Use the new interface with pure metadata (no raw tensors cross this boundary)
        orchestrator = self._get_orchestrator()
        orchestrator.execute_remote_aten_operation(
            op_name, input_metadata, output_metadata, processed_args, processed_kwargs
        )

        log.debug(f"✅ Remote orchestrator execution completed for {op_name}")

    def _execute_meta_operation(self, op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> List:
        """Execute operation on meta tensors for shape inference."""
        # Convert all tensor arguments to meta tensors
        meta_args = []
        meta_kwargs = {}
        output_metadata = []

        def convert_to_meta(tensor):
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "remote":
                # Create meta tensor with same metadata
                meta_tensor = torch.empty(tensor.size(), dtype=tensor.dtype, device="meta")
                if tensor.stride() != meta_tensor.stride():
                    meta_tensor = torch.as_strided(
                        meta_tensor, tensor.shape, tensor.stride(), tensor.storage_offset()
                    )
                return meta_tensor
            return tensor

        meta_args, meta_kwargs = tree_map(convert_to_meta, (args, kwargs))

        # Execute meta operation
        try:
            meta_result = op(*meta_args, **meta_kwargs)

            # Extract metadata from results
            if isinstance(meta_result, torch.Tensor):
                meta_result = [meta_result]
            elif isinstance(meta_result, (list, tuple)):
                meta_result = list(meta_result)
            else:
                meta_result = []

            # Convert meta results to metadata
            for result_tensor in meta_result:
                if isinstance(result_tensor, torch.Tensor):
                    # Create RemoteTensorMeta from meta tensor result
                    from .._meta_parser import RemoteTensorMeta
                    metadata = RemoteTensorMeta(
                        storage_id=0,  # Will be assigned by remote execution
                        size=result_tensor.size(),
                        stride=result_tensor.stride(),
                        storage_offset=result_tensor.storage_offset(),
                        dtype=result_tensor.dtype,
                        nelem_in_bytes=result_tensor.numel() * result_tensor.element_size(),
                    )
                    output_metadata.append(metadata)

        except Exception as e:
            log.warning(f"Meta execution failed for {op.overloadpacket._qualified_op_name}: {e}")
            # If meta execution fails, we'll let remote execution handle it

        return output_metadata

    def _extract_outputs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], output_metadata: List) -> Any:
        """Extract output tensors from args/kwargs based on operation result."""
        # For now, assume single tensor output (most common case)
        # More sophisticated output extraction can be added later
        if len(output_metadata) == 1:
            # Find the first remote tensor in args that could be the output
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.device.type == "remote":
                    return arg

        # For multiple outputs or complex cases, return first remote tensor found
        remote_tensors = []
        def collect_remote(obj):
            if isinstance(obj, torch.Tensor) and obj.device.type == "remote":
                remote_tensors.append(obj)
            return obj

        tree_map(collect_remote, (args, kwargs))
        return remote_tensors[0] if remote_tensors else None


class ScalarExecutionStrategy(ExecutionStrategy):
    """Strategy for executing scalar extraction operations."""

    def execute(self, op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute scalar operation, typically extracting values from remote tensors."""
        op_name = op.overloadpacket._qualified_op_name
        log.info(f"📊 Scalar extraction: {op_name}")

        # Handle common scalar operations
        if op_name == "aten::item":
            tensor = args[0]
            if tensor.device.type == "remote":
                # Transfer to CPU and extract item
                cpu_tensor = tensor.cpu()
                return cpu_tensor.item()
            else:
                return tensor.item()

        elif op_name in {"aten::numel", "aten::size", "aten::stride", "aten::storage_offset", "aten::dim"}:
            # These can be computed locally from tensor metadata
            tensor = args[0]
            if op_name == "aten::numel":
                return tensor.numel()
            elif op_name == "aten::size":
                dim = args[1] if len(args) > 1 else None
                return tensor.size(dim) if dim is not None else tensor.size()
            elif op_name == "aten::stride":
                dim = args[1] if len(args) > 1 else None
                return tensor.stride(dim) if dim is not None else tensor.stride()
            elif op_name == "aten::storage_offset":
                return tensor.storage_offset()
            elif op_name == "aten::dim":
                return tensor.dim()

        # For other scalar operations, fall back to remote execution
        log.warning(f"Unhandled scalar operation {op_name}, falling back to remote execution")
        remote_strategy = RemoteExecutionStrategy()
        return remote_strategy.execute(op, args, kwargs)


class MemoryExecutionStrategy(ExecutionStrategy):
    """Strategy for executing memory management operations."""

    def execute(self, op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute memory operation, typically on remote device."""
        op_name = op.overloadpacket._qualified_op_name
        log.info(f"💾 Memory operation: {op_name}")

        # Memory operations typically need to be executed remotely
        # as they modify the underlying storage
        remote_strategy = RemoteExecutionStrategy()
        return remote_strategy.execute(op, args, kwargs)


class ExecutionStrategyFactory:
    """Factory for creating appropriate execution strategies."""

    _strategies = {
        OperationType.VIEW_OPERATION: ViewExecutionStrategy,
        OperationType.COMPUTE_OPERATION: RemoteExecutionStrategy,
        OperationType.SCALAR_OPERATION: ScalarExecutionStrategy,
        OperationType.MEMORY_OPERATION: MemoryExecutionStrategy,
        OperationType.CREATION_OPERATION: RemoteExecutionStrategy,  # Default to remote
    }

    _strategy_instances = {}

    @classmethod
    def get_strategy(cls, operation_type: OperationType) -> ExecutionStrategy:
        """Get execution strategy for the given operation type.
        
        Args:
            operation_type: Type of operation to get strategy for
            
        Returns:
            ExecutionStrategy instance for the operation type
        """
        if operation_type not in cls._strategy_instances:
            strategy_class = cls._strategies.get(operation_type, RemoteExecutionStrategy)
            cls._strategy_instances[operation_type] = strategy_class()

        return cls._strategy_instances[operation_type]

    @classmethod
    def register_strategy(cls, operation_type: OperationType, strategy_class: type) -> None:
        """Register a custom strategy for an operation type.
        
        Args:
            operation_type: Operation type to register strategy for
            strategy_class: Strategy class to use for this operation type
        """
        cls._strategies[operation_type] = strategy_class
        # Clear cached instance so new strategy will be used
        cls._strategy_instances.pop(operation_type, None)
        log.info(f"Registered custom strategy {strategy_class.__name__} for {operation_type}")

    @classmethod
    def get_available_strategies(cls) -> Dict[OperationType, str]:
        """Get information about available strategies.
        
        Returns:
            Dictionary mapping operation types to strategy class names
        """
        return {op_type: strategy_class.__name__ for op_type, strategy_class in cls._strategies.items()}
