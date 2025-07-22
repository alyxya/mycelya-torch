# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tensor utilities for metadata handling, serialization, and device transfers.

This module consolidates all tensor-related operations:
- Tensor metadata extraction and conversion
- Serialization/deserialization of tensors
- Remote-to-CPU and CPU-to-remote transfers
- Tensor creation from metadata
- Argument processing for remote operations
"""

import io
import logging
from pprint import pformat
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from torch.utils._pytree import tree_map

if TYPE_CHECKING:
    from .device import RemoteMachine

log = logging.getLogger(__name__)


class RemoteTensorMeta:
    """Metadata representation of a remote tensor."""

    def __init__(
        self,
        storage_id: int,
        size: torch.Size,
        stride: Tuple[int, ...],
        storage_offset: int,
        dtype: torch.dtype,
        nelem_in_bytes: int,
    ) -> None:
        """Create RemoteTensorMeta with explicit metadata.

        Args:
            storage_id: Tensor storage ID
            size: Tensor shape
            stride: Tensor stride
            storage_offset: Storage offset
            dtype: Data type
            nelem_in_bytes: Number of elements in bytes
        """
        self.storage_id = storage_id
        self.size = size
        self.stride = stride
        self.storage_offset = storage_offset
        self.dtype = dtype
        self.nelem_in_bytes = nelem_in_bytes

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, checked: bool = True
    ) -> "RemoteTensorMeta":
        """Create RemoteTensorMeta from existing tensor.

        Args:
            tensor: Source tensor to extract metadata from
            checked: Whether to validate tensor is on remote device

        Returns:
            RemoteTensorMeta instance with tensor's metadata

        Raises:
            RuntimeError: If checked=True and tensor is not on remote device
        """
        if checked and tensor.device.type != "remote":
            raise RuntimeError(
                "Creating RemoteTensorMeta is only for Tensors on remote device"
            )

        return cls(
            storage_id=tensor.untyped_storage().data_ptr(),
            size=tensor.size(),
            stride=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            nelem_in_bytes=tensor.nelement() * tensor.element_size(),
        )

    @classmethod
    def from_remote_tensor(cls, tensor: torch.Tensor) -> "RemoteTensorMeta":
        """Create RemoteTensorMeta for remote tensor serialization.

        Args:
            tensor: Remote tensor to create metadata from

        Returns:
            RemoteTensorMeta instance with storage_id set for view tracking
        """
        storage_id = tensor.untyped_storage().data_ptr()
        return cls(
            storage_id=storage_id,
            size=tensor.size(),
            stride=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            nelem_in_bytes=tensor.numel() * tensor.element_size(),
        )

    def __repr__(self) -> str:
        return (
            f"RemoteTensorMeta({self.storage_id=}, {self.size=}, {self.stride=}, "
            f"{self.storage_offset=}, {self.dtype=}, {self.nelem_in_bytes=})"
        )


# Tensor serialization and transfer utilities
def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize tensor to bytes, ensuring view data is contiguous.

    Args:
        tensor: Tensor to serialize

    Returns:
        Serialized tensor data as bytes
    """
    buffer = io.BytesIO()
    # Convert to pure CPU tensor and make contiguous to serialize only the view's data
    # This ensures views are serialized as their actual data, not the full underlying storage
    cpu_tensor = tensor.cpu().detach().contiguous()
    torch.save(cpu_tensor, buffer)
    return buffer.getvalue()


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """Deserialize tensor from bytes as a contiguous tensor.

    Args:
        data: Serialized tensor data

    Returns:
        Deserialized tensor on CPU
    """
    buffer = io.BytesIO(data)
    tensor = torch.load(buffer, map_location="cpu")

    # Since we serialize with .contiguous(), the deserialized tensor should already be contiguous
    # with storage_offset=0 and optimal stride. No view reconstruction needed.
    # Just ensure we have untyped storage for consistency with the remote tensor system.
    if hasattr(tensor, "untyped_storage"):
        untyped_storage = tensor.untyped_storage()
        # The tensor should already be contiguous, so we preserve its natural layout
        tensor = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
            untyped_storage,
            0,  # storage_offset should be 0 for contiguous tensors
            tensor.shape,
            tensor.stride(),
        )
    return tensor


def remote_tensor_to_cpu(remote_tensor: torch.Tensor) -> torch.Tensor:
    """Convert remote tensor to CPU tensor by retrieving data from remote GPU.

    Args:
        remote_tensor: Tensor on remote device to transfer

    Returns:
        Tensor on CPU with same data

    Raises:
        RuntimeError: If no machine found for device or client not available
    """
    from .device import get_device_registry

    # Get the machine backend
    registry = get_device_registry()
    machine = registry.get_device_by_index(remote_tensor.device.index)

    if machine is None:
        raise RuntimeError(
            f"No RemoteMachine found for remote device index {remote_tensor.device.index}"
        )

    # Get the client for this machine
    client = machine.get_client()
    if client is None or not client.is_running():
        raise RuntimeError(f"Client not available for machine {machine.machine_id}")

    # Get tensor data using storage ID
    storage_id = remote_tensor.untyped_storage().data_ptr()

    # Use client to get tensor data by ID with view information
    # Pass tensor metadata so remote side can serialize just the view's data
    tensor_data = client.get_storage_data(
        storage_id,
        shape=list(remote_tensor.shape),
        stride=list(remote_tensor.stride()),
        storage_offset=remote_tensor.storage_offset(),
        dtype=str(remote_tensor.dtype),
    )

    # Deserialize the tensor
    return deserialize_tensor(tensor_data)


def cpu_tensor_to_remote(
    cpu_tensor: torch.Tensor, machine: "RemoteMachine"
) -> torch.Tensor:
    """Convert CPU tensor to remote tensor.

    Args:
        cpu_tensor: Tensor on CPU to transfer
        machine: Target remote machine

    Returns:
        Tensor on remote device

    Raises:
        RuntimeError: If machine is None
    """
    if machine is None:
        raise RuntimeError("RemoteMachine must be specified for remote tensor creation")

    # Convert to remote device - delegates to PyTorch's device transfer system
    result = cpu_tensor.to(machine.device())
    return result


def get_tensor_metadata(tensor: torch.Tensor) -> Dict[str, Any]:
    """Get tensor metadata as a dictionary.

    Args:
        tensor: Tensor to extract metadata from

    Returns:
        Dictionary containing tensor metadata
    """
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "size": tensor.numel(),
        "element_size": tensor.element_size(),
    }


# Argument processing utilities
VALID_QUEUE_TYPES_IN = {torch.Tensor, int, float, torch.dtype}
VALID_QUEUE_TYPES_OUT = {RemoteTensorMeta, int, float, str, torch.dtype}


def safe_str(args: Any) -> str:
    """Convert arguments to a safe string representation for logging.

    Converts torch.Tensor objects to RemoteTensorMeta strings to avoid
    potential issues with remote tensor string representation.

    Args:
        args: Arguments to convert to string

    Returns:
        Safe string representation of the arguments
    """

    def convert(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return str(RemoteTensorMeta.from_tensor(obj, checked=False))
        else:
            return obj

    new_args = tree_map(convert, args)
    return pformat(new_args)


def validate_send_queue_args(cmd: str, args: Any) -> None:
    """Validate that arguments are safe to send through the remote queue.

    Ensures that only supported object types are sent over the remote
    communication channel to prevent serialization errors.

    Args:
        cmd: Command name for context in error messages
        args: Arguments to validate

    Raises:
        RuntimeError: If invalid object types are found
    """

    def check(obj: Any) -> None:
        if type(obj) not in VALID_QUEUE_TYPES_OUT:
            if (
                cmd == "recv_data"
                and type(obj) in [torch.Tensor]
                and obj.device.type == "cpu"
            ):
                # Only HtoD copy command can send cpu Tensors over
                return
            raise RuntimeError(
                f"Trying to send invalid object through queue: {type(obj)}"
            )

    tree_map(check, args)


def prepare_for_sending(args: Any, kwargs: Any) -> Any:
    """Prepare arguments for sending to remote device.

    Converts torch.Tensor objects to RemoteTensorMeta for efficient
    transmission. Remote tensors are converted to metadata only,
    while CPU tensors include full tensor data.

    Args:
        args: Positional arguments to prepare
        kwargs: Keyword arguments to prepare

    Returns:
        Converted arguments ready for remote transmission

    Raises:
        RuntimeError: If unsupported object types are found
    """

    def convert(obj: Any) -> Any:
        if type(obj) not in VALID_QUEUE_TYPES_IN:
            raise RuntimeError(
                f"Cannot send object of type {type(obj)} over remote device pipe."
            )

        if isinstance(obj, torch.Tensor):
            if obj.device.type == "remote":
                # For remote tensors, send storage_id + metadata instead of full tensor data
                return RemoteTensorMeta.from_remote_tensor(obj)
            else:
                # For non-remote tensors, use original behavior
                return RemoteTensorMeta.from_tensor(obj)
        else:
            return obj

    return tree_map(convert, (args, kwargs))


def receive_after_sending(allocator: Any, args: Any, kwargs: Any) -> Any:
    """Process arguments received from remote device.

    Converts RemoteTensorMeta objects back to torch.Tensor using
    the provided allocator. Handles reconstruction of tensor objects
    from metadata.

    Args:
        allocator: Allocator to create tensors from metadata
        args: Received positional arguments
        kwargs: Received keyword arguments

    Returns:
        Reconstructed arguments with tensor objects

    Raises:
        RuntimeError: If invalid object types are received
    """

    def convert(obj: Any) -> Any:
        if type(obj) not in VALID_QUEUE_TYPES_OUT:
            raise RuntimeError(
                f"Received invalid object of type {type(obj)} over remote device pipe."
            )

        if isinstance(obj, RemoteTensorMeta):
            return allocator.tensor_from_meta(obj)
        else:
            return obj

    return tree_map(convert, (args, kwargs))


class TensorMetadataConverter:
    """Centralized converter for tensor-to-metadata transformations.

    This class provides a clean boundary for all tensor conversion operations,
    ensuring consistent handling across the entire execution pipeline.
    """

    @staticmethod
    def tensor_to_metadata(
        tensor: torch.Tensor, context: str = "default"
    ) -> RemoteTensorMeta:
        """Convert a single tensor to metadata.

        Args:
            tensor: PyTorch tensor to convert
            context: Context for conversion (for debugging/logging)

        Returns:
            RemoteTensorMeta object containing tensor metadata
        """
        if tensor.device.type == "remote":
            return RemoteTensorMeta.from_remote_tensor(tensor)
        else:
            return RemoteTensorMeta.from_tensor(tensor, checked=False)

    @staticmethod
    def metadata_to_dict(meta: RemoteTensorMeta, **operation_flags) -> Dict[str, Any]:
        """Convert metadata to serializable dictionary.

        Args:
            meta: RemoteTensorMeta object to convert
            **operation_flags: Additional flags (is_input, is_output, etc.)

        Returns:
            Dictionary ready for serialization/transport
        """
        result = {
            "storage_id": meta.storage_id,
            "shape": list(meta.size),
            "stride": list(meta.stride),
            "storage_offset": meta.storage_offset,
            "dtype": str(meta.dtype),
            "numel": meta.nelem_in_bytes
            // torch.tensor([], dtype=meta.dtype).element_size(),
        }
        result.update(operation_flags)
        return result

    @staticmethod
    def args_to_metadata_with_placeholders(
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        operation_context: str = "default",
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any], List[RemoteTensorMeta]]:
        """Convert args/kwargs, replacing tensors with placeholders and collecting metadata.

        This is the core method for the early conversion boundary - it processes
        nested structures, extracts all remote tensors, and returns both the
        placeholder structure and collected metadata.

        Args:
            args: Original arguments containing tensors
            kwargs: Original keyword arguments containing tensors
            operation_context: Context string for debugging

        Returns:
            Tuple of (processed_args, processed_kwargs, collected_metadata)
        """
        collected_metadata = []

        def convert_tensor_to_placeholder(obj):
            if isinstance(obj, torch.Tensor) and obj.device.type == "remote":
                # Convert tensor to metadata and add to collection
                metadata = TensorMetadataConverter.tensor_to_metadata(
                    obj, operation_context
                )
                tensor_index = len(collected_metadata)
                collected_metadata.append(metadata)
                return f"__TENSOR_{tensor_index}"
            return obj

        # Use tree_map to handle nested structures consistently
        processed_args, processed_kwargs = tree_map(
            convert_tensor_to_placeholder, (args, kwargs)
        )

        return processed_args, processed_kwargs, collected_metadata

    @staticmethod
    def metadata_list_to_dicts(
        metadata_list: List[RemoteTensorMeta],
        input_indices: Optional[set] = None,
        output_indices: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Convert metadata list to serializable dictionaries with operation flags.

        Args:
            metadata_list: List of RemoteTensorMeta objects
            input_indices: Set of indices that are input tensors
            output_indices: Set of indices that are output tensors

        Returns:
            List of metadata dictionaries ready for transport
        """
        result = []
        for i, meta in enumerate(metadata_list):
            operation_flags = {}
            if input_indices and i in input_indices:
                operation_flags["is_input"] = True
            if output_indices and i in output_indices:
                operation_flags["is_output"] = True

            result.append(
                TensorMetadataConverter.metadata_to_dict(meta, **operation_flags)
            )

        return result

    @staticmethod
    def collect_remote_tensors(
        args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> List[torch.Tensor]:
        """Collect all remote tensors from nested args/kwargs structure.

        Args:
            args: Arguments to search
            kwargs: Keyword arguments to search

        Returns:
            List of remote tensors found in the structure
        """
        tensors = []

        def collect_tensor(obj):
            if isinstance(obj, torch.Tensor) and obj.device.type == "remote":
                tensors.append(obj)
            return obj

        tree_map(collect_tensor, (args, kwargs))
        return tensors

