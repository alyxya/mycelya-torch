# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Remote state dict loading for HuggingFace models.

This module provides functionality to load HuggingFace model weights directly
onto remote machines and return them as a state dict of mycelya tensors.
"""

from typing import Dict

import torch

from ._logging import get_logger
from ._orchestrator import orchestrator
from ._utils import TensorMetadata

log = get_logger(__name__)


def _create_remote_tensor_from_metadata(
    metadata: TensorMetadata, device: torch.device, requires_grad: bool = False
) -> torch.Tensor:
    """Create a local tensor stub that will be linked to remote storage.

    This creates a tensor with the correct shape, stride, and dtype that can
    be used as a stub before linking to actual remote tensor data.

    Args:
        metadata: Tensor metadata containing shape, dtype, stride, storage_offset
        device: Mycelya device where the tensor should appear to be located
        requires_grad: Whether the tensor requires gradients

    Returns:
        Local tensor stub ready for linking to remote storage
    """
    # Convert string dtype to torch dtype
    if isinstance(metadata["dtype"], str):
        torch_dtype = getattr(torch, metadata["dtype"])
    else:
        torch_dtype = metadata["dtype"]

    # Calculate storage elements needed for this tensor view
    shape = metadata["shape"]
    stride = metadata["stride"]
    storage_offset = metadata["storage_offset"]

    # Calculate the maximum memory address this tensor accesses
    if not shape:  # Empty tensor
        storage_elements_needed = 0
    else:
        max_address = storage_offset
        for dim_size, dim_stride in zip(shape, stride):
            if dim_size > 1:
                max_address += (dim_size - 1) * abs(dim_stride)
        storage_elements_needed = max_address + 1

    # Create untyped storage with sufficient space
    storage_bytes = storage_elements_needed * torch_dtype.itemsize
    untyped_storage = torch.UntypedStorage(storage_bytes, device=device)

    # Create base tensor from untyped storage
    base_tensor = torch.empty(0, dtype=torch_dtype, device=device)
    base_tensor.set_(untyped_storage, 0, [storage_elements_needed], [1])

    # Create tensor view with exact shape/stride/offset
    remote_tensor = torch.as_strided(base_tensor, shape, stride, storage_offset)
    remote_tensor.requires_grad_(requires_grad)

    return remote_tensor


def load_huggingface_state_dict(
    repo_id: str,
    device: torch.device,
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
    path: str = "",
) -> Dict[str, torch.Tensor]:
    """Load HuggingFace model weights directly as a state dict of remote mycelya tensors.

    This function implements a two-step process:
    1. First RPC loads weights on remote machine and creates temporary links
    2. Second step creates local tensor stubs and links them to remote tensors

    Args:
        repo_id: HuggingFace repository ID (e.g., "microsoft/DialoGPT-medium")
        device: Mycelya device where weights should be loaded
        torch_dtype: Data type for model weights (ignored, uses model defaults)
        trust_remote_code: Whether to trust remote code (ignored)
        path: Path within repository to load from (default: whole repo)

    Returns:
        State dict mapping parameter names to mycelya tensors with remote data

    Raises:
        RuntimeError: If device is not a mycelya device or model loading fails
    """
    if device.type != "mycelya":
        raise RuntimeError(
            f"load_huggingface_state_dict() only supports mycelya devices, got {device.type}"
        )

    log.info(f"Loading HuggingFace model '{repo_id}' state dict on {device}")

    # Step 1: Load model weights remotely through orchestrator
    # The orchestrator will handle client management and ensure the machine is running
    log.debug(f"Requesting remote model loading for {repo_id}")
    future = orchestrator.prepare_huggingface_model(
        device_index=device.index,
        checkpoint=repo_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        path=path,
    )

    # Wait for remote loading to complete
    remote_metadata = future.result()
    log.debug(
        f"Received metadata for {len(remote_metadata.get('state_dict_metadata', {}))} parameters"
    )

    state_dict_metadata = remote_metadata["state_dict_metadata"]

    # Step 2: Create local tensor stubs and collect temp_keys for linking
    state_dict = {}
    local_tensors = []
    temp_keys = []

    for param_name, metadata in state_dict_metadata.items():
        # Create local tensor stub with proper shape/stride/dtype
        requires_grad = (
            "weight" in param_name
        )  # Only weights require gradients by default
        local_tensor = _create_remote_tensor_from_metadata(
            metadata, device, requires_grad
        )

        state_dict[param_name] = local_tensor
        local_tensors.append(local_tensor)
        temp_keys.append(metadata["temp_key"])

    # Step 3: Link all tensors at once using the orchestrator
    log.debug(f"Linking {len(local_tensors)} tensors to remote storage")
    orchestrator.link_tensors(local_tensors, temp_keys)

    log.info(f"Successfully loaded {len(state_dict)} parameters for {repo_id}")
    return state_dict



