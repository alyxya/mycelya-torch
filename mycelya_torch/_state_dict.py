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
from ._utils import TensorMetadata, create_mycelya_tensor_from_metadata

log = get_logger(__name__)


def load_huggingface_state_dict(
    repo_id: str,
    device: torch.device,
    path: str = "",
) -> Dict[str, torch.Tensor]:
    """Load HuggingFace model weights directly as a state dict of remote mycelya tensors.

    This function implements a two-step process:
    1. First RPC loads weights on remote machine and creates temporary links
    2. Second step creates local tensor stubs and links them to remote tensors

    Args:
        repo_id: HuggingFace repository ID (e.g., "microsoft/DialoGPT-medium")
        device: Mycelya device where weights should be loaded
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

    # Step 1: Load model weights remotely through orchestrator
    # The orchestrator will handle client management and ensure the machine is running
    state_dict_metadata = orchestrator.load_huggingface_state_dict(
        device_index=device.index,
        checkpoint=repo_id,
        path=path,
    )

    # Step 2: Create local tensor stubs and collect temp_keys for linking
    state_dict = {}
    tensors = []
    temp_keys = []

    for param_name, metadata in state_dict_metadata.items():
        # Create local tensor stub with proper shape/stride/dtype
        tensor = create_mycelya_tensor_from_metadata(metadata, device)

        state_dict[param_name] = tensor
        tensors.append(tensor)
        temp_keys.append(metadata["temp_key"])

    # Step 3: Link all tensors at once using the orchestrator
    orchestrator.link_tensors(tensors, temp_keys)

    return state_dict
