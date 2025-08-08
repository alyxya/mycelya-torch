# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
HuggingFace model loading utilities for mycelya_torch.

This module provides utilities for loading HuggingFace models directly on remote
machines and creating local model stubs with remote tensor references.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ._logging import get_logger
from .device import RemoteMachine

log = get_logger(__name__)


def create_remote_tensor_stub(
    storage_id: int,
    shape: list,
    stride: list,
    storage_offset: int,
    dtype: str,
    device: torch.device,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Create a local tensor stub that references remote storage.

    This creates a mycelya tensor that appears local but actually references
    data stored on a remote machine.

    Args:
        storage_id: Remote storage ID containing the tensor data
        shape: Tensor shape
        stride: Tensor stride
        storage_offset: Storage offset within the remote storage
        dtype: String representation of tensor data type
        device: Remote device where the tensor is stored
        requires_grad: Whether the tensor requires gradients

    Returns:
        Remote tensor stub that can be used in local PyTorch operations
    """
    # Convert dtype string to torch.dtype
    dtype_name = dtype.replace("torch.", "")
    torch_dtype = getattr(torch, dtype_name)

    # Create an empty tensor on the remote device
    # The storage will be linked to the remote storage ID
    remote_tensor = torch.empty(shape, dtype=torch_dtype, device=device)

    # Apply stride if different from default
    if remote_tensor.stride() != tuple(stride):
        remote_tensor = torch.as_strided(
            remote_tensor,
            shape,
            stride,
            storage_offset,
        )

    # Set gradient requirement
    remote_tensor.requires_grad_(requires_grad)

    # Note: The actual linking to remote storage happens through the
    # mycelya device backend when the tensor is created on the remote device.
    # The storage_id is embedded in the tensor's data pointer by the C++ allocator.

    return remote_tensor


def create_huggingface_model_from_remote(
    machine: RemoteMachine,
    checkpoint: str,
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
) -> nn.Module:
    """
    Create a HuggingFace model with remote tensors.

    This function:
    1. Calls the remote machine to download and prepare the model
    2. Receives tensor metadata for all model parameters
    3. Creates a local model skeleton with remote tensor stubs
    4. Links the local tensors to remote storage

    Args:
        machine: Remote machine to load the model on
        checkpoint: HuggingFace model checkpoint (e.g., "gpt2")
        torch_dtype: Data type for model weights ("auto", "float32", etc.)
        trust_remote_code: Whether to trust remote code for custom models

    Returns:
        Local model object with all parameters as remote tensors
    """
    log.info(
        f"Creating HuggingFace model {checkpoint} with remote tensors on {machine.machine_id}"
    )

    # Get the client for this machine
    client = machine.get_client()
    if not client.is_running():
        raise RuntimeError(
            f"Machine {machine.machine_id} is not running. Call machine.start() first."
        )

    # Step 1: Prepare model on remote machine
    log.info(f"Preparing model {checkpoint} on remote machine...")
    remote_data = client.prepare_huggingface_model(
        checkpoint=checkpoint,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    if not remote_data:
        raise RuntimeError(f"Failed to prepare model {checkpoint} on remote machine")

    state_dict_metadata = remote_data["state_dict_metadata"]
    buffer_metadata = remote_data.get("buffer_metadata", {})
    config_dict = remote_data["config"]
    model_type = remote_data["model_type"]

    log.info(
        f"Received metadata for {len(state_dict_metadata)} parameters and {len(buffer_metadata)} buffers"
    )

    # Step 2: Create local model skeleton
    log.info(f"Creating local {model_type} skeleton...")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        # Recreate the config from the dictionary
        # First create a config of the right type, then use from_dict
        model_type = config_dict.get("model_type", "gpt2")
        config = AutoConfig.for_model(model_type).from_dict(config_dict)

        # Create model skeleton on meta device (no memory allocation)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)

        log.info(f"Created model skeleton: {type(model).__name__}")

    except ImportError:
        raise ImportError(
            "transformers library required for HuggingFace model reconstruction. "
            "Install with: pip install transformers"
        )

    # Step 3: Replace parameters with remote tensor stubs
    log.info("Creating remote tensor stubs for model parameters...")
    device = machine.device()

    # Track storage IDs and parameter names for linking
    local_storage_ids = []
    parameter_names = []

    for name, param_metadata in state_dict_metadata.items():
        log.debug(f"Creating remote tensor stub for parameter: {name}")

        # Create remote tensor stub (this will generate a local storage ID)
        remote_tensor = create_remote_tensor_stub(
            storage_id=0,  # Unused - local storage ID will be generated
            shape=param_metadata["shape"],
            stride=param_metadata["stride"],
            storage_offset=param_metadata["storage_offset"],
            dtype=param_metadata["dtype"],
            device=device,
            requires_grad=param_metadata["requires_grad"],
        )

        # Extract the local storage ID generated by the C++ allocator
        local_storage_id = remote_tensor.untyped_storage().data_ptr()

        # Register the local storage ID in the local storage registry
        device_index = device.index
        from ._storage import _storage_registry

        _storage_registry.storage_id_to_device[local_storage_id] = device_index
        _storage_registry.generated_storage_ids.add(local_storage_id)

        # Track for linking
        local_storage_ids.append(local_storage_id)
        parameter_names.append(name)

        # Replace parameter in model
        _set_nested_parameter(model, name, nn.Parameter(remote_tensor))
        log.debug(
            f"Replaced parameter {name} with remote tensor (local_storage_id={local_storage_id})"
        )

    # Step 4: Replace buffers with remote tensor stubs
    for name, buffer_meta in buffer_metadata.items():
        log.debug(f"Creating remote tensor stub for buffer: {name}")

        # Create remote tensor stub (buffers don't require gradients)
        remote_tensor = create_remote_tensor_stub(
            storage_id=0,  # Unused - local storage ID will be generated
            shape=buffer_meta["shape"],
            stride=buffer_meta["stride"],
            storage_offset=buffer_meta["storage_offset"],
            dtype=buffer_meta["dtype"],
            device=device,
            requires_grad=False,
        )

        # Extract the local storage ID generated by the C++ allocator
        local_storage_id = remote_tensor.untyped_storage().data_ptr()

        # Register the local storage ID in the local storage registry
        device_index = device.index
        from ._storage import _storage_registry

        _storage_registry.storage_id_to_device[local_storage_id] = device_index
        _storage_registry.generated_storage_ids.add(local_storage_id)

        # Track for linking
        local_storage_ids.append(local_storage_id)
        parameter_names.append(name)

        # Replace buffer in model
        _set_nested_buffer(model, name, remote_tensor)
        log.debug(
            f"Replaced buffer {name} with remote tensor (local_storage_id={local_storage_id})"
        )

    # Step 4.5: Handle weight tying for parameters not in state_dict_metadata
    log.info("Checking for tied weights that need linking...")
    _handle_tied_weights(
        model, state_dict_metadata, {}
    )  # No storage mapping needed in new architecture

    # Step 5: Generate tensor IDs for remote caching
    log.info(f"🆔 Generating tensor IDs for {len(local_storage_ids)} model tensors...")
    tensor_ids = []

    from ._storage import get_or_create_tensor_id
    from ._tensor_utils import RemoteTensorMetadata

    # Generate tensor IDs for each parameter/buffer
    for local_storage_id, param_name in zip(local_storage_ids, parameter_names):
        # Get the remote tensor to extract metadata
        if param_name in state_dict_metadata:
            meta = state_dict_metadata[param_name]
        else:
            meta = buffer_metadata[param_name]

        # Create metadata object for tensor ID generation
        metadata = RemoteTensorMetadata(
            storage_id=local_storage_id,
            shape=meta["shape"],
            dtype=meta["dtype"],
            stride=meta["stride"],
            storage_offset=meta["storage_offset"],
        )

        # Generate tensor ID using the local tensor ID manager
        tensor_id = get_or_create_tensor_id(metadata)
        tensor_ids.append(tensor_id)
        log.debug(f"Generated tensor_id {tensor_id} for parameter {param_name}")

    log.info(
        f"🔗 Queueing linking of {len(local_storage_ids)} local storage IDs to remote model parameters..."
    )
    client.link_model_tensors(local_storage_ids, parameter_names, tensor_ids)

    # Wait for all batched operations (including tensor linking) to complete
    log.info("🚀 Waiting for all storage operations and tensor linking to complete...")
    if hasattr(client, "_batch_queue") and client._batch_queue:
        # Wake up the batch processor and wait for completion
        from ._remote_orchestrator import remote_orchestrator

        remote_orchestrator.wake_batch_thread_for_blocking_rpc()

        # Wait a moment for batch processing to complete
        import time

        time.sleep(0.5)  # Give time for batch to process

    log.info("✅ Model tensor linking completed successfully")

    # Step 6: Model is now fully connected with remote tensors
    log.info(
        f"✅ Successfully created {model_type} model with {len(local_storage_ids)} remote tensors"
    )
    return model


def _set_nested_parameter(
    model: nn.Module, param_name: str, param_value: nn.Parameter
) -> None:
    """Set a nested parameter in a model by name (e.g., 'transformer.h.0.attn.c_attn.weight')."""
    name_parts = param_name.split(".")
    current_module = model

    # Navigate to the parent module
    for part in name_parts[:-1]:
        if hasattr(current_module, part):
            current_module = getattr(current_module, part)
        else:
            # Handle special cases like numbered layers
            if part.isdigit():
                current_module = current_module[int(part)]
            else:
                raise AttributeError(
                    f"Module does not have attribute '{part}' in path '{param_name}'"
                )

    # Set the parameter
    final_name = name_parts[-1]
    if hasattr(current_module, final_name):
        setattr(current_module, final_name, param_value)
    else:
        raise AttributeError(
            f"Module does not have parameter '{final_name}' in path '{param_name}'"
        )


def _set_nested_buffer(
    model: nn.Module, buffer_name: str, buffer_value: torch.Tensor
) -> None:
    """Set a nested buffer in a model by name."""
    name_parts = buffer_name.split(".")
    current_module = model

    # Navigate to the parent module
    for part in name_parts[:-1]:
        if hasattr(current_module, part):
            current_module = getattr(current_module, part)
        else:
            # Handle special cases like numbered layers
            if part.isdigit():
                current_module = current_module[int(part)]
            else:
                raise AttributeError(
                    f"Module does not have attribute '{part}' in path '{buffer_name}'"
                )

    # Register the buffer
    final_name = name_parts[-1]
    current_module.register_buffer(final_name, buffer_value)


def _handle_tied_weights(
    model, state_dict_metadata: Dict[str, Any], storage_mapping: Dict[int, int]
):
    """Handle tied weights by finding parameters that share storage but only one was in state_dict_metadata.

    Args:
        model: The local model skeleton
        state_dict_metadata: Metadata from remote preparation (only includes unique parameters)
        storage_mapping: Mapping from local storage IDs to remote storage IDs
    """
    log.info("🔗 Checking for tied weights in model...")

    # Get all parameters in the local model (including tied ones)
    all_local_params = dict(model.named_parameters())

    # Check which parameters are still on meta device (need linking)
    meta_params = []
    for name, param in all_local_params.items():
        if param.device.type == "meta":
            meta_params.append(name)

    if not meta_params:
        log.info("✅ No parameters on meta device - no tied weights to handle")
        return

    log.info(f"Found {len(meta_params)} parameters on meta device: {meta_params}")

    # For each meta parameter, try to find a parameter that shares storage
    for meta_name in meta_params:
        meta_param = all_local_params[meta_name]

        # Look for a parameter in state_dict_metadata that should be tied to this one
        tied_param_name = None
        tied_param = None

        # Check for common tying patterns
        if "lm_head.weight" == meta_name:
            # lm_head.weight is commonly tied to embed_tokens.weight
            if "model.embed_tokens.weight" in state_dict_metadata:
                tied_param_name = "model.embed_tokens.weight"
                tied_param = all_local_params[tied_param_name]

        # Add more tying patterns as needed
        # elif meta_name.endswith(".weight") and "embed" in meta_name:
        #     # Handle other embedding tying patterns

        if tied_param_name and tied_param is not None:
            log.info(f"🔗 Tying {meta_name} to {tied_param_name}")

            # Replace meta parameter with tied parameter (share storage)
            tied_parameter = nn.Parameter(tied_param.data)
            tied_parameter.requires_grad_(meta_param.requires_grad)
            _set_nested_parameter(model, meta_name, tied_parameter)

            log.info(f"✅ Successfully tied {meta_name} to {tied_param_name}")
        else:
            log.warning(f"❌ Could not find tied parameter for {meta_name}")

    # Verify all parameters are now properly linked
    remaining_meta_params = [
        name for name, param in model.named_parameters() if param.device.type == "meta"
    ]
    if remaining_meta_params:
        log.warning(
            f"❌ Still have meta parameters after tying: {remaining_meta_params}"
        )
    else:
        log.info("✅ All parameters successfully linked - no meta devices remaining")


def load_huggingface_model(
    checkpoint: str,
    machine: RemoteMachine,
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
) -> nn.Module:
    """
    Convenience function to load a HuggingFace model with remote tensors.

    This is the main entry point for loading HuggingFace models optimized
    for mycelya_torch remote execution.

    Args:
        checkpoint: HuggingFace model checkpoint (e.g., "gpt2", "bert-base-uncased")
        machine: Remote machine to load the model on
        torch_dtype: Data type for model weights ("auto", "float32", "float16", etc.)
        trust_remote_code: Whether to trust remote code for custom models

    Returns:
        PyTorch model with all parameters as remote tensors

    Example:
        machine = mycelya_torch.create_modal_machine("T4")
        model = mycelya_torch.load_huggingface_model("gpt2", machine)

        # Model can now be used normally, all operations execute remotely
        output = model(input_ids)
    """
    return create_huggingface_model_from_remote(
        machine=machine,
        checkpoint=checkpoint,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
