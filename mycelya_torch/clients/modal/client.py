# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Modal client implementation for mycelya_torch.

This module provides the ModalClient class for interfacing with Modal cloud GPUs,
along with related functionality for creating and managing Modal applications.
"""

from typing import Any

from ..._logging import get_logger
from ..._utils import TensorMetadata
from ...servers.modal.server import create_modal_app_for_gpu
from ..base_client import BatchCall, Client

log = get_logger(__name__)


class ModalClient(Client):
    """
    Client interface for Modal cloud GPU execution.

    This class provides a client-side interface to Modal's cloud GPU infrastructure,
    encapsulating Modal app management, server instances, and communication
    protocols while maintaining state and connection management.
    """

    def __init__(self, machine_id: str, timeout: int | None = None):
        super().__init__(machine_id)
        self._server_instance = None
        self._app_context = None
        self._timeout = timeout
        self._initial_packages: list[str] | None = None

    def start(
        self,
        gpu_type: str,
        gpu_count: int,
        packages: list[str],
        python_version: str,
    ):
        """Start the Modal app context for this machine."""
        if self._app_context is None:
            # First time starting - save initial packages and use them for app creation
            if self._initial_packages is None:
                self._initial_packages = packages.copy()
                extra_packages = []
            else:
                # Subsequent start - reuse initial packages for app, install extras separately
                extra_packages = [pkg for pkg in packages if pkg not in self._initial_packages]

            # Format gpu_type with count for Modal
            modal_gpu_type = f"{gpu_type}:{gpu_count}" if gpu_count > 1 else gpu_type

            # Create the Modal app and server class
            app, server_class = create_modal_app_for_gpu(
                machine_id=self.machine_id,
                gpu_type=modal_gpu_type,
                packages=self._initial_packages,
                python_version=python_version,
                timeout=self._timeout,
            )

            # Start the app context
            self._app_context = app.run()
            self._app_context.__enter__()
            # Create server instance when app starts
            self._server_instance = server_class()

            # Install extra packages if any
            if extra_packages:
                self.pip_install(extra_packages)

    def stop(self):
        """Stop the Modal app context for this machine."""
        if self._app_context is not None:
            try:
                # Call server's stop method synchronously to disable offload_on_exit for handling preemptions
                self._server_instance.stop.remote()
                self._app_context.__exit__(None, None, None)
            except Exception:
                # Silently ignore cleanup errors during atexit
                pass
            finally:
                self._app_context = None
                self._server_instance = None

    def get_rpc_result(self, rpc_result: Any, blocking: bool) -> Any | None:
        """Get the result from an RPC call."""
        # For Modal, rpc_result is a FunctionCall object
        try:
            timeout = None if blocking else 0
            return rpc_result.get(timeout=timeout)
        except TimeoutError:
            return None

    def execute_batch(self, batch_calls: list[BatchCall]) -> Any:
        """Execute a batch of operations via Modal."""
        return self._server_instance.execute_batch.spawn(batch_calls)

    # Tensor management methods
    def create_tensor(self, metadata: TensorMetadata) -> None:
        """Implementation: Create a tensor on the remote machine.

        Creates either a new empty tensor or a tensor view based on metadata.alias_id:
        - If alias_id is None: Creates new empty tensor
        - If alias_id is int: Creates tensor view using alias_id as base tensor
        """
        self._server_instance.create_tensor.spawn(metadata)

    def update_tensor(
        self,
        tensor_id: int,
        raw_data: bytes,
        source_shape: list[int],
        source_stride: list[int],
        source_storage_offset: int,
        source_dtype: str,
    ) -> None:
        """Implementation: Update an existing tensor with new data and source metadata."""
        # Call Modal method directly (fire-and-forget)
        self._server_instance.update_tensor.spawn(
            tensor_id,
            raw_data,
            source_shape,
            source_stride,
            source_storage_offset,
            source_dtype,
        )

    def get_storage_data(self, tensor_id: int) -> Any:
        """Implementation: Get raw storage data by tensor ID."""
        return self._server_instance.get_storage_data.spawn(tensor_id)

    def remove_tensors(self, tensor_ids: list[int]) -> None:
        """Implementation: Remove multiple tensors from the remote machine."""
        self._server_instance.remove_tensors.spawn(tensor_ids)

    def resize_storage(self, tensor_id: int, nbytes: int) -> None:
        """Implementation: Resize the underlying storage for a tensor."""
        self._server_instance.resize_storage.spawn(tensor_id, nbytes)

    def copy_tensor(
        self,
        source_tensor_id: int,
        target_tensor_id: int,
    ) -> None:
        """Implementation: Copy tensor data from source to target on the remote machine."""
        self._server_instance.copy_tensor.spawn(source_tensor_id, target_tensor_id)

    # Operation execution methods
    def execute_aten_operation(
        self,
        op_name: str,
        args: list[Any],
        kwargs: dict[str, Any],
        tensor_mask: list[bool],
        output_tensor_ids: list[int] | None = None,
    ) -> Any:
        """Implementation: Execute an aten operation on the remote machine with tensor IDs."""
        return self._server_instance.execute_aten_operation.spawn(
            op_name,
            args,
            kwargs,
            tensor_mask,
            output_tensor_ids,
        )

    def link_tensors(
        self,
        tensor_ids: list[int],
        temp_ids: list[str],
    ) -> None:
        """Implementation: Link local mycelya tensor IDs to remote tensors from temporary registry."""
        self._server_instance.link_tensors.spawn(tensor_ids, temp_ids)

    def execute_function(self, pickled_function: bytes) -> Any:
        """Implementation: Execute a pickled function remotely."""
        return self._server_instance.execute_function.spawn(pickled_function)

    def pip_install(self, packages: list[str]) -> None:
        """Implementation: Install packages using pip on the remote machine."""
        self._server_instance.pip_install.spawn(packages)

    def offload(self) -> None:
        """Implementation: Offload tensor registry to disk."""
        self._server_instance.offload.spawn()

    def reload(self) -> None:
        """Implementation: Reload tensor registry from disk."""
        self._server_instance.reload.spawn()
