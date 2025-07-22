# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Services module for torch-remote.

This module contains extracted services that handle specific concerns:
- TensorTransferService: Handles tensor serialization and device transfers
- StorageMachineResolver: Tracks storage-to-machine mappings

These services provide better separation of concerns and cleaner boundaries
compared to having all functionality mixed in the orchestrator.
"""

from .storage_resolver import StorageMachineResolver
from .tensor_transfer import TensorTransferService

__all__ = ["TensorTransferService", "StorageMachineResolver"]
