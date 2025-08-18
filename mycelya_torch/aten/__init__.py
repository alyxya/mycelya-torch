# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ATen operation registrations for mycelya device backend.

This module contains only the PyTorch library registrations that connect
the ATen operations to their mycelya implementations. The actual operation
implementations are organized in separate modules by functionality.
"""

import torch

from .copy import _copy_from
from .dispatch import _remote_kernel_fallback
from .scalar import _equal, _local_scalar_dense
from .views import _set_source_tensor

# Register the fallback kernel for all unspecified operations
_remote_lib = torch.library.Library("_", "IMPL")
_remote_lib.fallback(_remote_kernel_fallback, dispatch_key="PrivateUse1")

# Register specific ATen operation implementations
_remote_lib_aten = torch.library.Library("aten", "IMPL")
_remote_lib_aten.impl("_copy_from", _copy_from, dispatch_key="PrivateUse1")
_remote_lib_aten.impl(
    "set_.source_Tensor", _set_source_tensor, dispatch_key="PrivateUse1"
)
_remote_lib_aten.impl(
    "_local_scalar_dense", _local_scalar_dense, dispatch_key="PrivateUse1"
)
_remote_lib_aten.impl("equal", _equal, dispatch_key="PrivateUse1")
