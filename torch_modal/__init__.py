import torch
import torch_modal._C

# Register modal as the privateuse1 backend
torch.utils.rename_privateuse1_backend("modal")

# Import modal module
from . import modal

# Register the modal device module
torch._register_device_module('modal', modal)

# Generate methods for privateuse1 backend
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True, 
    for_module=True, 
    for_storage=True
)

# Add tensor methods
from .utils import _add_tensor_methods
_add_tensor_methods()

# Make modal available at package level
__all__ = ['modal']