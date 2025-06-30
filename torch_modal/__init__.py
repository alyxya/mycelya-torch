import torch

# Import modal module first (before C extension)
from . import modal

# Register modal as the privateuse1 backend
torch.utils.rename_privateuse1_backend("modal")

# Register the modal device module
torch._register_device_module('modal', modal)

# Generate methods for privateuse1 backend
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True, 
    for_module=True, 
    for_storage=True
)

# Import C extension after device setup
try:
    import torch_modal._C
    # Add tensor methods only if C extension loads successfully
    from .utils import _add_tensor_methods
    _add_tensor_methods()
    
    # Force initialization of modal device hooks
    try:
        # Trigger device initialization by checking if modal is available
        torch.device('modal:0')
    except Exception:
        pass  # This might fail but will trigger hooks initialization
        
except ImportError as e:
    print(f"Warning: C extension failed to load: {e}")

# Make modal available at package level
__all__ = ['modal']