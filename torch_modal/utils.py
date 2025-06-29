import torch
import torch_modal._C

def _modal(self, device=None, dtype=None, non_blocking=False, copy=False):
    """Move tensor to modal device."""
    return torch_modal._C.modal(self, device, dtype, non_blocking, copy)

def _add_tensor_methods():
    """Add .modal() method to torch.Tensor."""
    torch.Tensor.modal = _modal