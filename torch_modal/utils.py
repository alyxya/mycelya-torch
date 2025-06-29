import torch
import torch_modal._C

def _modal(self, *args, **kwargs):
    """Move tensor to modal device."""
    return torch_modal._C.modal(self, *args, **kwargs)

def _add_tensor_methods():
    """Add .modal() method to torch.Tensor."""
    torch.Tensor.modal = _modal