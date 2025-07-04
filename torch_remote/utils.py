import torch

def _remote(self, device=None, dtype=None, non_blocking=False, copy=False):
    """Move tensor to remote device."""
    if device is None:
        device = torch.device("remote", 0)
    elif isinstance(device, int):
        device = torch.device("remote", device)
    elif isinstance(device, str) and not device.startswith("remote"):
        device = torch.device("remote", 0)  # Default to remote:0 if just "remote"
    elif isinstance(device, str):
        device = torch.device(device)
    
    return self.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)

def _add_tensor_methods():
    """Add .remote() method to torch.Tensor."""
    torch.Tensor.remote = _remote